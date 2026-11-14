"""Tests for the cross-model fusion, compression and feature-embedding commands.

All hermetic: no HF download, no heavy model. Uses synthetic predictions /
cached features / tiny images.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from fauxtography import cli as c


def _dinov2_yaml(tmp_path):
    return {
        "name": "dinov2",
        "arch": "dinov2_vitb14",
        "feature_batch_size": 4,
        "probe_batch_size": 8,
        "probe_epochs": 1,
        "probe_lr": 0.01,
        "probe_weight_decay": 0.0,
        "num_workers": 0,
        "square_resize": False,
        "data": {
            "root": str(tmp_path / "data"),
            "image_size": 32,
            "num_workers": 0,
            "seed": 42,
        },
        "output": {
            "run_dir": str(tmp_path / "exp" / "runs"),
            "results_dir": str(tmp_path / "results"),
            "report_name": "metrics",
        },
    }


def _clear_env(monkeypatch):
    for var in (
        "FAUXTOGRAPHY_DATA_ROOT",
        "FAUXTOGRAPHY_RUN_DIR",
        "FAUXTOGRAPHY_RESULTS_DIR",
        "FAUXTOGRAPHY_HELDOUT_TEST",
        "FAUXTOGRAPHY_CROSS_DIR",
        "FAUXTOGRAPHY_RAW_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


def test_analyze_fusion_fused_accuracy(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    results = tmp_path / "results"
    n = 20
    y_true = np.tile([0, 1], n // 2)
    # baseline & dinov2 perfect, clip disagree on some samples.
    rng = np.random.default_rng(0)
    yp_base = y_true.copy()
    yp_dino = y_true.copy()
    yp_clip = y_true.copy()
    flip = rng.choice(n, 6, replace=False)
    yp_clip[flip] = 1 - yp_clip[flip]
    for m, yp in (("baseline", yp_base), ("dinov2", yp_dino), ("clip", yp_clip)):
        d = results / m
        d.mkdir(parents=True)
        np.save(d / "y_true.npy", y_true)
        np.save(d / "y_pred.npy", yp)
        np.save(d / "heldout_y_true.npy", y_true)
        np.save(d / "heldout_y_pred.npy", yp)

    class _NS:
        results_dir = str(results)

    assert c.cmd_analyze_fusion(_NS()) == 0
    out = results / "fusion"
    assert (out / "summary.json").is_file()
    assert (out / "summary.csv").is_file()
    rows = json.loads((out / "summary.json").read_text())
    assert len(rows) == 2  # internal + heldout
    assert rows[0]["split"] == "internal"
    # majority vote: clip imperfect but majority is correct on all samples.
    assert rows[0]["fused_accuracy"] == 1.0
    # clip alone is imperfect.
    assert rows[0]["model_accuracies"]["clip"] < 1.0


def test_visualize_features_writes_embedding(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    tmp = tmp_path
    cfg = _dinov2_yaml(tmp)
    cfg_path = tmp / "dinov2.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    feats_dir = tmp / "exp" / "runs" / "dinov2" / "features" / "canonical"
    feats_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    # 8-dim features, two well-separated clusters for fake vs real.
    x = np.vstack([rng.normal(-3, 0.5, (60, 8)), rng.normal(3, 0.5, (60, 8))])
    y = np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)])
    torch.save({"features": torch.tensor(x, dtype=torch.float32), "labels": torch.tensor(y)}, feats_dir / "train.pt")
    torch.save({"features": torch.tensor(x, dtype=torch.float32), "labels": torch.tensor(y)}, feats_dir / "cross_Midjourney.pt")
    torch.save({"features": torch.tensor(x, dtype=torch.float32), "labels": torch.tensor(y)}, feats_dir / "cross_BigGAN.pt")

    _NS = SimpleNamespace(config=cfg_path, generators=["Midjourney"], max_points=20)

    assert c.cmd_visualize_features(_NS) == 0
    out = tmp / "results" / "dinov2" / "embedding"
    assert (out / "embedding.svg").is_file()
    assert (out / "embedding_xy.npy").is_file()
    names = json.loads((out / "embedding_classes.json").read_text())["classes"]
    # 2 generators x 2 classes
    assert len([n for n in names if "Midjourney" in n]) == 2


def _make_rgb_images(folder: Path, n: int, color):
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (32, 32), color).save(folder / f"{i:03d}.jpg")


def test_probe_compression_recompresses_and_scores(tmp_path, monkeypatch):
    """Compression probe with the DINOv2 extractor mocked out."""
    import torch

    _clear_env(monkeypatch)
    tmp = tmp_path
    cfg = _dinov2_yaml(tmp)
    cfg_path = tmp / "dinov2.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Minimal probe checkpoint using the real LinearProbe key layout.
    from fauxtography.models.linear_probe import LinearProbe

    probe = LinearProbe(in_features=4, num_classes=2)
    probe_path = tmp / "exp" / "runs" / "dinov2"
    probe_path.mkdir(parents=True)
    torch.save({"probe_state_dict": probe.state_dict()}, probe_path / "probe.pt")

    # Source set: small real files, big fake files.
    src = tmp / "test"
    _make_rgb_images(src / "real", 4, (240, 240, 240))
    _make_rgb_images(src / "fake", 4, (10, 10, 10))

    # Mock the HF extractor with a fixed 4-dim [CLS]-like transform.
    class _FakeExt:
        hidden_size = 4

        def __init__(self, arch=None, device=None):
            pass

        def extract(self, images):
            # Deterministic 4-dim from the channel-mean gray level.
            mean = images.mean(dim=(-3, -2, -1))
            return torch.stack([mean, mean, mean, mean], dim=1)

    monkeypatch.setattr("fauxtography.models.dinov2.DINOv2FeatureExtractor", _FakeExt)

    class _NS:
        config = cfg_path
        source = str(src)
        target_bytes = 0  # real-class median

    assert c.cmd_probe_compression(_NS()) == 0
    out = tmp / "results" / "dinov2" / "compression"
    assert (out / "metrics.json").is_file()
    assert (out / "confusion_matrix.svg").is_file()
    assert (out / "y_pred.npy").is_file()
    data = json.loads((out / "metrics.json").read_text())
    assert data["n_samples"] == 8
    # Recompressed fakes should be pulled toward the real-class budget.
    after = data["median_bytes_after"]
    assert after["fake"] <= max(1200, after["real"] * 3)
