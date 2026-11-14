"""Functional test for the cross-generator eval command.

Hermetic: exercises ``cmd_eval_cross`` end-to-end using a cached feature file
plus a minimal probe checkpoint, with the HF DINOv2 backbone mocked out so no
weights are downloaded and no GPU is needed.
"""

import json

import torch
import yaml

from fauxtography.cli import cmd_eval_cross, cmd_probe_file_size

# Minimum config needed by load_config("dinov2", ...): matching the real
# configs/dinov2.yaml keys so overrides apply cleanly.
MIN_YAML = {
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
        "root": "data",
        "image_size": 224,
        "num_workers": 0,
        "seed": 42,
    },
    "output": {
        "run_dir": "experiments/runs",
        "results_dir": "results",
        "report_name": "metrics",
    },
}


# Minimum baseline config (used by eval-cross --model baseline and the
# file-size probe).
def _baseline_yaml(tmp_path):
    return {
        "name": "baseline",
        "arch": "resnet18",
        "pretrained": False,
        "freeze_backbone": False,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.001,
        "weight_decay": 0.0,
        "warmup_epochs": 0,
        "data": {
            "root": str(tmp_path / "data"),
            "image_size": 32,
            "num_workers": 0,
            "seed": 42,
            "heldout_test": str(tmp_path / "heldout_test"),
            "cross_dir": str(tmp_path / "cross"),
        },
        "output": {
            "run_dir": str(tmp_path / "exp" / "runs"),
            "results_dir": str(tmp_path / "results"),
            "report_name": "metrics",
        },
    }


def _write_minimal_probe(path):
    """A LinearProbe-shaped checkpoint (weight + bias) loadable by eval-cross."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from fauxtography.models.linear_probe import LinearProbe

    probe = LinearProbe(in_features=3, num_classes=2)
    torch.save({"probe_state_dict": probe.state_dict()}, path)


def _seed_feature_cache(feats_dir, gen, x, y):
    feats_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"features": x, "labels": y}, feats_dir / f"cross_{gen}.pt")


def test_eval_cross_writes_per_generator_outputs(tmp_path, monkeypatch):
    cfg = dict(MIN_YAML)
    cfg["data"] = dict(MIN_YAML["data"])
    cfg["data"]["cross_dir"] = str(tmp_path / "cross")
    cfg["output"] = dict(MIN_YAML["output"])
    cfg["output"]["run_dir"] = str(tmp_path / "exp" / "runs")
    cfg["output"]["results_dir"] = str(tmp_path / "results")
    config_path = tmp_path / "dinov2.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    cross_root = tmp_path / "cross"
    for gen in ("Midjourney", "ADM", "BigGAN"):
        (cross_root / gen / "real").mkdir(parents=True)
        (cross_root / gen / "fake").mkdir(parents=True)

    run_dir = tmp_path / "exp" / "runs"
    results_dir = tmp_path / "results"
    variant = "canonical"  # square_resize: false
    feats_dir = run_dir / "dinov2" / "features" / variant

    _write_minimal_probe(run_dir / "dinov2" / "probe.pt")

    # Stable, perfectly-separable fake data (3-dim features) for each gen so
    # accuracy is deterministic (perfect) and only the cached-features path
    # runs (the DINOv2 extractor is never reached thanks to the cache).
    x = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y = torch.tensor([0, 1])
    for gen in ("Midjourney", "ADM", "BigGAN"):
        _seed_feature_cache(feats_dir, gen, x, y)

    # Keep load_config from touching real env overrides that may be set.
    for var in (
        "FAUXTOGRAPHY_DATA_ROOT",
        "FAUXTOGRAPHY_RUN_DIR",
        "FAUXTOGRAPHY_RESULTS_DIR",
        "FAUXTOGRAPHY_HELDOUT_TEST",
        "FAUXTOGRAPHY_CROSS_DIR",
        "FAUXTOGRAPHY_RAW_DIR",
    ):
        monkeypatch.delenv(var, raising=False)

    class _NS:
        config = config_path
        model = "dinov2"

    rc = cmd_eval_cross(_NS())
    assert rc == 0

    for gen in ("Midjourney", "ADM", "BigGAN"):
        out = results_dir / "dinov2" / "cross" / gen
        assert (out / "metrics.json").is_file(), f"missing metrics for {gen}"
        assert (out / "confusion_matrix.svg").is_file()
        assert (out / "y_pred.npy").is_file()
        assert (out / "y_true.npy").is_file()
        data = json.loads((out / "metrics.json").read_text())
        assert data["generator"] == gen
        assert data["variant"] == variant
        assert data["model"] == "dinov2"
        assert data["n_samples"] == 2

def _clear_dataloader_env(monkeypatch):
    """Neutralize host env vars that would redirect config paths."""
    for var in (
        "FAUXTOGRAPHY_DATA_ROOT",
        "FAUXTOGRAPHY_RUN_DIR",
        "FAUXTOGRAPHY_RESULTS_DIR",
        "FAUXTOGRAPHY_HELDOUT_TEST",
        "FAUXTOGRAPHY_CROSS_DIR",
        "FAUXTOGRAPHY_RAW_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


def test_eval_cross_baseline_writes_outputs(tmp_path, monkeypatch):
    """ResNet branch: real tiny checkpoint + real tiny images, no HF needed."""
    from PIL import Image

    cfg_yaml = _baseline_yaml(tmp_path)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(cfg_yaml))

    data_root = tmp_path / "data"
    for cls in ("real", "fake"):
        folder = data_root / "train" / cls
        folder.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (32, 32), (250, 250, 250) if cls == "real" else (5, 5, 5)).save(
                folder / f"{cls}{i}.jpg"
            )
    (tmp_path / "heldout_test" / "real").mkdir(parents=True)
    (tmp_path / "heldout_test" / "fake").mkdir(parents=True)
    for gen in ("Midjourney", "ADM"):
        for cls in ("real", "fake"):
            folder = tmp_path / "cross" / gen / cls
            folder.mkdir(parents=True)
            color = (250, 250, 250) if cls == "real" else (5, 5, 5)
            for i in range(2):
                Image.new("RGB", (32, 32), color).save(folder / f"{gen}{cls}{i}.jpg")

    # A trained checkpoint: one epoch against the trivially separable images.
    ckpt_dir = tmp_path / "exp" / "runs" / "baseline"
    ckpt_dir.mkdir(parents=True)
    _write_minimal(ckpt_dir / "baseline_best.pt")

    class _NS:
        config = config_path
        model = "baseline"

    rc = cmd_eval_cross(_NS())
    assert rc == 0
    out = tmp_path / "results" / "baseline" / "cross"
    for gen in ("Midjourney", "ADM"):
        assert (out / gen / "metrics.json").is_file(), gen
        assert (out / gen / "confusion_matrix.svg").is_file()
        data = json.loads((out / gen / "metrics.json").read_text())
        assert data["model"] == "baseline"
        assert data["generator"] == gen


def _write_minimal(path):
    """Baseline-shaped checkpoint (model_state_dict) with the arch key."""
    from fauxtography.models.baseline import build_baseline_model

    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_baseline_model(
        arch="resnet18", pretrained=False, freeze_backbone=False, num_classes=2, device="cpu"
    )
    torch.save({"arch": "resnet18", "model_state_dict": model.state_dict()}, path)


def test_probe_file_size_measures_shortcut(tmp_path, monkeypatch):
    """File-size probe: tiny real < huge fake => threshold near 100% acc."""
    cfg_yaml = _baseline_yaml(tmp_path)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(cfg_yaml))

    data_root = tmp_path / "data"
    for cls, size in (("real", 512), ("fake", 4096)):
        folder = data_root / "train" / cls
        folder.mkdir(parents=True)
        for i in range(4):
            path = folder / f"img{i}.bin"
            with open(path, "wb") as fh:
                fh.write(b"\0" * size)
    for cls, size in (("real", 512), ("fake", 4096)):
        path = tmp_path / "heldout_test" / cls
        path.mkdir(parents=True)
        for i in range(4):
            with open(path / f"img{i}.bin", "wb") as fh:
                fh.write(b"\0" * size)

    class _NS:
        config = config_path

    rc = cmd_probe_file_size(_NS())
    assert rc == 0
    out = tmp_path / "results" / "file_size"
    assert (out / "metrics.json").is_file()
    assert (out / "size_histogram.svg").is_file()
    data = json.loads((out / "metrics.json").read_text())
    # Perfectly separated by file size -> accuracy 1.0 on both sets.
    assert data["train_accuracy"] == 1.0
    assert data["sets"][0]["accuracy"] == 1.0  # train set
    heldout_set = [s for s in data["sets"] if s["name"] == "heldout_test"]
    assert heldout_set and heldout_set[0]["accuracy"] == 1.0
