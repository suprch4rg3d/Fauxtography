"""Functional test for the cross-generator eval command.

Hermetic: exercises ``cmd_eval_cross`` end-to-end using a cached feature file
plus a minimal probe checkpoint, with the HF DINOv2 backbone mocked out so no
weights are downloaded and no GPU is needed.
"""

import json

import torch
import yaml

from fauxtography.cli import cmd_eval_cross

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
        assert data["preprocess"] == variant
        assert data["n_samples"] == 2