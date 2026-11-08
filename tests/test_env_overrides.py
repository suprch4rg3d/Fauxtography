"""Env-var path overrides and device resolution (Colab-related)."""

import torch

from fauxtography.config import load_config
from fauxtography.utils.device import device_report, get_device


def test_path_override_env_vars(monkeypatch):
    monkeypatch.setenv("FAUXTOGRAPHY_DATA_ROOT", "/content/data")
    monkeypatch.setenv("FAUXTOGRAPHY_RUN_DIR", "/content/exp/runs")
    monkeypatch.setenv("FAUXTOGRAPHY_RESULTS_DIR", "/content/exp/results")

    for name in ("baseline", "clip", "dinov2"):
        cfg = load_config(name)
        assert cfg.data.root == "/content/data"
        assert cfg.output.run_dir == "/content/exp/runs"
        assert cfg.output.results_dir == "/content/exp/results"


def test_heldout_test_env_override(monkeypatch):
    monkeypatch.setenv(
        "FAUXTOGRAPHY_HELDOUT_TEST", "/content/drive/MyDrive/fauxtography/data/raw/test"
    )
    for name in ("baseline", "clip", "dinov2"):
        cfg = load_config(name)
        assert cfg.data.heldout_test == "/content/drive/MyDrive/fauxtography/data/raw/test"


def test_heldout_test_absent_env_uses_yaml(monkeypatch):
    monkeypatch.delenv("FAUXTOGRAPHY_HELDOUT_TEST", raising=False)
    cfg = load_config("baseline")
    assert cfg.data.heldout_test == "data/raw/test"  # from configs/baseline.yaml


def test_path_override_absent_env_vars(monkeypatch):
    monkeypatch.delenv("FAUXTOGRAPHY_DATA_ROOT", raising=False)
    cfg = load_config("baseline")
    assert cfg.data.root == "data"  # untouched default


def test_get_device_auto_returns_torch_device():
    dev = get_device("auto")
    assert isinstance(dev, torch.device)


def test_get_device_forces_explicit_choice():
    assert str(get_device("cpu")) == "cpu"


def test_device_report_shape():
    report = device_report()
    for key in ("python", "torch", "device", "cuda_available", "mps_available"):
        assert key in report
