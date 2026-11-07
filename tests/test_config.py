"""Smoke tests for config loading (no torch/GPU needed)."""

from fauxtography.config import load_config


def test_default_configs_load():
    for name in ("baseline", "clip", "dinov2"):
        cfg = load_config(name)
        assert cfg.name == name
        assert cfg.data.seed == 42


def test_baseline_defaults():
    cfg = load_config("baseline")
    assert cfg.arch == "resnet18"
    assert cfg.epochs == 12
    assert cfg.data.image_size == 224


def test_clip_prompts_nonempty():
    cfg = load_config("clip")
    assert all(len(p) > 0 for p in cfg.prompts_real)
    assert all(len(p) > 0 for p in cfg.prompts_fake)


def test_dinov2_defaults():
    cfg = load_config("dinov2")
    assert cfg.arch == "dinov2_vitb14"