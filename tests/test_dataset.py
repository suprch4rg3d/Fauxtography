"""Tests for the canonical folder layout and class ordering."""

from pathlib import Path

from PIL import Image
from torchvision.transforms import ToTensor

from fauxtography.data.dataset import CLASS_NAMES, FAKE_IDX, REAL_IDX, FauxtoDataset


def _make_tree(root: Path):
    for cls in CLASS_NAMES:
        p = root / "train" / cls
        p.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color="red").save(p / "a.png")
        Image.new("RGB", (8, 8), color="blue").save(p / "b.png")


def test_class_order_is_stable(tmp_path):
    _make_tree(tmp_path)
    ds = FauxtoDataset(tmp_path / "train", transform=ToTensor())
    assert ds.classes == ("real", "fake")
    # labels must be 0 for real, 1 for fake regardless of folder ordering
    for i in range(len(ds)):
        img, label = ds[i]
        path = ds.folder.samples[i][0]
        if "real" in str(path):
            assert label == REAL_IDX
        else:
            assert label == FAKE_IDX