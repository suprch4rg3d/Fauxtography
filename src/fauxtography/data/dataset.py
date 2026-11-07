"""Dataset handling (shared infra).

Standard class-folder layout created by `fauxtography prepare-data`:

    data/
      train/real/*  train/fake/*
      val/real/*    val/fake/*
      test/real/*   test/fake/*

Contracts:
    CLASS_NAMES  = ("real", "fake")        stable ordering
    REAL_IDX = 0, FAKE_IDX = 1
    FauxtoDataset(root, transform)         labels follow REAL_IDX/FAKE_IDX
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

# Canonical class order across the whole project.
CLASS_NAMES = ("real", "fake")
REAL_IDX, FAKE_IDX = 0, 1


class FauxtoDataset(Dataset):
    """ImageFolder wrapper exposing the stable class ordering (real=0, fake=1)."""

    def __init__(self, root: str | Path, transform=None) -> None:
        self.folder = ImageFolder(root=str(root), transform=transform)
        self._target_map = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.folder)

    def __getitem__(self, index: int):
        img, _cls_id = self.folder[index]
        class_name = self.folder.classes[_cls_id]
        return img, self._target_map[class_name]

    @property
    def classes(self) -> tuple[str, ...]:
        return CLASS_NAMES


def get_datasets(data_cfg, train_transform, eval_transform):
    """Return ``{train, val, test}`` datasets; raise if a split is missing."""
    root = Path(data_cfg.root)
    splits: dict[str, FauxtoDataset] = {}
    for split in ("train", "val", "test"):
        folder = root / split
        if folder.exists() and any(folder.iterdir()):
            transform = train_transform if split == "train" else eval_transform
            splits[split] = FauxtoDataset(folder, transform=transform)
    missing = {"train", "val", "test"} - set(splits)
    if missing:
        raise FileNotFoundError(
            f"Data root {root} is missing split(s) {sorted(missing)}. "
            "Run `fauxtography prepare-data --help` first."
        )
    return splits