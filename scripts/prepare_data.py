#!/usr/bin/env python3
"""Build the canonical train/val/test split directory structure.

Joint infrastructure (shared, no model owner): used by both collaborators so
everyone evaluates on the exact same train/val/test images.

Input:
    --source-real DIR      folder of real photographs
    --source-fake DIR      folder of AI-generated images
Output (default data root):
    data/train/{real,fake}  data/val/{real,fake}  data/test/{real,fake}

Splits are stratified per class and reproducible via --seed; files are copied
so the source folders remain untouched.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from fauxtography.utils.logging import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")
CLASSES = ("real", "fake")


def _iter_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-real", required=True, type=Path)
    parser.add_argument("--source-fake", required=True, type=Path)
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", type=float, default=[0.7, 0.15, 0.15])
    args = parser.parse_args()

    assert len(args.splits) == 3 and abs(sum(args.splits) - 1.0) < 1e-6

    rng = random.Random(args.seed)
    root = args.data_root
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (root / split).mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        source = {"real": args.source_real, "fake": args.source_fake}[class_name]
        if not source.exists():
            raise FileNotFoundError(f"source folder {source} does not exist")
        files = _iter_images(source)
        rng.shuffle(files)
        n = len(files)
        counts = [int(round(n * f)) for f in args.splits[:2]]
        counts.append(n - sum(counts))
        logger.info("class=%s files=%d -> train=%d val=%d test=%d", class_name, n, *counts)

        offset = 0
        for split, count in zip(SPLITS, counts):
            dst = root / split / class_name
            dst.mkdir(parents=True, exist_ok=True)
            for f in files[offset : offset + count]:
                shutil.copy2(f, dst / f.name)
            offset += count

    logger.info("Done. Structure written under %s", root.resolve())


if __name__ == "__main__":
    main()