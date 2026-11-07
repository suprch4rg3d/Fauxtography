#!/usr/bin/env python3
"""SKELETON — extract DINOv2 [CLS] features for train/val/test; cache to disk.

Owner: DINOv2 collaborator. Outputs under runs/dinov2/features/<split>.pt so
the linear probe can be trained independently and cheaply.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fauxtography.config import load_config
from fauxtography.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="configs/dinov2.yaml", type=Path)
    args = parser.parse_args()

    cfg = load_config("dinov2", args.config)
    logger.info("TODO: implement DINOv2 feature extraction (config=%s)", args.config)
    raise NotImplementedError("DINOv2 milestone — see docs/milestones.")


if __name__ == "__main__":
    main()