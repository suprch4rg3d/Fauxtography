#!/usr/bin/env python3
"""SKELETON — fine-tune the shared ResNet baseline, report loss/acc curves.

Milestone: baseline (joint/shared reference). Fills in the training loop and
saves checkpoints under <run_dir>/baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fauxtography.config import load_config
from fauxtography.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="configs/baseline.yaml", type=Path)
    args = parser.parse_args()

    cfg = load_config("baseline", args.config)
    logger.info("TODO: implement baseline training (config=%s)", args.config)
    raise NotImplementedError("Baseline training — see docs/milestones.")


if __name__ == "__main__":
    main()