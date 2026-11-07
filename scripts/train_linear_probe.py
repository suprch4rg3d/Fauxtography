#!/usr/bin/env python3
"""SKELETON — train and evaluate a linear probe on cached DINOv2 features.

Owner: DINOv2 collaborator. Produces results/dinov2/metrics.json, confusion
matrix, and y_true/y_pred arrays for analysis.
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
    logger.info("TODO: implement linear probe training (config=%s)", args.config)
    raise NotImplementedError("DINOv2 milestone — see Docs/milestones.")


if __name__ == "__main__":
    main()