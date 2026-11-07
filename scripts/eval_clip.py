#!/usr/bin/env python3
"""SKELETON — zero-shot CLIP evaluation on the test split.

Owner: CLIP collaborator. Produces results/clip/metrics.json,
confusion matrix figure, and the y_true/y_pred arrays for analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fauxtography.config import load_config
from fauxtography.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default="configs/clip.yaml", type=Path)
    args = parser.parse_args()

    cfg = load_config("clip", args.config)
    logger.info("TODO: implement CLIP zero-shot evaluation (config=%s)", args.config)
    raise NotImplementedError("CLIP milestone — see Docs/milestones.")


if __name__ == "__main__":
    main()