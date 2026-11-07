#!/usr/bin/env python3
"""SKELETON — aggregate per-model metrics into one comparison + figures.

Joint step: consumes results/{baseline,clip,dinov2}/metrics.json and emits
results/comparison.csv plus the side-by-side figure used in the presentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fauxtography.utils.logging import get_logger

logger = get_logger(__name__)

MODELS = ("baseline", "clip", "dinov2")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__.strip().splitlines()[0])
    parser.add_argument("--results-dir", default="results", type=Path)
    args = parser.parse_args()
    logger.info("TODO: implement model comparison (results-dir=%s)", args.results_dir)
    raise NotImplementedError("Joint comparison — see Docs/milestones.")


if __name__ == "__main__":
    main()