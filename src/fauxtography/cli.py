"""Command-line interface for fauxtography.

Exposes every pipeline step as a subcommand so it can be run through Poetry's
console script:

    poetry run fauxtography prepare-data --source-real R --source-fake F
    poetry run fauxtography train-baseline
    poetry run fauxtography eval-clip
    poetry run fauxtography extract-dinov2
    poetry run fauxtography train-linear-probe
    poetry run fauxtography compare
    poetry run fauxtography env

Use ``fauxtography env`` first in a Colab session to confirm the GPU/accelerator
is available before running any training command.

Path override env vars (set once, redirects every command):
    FAUXTOGRAPHY_DATA_ROOT, FAUXTOGRAPHY_RUN_DIR, FAUXTOGRAPHY_RESULTS_DIR
(see fauxtography.config).


This is the only entry point — the old per-step ``scripts/*.py`` header
remainders have been folded into it.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from .utils.logging import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")
CLASSES = ("real", "fake")

# Default relative config locations (relative to CWD, like the old scripts).
DEFAULT_CONFIG = {
    "train-baseline": "configs/baseline.yaml",
    "eval-clip": "configs/clip.yaml",
    "extract-dinov2": "configs/dinov2.yaml",
    "train-linear-probe": "configs/dinov2.yaml",
}


# --------------------------------------------------------------------------- #
# Command implementations
# --------------------------------------------------------------------------- #
def _iter_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file())


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Build the canonical train/val/test split directory structure."""
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
        counts = [round(n * fraction) for fraction in args.splits[:2]]
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
    return 0


def cmd_train_baseline(args: argparse.Namespace) -> int:
    from .config import load_config

    load_config("baseline", args.config)  # validate+warm the config at call time
    logger.info("TODO: implement baseline training (config=%s)", args.config)
    raise NotImplementedError("Baseline training — see docs/milestones.")


def cmd_eval_clip(args: argparse.Namespace) -> int:
    from .config import load_config

    load_config("clip", args.config)  # validate+warm the config at call time
    logger.info("TODO: implement CLIP zero-shot evaluation (config=%s)", args.config)
    raise NotImplementedError("CLIP milestone — see docs/milestones.")


def cmd_extract_dinov2(args: argparse.Namespace) -> int:
    from .config import load_config

    load_config("dinov2", args.config)  # validate+warm the config at call time
    logger.info("TODO: implement DINOv2 feature extraction (config=%s)", args.config)
    raise NotImplementedError("DINOv2 milestone — see docs/milestones.")


def cmd_train_linear_probe(args: argparse.Namespace) -> int:
    from .config import load_config

    load_config("dinov2", args.config)  # validate+warm the config at call time
    logger.info("TODO: implement linear probe training (config=%s)", args.config)
    raise NotImplementedError("DINOv2 milestone — see docs/milestones.")


def cmd_compare(args: argparse.Namespace) -> int:
    logger.info("TODO: implement model comparison (results-dir=%s)", args.results_dir)
    raise NotImplementedError("Joint comparison — see docs/milestones.")


def cmd_env(args: argparse.Namespace) -> int:
    """Print the runtime/environment report (tensor stack + accelerator).

    The first thing to run in a Colab session — confirms whether a GPU is
    actually attached before spending time on training.
    """
    from .utils.device import device_report

    for key, value in device_report().items():
        print(f"{key:>16}: {value}")
    return 0


# --------------------------------------------------------------------------- #
# Parser + dispatch
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fauxtography",
        description="Detect real photographs vs AI-generated images.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("prepare-data", help="build train/val/test split dirs")
    p.add_argument("--source-real", required=True, type=Path, help="folder with real photographs")
    p.add_argument("--source-fake", required=True, type=Path, help="folder with AI-generated images")
    p.add_argument("--data-root", default="data", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", nargs="+", type=float, default=[0.7, 0.15, 0.15])
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("train-baseline", help="fine-tune the ResNet baseline")
    p.add_argument("--config", default=DEFAULT_CONFIG["train-baseline"], type=Path)
    p.set_defaults(func=cmd_train_baseline)

    p = sub.add_parser("eval-clip", help="zero-shot CLIP evaluation")
    p.add_argument("--config", default=DEFAULT_CONFIG["eval-clip"], type=Path)
    p.set_defaults(func=cmd_eval_clip)

    p = sub.add_parser("extract-dinov2", help="cache DINOv2 features")
    p.add_argument("--config", default=DEFAULT_CONFIG["extract-dinov2"], type=Path)
    p.set_defaults(func=cmd_extract_dinov2)

    p = sub.add_parser("train-linear-probe", help="train linear probe on DINOv2 features")
    p.add_argument("--config", default=DEFAULT_CONFIG["train-linear-probe"], type=Path)
    p.set_defaults(func=cmd_train_linear_probe)

    p = sub.add_parser("compare", help="aggregate per-model metrics")
    p.add_argument("--results-dir", default="results", type=Path)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("env", help="print environment / accelerator report")
    p.set_defaults(func=cmd_env)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``fauxtography`` console script."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())