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


This is the only entry point - the old per-step ``scripts/*.py`` header
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
    """Fine-tune the ResNet baseline on real/fake and write metrics."""
    import json

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from .config import load_config
    from .data.dataset import FauxtoDataset
    from .data.transforms import baseline_transforms
    from .evaluate.metrics import classification_report, confusion_matrix_data, evaluate_predictions
    from .evaluate.visual import plot_confusion_matrix
    from .models.baseline import build_baseline_model
    from .utils.device import get_device
    from .utils.seed import set_seed

    cfg = load_config("baseline", args.config)
    set_seed(cfg.data.seed)
    device = get_device()

    tf = baseline_transforms(cfg.data.image_size)
    splits = {
        split: FauxtoDataset(
            Path(cfg.data.root) / split,
            transform=tf.train_transform if split == "train" else tf.eval_transform,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.data.num_workers,
            pin_memory=device.type == "cuda",
        )
        for split, ds in splits.items()
    }

    model = build_baseline_model(
        arch=cfg.arch,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
        num_classes=2,
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    def run_epoch(train: bool):
        model.train(train)
        total = correct = running = 0
        for images, labels in loaders["train"] if train else loaders["val"]:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                logits = model(images)
                loss = criterion(logits, labels)
                if train:
                    loss.backward()
                    optimizer.step()
            running += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return running / total, correct / total

    # ---- Training: per-epoch + best-val checkpoints, restore best at end ----
    out = Path(cfg.output.run_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_state = None

    logger.info(
        "Baseline %s on %s (%s)",
        cfg.arch,
        device,
        "frozen backbone" if cfg.freeze_backbone else "full fine-tune",
    )
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(train=True)
        scheduler.step()
        val_loss, val_acc = run_epoch(train=False)
        logger.info(
            "epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            tr_loss,
            tr_acc,
            val_loss,
            val_acc,
        )
        # Per-epoch checkpoint (keeps the most recent, so a Colab disconnect
        # loses at most one epoch of progress).
        torch.save(
            {"arch": cfg.arch, "model_state_dict": model.state_dict(), "epoch": epoch},
            out / "baseline_latest.pt",
        )
        # Track the best validation model and keep its weights.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(
                {"arch": cfg.arch, "model_state_dict": best_state, "epoch": epoch},
                out / "baseline_best.pt",
            )

    def _save_compressed(path: Path, payload: dict) -> None:
        """Save a .pt checkpoint plus a gzipped .pt.gz for easy sharing."""
        torch.save(payload, path)
        # Small helper: gzip the file on disk so the shared copy is smaller.
        import gzip
        import shutil

        gz = path.with_suffix(path.suffix + ".gz")
        with open(path, "rb") as fsrc, gzip.open(gz, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)

    # Restore the best-val weights for final evaluation and reporting.
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("restored best-val model (val_acc=%.4f)", best_val_acc)
    if best_state is not None:
        _save_compressed(
            out / "baseline_best.pt", {"arch": cfg.arch, "model_state_dict": best_state}
        )
    _save_compressed(
        out / "baseline.pt", {"arch": cfg.arch, "model_state_dict": model.state_dict()}
    )

    # ---- Evaluation on the internal test split -----------------------------
    def _predict(loader):
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                yt.extend(labels.tolist())
                yp.extend(logits.argmax(1).cpu().tolist())
        return np.asarray(yt), np.asarray(yp)

    import numpy as np

    results_dir = Path(cfg.output.results_dir) / cfg.name
    results_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = _predict(loaders["test"])
    metrics = evaluate_predictions(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix_data(y_true, y_pred)
    (results_dir / "metrics.json").write_text(
        json.dumps({"accuracy": metrics["accuracy"], "classification_report": report}, indent=2)
    )
    np.save(results_dir / "y_true.npy", y_true)
    np.save(results_dir / "y_pred.npy", y_pred)
    plot_confusion_matrix(
        cm, results_dir / "confusion_matrix.svg", title=f"{cfg.name} - internal split"
    )
    logger.info("internal test accuracy=%.4f -> %s", metrics["accuracy"], results_dir)

    # ---- Optional evaluation on the official held-out test ----------------
    heldout = cfg.data.heldout_test
    if heldout:
        heldout_root = Path(heldout)
        ds = FauxtoDataset(heldout_root, transform=tf.eval_transform)
        heldout_loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=device.type == "cuda",
        )
        hy_true, hy_pred = _predict(heldout_loader)
        h_metrics = evaluate_predictions(hy_true, hy_pred)
        h_report = classification_report(hy_true, hy_pred)
        h_cm = confusion_matrix_data(hy_true, hy_pred)
        (results_dir / "heldout_metrics.json").write_text(
            json.dumps(
                {
                    "accuracy": h_metrics["accuracy"],
                    "n_samples": h_metrics["n_samples"],
                    "classification_report": h_report,
                },
                indent=2,
            )
        )
        np.save(results_dir / "heldout_y_true.npy", hy_true)
        np.save(results_dir / "heldout_y_pred.npy", hy_pred)
        plot_confusion_matrix(
            h_cm,
            results_dir / "heldout_confusion_matrix.svg",
            title=f"{cfg.name} - official held-out test",
        )
        logger.info(
            "held-out test accuracy=%.4f (n=%d) -> %s",
            h_metrics["accuracy"],
            h_metrics["n_samples"],
            results_dir,
        )

    return 0


def cmd_eval_clip(args: argparse.Namespace) -> int:
    from .config import load_config

    load_config("clip", args.config)  # validate+warm the config at call time
    logger.info("TODO: implement CLIP zero-shot evaluation (config=%s)", args.config)
    raise NotImplementedError("CLIP milestone - see docs/milestones.")


def cmd_extract_dinov2(args: argparse.Namespace) -> int:
    from .config import load_config

    load_config("dinov2", args.config)  # validate+warm the config at call time
    logger.info("TODO: implement DINOv2 feature extraction (config=%s)", args.config)
    raise NotImplementedError("DINOv2 milestone - see docs/milestones.")


def cmd_train_linear_probe(args: argparse.Namespace) -> int:
    from .config import load_config

    load_config("dinov2", args.config)  # validate+warm the config at call time
    logger.info("TODO: implement linear probe training (config=%s)", args.config)
    raise NotImplementedError("DINOv2 milestone - see docs/milestones.")


def cmd_compare(args: argparse.Namespace) -> int:
    """Aggregate each model's metrics.json into one comparison CSV/report."""
    import csv
    import json

    results_dir = Path(args.results_dir)
    rows = []
    for name in ("baseline", "clip", "dinov2"):
        path = results_dir / name / "metrics.json"
        if not path.exists():
            logger.warning("missing %s - run its experiment first", path)
            continue
        data = json.loads(path.read_text())
        report = data.get("classification_report", {})
        rows.append(
            {
                "model": name,
                "accuracy": round(float(data.get("accuracy", 0.0)), 4),
                "f1_real": round(float(report.get("real", {}).get("f1-score", 0.0)), 4),
                "f1_fake": round(float(report.get("fake", {}).get("f1-score", 0.0)), 4),
                "macro_f1": round(float(report.get("macro avg", {}).get("f1-score", 0.0)), 4),
            }
        )

    if not rows:
        raise SystemExit(f"no metrics found under {results_dir}")

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "comparison.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        logger.info(
            "model=%-8s accuracy=%.4f macro_f1=%.4f", row["model"], row["accuracy"], row["macro_f1"]
        )
    logger.info("wrote %s", results_dir / "comparison.csv")
    return 0


def cmd_fetch_genimage(args: argparse.Namespace) -> int:
    """Download a size-bounded GenImage slice into data/raw/{real,fake}."""
    from .config import load_config
    from .data.download_genimage import fetch_genimage

    cfg = load_config("genimage", args.config)
    if cfg.train_rows_per_class <= 0 and not cfg.cross_generator_rows:
        logger.warning(
            "nothing configured to fetch - set train_rows_per_class or cross_generator_rows"
        )
    report = fetch_genimage(cfg)
    if report:
        logger.info("done:\n %s", report)
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    """Print the runtime/environment report (tensor stack + accelerator).

    The first thing to run in a Colab session - confirms whether a GPU is
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
        description="Detecting AI-Generated Images - distinguish real from fake.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("prepare-data", help="build train/val/test split dirs")
    p.add_argument("--source-real", required=True, type=Path, help="folder with real photographs")
    p.add_argument(
        "--source-fake", required=True, type=Path, help="folder with AI-generated images"
    )
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

    p = sub.add_parser("fetch-genimage", help="download a size-bounded GenImage slice")
    p.add_argument("--config", default="configs/genimage.yaml", type=Path)
    p.set_defaults(func=cmd_fetch_genimage)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``fauxtography`` console script."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
