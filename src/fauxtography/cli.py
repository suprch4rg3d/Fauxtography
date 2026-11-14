"""Command-line interface for fauxtography.

Exposes every pipeline step as a subcommand so it can be run through Poetry's
console script:

    poetry run fauxtography prepare-data --source-real R --source-fake F
    poetry run fauxtography train-baseline
    poetry run fauxtography eval-clip
    poetry run fauxtography extract-dinov2
    poetry run fauxtography train-linear-probe
    poetry run fauxtography eval-cross --model dinov2|baseline
    poetry run fauxtography probe-file-size
    poetry run fauxtography probe-compression
    poetry run fauxtography visualize-features
    poetry run fauxtography analyze-fusion
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

import numpy as np

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
    "eval-cross": "configs/dinov2.yaml",
    "probe-file-size": "configs/baseline.yaml",
    "probe-compression": "configs/dinov2.yaml",
    "visualize-features": "configs/dinov2.yaml",
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
    """Cache frozen DINOv2 [CLS] features for train/val/test (+ heldout)."""
    import json

    import torch
    from torch.utils.data import DataLoader

    from .config import load_config
    from .data.dataset import FauxtoDataset, get_datasets
    from .data.transforms import dinov2_transforms
    from .models.dinov2 import DINOv2FeatureExtractor
    from .utils.device import get_device
    from .utils.seed import set_seed

    cfg = load_config("dinov2", args.config)
    set_seed(cfg.data.seed)
    device = get_device()

    tf = dinov2_transforms(cfg.data.image_size, square=cfg.square_resize)
    splits = get_datasets(cfg.data, tf.train_transform, tf.eval_transform)

    heldout_root = cfg.data.heldout_test
    if heldout_root:
        splits["heldout"] = FauxtoDataset(Path(heldout_root), transform=tf.eval_transform)

    extractor = DINOv2FeatureExtractor(arch=cfg.arch, device=device)
    logger.info(
        "extracting with %s on %s (%d-dim features, square_resize=%s)",
        cfg.arch,
        device,
        extractor.hidden_size,
        cfg.square_resize,
    )

    # Variant-keyed cache so square vs canonical preproc never reuse each
    # other's features (and re-running with a tweak re-extracts).
    variant = "square" if cfg.square_resize else "canonical"
    out = Path(cfg.output.run_dir) / "dinov2" / "features" / variant
    out.mkdir(parents=True, exist_ok=True)

    cache_path = out / "metadata.json"
    with open(cache_path, "w") as fh:
        json.dump(
            {
                "arch": cfg.arch,
                "image_size": cfg.data.image_size,
                "preprocess": variant,
                "square_resize": cfg.square_resize,
                "pretrained": True,
                "note": "frozen DINOv2 [CLS] features; per-split .pt files next to this",
            },
            fh,
            indent=2,
        )

    for split, ds in splits.items():
        feats_path = out / f"{split}.pt"
        if feats_path.exists():
            logger.info("skip %s - features already cached (%s)", split, feats_path)
            continue
        loader = DataLoader(
            ds,
            batch_size=cfg.feature_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        feats, labels = [], []
        for images, _labels in loader:
            feats.append(extractor.extract(images).cpu())
            labels.append(_labels)
        torch.save(
            {"features": torch.cat(feats), "labels": torch.cat(labels)},
            feats_path,
        )
        logger.info("cached %s features (%d x %d)", split, len(ds), extractor.hidden_size)
    logger.info("features under %s", out)
    return 0


def cmd_train_linear_probe(args: argparse.Namespace) -> int:
    """Train a linear head on cached DINOv2 features and write metrics."""
    import json

    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from .config import load_config
    from .evaluate.metrics import classification_report, confusion_matrix_data, evaluate_predictions
    from .evaluate.visual import plot_confusion_matrix
    from .models.linear_probe import LinearProbe
    from .utils.device import get_device
    from .utils.seed import set_seed

    cfg = load_config("dinov2", args.config)
    set_seed(cfg.data.seed)
    device = get_device()

    variant = "square" if cfg.square_resize else "canonical"
    feats_dir = Path(cfg.output.run_dir) / "dinov2" / "features" / variant

    def _load(split: str):
        path = feats_dir / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"no cached features at {path} - run extract-dinov2 first")
        data = torch.load(path, map_location="cpu", weights_only=True)
        return data["features"], data["labels"].long()

    train_x, train_y = _load("train")
    val_x, val_y = _load("val")

    probe = LinearProbe(train_x.shape[1], num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.probe_epochs)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=cfg.probe_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=cfg.probe_batch_size)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, cfg.probe_epochs + 1):
        probe.train()
        total = correct = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(feats), labels)
            loss.backward()
            optimizer.step()
            correct += (probe(feats).argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        probe.eval()
        with torch.no_grad():
            v_correct = sum(
                (probe(f.to(device)).argmax(1) == l.to(device)).sum().item() for f, l in val_loader
            )
        val_acc = v_correct / len(val_x)
        logger.info(
            "probe epoch=%d train_acc=%.4f val_acc=%.4f",
            epoch,
            correct / total,
            val_acc,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
        logger.info("restored best-val probe (val_acc=%.4f)", best_val_acc)

    probe_path = Path(cfg.output.run_dir) / "dinov2" / "probe.pt"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"probe_state_dict": probe.state_dict()}, probe_path)
    logger.info("saved probe -> %s", probe_path)

    def _evaluate(split: str, title: str, prefix: str):
        """Evaluate cached features for a split; ``prefix`` is "" (internal
        test) or "heldout_" (official held-out test) to mirror baseline."""
        x, y = _load(split)
        probe.eval()
        with torch.no_grad():
            logits = probe(x.to(device))
        y_true, y_pred = y.numpy(), logits.argmax(1).cpu().numpy()
        metrics = evaluate_predictions(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix_data(y_true, y_pred)
        np.save(results_dir / f"{prefix}y_true.npy", y_true)
        np.save(results_dir / f"{prefix}y_pred.npy", y_pred)
        (results_dir / f"{prefix}metrics.json").write_text(
            json.dumps(
                {
                    "accuracy": metrics["accuracy"],
                    "n_samples": int(metrics["n_samples"]),
                    "classification_report": report,
                },
                indent=2,
            )
        )
        plot_confusion_matrix(cm, results_dir / f"{prefix}confusion_matrix.svg", title=title)
        logger.info("%s accuracy=%.4f -> %s", title, metrics["accuracy"], results_dir)
        return metrics

    results_dir = Path(cfg.output.results_dir) / "dinov2"
    results_dir.mkdir(parents=True, exist_ok=True)

    _evaluate("test", "DINOv2 - internal split", "")
    if Path(feats_dir / "heldout.pt").exists():
        _evaluate("heldout", "DINOv2 - held-out test", "heldout_")
    else:
        logger.warning("no heldout features cached - skipping held-out eval")
    return 0


def _discover_cross_generators(cross_root: Path) -> list[str]:
    """All subfolders under ``cross_root`` that look like generator sets."""
    if not cross_root.is_dir():
        raise FileNotFoundError(f"cross-generator root missing: {cross_root}")
    generators = sorted(
        d.name
        for d in cross_root.iterdir()
        if d.is_dir() and (d / "real").is_dir() and (d / "fake").is_dir()
    )
    if not generators:
        raise FileNotFoundError(
            f"no generator folders (real/ + fake/) under {cross_root}"
        )
    return generators


def _write_cross_result(
    out: Path,
    gen: str,
    model: str,
    variant: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Persist one generator's cross-gen metrics in the canonical layout."""
    import json

    import numpy as np

    from .evaluate.metrics import classification_report, confusion_matrix_data, evaluate_predictions
    from .evaluate.visual import plot_confusion_matrix

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "y_true.npy", y_true)
    np.save(out / "y_pred.npy", y_pred)
    metrics = evaluate_predictions(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix_data(y_true, y_pred)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "generator": gen,
                "model": model,
                "variant": variant,
                "accuracy": metrics["accuracy"],
                "n_samples": int(metrics["n_samples"]),
                "classification_report": report,
            },
            indent=2,
        )
    )
    plot_confusion_matrix(
        cm, out / "confusion_matrix.svg", title=f"{model} cross-gen: {gen}"
    )
    logger.info("%s accuracy=%.4f -> %s", gen, metrics["accuracy"], out)


def _eval_cross_dinov2(args: argparse.Namespace) -> int:
    """Score the SD-trained DINOv2 probe against cross-generator sets.

    No retraining: reuses the frozen backbone and the probe checkpoint written
    by `train-linear-probe`, and applies the same preprocessing as the main
    run (variant-keyed feature cache). One output dir per generator under
    `results/dinov2/cross/<generator>/`.
    """
    import torch
    from torch.utils.data import DataLoader

    from .config import load_config
    from .data.dataset import FauxtoDataset
    from .data.transforms import dinov2_transforms
    from .models.dinov2 import DINOv2FeatureExtractor
    from .models.linear_probe import LinearProbe
    from .utils.device import get_device
    from .utils.seed import set_seed

    cfg = load_config("dinov2", args.config)
    set_seed(cfg.data.seed)
    device = get_device()

    variant = "square" if cfg.square_resize else "canonical"
    tf = dinov2_transforms(cfg.data.image_size, square=cfg.square_resize)

    # Probe checkpoint produced by train-linear-probe (variant-consistent
    # training features are guaranteed by the same square_resize config).
    probe_path = Path(cfg.output.run_dir) / "dinov2" / "probe.pt"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"no probe at {probe_path} - run `fauxtography train-linear-probe` first"
        )
    probe_state = torch.load(probe_path, map_location="cpu", weights_only=True)[
        "probe_state_dict"
    ]
    in_features = next(iter(probe_state.values())).shape[1]
    probe = LinearProbe(in_features, num_classes=2).to(device)
    probe.load_state_dict(probe_state)
    probe.eval()

    generators = _discover_cross_generators(Path(cfg.data.cross_dir))
    extractor = DINOv2FeatureExtractor(arch=cfg.arch, device=device)
    logger.info(
        "cross-generator eval: %s on %s, generators=%s, preprocess=%s",
        cfg.arch,
        device,
        generators,
        variant,
    )

    feats_dir = Path(cfg.output.run_dir) / "dinov2" / "features" / variant
    feats_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(cfg.output.results_dir) / "dinov2" / "cross"
    results_root.mkdir(parents=True, exist_ok=True)

    for gen in generators:
        gen_root = Path(cfg.data.cross_dir) / gen
        gen_feats = feats_dir / f"cross_{gen}.pt"
        if gen_feats.exists():
            logger.info("reuse cached features -> %s", gen_feats)
            feats = torch.load(gen_feats, map_location="cpu", weights_only=True)
            x = feats["features"]
            y_true = feats["labels"].long()
        else:
            ds = FauxtoDataset(gen_root, transform=tf.eval_transform)
            loader = DataLoader(
                ds,
                batch_size=cfg.feature_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=device.type == "cuda",
            )
            feats, labels = [], []
            for images, _labels in loader:
                feats.append(extractor.extract(images).cpu())
                labels.append(_labels)
            x = torch.cat(feats)
            y_true = torch.cat(labels).long()
            torch.save({"features": x, "labels": y_true}, gen_feats)
            logger.info("cached %s features (%d x %d)", gen, len(ds), extractor.hidden_size)

        with torch.no_grad():
            logits = probe(x.to(device))
        y_pred = logits.argmax(1).cpu().numpy()
        _write_cross_result(
            results_root / gen, gen, "dinov2", variant, y_true.numpy(), y_pred
        )
    return 0


def _eval_cross_baseline(args: argparse.Namespace) -> int:
    """Score the fine-tuned ResNet baseline against cross-generator sets.

    Reuses the best-val baseline checkpoint written by `train-baseline` and
    the baseline eval transform (no retraining, no extra caching). Outputs
    land under `results/baseline/cross/<generator>/` so the paper comparison
    table gets an in-project ResNet row.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from .config import load_config
    from .data.dataset import FauxtoDataset
    from .data.transforms import baseline_transforms
    from .models.baseline import build_baseline_model
    from .utils.device import get_device

    cfg = load_config("baseline", args.config)
    device = get_device()

    ckpt = Path(cfg.output.run_dir) / "baseline" / "baseline_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"no baseline checkpoint at {ckpt} - run `fauxtography train-baseline` first"
        )
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = build_baseline_model(
        arch=cfg.arch,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
        num_classes=2,
        device=device,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    tf = baseline_transforms(cfg.data.image_size)
    generators = _discover_cross_generators(Path(cfg.data.cross_dir))
    logger.info(
        "baseline cross-gen eval: %s on %s, generators=%s",
        cfg.arch,
        device,
        generators,
    )

    results_root = Path(cfg.output.results_dir) / "baseline" / "cross"
    results_root.mkdir(parents=True, exist_ok=True)

    for gen in generators:
        ds = FauxtoDataset(Path(cfg.data.cross_dir) / gen, transform=tf.eval_transform)
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=device.type == "cuda",
        )
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                y_true.extend(labels.tolist())
                y_pred.extend(logits.argmax(1).cpu().tolist())
        _write_cross_result(
            results_root / gen,
            gen,
            "baseline",
            cfg.arch,
            np.asarray(y_true),
            np.asarray(y_pred),
        )
    return 0


def cmd_eval_cross(args: argparse.Namespace) -> int:
    """Score a trained model against the cross-generator sets.

    ``--model dinov2`` (default): the frozen DINOv2 backbone + linear probe.
    ``--model baseline``: the fine-tuned ResNet checkpoint.
    """
    cfg_default = DEFAULT_CONFIG["eval-cross"] if args.model == "dinov2" else DEFAULT_CONFIG["train-baseline"]
    args.config = Path(args.config) if args.config else Path(cfg_default)

    if args.model == "baseline":
        return _eval_cross_baseline(args)
    return _eval_cross_dinov2(args)


def cmd_probe_file_size(args: argparse.Namespace) -> int:
    """Measure how well file size alone separates real from fake.

    GenImage has a documented compression short-cut: real photographs are
    heavily compressed (small files) while generated images compress poorly
    (large files). This probe fits a threshold on file size using the train
    split, then reports accuracy on the held-out test and each cross-generator
    set. It turns the bias from a corpus observation into a measured number.
    """
    import json

    import numpy as np

    from .config import load_config
    from .data.transforms import baseline_transforms  # noqa: F401  (standalone probe)
    from .evaluate.visual import plot_size_histogram

    cfg = load_config("baseline", args.config)

    def _sizes(folder: Path) -> np.ndarray:
        """Image file sizes (bytes) under a real/ or fake/ folder."""
        if not folder.is_dir():
            return np.array([], dtype=np.int64)
        return np.array([p.stat().st_size for p in folder.iterdir() if p.is_file()])

    train_root = Path(cfg.data.root)
    real_train = _sizes(train_root / "train" / "real")
    fake_train = _sizes(train_root / "train" / "fake")
    if len(real_train) == 0 or len(fake_train) == 0:
        raise FileNotFoundError(
            f"train real/fake missing under {train_root} - run prepare-data first"
        )

    def _best_threshold(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
        """Pick the file-size threshold that maximises accuracy (fake > thr)."""
        candidates = np.unique(np.concatenate([real, fake]))
        best_acc, best_thr = -1.0, float(candidates[len(candidates) // 2])
        for thr in candidates:
            acc = (np.sum(real <= thr) + np.sum(fake > thr)) / (len(real) + len(fake))
            if acc > best_acc:
                best_acc, best_thr = acc, float(thr)
        return best_thr, best_acc

    thr, train_acc = _best_threshold(real_train, fake_train)

    def _score(name: str, real: np.ndarray, fake: np.ndarray) -> dict:
        n = len(real) + len(fake)
        if n == 0:
            return {"name": name, "n_samples": 0, "accuracy": None}
        acc = (np.sum(real <= thr) + np.sum(fake > thr)) / n
        # Confusion counts: real=0, fake=1; predicted fake when file larger.
        tn = int(np.sum(real <= thr))
        fp = int(np.sum(real > thr))
        fn = int(np.sum(fake <= thr))
        tp = int(np.sum(fake > thr))
        return {
            "name": name,
            "n_samples": n,
            "n_real": len(real),
            "n_fake": len(fake),
            "accuracy": round(float(acc), 4),
            "confusion": {"tn_real": tn, "fp_real": fp, "fn_fake": fn, "tp_fake": tp},
        }

    sets: list[dict] = [
        _score("train", real_train, fake_train),
    ]

    # Held-out official test (from the baseline config, env-overridable).
    heldout = Path(cfg.data.heldout_test)
    if heldout.is_dir():
        sets.append(_score(heldout.name, _sizes(heldout / "real"), _sizes(heldout / "fake")))

    # Cross-generator sets: same threshold (fit only on SD v1.4 train).
    # Tolerate a missing cross root (cross data not downloaded yet).
    cross_root = Path(cfg.data.cross_dir)
    if cross_root.is_dir():
        for gen in _discover_cross_generators(cross_root):
            sets.append(
                _score(gen, _sizes(cross_root / gen / "real"), _sizes(cross_root / gen / "fake"))
            )

    out_dir = Path(cfg.output.results_dir) / "file_size"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "probe": "file-size threshold (fake > thr => fake)",
                "threshold_bytes": int(thr),
                "threshold_kb": round(thr / 1024, 1),
                "train_accuracy": round(float(train_acc), 4),
                "mean_size_bytes": {
                    "real_train": float(real_train.mean()),
                    "fake_train": float(fake_train.mean()),
                    "real_train_kb": round(float(real_train.mean()) / 1024, 1),
                    "fake_train_kb": round(float(fake_train.mean()) / 1024, 1),
                },
                "sets": sets,
            },
            indent=2,
        )
    )
    plot_size_histogram(
        real_train,
        fake_train,
        threshold=thr,
        save_path=out_dir / "size_histogram.svg",
    )
    for s in sets:
        acc = s["accuracy"]
        acc_s = f"{acc:.4f}" if acc is not None else "n/a"
        logger.info("file-size probe %-12s acc=%s (thr %d bytes)", s["name"], acc_s, thr)
    logger.info("file-size probe -> %s", out_dir)
    return 0


def cmd_visualize_features(args: argparse.Namespace) -> int:
    """Project cached DINOv2 features to 2-D and color by generator.

    Reads the variant-keyed feature cache (train/val/test + cross_<gen>) and
    produces a PCA then TSNE embedding. Illuminates whether each generator
    has its own signature inside a self-supervised representation - the
    geometric counterpart of the cross-generator accuracy collapse.
    """
    import json

    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from .config import load_config
    from .evaluate.visual import plot_feature_scatter

    cfg = load_config("dinov2", args.config)
    variant = "square" if cfg.square_resize else "canonical"
    feats_dir = Path(cfg.output.run_dir) / "dinov2" / "features" / variant
    if not feats_dir.is_dir():
        raise FileNotFoundError(f"no feature cache at {feats_dir} - run extract-dinov2 first")

    # Generators to color (split name -> display label). Cross sets are the
    # distinct generators; SD pool is the in-domain (real/fake) reference.
    generators = args.generators or sorted(
        p.stem for p in feats_dir.glob("cross_*.pt")
    )
    splits = ["train"] + [f"cross_{g}" for g in generators] if generators else ["train"]

    feats_list, labels_list, group = [], [], []
    ref_dims = None
    for i, split in enumerate(splits):
        path = feats_dir / f"{split}.pt"
        if not path.exists():
            logger.warning("skip %s - no cached features", path)
            continue
        data = torch.load(path, map_location="cpu", weights_only=True)
        x = data["features"]
        if ref_dims is None:
            ref_dims = x.shape[1]
        # Speed + readability: subsample large splits.
        n = min(len(x), args.max_points)
        idx = np.random.default_rng(cfg.data.seed).choice(len(x), n, replace=False)
        feats_list.append(x[idx])
        labels_list.append(data["labels"][idx])
        group.append(np.full(n, split))

    X = torch.cat(feats_list).numpy()
    y = torch.cat(labels_list).numpy()
    groups = np.concatenate(group)

    pca = PCA(n_components=min(50, X.shape[1]), random_state=cfg.data.seed)
    X_pca = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=min(30, len(X) - 1), random_state=cfg.data.seed)
    xy = tsne.fit_transform(X_pca)

    # Color by (generator, real/fake): 2 classes per generator.
    class_map = {}
    labels2 = np.zeros(len(groups), dtype=int)
    names = []
    for i, g in enumerate(sorted(set(groups.tolist()))):
        mask = groups == g
        for cls in ("real", "fake"):
            key = f"{g}:{cls}"
            class_map[key] = len(names)
            names.append(key)
            labels2[mask & (y == (1 if cls == "fake" else 0))] = class_map[key]

    out = Path(cfg.output.results_dir) / "dinov2" / "embedding"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embedding_xy.npy", xy)
    np.save(out / "embedding_labels.npy", labels2)
    (out / "embedding_classes.json").write_text(
        json.dumps({"classes": names, "method": "PCA->TSNE", "n_points": len(X)})
    )
    plot_feature_scatter(
        xy,
        labels2,
        names,
        title=f"DINOv2 feature geometry ({variant})",
        save_path=out / "embedding.svg",
    )
    logger.info("feature embedding (%d points) -> %s", len(X), out)
    return 0


def cmd_analyze_fusion(args: argparse.Namespace) -> int:
    """Fusion and disagreement across the three model families.

    Reads each model's predictions (baseline/dinov2/clip) on the internal and
    held-out splits, computes majority-vote fusion accuracy, and counts where
    the families disagree. Requires prior eval runs for whichever models are
    present; missing models are skipped with a warning.

    Outputs under results/fusion/{summary.json, disagreement.csv, scatter.csv}.
    """
    import csv
    import json

    import numpy as np

    results_dir = Path(args.results_dir)
    models = [m for m in ("baseline", "dinov2", "clip") if (results_dir / m).is_dir()]
    if not models:
        raise SystemExit(f"no model result dirs under {results_dir}")

    def _load_preds(model: str, split: str):
        base = results_dir / model
        yt = base / f"{split}y_true.npy"
        yp = base / f"{split}y_pred.npy"
        if not (yt.exists() and yp.exists()):
            return None
        return np.load(yt), np.load(yp)

    # Same split naming as the eval outputs: "" = internal, "heldout_" = held-out.
    splits = {"internal": "", "heldout": "heldout_"}
    rows = []
    for split_label, prefix in splits.items():
        preds = {m: _load_preds(m, prefix) for m in models}
        preds = {m: p for m, p in preds.items() if p is not None}
        if not preds:
            continue
        # Align on a common ordering (same split => same samples across models).
        sample = next(iter(preds.values()))
        n = len(sample[0])
        y_true = sample[0]
        stack = np.stack([p[1] for p in preds.values()])
        # Majority vote per sample.
        fused = np.apply_along_axis(lambda v: np.bincount(v).argmax(), 0, stack)
        acc_fused = float((fused == y_true).mean())
        acc_model = {m: float((p[1] == y_true).mean()) for m, p in preds.items()}

        # Agreement / disagreement stats.
        names = list(preds.keys())
        agreement = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                equal = float((preds[a][1] == preds[b][1]).mean())
                agreement[f"{a}_vs_{b}"] = round(equal, 4)
        rows.append(
            {
                "split": split_label,
                "n_samples": n,
                "fused_accuracy": round(acc_fused, 4),
                "model_accuracies": acc_model,
                "pairwise_agreement": agreement,
            }
        )

    out = Path(args.results_dir) / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))

    with open(out / "summary.csv", "w", newline="") as fh:
        fieldnames = ["split", "n_samples", "fused_accuracy", "model_accuracies", "pairwise_agreement"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r = dict(r)
            r["model_accuracies"] = json.dumps(r["model_accuracies"])
            r["pairwise_agreement"] = json.dumps(r["pairwise_agreement"])
            writer.writerow(r)

    for r in rows:
        logger.info(
            "%s: fused_acc=%.4f models=%s",
            r["split"],
            r["fused_accuracy"],
            r["model_accuracies"],
        )
    logger.info("fusion analysis -> %s", out)
    return 0


def _recompress_to_target(path: Path, out_path: Path, target_bytes: int) -> int:
    """Re-encode an image as JPEG so its file size lands near ``target_bytes``.

    Adjusts JPEG quality down until the encoded size fits the budget (or a
    floor quality is hit), then saves to ``out_path``. Returns bytes written.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    # Try progressively lower quality (min 10) to hit the budget.
    quality = 95
    for _ in range(12):
        trial = out_path.with_suffix(f".q{quality}.jpg")
        img.save(trial, "JPEG", quality=quality)
        size = trial.stat().st_size
        if size <= target_bytes or quality <= 10:
            trial.replace(out_path)
            return size
        quality -= 8
    # Fall back: best-effort copy.
    img.save(out_path, "JPEG", quality=10)
    return out_path.stat().st_size


def cmd_probe_compression(args: argparse.Namespace) -> int:
    """Causal test of the compression short-cut: re-encode and re-evaluate.

    GenImage fakes are ~4x larger files than real photos. This probe
    re-compresses both classes of a source set (held-out test by default) to
    a similar file size, re-extracts frozen DINOv2 features, and scores the
    unchanged probe. If accuracy collapses, the shortcut was load-bearing -
    a causal result, not just a correlation.

    Quality target: real-class median file size (bytes), or --target-bytes.
    Outputs under results/dinov2/compression/.
    """
    import json

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from .config import load_config
    from .data.dataset import FauxtoDataset
    from .data.transforms import dinov2_transforms
    from .evaluate.metrics import classification_report, confusion_matrix_data, evaluate_predictions
    from .evaluate.visual import plot_confusion_matrix
    from .models.dinov2 import DINOv2FeatureExtractor
    from .models.linear_probe import LinearProbe
    from .utils.device import get_device

    cfg = load_config("dinov2", args.config)
    device = get_device()
    variant = "square" if cfg.square_resize else "canonical"
    tf = dinov2_transforms(cfg.data.image_size, square=cfg.square_resize)

    # Probe from the SD-trained run (same probe as eval-cross).
    probe_path = Path(cfg.output.run_dir) / "dinov2" / "probe.pt"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"no probe at {probe_path} - run `fauxtography train-linear-probe` first"
        )
    probe_state = torch.load(probe_path, map_location="cpu", weights_only=True)[
        "probe_state_dict"
    ]
    in_features = next(iter(probe_state.values())).shape[1]
    probe = LinearProbe(in_features, num_classes=2).to(device)
    probe.load_state_dict(probe_state)
    probe.eval()

    # Source set to recompress (real/fake pair). Defaults to the official
    # held-out test from config (env-overridable), so the Drive layout works
    # without spelling out a path.
    src = Path(args.source) if str(args.source) not in ("", ".") else Path(cfg.data.heldout_test)
    if not ((src / "real").is_dir() and (src / "fake").is_dir()):
        raise FileNotFoundError(f"source needs real/ + fake/: {src}")

    # Real-class median size = the budget we force BOTH classes to hit.
    if args.target_bytes:
        target = int(args.target_bytes)
    else:
        real_sizes = [p.stat().st_size for p in (src / "real").iterdir() if p.is_file()]
        if not real_sizes:
            raise FileNotFoundError(f"no images in {src / 'real'}")
        target = int(np.median(real_sizes))

    # Recompress into a run-dir mirror (git-ignored), variant-shaped cache.
    mirror = Path(cfg.output.run_dir) / "dinov2" / "recompressed" / src.name
    (mirror / "real").mkdir(parents=True, exist_ok=True)
    (mirror / "fake").mkdir(parents=True, exist_ok=True)

    sizes_before = {}
    sizes_after = {}
    for cls in ("real", "fake"):
        before = [p.stat().st_size for p in (src / cls).iterdir() if p.is_file()]
        sizes_before[cls] = int(np.median(before)) if before else 0
        for i, p in enumerate(sorted(p for p in (src / cls).iterdir() if p.is_file())):
            out_path = mirror / cls / f"{i:05d}.jpg"
            bytes_w = _recompress_to_target(p, out_path, target)
            sizes_after.setdefault(cls, []).append(bytes_w)

    logger.info(
        "recompressed %s -> median real %d bytes / fake %d bytes (target %d)",
        src.name,
        int(np.median(sizes_after["real"])),
        int(np.median(sizes_after["fake"])),
        target,
    )

    # Re-extract frozen features on the recompressed images + score probe.
    extractor = DINOv2FeatureExtractor(arch=cfg.arch, device=device)
    ds = FauxtoDataset(mirror, transform=tf.eval_transform)
    loader = DataLoader(
        ds,
        batch_size=cfg.feature_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    feats, labels = [], []
    for images, _l in loader:
        feats.append(extractor.extract(images).cpu())
        labels.append(_l)
    x = torch.cat(feats)
    y_true = torch.cat(labels).long().numpy()
    with torch.no_grad():
        y_pred = probe(x.to(device)).argmax(1).cpu().numpy()

    metrics = evaluate_predictions(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix_data(y_true, y_pred)

    out = Path(cfg.output.results_dir) / "dinov2" / "compression"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "y_true.npy", y_true)
    np.save(out / "y_pred.npy", y_pred)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "probe": "recompress-and-eval (compression short-cut test)",
                "source": src.name,
                "variant": variant,
                "target_bytes": target,
                "median_bytes_before": sizes_before,
                "median_bytes_after": {
                    c: int(np.median(v)) for c, v in sizes_after.items()
                },
                "accuracy": metrics["accuracy"],
                "n_samples": int(metrics["n_samples"]),
                "classification_report": report,
            },
            indent=2,
        )
    )
    plot_confusion_matrix(
        cm, out / "confusion_matrix.svg", title=f"DINOv2 recompressed ({src.name})"
    )
    logger.info(
        "recompressed accuracy=%.4f (before: %s) -> %s",
        metrics["accuracy"],
        sizes_before,
        out,
    )
    return 0


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

    p = sub.add_parser("eval-cross", help="score a trained model on cross-generator sets")
    p.add_argument("--model", default="dinov2", choices=["dinov2", "baseline"])
    p.add_argument("--config", type=Path)
    p.set_defaults(func=cmd_eval_cross)

    p = sub.add_parser("probe-file-size", help="measure the file-size shortcut")
    p.add_argument("--config", default=DEFAULT_CONFIG["probe-file-size"], type=Path)
    p.set_defaults(func=cmd_probe_file_size)

    p = sub.add_parser("probe-compression", help="recompress-and-eval causal test")
    p.add_argument("--config", default=DEFAULT_CONFIG["probe-compression"], type=Path)
    p.add_argument("--source", default="", type=Path)
    p.add_argument("--target-bytes", type=int, default=0)
    p.set_defaults(func=cmd_probe_compression)

    p = sub.add_parser("visualize-features", help="TSNE embedding of DINOv2 features")
    p.add_argument("--config", default=DEFAULT_CONFIG["visualize-features"], type=Path)
    p.add_argument("--generators", nargs="*", default=None)
    p.add_argument("--max-points", type=int, default=2000)
    p.set_defaults(func=cmd_visualize_features)

    p = sub.add_parser("analyze-fusion", help="majority vote across model families")
    p.add_argument("--results-dir", default="results", type=Path)
    p.set_defaults(func=cmd_analyze_fusion)

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
