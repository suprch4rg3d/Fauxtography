"""Fetch a size-bounded slice of the GenImage dataset from the HuggingFace mirror.

GenImage (Huawei Noah's Ark Lab) stores real photographs (ImageNet "nature")
and AI-generated images in per-generator leaves of the Arrow mirror. Each row
carries ``label`` = real | fake and an embedded full-res image.

This downloader turns the mirror into the ``real/`` + ``fake/`` folder layout
that ``prepare-data`` expects, fetching **by row count** (not raw GB) so the
amount of data is decided by image count - the quantity that actually drives
training cost and accuracy.

Ownership/notes:
- ``train_rows_per_class`` + the generator determine the main train pool.
- ``fetch_full_test`` grabs the generator's official test subset.
- ``cross_generator_rows`` pulls small sets from other generators for the
  cross-generator generalisation analysis.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from datasets import load_dataset

from ..utils.logging import get_logger

logger = get_logger(__name__)

ROWS_PER_SHARD = 1720  # observed in the SD v1.4 train leaves

# Mirror label encoding: ClassLabel(int) - 0 = real, 1 = fake.
_LABEL_TO_CLASS = {"0": "real", 0: "real", "1": "fake", 1: "fake"}


def _leaf_shards(repo_files: list[str], prefix: str) -> list[str]:
    """The ordered ``.arrow`` files of a mirror leaf."""
    files = [f for f in repo_files if f.startswith(prefix) and f.endswith(".arrow")]
    return sorted(files)


def _shards_needed(rows: int) -> int:
    """Number of mirror shards supplying at least ``rows`` real+fake images."""
    if rows <= 0:
        return 0
    per_shard_real = ROWS_PER_SHARD // 2
    return max(1, (rows + per_shard_real - 1) // per_shard_real)


def _list_mirror_files(repo_id: str) -> list[str]:
    from huggingface_hub import list_repo_files

    try:
        return list(list_repo_files(repo_id, repo_type="dataset"))
    except Exception as exc:  # noqa: BLE001 - fall back to a plain listing
        logger.warning("listing mirror files failed (%s); assuming shard layout", exc)
        return []


def _stream_split(
    dataset,
    train_rows_per_class: int,
    real_dir: Path,
    fake_dir: Path,
) -> dict[str, int]:
    """Write up to N real and N fake images from a dataset into the folders."""
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = Counter()
    saved = 0
    for row in dataset:
        label = _LABEL_TO_CLASS.get(row["label"])
        if label is None:
            continue
        if counts[label] >= train_rows_per_class:
            continue
        # row["image"] is binary-encoded; datasets exposes it as bytes-like.
        img = row["image"]
        data = img if isinstance(img, bytes) else bytes(img)
        if not data:
            continue
        ext = ".jpg"
        target = (real_dir if label == "real" else fake_dir) / f"{saved:06d}{ext}"
        target.write_bytes(data)
        counts[label] += 1
        saved += 1
        if counts["real"] >= train_rows_per_class and counts["fake"] >= train_rows_per_class:
            break
    return dict(counts)


def fetch_genimage(cfg) -> dict:
    """Download the configured GenImage slice; return per-split counts."""
    import os

    cache = Path(os.path.expanduser(cfg.repo_cache or "~/.cache/huggingface"))
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    raw = Path(cfg.raw_dir)

    mirror = cfg.mirror
    gen = cfg.train_generator
    rows = cfg.train_rows_per_class

    report: dict[str, dict[str, int]] = {}
    files = _list_mirror_files(mirror)

    def resolve(repo_file: str) -> str:
        """Download the mirror file into the HF cache, return its local path."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(mirror, repo_file, repo_type="dataset")

    def take(prefix: str, n_files: int | None) -> list[str]:
        """Resolve up to ``n_files`` (all if None) shards of a leaf."""
        shards = _leaf_shards(files, prefix) if files else []
        if not shards:
            raise FileNotFoundError(
                f"no shards found on the mirror for {prefix!r}; mirror layout may have changed"
            )
        chosen = shards if n_files is None else shards[:n_files]
        return [resolve(f) for f in chosen]

    if rows > 0:
        n_shards = _shards_needed(rows)
        paths = take(f"data/train/{gen}", n_shards)
        logger.info("fetching %d rows/class of %s train (shards %d)", rows, gen, n_shards)
        ds = load_dataset("arrow", data_files=paths, split="train")
        counts = _stream_split(ds, rows, raw / "real", raw / "fake")
        logger.info("train %s: %s", gen, counts)
        report["train"] = counts

    if cfg.fetch_full_test:
        paths = take(f"data/test/{gen}", None)
        ds = load_dataset("arrow", data_files=paths, split="train")
        counts = _stream_split(ds, 50_000, raw / "real", raw / "fake")
        logger.info("test %s: %s", gen, counts)
        report["test"] = counts

    for g, g_rows in (cfg.cross_generator_rows or {}).items():
        if g_rows <= 0:
            continue
        paths = take(f"data/test/{g}", 1)  # enough for small cross-gene sets
        ds = load_dataset("arrow", data_files=paths, split="train")
        counts = _stream_split(ds, g_rows, raw / "real", raw / "fake")
        logger.info("cross %s: %s", g, counts)
        report[f"cross_{g}"] = counts

    return report
