"""Plots for results: confusion matrix and error gallery.

Shared by both collaborators (each plots their own model's matrix / error
gallery; the compare step produces the side-by-side figure).
"""

from __future__ import annotations

import os
from pathlib import Path

# Force the non-interactive Agg backend so headless runs (Colab, CI) work even
# when an inherited MPLBACKEND (e.g. Colab's matplotlib_inline, which isn't in
# the project venv) would otherwise break matplotlib at import time.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np

from ..data.dataset import CLASS_NAMES


def plot_confusion_matrix(cm: np.ndarray, save_path: str | Path, title: str = "") -> Path:
    """Save a confusion-matrix figure (SVG by extension), return the path."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_feature_scatter(
    xy: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str | Path,
) -> Path:
    """2-D embedding scatter coloured by class (e.g. DINOv2 PCA/TSNE)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    classes = np.unique(labels)
    for c in classes:
        idx = labels == c
        name = class_names[int(c)] if int(c) < len(class_names) else str(c)
        ax.scatter(xy[idx, 0], xy[idx, 1], s=8, alpha=0.55, label=name)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_size_histogram(
    real_sizes: np.ndarray,
    fake_sizes: np.ndarray,
    threshold: float | None,
    save_path: str | Path,
) -> Path:
    """Histogram of real vs fake file sizes (bytes, log-x), with the fitted
    threshold marked. Used by `probe-file-size` to show the compression bias."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.logspace(3.5, 6.4, 60)  # ~3 KB .. ~2.5 MB
    ax.hist(real_sizes, bins=bins, alpha=0.65, label="real", color="tab:blue")
    ax.hist(fake_sizes, bins=bins, alpha=0.65, label="fake", color="tab:orange")
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.4, label="threshold")
    ax.set_xscale("log")
    ax.set_xlabel("File size (bytes, log scale)")
    ax.set_ylabel("Count")
    ax.set_title("File-size shortcut: real vs fake")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path
