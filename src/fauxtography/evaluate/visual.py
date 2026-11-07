"""STUB — Plots for results: confusion matrix and error gallery.

Shared by both collaborators (each plots their own model's matrix / error
gallery; the compare step produces the side-by-side figure).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_confusion_matrix(cm: np.ndarray, save_path: str | Path, title: str = "") -> Path:
    """TODO(impl): save a confusion-matrix figure (SVG), return the path."""
    raise NotImplementedError


def plot_error_gallery(images, y_true, y_pred, save_path: str | Path, max_images: int = 16):
    """TODO(impl): grid of misclassified images with true/predicted labels."""
    raise NotImplementedError