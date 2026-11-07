"""STUB — Metrics: accuracy, per-class precision/recall/F1, confusion matrix.

All functions are model-agnostic: they consume ``(y_true, y_pred)`` arrays from
any of the three methods so comparison is apples-to-apples. Shared by both
collaborators for their per-model analysis and the joint comparison.
"""

from __future__ import annotations

import numpy as np

from ..data.dataset import CLASS_NAMES

LABELS = list(CLASS_NAMES)  # ["real", "fake"]


def evaluate_predictions(y_true, y_pred) -> dict:
    """TODO(impl): return dict of headline metrics (accuracy, counts...)."""
    raise NotImplementedError


def classification_report(y_true, y_pred) -> dict:
    """TODO(impl): per-class precision/recall/F1 as dict label -> metrics."""
    raise NotImplementedError


def confusion_matrix_data(y_true, y_pred) -> np.ndarray:
    """TODO(impl): confusion matrix in canonical order rows/cols = [real, fake]."""
    raise NotImplementedError