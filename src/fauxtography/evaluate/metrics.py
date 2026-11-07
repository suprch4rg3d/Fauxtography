"""Metrics: accuracy, per-class precision/recall/F1, confusion matrix.

All functions are model-agnostic: they consume ``(y_true, y_pred)`` arrays from
any of the three methods so comparison is apples-to-apples. Shared by both
collaborators for their per-model analysis and the joint comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from ..data.dataset import CLASS_NAMES

LABELS = list(CLASS_NAMES)  # ["real", "fake"]


def _as_array(y) -> np.ndarray:
    return np.asarray(y)


def evaluate_predictions(y_true, y_pred) -> dict:
    """Headline metrics: accuracy plus per-class sample counts."""
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_samples": len(y_true),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
    }


def classification_report(y_true, y_pred) -> dict:
    """Per-class precision/recall/F1 as a dict (JSON-serialisable)."""
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    report = sk_classification_report(
        y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True
    )
    return report


def confusion_matrix_data(y_true, y_pred) -> np.ndarray:
    """Confusion matrix in canonical order: rows/cols = [real, fake]."""
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    return sk_confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
