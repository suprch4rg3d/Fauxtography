"""Evaluation metrics and error-analysis helpers."""

from .metrics import classification_report, confusion_matrix_data, evaluate_predictions
from .visual import plot_confusion_matrix

__all__ = ["classification_report", "confusion_matrix_data", "evaluate_predictions", "plot_confusion_matrix"]