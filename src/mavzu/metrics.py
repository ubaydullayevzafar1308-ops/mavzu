"""Shared metric helpers. Macro-F1 is the headline metric because classes are imbalanced."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from mavzu.labels import LABELS


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(LABELS)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "per_class": {name: {k: round(float(v), 4) for k, v in report[name].items()} for name in LABELS},
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, normalize: str | None = "true") -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))), normalize=normalize)


def top_confusions(cm: np.ndarray, k: int = 10) -> list[dict]:
    """Largest off-diagonal cells of a row-normalized confusion matrix."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:k]
    return [
        {"true": LABELS[i], "pred": LABELS[j], "rate": round(float(off[i, j]), 4)}
        for i, j in idx
        if off[i, j] > 0
    ]
