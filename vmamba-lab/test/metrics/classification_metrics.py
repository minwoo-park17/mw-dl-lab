"""Classification metrics for face forgery detection."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_auc(
    y_true: Union[np.ndarray, List],
    y_scores: Union[np.ndarray, List],
) -> float:
    """Compute Area Under ROC Curve (AUC).

    Args:
        y_true: Ground truth labels (binary)
        y_scores: Prediction scores/probabilities

    Returns:
        AUC score
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return 0.5


def compute_eer(
    y_true: Union[np.ndarray, List],
    y_scores: Union[np.ndarray, List],
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER).

    EER is the point where False Positive Rate equals False Negative Rate.

    Args:
        y_true: Ground truth labels (binary)
        y_scores: Prediction scores/probabilities

    Returns:
        Tuple of (EER, threshold at EER)
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr

        # Find the point where FPR and FNR are closest
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]

        return float(eer), float(eer_threshold)
    except Exception:
        return 0.5, 0.5


def compute_accuracy(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return accuracy_score(y_true, y_pred)


def compute_ap(
    y_true: Union[np.ndarray, List],
    y_scores: Union[np.ndarray, List],
) -> float:
    """Compute Average Precision (AP).

    Args:
        y_true: Ground truth labels (binary)
        y_scores: Prediction scores/probabilities

    Returns:
        Average Precision score
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        return average_precision_score(y_true, y_scores)
    except ValueError:
        return 0.5


def compute_confusion_matrix(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    normalize: bool = False,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: Normalize the confusion matrix

    Returns:
        Confusion matrix array
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    return cm


class ClassificationMetrics:
    """Class to compute and store classification metrics."""

    def __init__(self):
        """Initialize metrics storage."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.y_true = []
        self.y_pred = []
        self.y_scores = []

    def update(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        y_scores: Optional[Union[np.ndarray, List]] = None,
    ):
        """Update metrics with new predictions.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_scores: Optional prediction scores
        """
        if isinstance(y_true, np.ndarray):
            y_true = y_true.tolist()
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()

        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

        if y_scores is not None:
            if isinstance(y_scores, np.ndarray):
                y_scores = y_scores.tolist()
            self.y_scores.extend(y_scores)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}

        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)

        # Basic metrics
        metrics["accuracy"] = compute_accuracy(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        # Score-based metrics
        if self.y_scores:
            y_scores = np.array(self.y_scores)
            metrics["auc"] = compute_auc(y_true, y_scores)
            metrics["ap"] = compute_ap(y_true, y_scores)
            eer, eer_threshold = compute_eer(y_true, y_scores)
            metrics["eer"] = eer
            metrics["eer_threshold"] = eer_threshold

        return metrics

    def get_confusion_matrix(self, normalize: bool = False) -> np.ndarray:
        """Get confusion matrix.

        Args:
            normalize: Normalize the confusion matrix

        Returns:
            Confusion matrix array
        """
        return compute_confusion_matrix(self.y_true, self.y_pred, normalize)
