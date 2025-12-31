"""Segmentation metrics for image tampering localization."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_f1(
    pred: Union[np.ndarray, List],
    target: Union[np.ndarray, List],
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute F1 score for binary segmentation.

    Args:
        pred: Predicted mask (probabilities or binary)
        target: Ground truth mask
        threshold: Threshold for binarization
        smooth: Smoothing factor

    Returns:
        F1 score
    """
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()

    # Binarize if needed
    if pred.max() > 1 or (pred.min() >= 0 and pred.max() <= 1 and not np.all(np.isin(pred, [0, 1]))):
        pred = (pred > threshold).astype(np.float32)

    target = target.astype(np.float32)

    # Compute TP, FP, FN
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))
    fn = np.sum((1 - pred) * target)

    # Compute precision and recall
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    # Compute F1
    f1 = 2 * precision * recall / (precision + recall + smooth)

    return float(f1)


def compute_iou(
    pred: Union[np.ndarray, List],
    target: Union[np.ndarray, List],
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute Intersection over Union (IoU / Jaccard Index).

    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold: Threshold for binarization
        smooth: Smoothing factor

    Returns:
        IoU score
    """
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()

    # Binarize if needed
    if pred.max() > 1 or (pred.min() >= 0 and pred.max() <= 1 and not np.all(np.isin(pred, [0, 1]))):
        pred = (pred > threshold).astype(np.float32)

    target = target.astype(np.float32)

    # Compute intersection and union
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection

    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)

    return float(iou)


def compute_pixel_auc(
    pred: Union[np.ndarray, List],
    target: Union[np.ndarray, List],
) -> float:
    """Compute pixel-level AUC.

    Args:
        pred: Predicted probability map
        target: Ground truth binary mask

    Returns:
        Pixel-level AUC score
    """
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()

    # Handle edge cases
    if len(np.unique(target)) < 2:
        return 0.5

    try:
        return roc_auc_score(target, pred)
    except ValueError:
        return 0.5


def compute_pixel_accuracy(
    pred: Union[np.ndarray, List],
    target: Union[np.ndarray, List],
    threshold: float = 0.5,
) -> float:
    """Compute pixel-level accuracy.

    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold: Threshold for binarization

    Returns:
        Pixel accuracy
    """
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()

    # Binarize if needed
    if pred.max() > 1 or (pred.min() >= 0 and pred.max() <= 1 and not np.all(np.isin(pred, [0, 1]))):
        pred = (pred > threshold).astype(np.float32)

    target = target.astype(np.float32)

    correct = np.sum(pred == target)
    total = len(target)

    return float(correct / total)


def compute_dice(
    pred: Union[np.ndarray, List],
    target: Union[np.ndarray, List],
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute Dice coefficient.

    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold: Threshold for binarization
        smooth: Smoothing factor

    Returns:
        Dice coefficient
    """
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()

    # Binarize if needed
    if pred.max() > 1 or (pred.min() >= 0 and pred.max() <= 1 and not np.all(np.isin(pred, [0, 1]))):
        pred = (pred > threshold).astype(np.float32)

    target = target.astype(np.float32)

    intersection = np.sum(pred * target)
    dice = (2 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

    return float(dice)


class SegmentationMetrics:
    """Class to compute and store segmentation metrics."""

    def __init__(self, threshold: float = 0.5):
        """Initialize metrics storage.

        Args:
            threshold: Threshold for binarization
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.pred_masks = []
        self.gt_masks = []
        self.pred_probs = []

    def update(
        self,
        pred: Union[np.ndarray, List],
        target: Union[np.ndarray, List],
        pred_prob: Optional[Union[np.ndarray, List]] = None,
    ):
        """Update metrics with new predictions.

        Args:
            pred: Predicted binary masks
            target: Ground truth masks
            pred_prob: Optional probability predictions
        """
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()
        if isinstance(target, np.ndarray):
            target = target.tolist()

        self.pred_masks.append(pred)
        self.gt_masks.append(target)

        if pred_prob is not None:
            if isinstance(pred_prob, np.ndarray):
                pred_prob = pred_prob.tolist()
            self.pred_probs.append(pred_prob)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}

        # Compute per-sample metrics and average
        f1_scores = []
        iou_scores = []
        dice_scores = []
        pixel_acc_scores = []
        pixel_auc_scores = []

        for i, (pred, gt) in enumerate(zip(self.pred_masks, self.gt_masks)):
            f1_scores.append(compute_f1(pred, gt, self.threshold))
            iou_scores.append(compute_iou(pred, gt, self.threshold))
            dice_scores.append(compute_dice(pred, gt, self.threshold))
            pixel_acc_scores.append(compute_pixel_accuracy(pred, gt, self.threshold))

            if self.pred_probs:
                pixel_auc_scores.append(compute_pixel_auc(self.pred_probs[i], gt))

        metrics["f1"] = float(np.mean(f1_scores))
        metrics["iou"] = float(np.mean(iou_scores))
        metrics["dice"] = float(np.mean(dice_scores))
        metrics["pixel_acc"] = float(np.mean(pixel_acc_scores))

        if pixel_auc_scores:
            metrics["pixel_auc"] = float(np.mean(pixel_auc_scores))

        return metrics

    def compute_per_sample(self) -> Dict[str, List[float]]:
        """Compute per-sample metrics.

        Returns:
            Dictionary of metric names and lists of values
        """
        per_sample = {
            "f1": [],
            "iou": [],
            "dice": [],
            "pixel_acc": [],
        }

        for pred, gt in zip(self.pred_masks, self.gt_masks):
            per_sample["f1"].append(compute_f1(pred, gt, self.threshold))
            per_sample["iou"].append(compute_iou(pred, gt, self.threshold))
            per_sample["dice"].append(compute_dice(pred, gt, self.threshold))
            per_sample["pixel_acc"].append(compute_pixel_accuracy(pred, gt, self.threshold))

        return per_sample
