"""
Utility functions for deepfake classification.
"""
import os
import logging
from typing import Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_correct(
    preds: torch.Tensor,
    targets: torch.Tensor,
    classes: List[int]
) -> Tuple[int, int, int, np.ndarray]:
    """
    Calculate prediction correctness and confusion matrix.

    Args:
        preds: Model predictions (logits)
        targets: Ground truth labels
        classes: List of class labels [0, 1]

    Returns:
        Tuple of (correct count, positive predictions, negative predictions, confusion matrix)
    """
    preds_np = torch.sigmoid(preds).cpu().detach().numpy().round()
    targets_np = targets.cpu().detach().numpy().round()

    cm = confusion_matrix(y_true=targets_np, y_pred=preds_np, labels=classes)

    correct = 0
    positive_class = 0
    negative_class = 0

    for i in range(len(targets_np)):
        pred = int(preds_np[i])
        if targets_np[i] == pred:
            correct += 1
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1

    return correct, positive_class, negative_class, cm


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default value when denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_metrics(cm: np.ndarray, label_list: List[int], prob_list: List[float]) -> dict:
    """
    Calculate classification metrics from confusion matrix.

    Args:
        cm: Confusion matrix [[TN, FP], [FN, TP]]
        label_list: Ground truth labels
        prob_list: Predicted probabilities

    Returns:
        Dictionary containing all metrics
    """
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tn = cm[0, 0]

    n_data = np.sum(cm)

    # Accuracy
    accuracy = safe_divide(tp + tn, n_data)

    # Balanced Accuracy
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    # Precision
    precision = safe_divide(tp, tp + fp)

    # Recall (same as sensitivity)
    recall = sensitivity

    # F1-score
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    # AUC
    try:
        auc = roc_auc_score(y_true=label_list, y_score=prob_list)
    except ValueError as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = 0.0

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'auc': auc
    }


def postprocess(
    results_info: dict,
    save_dir: str,
    folder_name: Optional[str] = None
) -> None:
    """
    Post-process results and save visualizations.

    Args:
        results_info: Dictionary containing results for each data split
        save_dir: Directory to save results
        folder_name: Optional folder name for custom results
    """
    def _unit_postprocess(unit_result_info: dict, unit_save_dir: str) -> None:
        """Process and save results for a single data split."""
        path_list = unit_result_info["path_list"]
        label_list = unit_result_info["label_list"]
        prob_list = unit_result_info["prob_list"]
        pred_list = unit_result_info["pred_list"]
        cm = unit_result_info["cm_total"].astype(int)

        # Create results dataframe
        pd_preds_info = pd.DataFrame({
            "target_img_path": path_list,
            "labels": label_list,
            "preds": pred_list,
            "prob": prob_list
        })

        # Calculate metrics
        metrics = calculate_metrics(cm, label_list, prob_list)

        # Create performance dataframe
        performance = pd.DataFrame({
            "Performance": ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score", "AUC"],
            "MeasureValue": [
                metrics['accuracy'],
                metrics['balanced_accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics['auc']
            ]
        })

        # File paths
        cm_save_path = os.path.join(unit_save_dir, "confusion_matrix_heatmap.png")
        roc_save_path = os.path.join(unit_save_dir, "roc_curve.png")
        pd_save_path = os.path.join(unit_save_dir, "pred_results_info.csv")
        performance_path = os.path.join(unit_save_dir, "performance.csv")

        # Plot confusion matrix
        class_names = ['Real', 'Deepfake']
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.title('Confusion Matrix')
        plt.savefig(cm_save_path)
        plt.close()

        # Plot ROC curve
        try:
            fp_ratio, tp_ratio, _ = roc_curve(y_true=label_list, y_score=prob_list)
            plt.figure(figsize=(6, 5))
            plt.plot(fp_ratio, tp_ratio, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {metrics["auc"]:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(roc_save_path)
            plt.close()
        except ValueError as e:
            logger.warning(f"Could not plot ROC curve: {e}")

        # Save CSVs
        pd_preds_info.to_csv(pd_save_path, index=False, encoding="utf-8-sig")
        performance.to_csv(performance_path, index=False, encoding="utf-8-sig")

        logger.info(f"Results saved to {unit_save_dir}")

    # Process each data split
    splits = ["train", "validation", "test"]
    for split in splits:
        if split in results_info and results_info[split] is not None:
            split_save_dir = os.path.join(save_dir, f"{split}_results")
            os.makedirs(split_save_dir, exist_ok=True)
            _unit_postprocess(results_info[split], split_save_dir)

    # Process custom folder
    if folder_name is not None and folder_name in results_info:
        folder_save_dir = os.path.join(save_dir, folder_name)
        os.makedirs(folder_save_dir, exist_ok=True)
        _unit_postprocess(results_info[folder_name], folder_save_dir)


def mtcnn_crop_face(
    frame: np.ndarray,
    face_detector,
    crop_size: int
) -> Optional[Image.Image]:
    """
    Crop face from frame using MTCNN detector.

    Args:
        frame: Input frame (BGR format)
        face_detector: MTCNN face detector
        crop_size: Output crop size

    Returns:
        Cropped face image or None if no face detected
    """
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        batch_boxes, confidence = face_detector.detect(frame_pil, landmarks=False)

        if batch_boxes is None or confidence is None or len(batch_boxes) == 0:
            return None

        # Select largest face
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in batch_boxes]
        max_pos = np.argmax(areas)

        box = batch_boxes[max_pos]
        img_w, img_h = frame_pil.size
        w_min, h_min = max(0, int(box[0])), max(0, int(box[1]))
        w_max, h_max = min(img_w, int(box[2])), min(img_h, int(box[3]))

        # Make square crop
        width, height = w_max - w_min, h_max - h_min
        max_size = max(width, height)
        center_x, center_y = (w_min + w_max) // 2, (h_min + h_max) // 2

        new_w_min = max(0, center_x - max_size // 2)
        new_h_min = max(0, center_y - max_size // 2)
        new_w_max = min(img_w, center_x + max_size // 2)
        new_h_max = min(img_h, center_y + max_size // 2)

        cropped_img = frame_pil.crop((new_w_min, new_h_min, new_w_max, new_h_max))
        cropped_img = cropped_img.resize((crop_size, crop_size), Image.LANCZOS)

        return cropped_img

    except Exception as e:
        logger.error(f"Error in face cropping: {e}")
        return None
