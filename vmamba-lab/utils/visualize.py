"""Visualization utilities for model outputs and metrics."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def visualize_attention(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: Optional[str] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Visualize attention/activation maps using Grad-CAM style approach.

    Args:
        model: Model to visualize
        image: Input image tensor (B, C, H, W)
        target_layer: Name of target layer for visualization
        save_path: Optional path to save visualization

    Returns:
        Attention heatmap as numpy array
    """
    model.eval()
    activations = {}
    gradients = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook

    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if target_layer is None or target_layer in name:
            handles.append(module.register_forward_hook(save_activation(name)))
            handles.append(module.register_full_backward_hook(save_gradient(name)))
            if target_layer is not None:
                break

    # Forward pass
    output = model(image)
    if isinstance(output, dict):
        output = output.get("logits", output.get("pred", list(output.values())[0]))

    # Backward pass
    model.zero_grad()
    if output.dim() > 1 and output.size(1) > 1:
        target = output.argmax(dim=1)
        loss = output[range(len(target)), target].sum()
    else:
        loss = output.sum()
    loss.backward()

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Generate heatmap
    if activations and gradients:
        layer_name = list(activations.keys())[-1]
        activation = activations[layer_name]
        gradient = gradients[layer_name]

        # Global average pooling of gradients
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(
            cam, size=image.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros(image.shape[2:])

    # Visualization
    if save_path:
        img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap="jet")
        plt.title("Attention Map")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * img_np + 0.5 * heatmap
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return cam


def visualize_segmentation(
    image: Union[np.ndarray, torch.Tensor],
    pred_mask: Union[np.ndarray, torch.Tensor],
    gt_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Visualize segmentation prediction with ground truth.

    Args:
        image: Input image (H, W, C) or (C, H, W)
        pred_mask: Predicted mask (H, W)
        gt_mask: Optional ground truth mask (H, W)
        threshold: Threshold for binary mask
        save_path: Optional path to save visualization
        alpha: Overlay transparency

    Returns:
        Visualization as numpy array
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if gt_mask is not None and isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    # Normalize image
    if image.max() > 1:
        image = image / 255.0
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Binarize masks
    pred_binary = (pred_mask > threshold).astype(np.float32)

    # Create colored overlays
    pred_color = np.zeros((*pred_mask.shape, 3))
    pred_color[pred_binary > 0] = [1, 0, 0]  # Red for predictions

    num_plots = 3 if gt_mask is not None else 2

    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Prediction overlay
    overlay = image.copy()
    mask_region = pred_binary[..., None] > 0
    overlay = np.where(mask_region, overlay * (1 - alpha) + pred_color * alpha, overlay)
    axes[1].imshow(overlay)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Ground truth comparison
    if gt_mask is not None:
        gt_binary = (gt_mask > threshold).astype(np.float32)

        # Create comparison: Green=TP, Red=FP, Blue=FN
        comparison = np.zeros((*pred_mask.shape, 3))
        tp = (pred_binary > 0) & (gt_binary > 0)
        fp = (pred_binary > 0) & (gt_binary == 0)
        fn = (pred_binary == 0) & (gt_binary > 0)

        comparison[tp] = [0, 1, 0]  # Green: True Positive
        comparison[fp] = [1, 0, 0]  # Red: False Positive
        comparison[fn] = [0, 0, 1]  # Blue: False Negative

        overlay_gt = image.copy()
        mask_any = (tp | fp | fn)[..., None]
        overlay_gt = np.where(mask_any, overlay_gt * 0.5 + comparison * 0.5, overlay_gt)
        axes[2].imshow(overlay_gt)
        axes[2].set_title("GT Comparison\n(G:TP, R:FP, B:FN)")
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Convert figure to numpy array
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return vis


def plot_roc_curve(
    y_true: Union[np.ndarray, List],
    y_scores: Union[np.ndarray, List],
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
) -> Tuple[float, np.ndarray]:
    """Plot ROC curve and calculate AUC.

    Args:
        y_true: Ground truth labels
        y_scores: Prediction scores
        save_path: Optional path to save plot
        title: Plot title

    Returns:
        Tuple of (AUC, EER threshold)
    """
    from sklearn.metrics import auc, roc_curve

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Calculate EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.scatter([fpr[eer_idx]], [tpr[eer_idx]], color="red", s=100, zorder=5, label=f"EER = {eer:.4f}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return roc_auc, eer_threshold


def plot_confusion_matrix(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> np.ndarray:
    """Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
        save_path: Optional path to save plot
        title: Plot title
        normalize: Whether to normalize the matrix

    Returns:
        Confusion matrix as numpy array
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return cm
