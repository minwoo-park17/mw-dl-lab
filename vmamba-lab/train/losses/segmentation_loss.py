"""Segmentation loss functions for image tampering localization."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """

    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = "mean",
        squared: bool = False,
    ):
        """Initialize Dice Loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('none', 'mean', 'sum')
            squared: Use squared denominators
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.squared = squared

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            inputs: Predictions (B, 1, H, W) or (B, H, W) probabilities or logits
            targets: Ground truth masks (B, 1, H, W) or (B, H, W)

        Returns:
            Dice loss value
        """
        # Handle logits
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Compute intersection and union
        intersection = (inputs * targets).sum(dim=-1)

        if self.squared:
            denominator = (inputs ** 2).sum(dim=-1) + (targets ** 2).sum(dim=-1)
        else:
            denominator = inputs.sum(dim=-1) + targets.sum(dim=-1)

        # Compute Dice coefficient
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)

        # Compute loss
        loss = 1 - dice

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss.

    Commonly used for binary segmentation tasks.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        """Initialize BCE + Dice Loss.

        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            smooth: Smoothing factor for Dice
            reduction: Reduction method
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.dice = DiceLoss(smooth=smooth, reduction=reduction)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined BCE and Dice loss.

        Args:
            inputs: Predictions (B, 1, H, W) logits
            targets: Ground truth masks (B, 1, H, W)

        Returns:
            Combined loss value
        """
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class IoULoss(nn.Module):
    """IoU (Intersection over Union) Loss.

    IoU = |A ∩ B| / |A ∪ B|
    Loss = 1 - IoU
    """

    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        """Initialize IoU Loss.

        Args:
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU loss.

        Args:
            inputs: Predictions (B, 1, H, W) probabilities or logits
            targets: Ground truth masks (B, 1, H, W)

        Returns:
            IoU loss value
        """
        # Handle logits
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Compute intersection and union
        intersection = (inputs * targets).sum(dim=-1)
        union = inputs.sum(dim=-1) + targets.sum(dim=-1) - intersection

        # Compute IoU
        iou = (intersection + self.smooth) / (union + self.smooth)

        # Compute loss
        loss = 1 - iou

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalDiceLoss(nn.Module):
    """Combined Focal and Dice Loss.

    Combines focal loss for hard example mining with Dice loss
    for overlap optimization.
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        gamma: float = 2.0,
        alpha: float = 0.25,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        """Initialize Focal + Dice Loss.

        Args:
            focal_weight: Weight for Focal loss
            dice_weight: Weight for Dice loss
            gamma: Focal loss focusing parameter
            alpha: Focal loss alpha parameter
            smooth: Dice loss smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth, reduction=reduction)
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined Focal and Dice loss.

        Args:
            inputs: Predictions (B, 1, H, W) logits
            targets: Ground truth masks (B, 1, H, W)

        Returns:
            Combined loss value
        """
        # Compute binary focal loss
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        bce = F.binary_cross_entropy_with_logits(
            inputs_flat, targets_flat, reduction="none"
        )

        p = torch.sigmoid(inputs_flat)
        p_t = p * targets_flat + (1 - p) * (1 - targets_flat)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets_flat + (1 - self.alpha) * (1 - targets_flat)

        focal_loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()

        # Compute Dice loss
        dice_loss = self.dice(inputs, targets)

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


class TverskyLoss(nn.Module):
    """Tversky Loss for imbalanced segmentation.

    Generalizes Dice loss with alpha and beta parameters
    to control false positive and false negative penalties.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        """Initialize Tversky Loss.

        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Tversky loss.

        Args:
            inputs: Predictions (B, 1, H, W) probabilities or logits
            targets: Ground truth masks (B, 1, H, W)

        Returns:
            Tversky loss value
        """
        # Handle logits
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # True positives, false positives, false negatives
        tp = (inputs * targets).sum(dim=-1)
        fp = ((1 - targets) * inputs).sum(dim=-1)
        fn = (targets * (1 - inputs)).sum(dim=-1)

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Loss
        loss = 1 - tversky

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):
    """Boundary Loss for segmentation.

    Focuses on boundary regions for better edge segmentation.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        reduction: str = "mean",
    ):
        """Initialize Boundary Loss.

        Args:
            kernel_size: Kernel size for boundary extraction
            reduction: Reduction method
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.reduction = reduction

        # Create Laplacian kernel for boundary detection
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary loss.

        Args:
            inputs: Predictions (B, 1, H, W) probabilities or logits
            targets: Ground truth masks (B, 1, H, W)

        Returns:
            Boundary loss value
        """
        # Handle logits
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)

        # Extract boundaries
        pred_boundary = F.conv2d(inputs, self.kernel, padding=1)
        gt_boundary = F.conv2d(targets, self.kernel, padding=1)

        # Compute boundary loss
        loss = F.mse_loss(pred_boundary, gt_boundary, reduction=self.reduction)

        return loss


def get_segmentation_loss(
    loss_type: str,
    **kwargs,
) -> nn.Module:
    """Get segmentation loss function by name.

    Args:
        loss_type: Loss type ('dice', 'bce_dice', 'iou', 'focal_dice', 'tversky')
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function module
    """
    if loss_type == "dice":
        return DiceLoss(**kwargs)
    elif loss_type == "bce_dice":
        return BCEDiceLoss(**kwargs)
    elif loss_type == "iou":
        return IoULoss(**kwargs)
    elif loss_type == "focal_dice":
        return FocalDiceLoss(**kwargs)
    elif loss_type == "tversky":
        return TverskyLoss(**kwargs)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type == "boundary":
        return BoundaryLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
