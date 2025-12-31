"""Classification loss functions for face forgery detection."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: int = 2,
    ):
        """Initialize Focal Loss.

        Args:
            alpha: Class weights tensor of shape (num_classes,)
            gamma: Focusing parameter (>= 0)
            reduction: Reduction method ('none', 'mean', 'sum')
            num_classes: Number of classes
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", alpha)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predictions (B, num_classes) logits
            targets: Ground truth labels (B,) integers

        Returns:
            Focal loss value
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)

        # Get probabilities for target classes
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        # Compute focal loss
        loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss for binary classification.

    More efficient implementation for binary classification.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """Initialize Binary Focal Loss.

        Args:
            alpha: Weight for positive class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary focal loss.

        Args:
            inputs: Predictions (B,) or (B, 1) logits
            targets: Ground truth labels (B,) binary

        Returns:
            Binary focal loss value
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Compute sigmoid probabilities
        p = torch.sigmoid(inputs)

        # Compute BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # Compute focal weight
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * focal_weight

        # Compute focal loss
        loss = focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy Loss.

    Regularizes the model by preventing overconfident predictions.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int = 2,
        reduction: str = "mean",
    ):
        """Initialize Label Smoothing Loss.

        Args:
            smoothing: Label smoothing factor (0 = no smoothing)
            num_classes: Number of classes
            reduction: Reduction method
        """
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute label smoothing loss.

        Args:
            inputs: Predictions (B, num_classes) logits
            targets: Ground truth labels (B,) integers

        Returns:
            Label smoothing loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smooth labels
        smooth_labels = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Compute loss
        loss = -smooth_labels * log_probs
        loss = loss.sum(dim=-1)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Paper: "Asymmetric Loss For Multi-Label Classification" (Ridnik et al., 2021)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        """Initialize Asymmetric Loss.

        Args:
            gamma_neg: Focusing parameter for negative samples
            gamma_pos: Focusing parameter for positive samples
            clip: Probability clipping threshold
            reduction: Reduction method
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute asymmetric loss.

        Args:
            inputs: Predictions (B, C) logits
            targets: Ground truth (B, C) multi-hot or (B,) integers

        Returns:
            Asymmetric loss value
        """
        # Handle integer targets
        if targets.dim() == 1:
            targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        else:
            targets_onehot = targets.float()

        # Compute probabilities
        p = torch.sigmoid(inputs)

        # Clip probabilities for negative samples
        p_neg = (p + self.clip).clamp(max=1)

        # Compute losses for positive and negative samples
        pos_loss = targets_onehot * torch.log(p.clamp(min=1e-8))
        neg_loss = (1 - targets_onehot) * torch.log(1 - p_neg.clamp(max=1 - 1e-8))

        # Apply asymmetric focusing
        pos_weight = (1 - p) ** self.gamma_pos
        neg_weight = p_neg ** self.gamma_neg

        loss = -pos_weight * pos_loss - neg_weight * neg_loss
        loss = loss.sum(dim=-1)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_classification_loss(
    loss_type: str,
    **kwargs,
) -> nn.Module:
    """Get classification loss function by name.

    Args:
        loss_type: Loss type ('focal', 'bce', 'ce', 'label_smoothing')
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function module
    """
    if loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "binary_focal":
        return BinaryFocalLoss(**kwargs)
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type == "ce" or loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == "asymmetric":
        return AsymmetricLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
