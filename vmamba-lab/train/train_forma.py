"""ForMa training script for image tampering localization."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import CASIADataset, get_segmentation_transforms
from model import ForMa
from train.trainer import BaseTrainer
from train.losses import BCEDiceLoss, DiceLoss, FocalDiceLoss, get_segmentation_loss
from utils.logger import AverageMeter
from utils.device import move_to_device


class ForMaTrainer(BaseTrainer):
    """Trainer for ForMa image tampering localization model."""

    def _setup_loss(self, train_config: Dict) -> nn.Module:
        """Setup segmentation loss function.

        Args:
            train_config: Training configuration

        Returns:
            Loss function module
        """
        loss_config = train_config.get("loss", {})
        loss_type = loss_config.get("type", "bce_dice")

        if loss_type == "bce_dice":
            return BCEDiceLoss(
                bce_weight=loss_config.get("bce_weight", 0.5),
                dice_weight=loss_config.get("dice_weight", 0.5),
            ).to(self.device)
        elif loss_type == "focal_dice":
            return FocalDiceLoss(
                focal_weight=loss_config.get("focal_weight", 0.5),
                dice_weight=loss_config.get("dice_weight", 0.5),
                gamma=loss_config.get("focal_gamma", 2.0),
            ).to(self.device)
        else:
            return get_segmentation_loss(loss_type).to(self.device)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        loss_meter = AverageMeter("Loss")
        f1_meter = AverageMeter("F1")

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Ensure mask has channel dimension
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(images)
                    pred_mask = output["logits"]
                    loss = self.criterion(pred_mask, masks)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(images)
                pred_mask = output["logits"]
                loss = self.criterion(pred_mask, masks)

                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()

            # Compute F1 score
            pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
            f1 = self._compute_f1(pred_binary, masks)

            # Update meters
            loss_meter.update(loss.item(), images.size(0))
            f1_meter.update(f1.item(), images.size(0))

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "f1": f"{f1_meter.avg:.4f}",
            })

            self.global_step += 1

        return {
            "loss": loss_meter.avg,
            "f1": f1_meter.avg,
        }

    def validate(self) -> Dict[str, float]:
        """Validate model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        loss_meter = AverageMeter("Loss")
        f1_meter = AverageMeter("F1")
        iou_meter = AverageMeter("IoU")
        pixel_auc_meter = AverageMeter("Pixel AUC")

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)

                output = self.model(images)
                pred_mask = output["logits"]
                pred_prob = output["mask"]

                loss = self.criterion(pred_mask, masks)
                loss_meter.update(loss.item(), images.size(0))

                # Compute metrics
                pred_binary = (pred_prob > 0.5).float()
                f1 = self._compute_f1(pred_binary, masks)
                iou = self._compute_iou(pred_binary, masks)

                f1_meter.update(f1.item(), images.size(0))
                iou_meter.update(iou.item(), images.size(0))

                # Pixel-level AUC
                try:
                    pixel_auc = self._compute_pixel_auc(pred_prob, masks)
                    pixel_auc_meter.update(pixel_auc, images.size(0))
                except Exception:
                    pass

        return {
            "loss": loss_meter.avg,
            "f1": f1_meter.avg,
            "iou": iou_meter.avg,
            "pixel_auc": pixel_auc_meter.avg,
        }

    def _compute_f1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-6,
    ) -> torch.Tensor:
        """Compute F1 score.

        Args:
            pred: Predicted binary mask
            target: Ground truth mask
            smooth: Smoothing factor

        Returns:
            F1 score
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()

        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)

        f1 = 2 * precision * recall / (precision + recall + smooth)
        return f1

    def _compute_iou(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-6,
    ) -> torch.Tensor:
        """Compute IoU score.

        Args:
            pred: Predicted binary mask
            target: Ground truth mask
            smooth: Smoothing factor

        Returns:
            IoU score
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou

    def _compute_pixel_auc(
        self,
        pred_prob: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Compute pixel-level AUC.

        Args:
            pred_prob: Predicted probability map
            target: Ground truth mask

        Returns:
            Pixel AUC score
        """
        from sklearn.metrics import roc_auc_score

        pred_flat = pred_prob.cpu().numpy().flatten()
        target_flat = target.cpu().numpy().flatten()

        # Skip if all same class
        if len(set(target_flat)) < 2:
            return 0.5

        return roc_auc_score(target_flat, pred_flat)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
) -> tuple:
    """Create training and validation data loaders.

    Args:
        config: Main configuration
        data_config: Data configuration

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_conf = config.get("data", {})
    train_conf = data_conf.get("train", {})
    train_config = config.get("train", {})

    input_size = data_conf.get("input_size", 512)

    # Create transforms
    aug_config = train_config.get("augmentation", {})
    train_transform = get_segmentation_transforms(
        input_size=input_size,
        split="train",
        horizontal_flip=aug_config.get("horizontal_flip", True),
        vertical_flip=aug_config.get("vertical_flip", True),
        rotation=aug_config.get("rotation", 90),
        scale=aug_config.get("scale"),
        color_jitter=aug_config.get("color_jitter") is not None,
        jpeg_compression=aug_config.get("jpeg_compression"),
    )
    val_transform = get_segmentation_transforms(
        input_size=input_size,
        split="val",
    )

    # Get data paths from data_config
    casia_path = data_config.get("tampering", {}).get("casia", {}).get("root", "")

    # Create datasets
    train_dataset = CASIADataset(
        root=casia_path,
        split="train",
        transform=train_transform,
        version="v2",
    )

    val_dataset = CASIADataset(
        root=casia_path,
        split="val",
        transform=val_transform,
        version="v2",
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=True,
        num_workers=config.get("train", {}).get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("eval", {}).get("batch_size", 16),
        shuffle=False,
        num_workers=config.get("eval", {}).get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ForMa for image tampering localization")
    parser.add_argument(
        "--config",
        type=str,
        default="config/forma_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="data/data_path.yaml",
        help="Path to data config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    data_config = load_config(args.data_config)

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(config, data_config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("Creating model...")
    model = ForMa(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = ForMaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)

    # Train
    results = trainer.train()

    print("\nTraining completed!")
    print(f"Best epoch: {results['best_epoch'] + 1}")
    print(f"Best F1: {results['best_metric']:.4f}")


if __name__ == "__main__":
    main()
