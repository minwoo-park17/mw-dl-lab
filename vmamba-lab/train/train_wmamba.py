"""WMamba training script for face forgery detection."""

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

from dataset import FFPPDataset, get_train_transforms, get_test_transforms
from model import WMamba
from train.trainer import BaseTrainer
from train.losses import FocalLoss, LabelSmoothingLoss, get_classification_loss
from utils.logger import AverageMeter
from utils.device import move_to_device


class WMambaTrainer(BaseTrainer):
    """Trainer for WMamba face forgery detection model."""

    def _setup_loss(self, train_config: Dict) -> nn.Module:
        """Setup classification loss function.

        Args:
            train_config: Training configuration

        Returns:
            Loss function module
        """
        loss_config = train_config.get("loss", {})
        loss_type = loss_config.get("type", "focal")

        if loss_type == "focal":
            return FocalLoss(
                gamma=loss_config.get("focal_gamma", 2.0),
                alpha=torch.tensor([1 - loss_config.get("focal_alpha", 0.25),
                                   loss_config.get("focal_alpha", 0.25)]),
            ).to(self.device)
        elif loss_type == "label_smoothing":
            return LabelSmoothingLoss(
                smoothing=loss_config.get("label_smoothing", 0.1),
            ).to(self.device)
        else:
            return get_classification_loss(loss_type).to(self.device)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Acc")

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(images)
                    logits = output["logits"]
                    loss = self.criterion(logits, labels)

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
                logits = output["logits"]
                loss = self.criterion(logits, labels)

                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()

            # Compute accuracy
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()

            # Update meters
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.4f}",
            })

            self.global_step += 1

        return {
            "loss": loss_meter.avg,
            "acc": acc_meter.avg,
        }

    def validate(self) -> Dict[str, float]:
        """Validate model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        loss_meter = AverageMeter("Loss")
        all_labels = []
        all_preds = []
        all_scores = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                output = self.model(images)
                logits = output["logits"]
                probs = output["probs"]

                loss = self.criterion(logits, labels)
                loss_meter.update(loss.item(), images.size(0))

                # Store predictions
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_scores.extend(probs[:, 1].cpu().numpy())

        # Compute metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

        all_labels = list(all_labels)
        all_preds = list(all_preds)
        all_scores = list(all_scores)

        acc = accuracy_score(all_labels, all_preds)

        try:
            auc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc = 0.0

        try:
            ap = average_precision_score(all_labels, all_scores)
        except ValueError:
            ap = 0.0

        return {
            "loss": loss_meter.avg,
            "acc": acc,
            "auc": auc,
            "ap": ap,
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
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
    val_conf = data_conf.get("val", {})
    train_config = config.get("train", {})

    input_size = data_conf.get("input_size", 256)

    # Create transforms
    aug_config = train_config.get("augmentation", {})
    train_transform = get_train_transforms(
        input_size=input_size,
        horizontal_flip=aug_config.get("horizontal_flip", True),
        rotation=aug_config.get("rotation", 15),
        color_jitter=aug_config.get("color_jitter") is not None,
        jpeg_compression=aug_config.get("jpeg_compression"),
    )
    val_transform = get_test_transforms(input_size=input_size)

    # Get data paths from data_config
    ff_path = data_config.get("face_forgery", {}).get("ff++", {}).get("root", "")

    # Create datasets
    train_dataset = FFPPDataset(
        root=ff_path,
        split="train",
        transform=train_transform,
        compression=train_conf.get("compression", "c23"),
        methods=train_conf.get("methods"),
    )

    val_dataset = FFPPDataset(
        root=ff_path,
        split="val",
        transform=val_transform,
        compression=val_conf.get("compression", "c23"),
        methods=val_conf.get("methods"),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=True,
        num_workers=config.get("train", {}).get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("eval", {}).get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("eval", {}).get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train WMamba for face forgery detection")
    parser.add_argument(
        "--config",
        type=str,
        default="config/wmamba_config.yaml",
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
    model = WMamba(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = WMambaTrainer(
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
    print(f"Best AUC: {results['best_metric']:.4f}")


if __name__ == "__main__":
    main()
