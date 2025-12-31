"""Base trainer class for model training."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.device import get_device, move_to_device, set_seed
from utils.logger import AverageMeter, get_logger, log_metrics, setup_wandb


class BaseTrainer(ABC):
    """Abstract base trainer class."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """Initialize base trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.config = config
        self.device = device or get_device()

        # Set seed for reproducibility
        seed = config.get("seed", 42)
        set_seed(seed)

        # Model
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training config
        train_config = config.get("train", {})
        self.epochs = train_config.get("epochs", 50)
        self.start_epoch = 0
        self.global_step = 0

        # Setup optimizer
        self.optimizer = self._setup_optimizer(train_config)

        # Setup scheduler
        self.scheduler = self._setup_scheduler(train_config)

        # Setup loss
        self.criterion = self._setup_loss(train_config)

        # Mixed precision
        self.use_amp = train_config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.grad_clip = train_config.get("gradient_clip", 1.0)

        # Early stopping
        early_stop_config = train_config.get("early_stopping", {})
        self.early_stopping_patience = early_stop_config.get("patience", 10)
        self.early_stopping_min_delta = early_stop_config.get("min_delta", 0.001)
        self.best_metric = None
        self.patience_counter = 0

        # Logging
        logging_config = config.get("logging", {})
        self.log_interval = logging_config.get("log_interval", 100)
        self.save_interval = logging_config.get("save_interval", 1)

        # Output directories
        output_config = config.get("output", {})
        self.output_dir = Path(output_config.get("base_dir", "outputs"))
        self.checkpoint_dir = self.output_dir / output_config.get("checkpoint_dir", "checkpoints")
        self.log_dir = self.output_dir / output_config.get("log_dir", "logs")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        model_name = config.get("model", {}).get("name", "model")
        self.logger = get_logger(
            model_name,
            log_file=str(self.log_dir / f"{model_name}_train.log"),
        )

        # Wandb
        self.wandb_run = None
        if logging_config.get("use_wandb", False):
            self.wandb_run = setup_wandb(
                config,
                project=logging_config.get("wandb_project", "mamba-forgery"),
                entity=logging_config.get("wandb_entity"),
            )

    def _setup_optimizer(self, train_config: Dict) -> torch.optim.Optimizer:
        """Setup optimizer.

        Args:
            train_config: Training configuration

        Returns:
            Optimizer instance
        """
        opt_config = train_config.get("optimizer", {})
        opt_type = opt_config.get("type", "adamw").lower()
        lr = opt_config.get("lr", 1e-4)
        weight_decay = opt_config.get("weight_decay", 0.05)

        if opt_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get("betas", (0.9, 0.999)),
            )
        elif opt_type == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=opt_config.get("momentum", 0.9),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        return optimizer

    def _setup_scheduler(self, train_config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler.

        Args:
            train_config: Training configuration

        Returns:
            Scheduler instance or None
        """
        sched_config = train_config.get("scheduler", {})
        sched_type = sched_config.get("type", "cosine").lower()

        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - sched_config.get("warmup_epochs", 5),
                eta_min=sched_config.get("min_lr", 1e-6),
            )
        elif sched_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get("step_size", 10),
                gamma=sched_config.get("gamma", 0.1),
            )
        elif sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=sched_config.get("patience", 5),
                factor=sched_config.get("factor", 0.5),
            )
        elif sched_type == "none":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")

        return scheduler

    @abstractmethod
    def _setup_loss(self, train_config: Dict) -> nn.Module:
        """Setup loss function. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Validate model. Must be implemented by subclasses."""
        pass

    def train(self) -> Dict[str, Any]:
        """Full training loop.

        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")

        best_epoch = 0
        history = {"train": [], "val": []}

        for epoch in range(self.start_epoch, self.epochs):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            self.logger.info(f"{'='*50}")

            # Train
            train_metrics = self.train_epoch(epoch)
            history["train"].append(train_metrics)

            # Validate
            val_metrics = self.validate()
            history["val"].append(val_metrics)

            # Log metrics
            log_metrics(
                train_metrics, epoch, "train",
                self.logger, self.wandb_run,
            )
            log_metrics(
                val_metrics, epoch, "val",
                self.logger, self.wandb_run,
            )

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("auc", val_metrics.get("f1", 0)))
                else:
                    self.scheduler.step()

            # Check for best model
            current_metric = val_metrics.get("auc", val_metrics.get("f1", 0))
            is_best = False

            if self.best_metric is None or current_metric > self.best_metric + self.early_stopping_min_delta:
                self.best_metric = current_metric
                best_epoch = epoch
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0 or is_best:
                self._save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        self.logger.info(f"\nTraining completed!")
        self.logger.info(f"Best epoch: {best_epoch + 1}, Best metric: {self.best_metric:.4f}")

        return {
            "best_epoch": best_epoch,
            "best_metric": self.best_metric,
            "history": history,
        }

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Save training checkpoint.

        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            metrics,
            checkpoint_path,
            self.scheduler,
            is_best,
            self.config,
        )

        if is_best:
            self.logger.info(f"Saved best model with metric: {self.best_metric:.4f}")

    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        info = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.device,
        )

        self.start_epoch = info["epoch"] + 1
        self.best_metric = info["metrics"].get("auc", info["metrics"].get("f1"))

        self.logger.info(f"Resumed from epoch {self.start_epoch}")
