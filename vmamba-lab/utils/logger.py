"""Logging utilities for training and evaluation."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Get a logger with console and optional file handlers.

    Args:
        name: Logger name
        log_file: Optional path to log file
        log_level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_wandb(
    config: Dict[str, Any],
    project: str,
    name: Optional[str] = None,
    entity: Optional[str] = None,
    resume: bool = False,
) -> Optional[Any]:
    """Setup Weights & Biases logging.

    Args:
        config: Configuration dictionary to log
        project: W&B project name
        name: Run name (auto-generated if None)
        entity: W&B entity/username
        resume: Whether to resume a previous run

    Returns:
        W&B run object or None if wandb not available
    """
    try:
        import wandb

        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = config.get("model", {}).get("name", "unknown")
            name = f"{model_name}_{timestamp}"

        run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            config=config,
            resume="allow" if resume else None,
        )
        return run

    except ImportError:
        print("wandb not installed. Skipping W&B logging.")
        return None


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = "",
    logger: Optional[logging.Logger] = None,
    wandb_run: Optional[Any] = None,
    tensorboard_writer: Optional[Any] = None,
) -> None:
    """Log metrics to various backends.

    Args:
        metrics: Dictionary of metric names and values
        step: Current step/epoch
        prefix: Prefix for metric names (e.g., "train/", "val/")
        logger: Python logger for console output
        wandb_run: W&B run object
        tensorboard_writer: TensorBoard SummaryWriter
    """
    # Add prefix to metric names
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

    # Console logging
    if logger:
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in prefixed_metrics.items()])
        logger.info(f"Step {step} | {metric_str}")

    # W&B logging
    if wandb_run is not None:
        try:
            import wandb

            wandb.log(prefixed_metrics, step=step)
        except Exception:
            pass

    # TensorBoard logging
    if tensorboard_writer is not None:
        for name, value in prefixed_metrics.items():
            tensorboard_writer.add_scalar(name, value, step)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display progress for training."""

    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int, logger: Optional[logging.Logger] = None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        message = "\t".join(entries)
        if logger:
            logger.info(message)
        else:
            print(message)

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
