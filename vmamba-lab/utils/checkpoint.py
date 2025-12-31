"""Checkpoint utilities for saving and loading models."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Union[str, Path],
    scheduler: Optional[Any] = None,
    is_best: bool = False,
    config: Optional[Dict] = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        is_best: Whether this is the best model so far
        config: Optional config dictionary
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Handle DataParallel or DistributedDataParallel
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, filepath)

    # Save best model separately
    if is_best:
        best_path = filepath.parent / "best_model.pth"
        torch.save(checkpoint, best_path)


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load checkpoint to
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Dictionary containing checkpoint info (epoch, metrics, etc.)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)

    # Handle DataParallel or DistributedDataParallel
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", None),
    }


def load_pretrained(
    model: nn.Module,
    pretrained_path: Union[str, Path],
    prefix: str = "",
    ignore_keys: Optional[list] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load pretrained weights with flexible key matching.

    Args:
        model: Model to load weights into
        pretrained_path: Path to pretrained weights
        prefix: Prefix to add/remove from state dict keys
        ignore_keys: List of keys to ignore
        device: Device to load weights to

    Returns:
        Model with loaded weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ignore_keys is None:
        ignore_keys = []

    # Load pretrained state dict
    if str(pretrained_path).startswith("http"):
        state_dict = torch.hub.load_state_dict_from_url(
            pretrained_path, map_location=device
        )
    else:
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    # Handle different key formats
    model_state = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        # Remove 'module.' prefix if present
        if k.startswith("module."):
            k = k[7:]

        # Add or remove prefix
        if prefix:
            if k.startswith(prefix):
                k = k[len(prefix):]
            else:
                k = prefix + k

        # Skip ignored keys
        if any(ignore_key in k for ignore_key in ignore_keys):
            continue

        # Check if key exists in model
        if k in model_state and v.shape == model_state[k].shape:
            new_state_dict[k] = v

    # Load matching weights
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print(f"Loaded {len(new_state_dict)} / {len(model_state)} parameters")
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")

    return model


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Get the latest checkpoint from a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0]
