"""Device management utilities."""

import os
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


def setup_device(
    gpu_ids: Optional[List[int]] = None,
    cudnn_benchmark: bool = True,
    cudnn_deterministic: bool = False,
) -> torch.device:
    """Setup computing device.

    Args:
        gpu_ids: List of GPU IDs to use. None for CPU.
        cudnn_benchmark: Enable cuDNN benchmark mode
        cudnn_deterministic: Enable cuDNN deterministic mode

    Returns:
        torch.device for computation
    """
    if gpu_ids is None or not torch.cuda.is_available():
        print("Using CPU")
        return torch.device("cpu")

    # Set CUDA device
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        torch.cuda.set_device(gpu_ids[0])

    # cuDNN settings
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic

    device = torch.device(f"cuda:{gpu_ids[0]}")
    print(f"Using GPU: {gpu_ids}")
    print(f"  - cuDNN benchmark: {cudnn_benchmark}")
    print(f"  - cuDNN deterministic: {cudnn_deterministic}")

    return device


def get_device() -> torch.device:
    """Get the current device.

    Returns:
        Current torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_to_device(
    data: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:
    """Move data to device recursively.

    Args:
        data: Data to move (tensor, list, tuple, or dict)
        device: Target device
        non_blocking: Use non-blocking transfer

    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device, non_blocking) for v in data)
    else:
        return data


def wrap_model(
    model: nn.Module,
    device: torch.device,
    gpu_ids: Optional[List[int]] = None,
    use_ddp: bool = False,
    sync_bn: bool = False,
) -> nn.Module:
    """Wrap model for multi-GPU training.

    Args:
        model: Model to wrap
        device: Target device
        gpu_ids: List of GPU IDs for DataParallel
        use_ddp: Use DistributedDataParallel instead of DataParallel
        sync_bn: Convert BatchNorm to SyncBatchNorm (for DDP)

    Returns:
        Wrapped model
    """
    model = model.to(device)

    if sync_bn and use_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if torch.cuda.is_available() and gpu_ids is not None and len(gpu_ids) > 1:
        if use_ddp:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        else:
            model = DataParallel(model, device_ids=gpu_ids)

    return model


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed
        deterministic: Enable deterministic algorithms (slower)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)


def get_gpu_memory_info() -> dict:
    """Get GPU memory information.

    Returns:
        Dictionary with memory info for each GPU
    """
    if not torch.cuda.is_available():
        return {}

    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2),
        }
    return info
