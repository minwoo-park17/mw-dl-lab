"""
Utility Module
학습에 필요한 유틸리티 함수들
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import yaml
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, SGD, AdamW
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau
)


def load_config(config_path: str) -> dict:
    """
    YAML 설정 파일을 로드합니다.

    Args:
        config_path: 설정 파일 경로

    Returns:
        설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, save_path: str):
    """설정을 YAML 파일로 저장합니다."""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def setup_device() -> torch.device:
    """사용 가능한 최적의 디바이스를 반환합니다."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def create_optimizer(model: nn.Module, config: dict) -> Optimizer:
    """
    설정에 따라 옵티마이저를 생성합니다.

    Args:
        model: 최적화할 모델
        config: 전체 설정 딕셔너리

    Returns:
        생성된 옵티마이저
    """
    training_config = config.get('training', {})
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    lr = training_config.get('learning_rate', 0.001)

    optimizers = {
        'adam': lambda: Adam(model.parameters(), lr=lr),
        'sgd': lambda: SGD(model.parameters(), lr=lr, momentum=0.9),
        'adamw': lambda: AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizers[optimizer_name]()


def create_scheduler(
    optimizer: Optimizer,
    config: dict
) -> Optional[_LRScheduler]:
    """
    설정에 따라 학습률 스케줄러를 생성합니다.

    Args:
        optimizer: 옵티마이저
        config: 전체 설정 딕셔너리

    Returns:
        생성된 스케줄러 또는 None
    """
    training_config = config.get('training', {})
    scheduler_name = training_config.get('scheduler', 'cosine').lower()
    epochs = training_config.get('epochs', 100)

    schedulers = {
        'cosine': lambda: CosineAnnealingLR(optimizer, T_max=epochs),
        'step': lambda: StepLR(optimizer, step_size=30, gamma=0.1),
        'plateau': lambda: ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        ),
        'none': lambda: None
    }

    if scheduler_name not in schedulers:
        print(f"Unknown scheduler: {scheduler_name}, using none")
        return None

    return schedulers[scheduler_name]()


# ============================================================================
# Loss Functions
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss"""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """BCE + Dice 결합 손실 함수"""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            preds, targets, reduction='none'
        )
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


def create_loss_function(config: dict) -> nn.Module:
    """
    설정에 따라 손실 함수를 생성합니다.

    Args:
        config: 전체 설정 딕셔너리

    Returns:
        손실 함수
    """
    loss_config = config.get('loss', {})
    loss_name = loss_config.get('name', 'bce_dice').lower()

    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'bce_dice':
        return BCEDiceLoss(
            bce_weight=loss_config.get('bce_weight', 0.5),
            dice_weight=loss_config.get('dice_weight', 0.5)
        )
    elif loss_name == 'focal':
        return FocalLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# ============================================================================
# Training Utilities
# ============================================================================

class AverageMeter:
    """평균값 추적을 위한 클래스"""

    def __init__(self):
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


class EarlyStopping:
    """조기 종료를 위한 클래스"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class CheckpointManager:
    """체크포인트 관리 클래스"""

    def __init__(
        self,
        base_dir: str = 'results',
        monitor: str = 'val_dice',
        mode: str = 'max',
        arch_config_path: str = None,
        data_config_path: str = None
    ):
        from datetime import datetime
        import shutil

        # 타임스탬프 폴더 생성 (train_YYMMDDHHMMSS)
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        self.save_dir = Path(base_dir) / f"train_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.best_value = float('-inf') if mode == 'max' else float('inf')

        # 설정 파일 복사
        if arch_config_path and Path(arch_config_path).exists():
            shutil.copy(arch_config_path, self.save_dir / "architecture.yaml")

        if data_config_path and Path(data_config_path).exists():
            shutil.copy(data_config_path, self.save_dir / "data.yaml")

        # augmentation.py 내용 복사
        aug_path = Path(__file__).parent / "augmentation.py"
        if aug_path.exists():
            with open(aug_path, 'r', encoding='utf-8') as f:
                aug_content = f.read()
            with open(self.save_dir / "augmentation.txt", 'w', encoding='utf-8') as f:
                f.write(aug_content)

        print(f"Results will be saved to: {self.save_dir}")

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        monitor_value: float
    ):
        """체크포인트 저장 (best + last)"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_value': self.best_value
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # 항상 last 모델 저장
        last_path = self.save_dir / 'last_model.pth'
        torch.save(checkpoint, last_path)

        # 개선 시 best 모델 저장
        if self.mode == 'max':
            improved = monitor_value > self.best_value
        else:
            improved = monitor_value < self.best_value

        if improved:
            self.best_value = monitor_value
            checkpoint['best_value'] = self.best_value
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  >> Saved best model ({self.monitor}: {monitor_value:.4f})")

    def get_best_checkpoint_path(self) -> Optional[str]:
        """최상의 체크포인트 경로 반환"""
        best_path = self.save_dir / 'best_model.pth'
        return str(best_path) if best_path.exists() else None

    def get_last_checkpoint_path(self) -> Optional[str]:
        """마지막 체크포인트 경로 반환"""
        last_path = self.save_dir / 'last_model.pth'
        return str(last_path) if last_path.exists() else None


class MetricTracker:
    """메트릭 추적 클래스"""

    def __init__(self):
        self.metrics = {}

    def update(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get(self, name: str) -> list:
        return self.metrics.get(name, [])

    def get_last(self, name: str) -> Optional[float]:
        values = self.metrics.get(name, [])
        return values[-1] if values else None

    def get_best(self, name: str, mode: str = 'max') -> Optional[float]:
        values = self.metrics.get(name, [])
        if not values:
            return None
        return max(values) if mode == 'max' else min(values)


def setup_logging(config: dict) -> logging.Logger:
    """로깅 설정"""
    logging_config = config.get('logging', {})
    log_dir = logging_config.get('log_dir', 'logs')

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('segmentation')
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    모델의 파라미터 수를 계산합니다.

    Returns:
        (전체 파라미터 수, 학습 가능 파라미터 수)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
