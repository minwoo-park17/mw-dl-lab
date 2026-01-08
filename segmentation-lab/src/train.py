"""
Training Module
Segmentation 모델 학습을 위한 메인 학습 루프 및 관련 기능
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

# 패키지 경로 추가 (직접 실행 지원)
sys.path.insert(0, str(Path(__file__).parent.parent))

# libpng warning 억제
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
warnings.filterwarnings("ignore", message=".*libpng.*")
warnings.filterwarnings("ignore", message=".*iCCP.*")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

# 절대 import (sys.path에 프로젝트 루트 추가됨)
from src.model import create_model
from src.dataset import create_dataloaders
from src.validation import evaluate, calculate_metrics
from src.utils import (
    load_config,
    setup_device,
    create_optimizer,
    create_scheduler,
    create_loss_function,
    EarlyStopping,
    CheckpointManager,
    AverageMeter
)


class Trainer:
    """
    Segmentation 모델 학습을 관리하는 클래스

    Args:
        arch_config: 아키텍처 설정 딕셔너리
        data_config: 데이터 설정 딕셔너리
        arch_config_path: 아키텍처 설정 파일 경로 (결과 저장용)
        data_config_path: 데이터 설정 파일 경로 (결과 저장용)
        model: 학습할 모델 (None이면 config에서 생성)
        device: 학습에 사용할 디바이스
    """

    def __init__(
        self,
        arch_config: dict,
        data_config: dict,
        arch_config_path: str = None,
        data_config_path: str = None,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        self.arch_config = arch_config
        self.data_config = data_config
        self.device = device or setup_device()

        # 모델 설정
        if model is None:
            self.model = create_model(arch_config)
        else:
            self.model = model
        self.model = self.model.to(self.device)

        # 손실 함수, 옵티마이저, 스케줄러
        self.criterion = create_loss_function(arch_config)
        self.optimizer = create_optimizer(self.model, arch_config)
        self.scheduler = create_scheduler(self.optimizer, arch_config)

        # 체크포인트 및 조기 종료
        checkpoint_config = arch_config.get('checkpoint', {})
        self.checkpoint_manager = CheckpointManager(
            base_dir=checkpoint_config.get('save_dir', 'results'),
            monitor=checkpoint_config.get('monitor', 'val_dice'),
            mode=checkpoint_config.get('mode', 'max'),
            arch_config_path=arch_config_path,
            data_config_path=data_config_path
        )

        early_stopping_config = arch_config.get('training', {}).get('early_stopping', {})
        self.early_stopping = None
        if early_stopping_config.get('enabled', False):
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                min_delta=early_stopping_config.get('min_delta', 0.001),
                mode=checkpoint_config.get('mode', 'max')
            )

        # 학습 설정
        training_config = arch_config.get('training', {})
        self.epochs = training_config.get('epochs', 100)
        self.print_freq = arch_config.get('logging', {}).get('print_freq', 10)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        한 에폭 동안 모델을 학습합니다.

        Args:
            train_loader: 학습 데이터 로더
            epoch: 현재 에폭 번호

        Returns:
            학습 메트릭 딕셔너리
        """
        self.model.train()

        loss_meter = AverageMeter()
        dice_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # 손실 계산
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # 메트릭 계산
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                metrics = calculate_metrics(preds, masks)

            # 메트릭 업데이트
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            dice_meter.update(metrics['dice'], batch_size)

            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'dice': f'{dice_meter.avg:.4f}'
            })

        return {
            'train_loss': loss_meter.avg,
            'train_dice': dice_meter.avg
        }

    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        검증 데이터로 모델을 평가합니다.

        Args:
            val_loader: 검증 데이터 로더

        Returns:
            검증 메트릭 딕셔너리
        """
        return evaluate(
            model=self.model,
            data_loader=val_loader,
            criterion=self.criterion,
            device=self.device
        )

    def _save_training_plot(self, history: Dict[str, list]):
        """학습 곡선 그래프를 저장합니다 (매 에폭 덮어쓰기)."""
        import matplotlib.pyplot as plt

        epochs = range(1, len(history['train_loss']) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss 그래프
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Dice 그래프
        axes[1].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
        axes[1].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Training & Validation Dice')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        save_path = self.checkpoint_manager.save_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=100)
        plt.close()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, list]:
        """
        모델을 학습합니다.

        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더

        Returns:
            학습 히스토리 딕셔너리
        """
        print(f"\nTraining on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("-" * 50)

        history = {
            'train_loss': [], 'train_dice': [],
            'val_loss': [], 'val_dice': [], 'val_iou': [],
            'lr': []
        }

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # 학습
            train_metrics = self.train_epoch(train_loader, epoch)

            # 검증
            val_metrics = self.validate(val_loader)

            # 스케줄러 업데이트
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_dice'])
                else:
                    self.scheduler.step()

            # 히스토리 업데이트
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_dice'].append(train_metrics['train_dice'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_dice'].append(val_metrics['val_dice'])
            history['val_iou'].append(val_metrics['val_iou'])
            history['lr'].append(current_lr)

            # 학습 곡선 그래프 저장 (매 에폭 덮어쓰기)
            self._save_training_plot(history)

            # 에폭 결과 출력
            epoch_time = time.time() - epoch_start
            print(
                f"\nEpoch {epoch + 1}/{self.epochs} "
                f"({epoch_time:.1f}s) - "
                f"loss: {train_metrics['train_loss']:.4f}, "
                f"dice: {train_metrics['train_dice']:.4f}, "
                f"val_loss: {val_metrics['val_loss']:.4f}, "
                f"val_dice: {val_metrics['val_dice']:.4f}, "
                f"val_iou: {val_metrics['val_iou']:.4f}, "
                f"lr: {current_lr:.6f}"
            )

            # 체크포인트 저장
            monitor_value = val_metrics[self.checkpoint_manager.monitor]
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=val_metrics,
                monitor_value=monitor_value
            )

            # 조기 종료 확인
            if self.early_stopping is not None:
                if self.early_stopping(monitor_value):
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        print("\nTraining completed!")
        print(f"Best {self.checkpoint_manager.monitor}: {self.checkpoint_manager.best_value:.4f}")

        return history

    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트에서 모델을 로드합니다."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', 0)


def train(
    arch_config_path: str = "config/architecture.yaml",
    data_config_path: str = "config/data.yaml"
):
    """
    설정 파일을 기반으로 학습을 실행합니다.

    Args:
        arch_config_path: 아키텍처 설정 파일 경로
        data_config_path: 데이터 설정 파일 경로
    """
    # 설정 로드
    arch_config = load_config(arch_config_path)
    data_config = load_config(data_config_path)

    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        data_config_path=data_config_path,
        arch_config_path=arch_config_path
    )

    # 트레이너 생성 및 학습
    trainer = Trainer(
        arch_config=arch_config,
        data_config=data_config,
        arch_config_path=arch_config_path,
        data_config_path=data_config_path
    )
    history = trainer.fit(train_loader, val_loader)

    # 테스트 데이터가 있으면 최종 평가
    if test_loader is not None:
        print("\n" + "=" * 50)
        print("Final Test Evaluation")
        print("=" * 50)

        # Best 모델 로드
        best_path = trainer.checkpoint_manager.get_best_checkpoint_path()
        if best_path and os.path.exists(best_path):
            trainer.load_checkpoint(best_path)

        test_metrics = trainer.validate(test_loader)
        print(f"Test Loss: {test_metrics['val_loss']:.4f}")
        print(f"Test Dice: {test_metrics['val_dice']:.4f}")
        print(f"Test IoU: {test_metrics['val_iou']:.4f}")

    return trainer, history


if __name__ == "__main__":
    # =========================================================================
    # 설정
    # =========================================================================
    ARCH_CONFIG = "config/architecture.yaml"
    DATA_CONFIG = "config/data.yaml"

    # =========================================================================
    # 실행
    # =========================================================================
    train(ARCH_CONFIG, DATA_CONFIG)
