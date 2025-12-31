"""
Validation Module
모델 평가 및 메트릭 계산 기능
"""

from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import AverageMeter


def calculate_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    세분화 메트릭을 계산합니다.

    Args:
        preds: 예측 확률 (sigmoid 적용된 상태)
        targets: 실제 마스크
        threshold: 이진화 임계값
        smooth: 0으로 나누기 방지를 위한 작은 값

    Returns:
        메트릭 딕셔너리 (dice, iou, precision, recall, accuracy)
    """
    # 이진화
    preds_binary = (preds > threshold).float()

    # Flatten
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)

    # True Positives, False Positives, False Negatives
    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()
    tn = ((1 - preds_flat) * (1 - targets_flat)).sum()

    # Dice Score (F1 Score)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    # IoU (Jaccard Index)
    iou = (tp + smooth) / (tp + fp + fn + smooth)

    # Precision
    precision = (tp + smooth) / (tp + fp + smooth)

    # Recall (Sensitivity)
    recall = (tp + smooth) / (tp + fn + smooth)

    # Accuracy
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item()
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    데이터셋에 대해 모델을 평가합니다.

    Args:
        model: 평가할 모델
        data_loader: 데이터 로더
        criterion: 손실 함수
        device: 평가에 사용할 디바이스
        threshold: 이진화 임계값

    Returns:
        평가 메트릭 딕셔너리
    """
    model.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.sigmoid(outputs)
            metrics = calculate_metrics(preds, masks, threshold)

            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            dice_meter.update(metrics['dice'], batch_size)
            iou_meter.update(metrics['iou'], batch_size)
            precision_meter.update(metrics['precision'], batch_size)
            recall_meter.update(metrics['recall'], batch_size)
            accuracy_meter.update(metrics['accuracy'], batch_size)

    return {
        'val_loss': loss_meter.avg,
        'val_dice': dice_meter.avg,
        'val_iou': iou_meter.avg,
        'val_precision': precision_meter.avg,
        'val_recall': recall_meter.avg,
        'val_accuracy': accuracy_meter.avg
    }


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    배치 단위로 예측을 수행합니다.

    Args:
        model: 예측에 사용할 모델
        data_loader: 데이터 로더
        device: 예측에 사용할 디바이스
        threshold: 이진화 임계값

    Returns:
        (predictions, probabilities, targets) 튜플
    """
    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Predicting"):
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(masks.numpy())

    return all_preds, all_probs, all_targets


def predict_single(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    단일 이미지에 대해 예측을 수행합니다.

    Args:
        model: 예측에 사용할 모델
        image: 입력 이미지 텐서 (C, H, W) 또는 (1, C, H, W)
        device: 예측에 사용할 디바이스
        threshold: 이진화 임계값

    Returns:
        (prediction, probability) 튜플
    """
    model.eval()

    # 배치 차원 추가
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output)
        pred = (prob > threshold).float()

    return pred.squeeze().cpu().numpy(), prob.squeeze().cpu().numpy()


def find_best_threshold(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    thresholds: Optional[List[float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    최적의 이진화 임계값을 찾습니다.

    Args:
        model: 평가할 모델
        data_loader: 데이터 로더
        device: 평가에 사용할 디바이스
        thresholds: 테스트할 임계값 목록

    Returns:
        (최적 임계값, 해당 임계값의 메트릭)
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    model.eval()

    # 모든 예측 수집
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Collecting predictions"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs)
            all_targets.append(masks.to(device))

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 각 임계값에 대해 메트릭 계산
    best_threshold = 0.5
    best_dice = 0
    best_metrics = None

    print("\nThreshold search results:")
    print("-" * 50)

    for threshold in thresholds:
        metrics = calculate_metrics(all_probs, all_targets, threshold)
        print(f"Threshold {threshold:.1f}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")

        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            best_threshold = threshold
            best_metrics = metrics

    print("-" * 50)
    print(f"Best threshold: {best_threshold:.1f} (Dice={best_dice:.4f})")

    return best_threshold, best_metrics


class ConfusionMatrix:
    """혼동 행렬 계산 및 시각화"""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """혼동 행렬 업데이트"""
        preds = preds.flatten().astype(np.int64)
        targets = targets.flatten().astype(np.int64)

        for p, t in zip(preds, targets):
            self.matrix[t, p] += 1

    def reset(self):
        """혼동 행렬 초기화"""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def get_metrics(self) -> Dict[str, float]:
        """혼동 행렬에서 메트릭 계산"""
        tp = self.matrix[1, 1]
        tn = self.matrix[0, 0]
        fp = self.matrix[0, 1]
        fn = self.matrix[1, 0]

        smooth = 1e-6

        return {
            'accuracy': (tp + tn) / (tp + tn + fp + fn + smooth),
            'precision': tp / (tp + fp + smooth),
            'recall': tp / (tp + fn + smooth),
            'specificity': tn / (tn + fp + smooth),
            'f1': 2 * tp / (2 * tp + fp + fn + smooth)
        }

    def __str__(self) -> str:
        return f"Confusion Matrix:\n{self.matrix}"
