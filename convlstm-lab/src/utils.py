"""
유틸리티 함수 모듈
"""

import os
import random
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML config 파일 로드

    Args:
        config_path: config 파일 경로

    Returns:
        config 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    여러 config 딕셔너리 병합

    Args:
        configs: 병합할 config들

    Returns:
        병합된 config
    """
    result = {}
    for config in configs:
        result.update(config)
    return result


def set_seed(seed: int):
    """
    재현성을 위한 시드 설정

    Args:
        seed: 랜덤 시드
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN 결정론적 동작
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    path: str,
    **kwargs
):
    """
    모델 체크포인트 저장

    Args:
        model: 저장할 모델
        optimizer: 옵티마이저
        epoch: 현재 에폭
        loss: 손실값
        accuracy: 정확도
        path: 저장 경로
        **kwargs: 추가 저장 항목
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        **kwargs
    }

    # 디렉토리 생성
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    체크포인트 로드

    Args:
        path: 체크포인트 경로
        model: 모델 인스턴스
        optimizer: 옵티마이저 (선택적)
        device: 디바이스

    Returns:
        체크포인트 정보 딕셔너리
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded: {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  Accuracy: {checkpoint.get('accuracy', 'N/A'):.4f}")

    return checkpoint


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
) -> int:
    """
    비디오에서 프레임 추출

    Args:
        video_path: 비디오 파일 경로
        output_dir: 프레임 저장 디렉토리
        fps: 추출할 FPS (None이면 원본 FPS)
        max_frames: 최대 프레임 수 (None이면 전체)
        resize: 리사이즈 크기 (width, height)

    Returns:
        추출된 프레임 수
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FPS 기반 프레임 간격 계산
    if fps is not None and fps < video_fps:
        frame_interval = int(video_fps / fps)
    else:
        frame_interval = 1

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize)

            # 프레임 저장
            frame_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    print(f"Extracted {saved_count} frames from {video_path}")
    return saved_count


def extract_frames_from_dir(
    video_dir: str,
    output_dir: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
):
    """
    디렉토리 내 모든 비디오에서 프레임 추출

    Args:
        video_dir: 비디오 디렉토리
        output_dir: 프레임 저장 디렉토리
        fps: 추출할 FPS
        max_frames: 비디오당 최대 프레임 수
        resize: 리사이즈 크기
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    for video_path in video_dir.iterdir():
        if video_path.suffix.lower() in video_extensions:
            video_output_dir = output_dir / video_path.stem
            extract_frames(
                str(video_path),
                str(video_output_dir),
                fps=fps,
                max_frames=max_frames,
                resize=resize
            )


class AverageMeter:
    """평균값 추적 클래스"""

    def __init__(self, name: str = ''):
        self.name = name
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
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """조기 종료 클래스"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        """
        Args:
            patience: 개선 없이 기다릴 에폭 수
            min_delta: 개선으로 인정할 최소 변화량
            mode: 'min' (loss) 또는 'max' (accuracy)
        """
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

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_device(device_str: str = 'cuda') -> torch.device:
    """
    디바이스 가져오기

    Args:
        device_str: 'cuda' 또는 'cpu'

    Returns:
        torch.device
    """
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    모델 파라미터 수 계산

    Args:
        model: PyTorch 모델

    Returns:
        (전체 파라미터 수, 학습 가능 파라미터 수)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def visualize_prediction(
    frames: np.ndarray,
    prediction: int,
    confidence: float,
    save_path: str,
    num_display: int = 8
):
    """
    예측 결과 시각화

    Args:
        frames: 프레임 시퀀스 (T, H, W, C)
        prediction: 예측 클래스 (0: real, 1: fake)
        confidence: 신뢰도
        save_path: 저장 경로
        num_display: 표시할 프레임 수
    """
    import matplotlib.pyplot as plt

    seq_len = len(frames)
    indices = np.linspace(0, seq_len - 1, min(num_display, seq_len), dtype=int)

    fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))

    if len(indices) == 1:
        axes = [axes]

    label = 'FAKE' if prediction == 1 else 'REAL'
    color = 'red' if prediction == 1 else 'green'

    for idx, ax in zip(indices, axes):
        ax.imshow(frames[idx])
        ax.axis('off')
        ax.set_title(f'Frame {idx}')

    fig.suptitle(f'Prediction: {label} ({confidence:.2%})', color=color, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved: {save_path}")
