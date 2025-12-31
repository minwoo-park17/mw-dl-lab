"""
프레임 시퀀스 데이터셋 모듈
연속된 프레임을 로드하여 ConvLSTM 입력 형태로 변환
"""

import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A

from .transforms import apply_sequence_transform, get_train_transforms, get_val_transforms


class FrameSequenceDataset(Dataset):
    """
    연속 프레임 시퀀스를 로드하는 데이터셋

    폴더 구조:
        data_dir/
            real/
                video001/
                    frame_0001.jpg
                    frame_0002.jpg
                    ...
                video002/
                    ...
            fake/
                video001/
                    ...

    입력: 폴더 내 정렬된 프레임 이미지들
    출력: (sequence, label)
        - sequence: (T, C, H, W) 텐서 (T=시퀀스 길이)
        - label: 0(real) or 1(fake)
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 16,
        stride: int = 8,
        transform: Optional[A.ReplayCompose] = None,
        is_train: bool = True
    ):
        """
        Args:
            data_dir: 데이터 루트 디렉토리 (real/, fake/ 하위 폴더 포함)
            sequence_length: 시퀀스당 프레임 수
            stride: 슬라이딩 윈도우 간격
            transform: albumentations 변환
            is_train: 학습 모드 여부 (True면 슬라이딩 윈도우로 다중 샘플 생성)
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.is_train = is_train

        self.samples: List[Tuple[Path, int, int]] = []  # (video_dir, start_idx, label)
        self.class_counts = {'real': 0, 'fake': 0}

        self._scan_videos()

    def _scan_videos(self):
        """데이터 디렉토리 스캔하여 샘플 목록 생성"""
        for label_idx, label_name in enumerate(['real', 'fake']):
            label_dir = self.data_dir / label_name

            if not label_dir.exists():
                print(f"Warning: {label_dir} not found")
                continue

            # 각 비디오 폴더 순회
            video_dirs = sorted([d for d in label_dir.iterdir() if d.is_dir()])

            for video_dir in video_dirs:
                frames = self._get_frame_list(video_dir)
                num_frames = len(frames)

                if num_frames < self.sequence_length:
                    print(f"Warning: {video_dir} has only {num_frames} frames, skipping")
                    continue

                # 슬라이딩 윈도우로 샘플 생성
                if self.is_train:
                    for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                        self.samples.append((video_dir, start_idx, label_idx))
                        self.class_counts[label_name] += 1
                else:
                    # 검증/테스트: 중앙에서 하나의 시퀀스만
                    start_idx = (num_frames - self.sequence_length) // 2
                    self.samples.append((video_dir, start_idx, label_idx))
                    self.class_counts[label_name] += 1

        print(f"Dataset loaded: {len(self.samples)} samples")
        print(f"  Real: {self.class_counts['real']}, Fake: {self.class_counts['fake']}")

    def _get_frame_list(self, video_dir: Path) -> List[Path]:
        """비디오 폴더에서 프레임 파일 목록 반환 (정렬됨)"""
        frames = []
        for ext in self.SUPPORTED_EXTENSIONS:
            frames.extend(video_dir.glob(f'*{ext}'))
            frames.extend(video_dir.glob(f'*{ext.upper()}'))

        return sorted(frames, key=lambda x: x.stem)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_dir, start_idx, label = self.samples[idx]

        # 프레임 로드
        frame_paths = self._get_frame_list(video_dir)
        selected_paths = frame_paths[start_idx:start_idx + self.sequence_length]

        frames = []
        for path in selected_paths:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            frames.append(img)

        # 변환 적용
        if self.transform is not None:
            frames = apply_sequence_transform(frames, self.transform)

        # 스택하여 (T, C, H, W) 텐서 생성
        sequence = torch.stack(frames, dim=0)

        return sequence, label

    def get_sample_weights(self) -> torch.Tensor:
        """클래스 불균형 처리를 위한 샘플 가중치 반환"""
        total = sum(self.class_counts.values())
        class_weights = {
            0: total / (2 * self.class_counts['real']) if self.class_counts['real'] > 0 else 1.0,
            1: total / (2 * self.class_counts['fake']) if self.class_counts['fake'] > 0 else 1.0
        }

        weights = [class_weights[sample[2]] for sample in self.samples]
        return torch.tensor(weights, dtype=torch.float)


def get_dataloader(
    data_dir: str,
    config: Dict[str, Any],
    is_train: bool = True,
    use_weighted_sampler: bool = True
) -> DataLoader:
    """
    데이터로더 생성 함수

    Args:
        data_dir: 데이터 디렉토리 경로
        config: 전체 config (data_config + model_config 병합)
        is_train: 학습 모드 여부
        use_weighted_sampler: 클래스 불균형 처리를 위한 가중치 샘플러 사용 여부

    Returns:
        DataLoader 인스턴스
    """
    # Config 파싱
    seq_config = config.get('sequence', {})
    sequence_length = seq_config.get('length', 16)
    stride = seq_config.get('stride', 8)

    dl_config = config.get('dataloader', {})
    num_workers = dl_config.get('num_workers', 4)
    pin_memory = dl_config.get('pin_memory', True)

    batch_size = config.get('training', {}).get('batch_size', 8)

    # 변환 생성
    if is_train:
        transform = get_train_transforms(config)
    else:
        transform = get_val_transforms(config)

    # 데이터셋 생성
    dataset = FrameSequenceDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        stride=stride if is_train else sequence_length,  # 검증시 중복 없이
        transform=transform,
        is_train=is_train
    )

    # 샘플러 설정
    sampler = None
    shuffle = is_train

    if is_train and use_weighted_sampler and len(dataset) > 0:
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        shuffle = False  # sampler 사용시 shuffle은 False

    # 데이터로더 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train
    )

    return dataloader


class VideoFrameDataset(Dataset):
    """
    단일 비디오/프레임 폴더에서 추론용 시퀀스 생성

    슬라이딩 윈도우 방식으로 모든 가능한 시퀀스 반환
    """

    def __init__(
        self,
        frame_dir: str,
        sequence_length: int = 16,
        stride: int = 1,
        transform: Optional[A.ReplayCompose] = None
    ):
        self.frame_dir = Path(frame_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform

        # 프레임 목록
        self.frames = self._get_frames()
        self.num_sequences = max(0, (len(self.frames) - sequence_length) // stride + 1)

    def _get_frames(self) -> List[Path]:
        frames = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            frames.extend(self.frame_dir.glob(f'*{ext}'))
            frames.extend(self.frame_dir.glob(f'*{ext.upper()}'))
        return sorted(frames, key=lambda x: x.stem)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        start_idx = idx * self.stride
        selected_paths = self.frames[start_idx:start_idx + self.sequence_length]

        frames = []
        for path in selected_paths:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            frames.append(img)

        if self.transform is not None:
            frames = apply_sequence_transform(frames, self.transform)

        sequence = torch.stack(frames, dim=0)

        return sequence, start_idx  # 시작 인덱스도 반환 (시각화용)
