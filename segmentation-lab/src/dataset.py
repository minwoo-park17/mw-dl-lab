"""
Segmentation Dataset Module
Real/Fake 폴더 구조 기반으로 마스크를 자동 생성합니다.
- Real 폴더: 마스크 전체 0
- Fake 폴더: 마스크 전체 1

지원하는 config 형식:
  # 단일 경로
  real_dir: "/path/to/real"

  # 다중 경로 (리스트)
  real_dir:
    - "/path1/real"
    - "/path2/real"
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Union

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A

from .augmentation import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
    get_inference_transform
)
from .utils import load_config


def _normalize_dirs(dir_config: Union[str, List[str], None]) -> List[str]:
    """
    경로 설정을 리스트로 정규화합니다.

    Args:
        dir_config: 단일 경로 문자열, 경로 리스트, 또는 None

    Returns:
        경로 리스트 (존재하는 경로만)
    """
    if dir_config is None:
        return []

    if isinstance(dir_config, str):
        dirs = [dir_config]
    else:
        dirs = dir_config

    # 존재하는 경로만 필터링
    valid_dirs = [d for d in dirs if d and Path(d).exists()]
    return valid_dirs


def _create_datasets_from_dirs(
    dirs: List[str],
    label: int,
    transform: Optional[A.Compose],
    output_size: int
) -> List[Dataset]:
    """
    여러 디렉토리에서 데이터셋들을 생성합니다.

    Args:
        dirs: 디렉토리 경로 리스트
        label: 레이블 값 (0=real, 1=fake)
        transform: albumentations 변환
        output_size: 출력 크기

    Returns:
        데이터셋 리스트
    """
    datasets = []
    for d in dirs:
        ds = BinarySegmentationDataset(
            image_dir=d,
            label=label,
            transform=transform,
            output_size=output_size
        )
        if len(ds) > 0:
            datasets.append(ds)
    return datasets


class BinarySegmentationDataset(Dataset):
    """
    Real/Fake 이진 세분화 데이터셋
    폴더 구조에 따라 마스크를 자동 생성합니다.

    Args:
        image_dir: 이미지 디렉토리 경로
        label: 레이블 값 (0=real, 1=fake)
        transform: albumentations 변환
        output_size: 출력 마스크 크기 (transform 후 크기)
    """

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(
        self,
        image_dir: str,
        label: int,
        transform: Optional[A.Compose] = None,
        output_size: int = 512
    ):
        self.image_dir = Path(image_dir)
        self.label = label  # 0 for real, 1 for fake
        self.transform = transform
        self.output_size = output_size

        # 파일 목록 생성
        self.images = self._get_image_list()

        if len(self.images) == 0:
            print(f"Warning: No images found in {image_dir}")
        else:
            label_name = "real" if label == 0 else "fake"
            print(f"Loaded {len(self.images)} {label_name} images from {image_dir}")

    def _get_image_list(self) -> List[Path]:
        """지원되는 이미지 파일 목록을 반환합니다."""
        if not self.image_dir.exists():
            return []

        # Windows 대소문자 무시로 인한 중복 방지
        images = set()
        for ext in self.SUPPORTED_EXTENSIONS:
            images.update(self.image_dir.glob(f"*{ext}"))
            images.update(self.image_dir.glob(f"*{ext.upper()}"))
        return sorted(images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[idx]

        # 이미지 로드 (PIL 사용 - libpng warning 없음)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # 마스크 생성 (이미지와 같은 크기, 레이블 값으로 채움)
        h, w = image.shape[:2]
        mask = np.full((h, w), self.label, dtype=np.float32)

        # 변환 적용
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # 마스크 차원 추가 (H, W) -> (1, H, W)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask


def create_dataloaders(
    data_config_path: str = "config/data.yaml",
    arch_config_path: str = "config/architecture.yaml"
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    설정 파일에 따라 DataLoader들을 생성합니다.

    Args:
        data_config_path: 데이터 설정 파일 경로
        arch_config_path: 아키텍처 설정 파일 경로

    Returns:
        train_loader, val_loader, test_loader (또는 None)
    """
    data_config = load_config(data_config_path)
    arch_config = load_config(arch_config_path)

    # 이미지 설정
    image_config = data_config['image']
    resize = image_config.get('resize', 1024)
    crop_size = image_config.get('crop_size', 512)
    min_downscale = image_config.get('min_downscale', 768)

    # 로더 설정
    loader_config = data_config.get('loader', {})
    batch_size = arch_config['training']['batch_size']
    num_workers = loader_config.get('num_workers', 4)
    pin_memory = loader_config.get('pin_memory', True)

    # 변환 생성 (1024 resize + 512 crop 방식)
    train_transform = get_train_transform(resize=resize, crop_size=crop_size, min_downscale=min_downscale)
    val_transform = get_val_transform(resize=resize, crop_size=crop_size)
    test_transform = get_test_transform(resize=resize, crop_size=crop_size)

    # =========================================================================
    # 학습 데이터셋 (다중 경로 지원)
    # =========================================================================
    train_config = data_config['train']
    train_real_dirs = _normalize_dirs(train_config.get('real_dir'))
    train_fake_dirs = _normalize_dirs(train_config.get('fake_dir'))

    train_datasets = []
    train_datasets.extend(_create_datasets_from_dirs(
        train_real_dirs, label=0, transform=train_transform, output_size=crop_size
    ))
    train_datasets.extend(_create_datasets_from_dirs(
        train_fake_dirs, label=1, transform=train_transform, output_size=crop_size
    ))

    if not train_datasets:
        raise ValueError("No training data found! Check your data paths in config.")

    train_dataset = ConcatDataset(train_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    # =========================================================================
    # 검증 데이터셋 (다중 경로 지원)
    # =========================================================================
    val_config = data_config['val']
    val_real_dirs = _normalize_dirs(val_config.get('real_dir'))
    val_fake_dirs = _normalize_dirs(val_config.get('fake_dir'))

    val_datasets = []
    val_datasets.extend(_create_datasets_from_dirs(
        val_real_dirs, label=0, transform=val_transform, output_size=resize
    ))
    val_datasets.extend(_create_datasets_from_dirs(
        val_fake_dirs, label=1, transform=val_transform, output_size=resize
    ))

    if not val_datasets:
        raise ValueError("No validation data found! Check your data paths in config.")

    val_dataset = ConcatDataset(val_datasets)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # =========================================================================
    # 테스트 데이터셋 (다중 경로 지원, 경로가 존재하는 경우만)
    # =========================================================================
    test_loader = None
    test_config = data_config.get('test', {})

    if test_config:
        test_real_dirs = _normalize_dirs(test_config.get('real_dir'))
        test_fake_dirs = _normalize_dirs(test_config.get('fake_dir'))

        test_datasets = []
        test_datasets.extend(_create_datasets_from_dirs(
            test_real_dirs, label=0, transform=test_transform, output_size=resize
        ))
        test_datasets.extend(_create_datasets_from_dirs(
            test_fake_dirs, label=1, transform=test_transform, output_size=resize
        ))

        if test_datasets:
            test_dataset = ConcatDataset(test_datasets)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

    print(f"\nDataset Summary:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    if test_loader:
        print(f"  Test: {len(test_loader.dataset)} images")

    return train_loader, val_loader, test_loader


def create_inference_dataset(
    image_dir: str,
    data_config_path: str = "config/data.yaml"
) -> DataLoader:
    """
    추론용 데이터셋을 생성합니다 (마스크 없이 이미지만).

    Args:
        image_dir: 이미지 디렉토리 경로
        data_config_path: 데이터 설정 파일 경로

    Returns:
        DataLoader
    """
    data_config = load_config(data_config_path)
    image_config = data_config['image']

    # 추론: 가변 크기 (1024보다 크면 축소, 작으면 그대로)
    transform = get_inference_transform(
        max_size=image_config.get('resize', 1024)
    )

    dataset = InferenceDataset(image_dir=image_dir, transform=transform)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )


class InferenceDataset(Dataset):
    """추론용 데이터셋 (마스크 없음)"""

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(
        self,
        image_dir: str,
        transform: Optional[A.Compose] = None
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = self._get_image_list()

        if len(self.images) == 0:
            raise ValueError(f"No images found in {image_dir}")

    def _get_image_list(self) -> List[Path]:
        # Windows 대소문자 무시로 인한 중복 방지
        images = set()
        for ext in self.SUPPORTED_EXTENSIONS:
            images.update(self.image_dir.glob(f"*{ext}"))
            images.update(self.image_dir.glob(f"*{ext.upper()}"))
        return sorted(images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.images[idx]

        # 이미지 로드 (PIL 사용 - libpng warning 없음)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, image_path.name


# 하위 호환성을 위한 별칭
SegmentationDataset = BinarySegmentationDataset
