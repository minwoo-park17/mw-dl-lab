"""
데이터 증강 및 전처리 모듈
시퀀스 단위로 동일한 변환을 적용
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import List, Dict, Any, Optional


class SequenceTransform:
    """
    시퀀스의 모든 프레임에 동일한 변환을 적용하는 래퍼 클래스
    """
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Args:
            frames: 프레임 리스트 [(H, W, C), ...]
        Returns:
            변환된 프레임 리스트
        """
        if len(frames) == 0:
            return frames

        # 첫 번째 프레임으로 변환 파라미터 결정
        replay = self.transform(image=frames[0])
        transformed_frames = [replay['image']]

        # 나머지 프레임에 동일한 변환 적용
        for frame in frames[1:]:
            result = A.ReplayCompose.replay(replay['replay'], image=frame)
            transformed_frames.append(result['image'])

        return transformed_frames


def get_train_transforms(config: Dict[str, Any]) -> A.ReplayCompose:
    """
    학습용 데이터 증강 변환 생성

    Args:
        config: data_config의 augmentation 및 image 설정

    Returns:
        albumentations ReplayCompose 객체
    """
    img_config = config.get('image', {})
    aug_config = config.get('augmentation', {})

    height = img_config.get('height', 224)
    width = img_config.get('width', 224)

    normalize = aug_config.get('normalize', {})
    mean = normalize.get('mean', [0.485, 0.456, 0.406])
    std = normalize.get('std', [0.229, 0.224, 0.225])

    transforms_list = [
        # 리사이즈
        A.Resize(height=height, width=width),
    ]

    if aug_config.get('enabled', True):
        # 랜덤 크롭
        if aug_config.get('random_crop', True):
            transforms_list.insert(0, A.RandomResizedCrop(
                height=height,
                width=width,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ))
            transforms_list.pop(1)  # 기본 Resize 제거

        # 수평 뒤집기
        if aug_config.get('horizontal_flip', True):
            transforms_list.append(A.HorizontalFlip(p=0.5))

        # 컬러 지터링
        if aug_config.get('color_jitter', True):
            transforms_list.append(A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ))

        # 추가 증강
        transforms_list.extend([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])

    # 정규화 및 텐서 변환
    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return A.ReplayCompose(transforms_list)


def get_val_transforms(config: Dict[str, Any]) -> A.ReplayCompose:
    """
    검증/테스트용 변환 생성 (증강 없음)

    Args:
        config: data_config 설정

    Returns:
        albumentations ReplayCompose 객체
    """
    img_config = config.get('image', {})
    aug_config = config.get('augmentation', {})

    height = img_config.get('height', 224)
    width = img_config.get('width', 224)

    normalize = aug_config.get('normalize', {})
    mean = normalize.get('mean', [0.485, 0.456, 0.406])
    std = normalize.get('std', [0.229, 0.224, 0.225])

    transforms_list = [
        A.Resize(height=height, width=width),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]

    return A.ReplayCompose(transforms_list)


def apply_sequence_transform(
    frames: List[np.ndarray],
    transform: A.ReplayCompose
) -> List[np.ndarray]:
    """
    프레임 시퀀스에 동일한 변환 적용

    Args:
        frames: BGR 이미지 리스트
        transform: ReplayCompose 변환

    Returns:
        변환된 텐서 리스트
    """
    if len(frames) == 0:
        return []

    # BGR -> RGB 변환
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    # 첫 프레임으로 변환 파라미터 결정
    first_result = transform(image=rgb_frames[0])
    transformed = [first_result['image']]

    # 동일한 변환을 나머지 프레임에 적용
    for frame in rgb_frames[1:]:
        result = A.ReplayCompose.replay(first_result['replay'], image=frame)
        transformed.append(result['image'])

    return transformed


def denormalize(
    tensor: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    정규화된 텐서를 원래 스케일로 복원

    Args:
        tensor: (C, H, W) 또는 (H, W, C) 정규화된 이미지
        mean: 정규화에 사용된 평균
        std: 정규화에 사용된 표준편차

    Returns:
        복원된 이미지 (0-255, uint8)
    """
    mean = np.array(mean)
    std = np.array(std)

    if tensor.shape[0] == 3:  # (C, H, W)
        tensor = tensor.transpose(1, 2, 0)

    img = tensor * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img
