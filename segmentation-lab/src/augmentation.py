"""
Augmentation Module
데이터 증강 변환 정의 - 1024 resize + 512 crop 방식
"""

import random
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


class DownscaleUpscaleAndResize(A.DualTransform):
    """
    정사각형 이미지 입력 가정
    1. 다운스케일-업스케일 (원본 크기 이하로만, 품질 저하 시뮬레이션)
    2. target_size로 리사이즈

    Args:
        target_size: 최종 출력 크기 (1024)
        min_downscale: 다운스케일 최소 크기 (512)
        downscale_p: 다운스케일-업스케일 적용 확률 (0.7)
    """
    def __init__(
        self,
        target_size: int = 1024,
        min_downscale: int = 512,
        downscale_p: float = 0.7,
        always_apply: bool = False,
        p: float = 1.0
    ):
        super().__init__(always_apply, p)
        self.target_size = target_size
        self.min_downscale = min_downscale
        self.downscale_p = downscale_p

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        original_size = min(h, w)

        # 1. 다운스케일-업스케일 (원본 크기 이하로만)
        #    예: 원본 640이면 512~640 범위로 다운 후 다시 원본 크기로 업
        result = img
        if original_size >= self.min_downscale and random.random() < self.downscale_p:
            down_size = random.randint(self.min_downscale, original_size)
            if down_size < original_size:
                downscaled = cv2.resize(img, (down_size, down_size), interpolation=cv2.INTER_AREA)
                result = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)

        # 2. target_size로 리사이즈
        resized = cv2.resize(result, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        return resized

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        # 마스크는 다운업 없이 리사이즈만
        resized = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        return resized

    def get_transform_init_args_names(self):
        return ("target_size", "min_downscale", "downscale_p")


class RandomDownscaleUpscale(ImageOnlyTransform):
    """
    이미지를 랜덤하게 축소 후 다시 원래 크기로 확대
    JPEG 압축 아티팩트와 유사한 효과를 시뮬레이션

    Args:
        scale_range: (min_scale, max_scale) 축소 비율 범위
        p: 적용 확률
    """
    def __init__(
        self,
        scale_range: tuple = (0.5, 1.0),
        always_apply: bool = False,
        p: float = 0.7
    ):
        super().__init__(always_apply, p)
        self.scale_range = scale_range

    def apply(self, img: np.ndarray, scale: float = 1.0, **params) -> np.ndarray:
        if scale >= 1.0:
            return img

        h, w = img.shape[:2]

        # 축소
        new_h, new_w = int(h * scale), int(w * scale)
        downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 다시 원래 크기로 확대
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)

        return upscaled

    def get_params(self):
        return {"scale": random.uniform(*self.scale_range)}

    def get_transform_init_args_names(self):
        return ("scale_range",)


def get_train_transform(resize: int = 1024, crop_size: int = 512, min_downscale: int = 768) -> A.Compose:
    """
    학습용 데이터 증강 변환 (정사각형 입력 가정)
    - 다운스케일-업스케일로 품질 저하 시뮬레이션 (70%)
    - resize 크기로 리사이즈
    - 랜덤 크롭으로 crop_size x crop_size 출력

    Note:
        입력 이미지는 전처리로 정사각형으로 크롭되어 있어야 함
        (tools/preprocess_face_crop.py 사용)

    Args:
        resize: 리사이즈 크기 (1024)
        crop_size: 학습 시 크롭할 크기 (512)
        min_downscale: 품질 저하 시뮬레이션 다운스케일 최소값 (768)

    Returns:
        albumentations Compose 객체
    """
    return A.Compose([
        # 1. 다운스케일-업스케일 + 리사이즈 (정사각형 입력 가정)
        #    - 모든 이미지에 70% 확률로 품질 저하 시뮬레이션
        #    - 다운스케일 범위: min_downscale ~ 원본크기 (원본 이상으로 업스케일 안함)
        DownscaleUpscaleAndResize(
            target_size=resize,
            min_downscale=min_downscale,
            downscale_p=0.7
        ),

        # 2. 랜덤 크롭 (512x512)
        A.RandomCrop(height=crop_size, width=crop_size),

        # 3. 기하학적 변환
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # 4. 정규화 및 텐서 변환
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_val_transform(resize: int = 1024, crop_size: int = 512) -> A.Compose:
    """
    검증용 변환 (정사각형 입력 가정, 증강 없음)
    - 전체 이미지를 resize 크기로 리사이즈 (크롭 없이 전체 이미지 사용)

    Args:
        resize: 출력 크기 (1024)
        crop_size: (사용 안함, 하위 호환성 유지)

    Returns:
        albumentations Compose 객체
    """
    return A.Compose([
        # 전체 이미지를 resize 크기로 리사이즈 (크롭 없이 전체 이미지 사용)
        A.Resize(height=resize, width=resize),

        # 정규화 및 텐서 변환
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_test_transform(resize: int = 1024, crop_size: int = 512) -> A.Compose:
    """
    테스트용 변환 (검증과 동일)
    """
    return get_val_transform(resize, crop_size)


def get_inference_transform(max_size: int = 1024) -> A.Compose:
    """
    추론용 변환 (가변 크기)
    - 1024보다 크면 축소
    - 작으면 그대로 유지

    Args:
        max_size: 최대 크기 (1024보다 크면 축소)

    Returns:
        albumentations Compose 객체
    """
    return A.Compose([
        # 큰 이미지만 축소, 작은 이미지는 그대로
        A.LongestMaxSize(max_size=max_size),

        # 정규화 및 텐서 변환
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


# =============================================================================
# 커스텀 증강 예시 (필요시 활성화)
# =============================================================================

def get_heavy_train_transform(resize: int = 1024, crop_size: int = 512) -> A.Compose:
    """
    강한 데이터 증강 (오버피팅 방지용)
    """
    return A.Compose([
        # 1. 다운스케일-업스케일 + 리사이즈 (정사각형 입력 가정)
        DownscaleUpscaleAndResize(
            target_size=resize,
            min_downscale=crop_size,
            downscale_p=0.7
        ),

        # 2. 랜덤 크롭
        A.RandomCrop(height=crop_size, width=crop_size),

        # 기하학적 변환
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=0,
            p=0.5
        ),

        # 왜곡
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3),
            A.ElasticTransform(alpha=1, sigma=50, approximate=True),
        ], p=0.3),

        # Cutout/CoarseDropout
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(32, 64),
            hole_width_range=(32, 64),
            fill=0,
            p=0.3
        ),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_light_train_transform(resize: int = 1024, crop_size: int = 512) -> A.Compose:
    """
    가벼운 데이터 증강 (빠른 실험용)
    """
    return A.Compose([
        # 1. 다운스케일-업스케일 + 리사이즈 (정사각형 입력 가정)
        DownscaleUpscaleAndResize(
            target_size=resize,
            min_downscale=crop_size,
            downscale_p=0.7
        ),

        # 2. 랜덤 크롭
        A.RandomCrop(height=crop_size, width=crop_size),

        A.HorizontalFlip(p=0.5),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
