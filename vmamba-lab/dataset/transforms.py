"""Data augmentation and transformation utilities."""

from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms as T


def get_train_transforms(
    input_size: int = 256,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    horizontal_flip: bool = True,
    rotation: int = 15,
    color_jitter: bool = True,
    random_erasing: float = 0.0,
    gaussian_blur: bool = False,
    jpeg_compression: Optional[Tuple[int, int]] = None,
) -> A.Compose:
    """Get training transforms for classification.

    Args:
        input_size: Output image size
        mean: Normalization mean
        std: Normalization std
        horizontal_flip: Enable horizontal flip
        rotation: Max rotation angle
        color_jitter: Enable color jitter
        random_erasing: Random erasing probability
        gaussian_blur: Enable gaussian blur
        jpeg_compression: JPEG compression quality range (min, max)

    Returns:
        Albumentations Compose transform
    """
    transform_list = [
        A.Resize(input_size, input_size),
    ]

    # Geometric transforms
    if horizontal_flip:
        transform_list.append(A.HorizontalFlip(p=0.5))

    if rotation > 0:
        transform_list.append(A.Rotate(limit=rotation, p=0.5))

    # Color transforms
    if color_jitter:
        transform_list.append(
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            )
        )

    # Blur
    if gaussian_blur:
        transform_list.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))

    # JPEG compression
    if jpeg_compression:
        transform_list.append(
            A.ImageCompression(
                quality_lower=jpeg_compression[0],
                quality_upper=jpeg_compression[1],
                p=0.5,
            )
        )

    # Random erasing (CoarseDropout in albumentations)
    if random_erasing > 0:
        transform_list.append(
            A.CoarseDropout(
                max_holes=1,
                max_height=int(input_size * 0.3),
                max_width=int(input_size * 0.3),
                min_height=int(input_size * 0.1),
                min_width=int(input_size * 0.1),
                fill_value=0,
                p=random_erasing,
            )
        )

    # Normalize and convert to tensor
    transform_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transform_list)


def get_test_transforms(
    input_size: int = 256,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Get test/validation transforms for classification.

    Args:
        input_size: Output image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_segmentation_transforms(
    input_size: int = 512,
    split: str = "train",
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotation: int = 90,
    scale: Optional[Tuple[float, float]] = None,
    color_jitter: bool = True,
    jpeg_compression: Optional[Tuple[int, int]] = None,
    gaussian_noise: float = 0.0,
) -> A.Compose:
    """Get transforms for segmentation tasks.

    Args:
        input_size: Output image size
        split: Data split ('train', 'val', 'test')
        mean: Normalization mean
        std: Normalization std
        horizontal_flip: Enable horizontal flip
        vertical_flip: Enable vertical flip
        rotation: Max rotation angle
        scale: Scale range (min, max)
        color_jitter: Enable color jitter
        jpeg_compression: JPEG compression range
        gaussian_noise: Gaussian noise probability

    Returns:
        Albumentations Compose transform
    """
    if split == "train":
        transform_list = []

        # Resize with optional random scale
        if scale:
            transform_list.append(
                A.RandomScale(scale_limit=(scale[0] - 1, scale[1] - 1), p=0.5)
            )
        transform_list.append(A.Resize(input_size, input_size))

        # Geometric transforms (applied to both image and mask)
        if horizontal_flip:
            transform_list.append(A.HorizontalFlip(p=0.5))
        if vertical_flip:
            transform_list.append(A.VerticalFlip(p=0.5))
        if rotation > 0:
            transform_list.append(
                A.Rotate(limit=rotation, p=0.5, border_mode=cv2.BORDER_CONSTANT)
            )

        # Color transforms (image only)
        if color_jitter:
            transform_list.append(
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=0.5,
                )
            )

        # Noise and compression
        if gaussian_noise > 0:
            transform_list.append(
                A.GaussNoise(var_limit=(10.0, 50.0), p=gaussian_noise)
            )
        if jpeg_compression:
            transform_list.append(
                A.ImageCompression(
                    quality_lower=jpeg_compression[0],
                    quality_upper=jpeg_compression[1],
                    p=0.5,
                )
            )

        # Normalize and convert
        transform_list.extend([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

        return A.Compose(transform_list)

    else:  # val or test
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class DualTransform:
    """Apply same random transforms to both image and reference."""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        if mask is not None:
            transformed = self.transform(image=image, mask=mask)
            return {
                "image": transformed["image"],
                "mask": transformed["mask"],
            }
        else:
            transformed = self.transform(image=image)
            return {
                "image": transformed["image"],
                "mask": torch.zeros(
                    transformed["image"].shape[1:], dtype=torch.float32
                ),
            }


class TorchvisionTransforms:
    """Torchvision-based transforms (alternative to albumentations)."""

    @staticmethod
    def get_train_transforms(
        input_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> T.Compose:
        """Get training transforms using torchvision."""
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    @staticmethod
    def get_test_transforms(
        input_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> T.Compose:
        """Get test transforms using torchvision."""
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Denormalize a tensor for visualization.

    Args:
        tensor: Normalized tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def tensor_to_numpy(
    tensor: torch.Tensor,
    denorm: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert tensor to numpy array for visualization.

    Args:
        tensor: Input tensor (C, H, W)
        denorm: Whether to denormalize
        mean: Normalization mean
        std: Normalization std

    Returns:
        Numpy array (H, W, C) in range [0, 255]
    """
    if denorm:
        tensor = denormalize(tensor, mean, std)

    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy
    img = tensor.cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return (img * 255).astype(np.uint8)
