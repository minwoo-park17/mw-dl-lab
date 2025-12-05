"""
Dataset classes for deepfake image classification.
"""
import os
import glob
import random
import logging
from typing import List, Tuple, Optional

import yaml
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def shuffle_and_balance(
    path_list: List[str],
    label_list: List[int]
) -> Tuple[List[str], List[int]]:
    """
    Shuffle paths and labels together.

    Args:
        path_list: List of file paths
        label_list: List of labels

    Returns:
        Shuffled paths and labels
    """
    combined = list(zip(path_list, label_list))
    random.shuffle(combined)
    shuffled_paths, shuffled_labels = zip(*combined)
    return list(shuffled_paths), list(shuffled_labels)


class LoadDataInfo:
    """Load data paths and labels from configuration file."""

    def __init__(
        self,
        data_config_path: str,
        data_class: str,
        printcheck: bool = False
    ):
        """
        Initialize data loader.

        Args:
            data_config_path: Path to data configuration YAML file
            data_class: Data split type ('train', 'validation', 'test')
            printcheck: Whether to print data counts
        """
        self.data_config_path = data_config_path
        self.data_class = data_class
        self.printcheck = printcheck

    def _search_path_label(
        self,
        data_dir_list: Optional[List[str]],
        label: int,
        support_type: List[str]
    ) -> Tuple[List[str], List[int]]:
        """
        Search for files in directories and assign labels.

        Args:
            data_dir_list: List of directory paths
            label: Label to assign (0: real, 1: fake)
            support_type: List of supported file extensions

        Returns:
            Tuple of (file paths, labels)
        """
        total_path_list = []
        if data_dir_list is None:
            data_dir_list = []

        for data_dir in data_dir_list:
            glob_dir = os.path.join(data_dir, "*")
            data_path_list = glob.glob(glob_dir, recursive=True)

            # Filter by supported file types
            data_path_list = [
                x for x in data_path_list
                for ext in support_type
                if x.lower().endswith(ext.lower())
            ]
            total_path_list += data_path_list

            if self.printcheck:
                logger.info(f"{data_dir}: {len(data_path_list)} files")

        label_list = [label] * len(total_path_list)
        return total_path_list, label_list

    def _get_path_label(self) -> Tuple[List[str], List[int]]:
        """Get all file paths and labels from config."""
        with open(self.data_config_path, 'r', encoding='utf-8') as ymlfile:
            data_config = yaml.safe_load(ymlfile)

        fake_dir_path = data_config[self.data_class].get("fake")
        real_dir_path = data_config[self.data_class].get("real")
        support_type = data_config.get("support_type", [".png", ".jpg", ".jpeg"])

        fake_path, fake_label = self._search_path_label(fake_dir_path, label=1, support_type=support_type)
        real_path, real_label = self._search_path_label(real_dir_path, label=0, support_type=support_type)

        path = fake_path + real_path
        label = fake_label + real_label

        return path, label

    def __call__(self) -> Tuple[List[str], List[int]]:
        """Return file paths and labels."""
        return self._get_path_label()


class RandomZoomOut:
    """Random zoom out augmentation."""

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.3, 0.8),
        fill: int = 0,
        min_height: int = 448
    ):
        self.scale_range = scale_range
        self.fill = fill
        self.min_height = min_height

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size

        if h <= self.min_height:
            return img

        scale = random.uniform(*self.scale_range)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

        pad_w = (w - new_w) // 2
        pad_h = (h - new_h) // 2

        img_padded = F.pad(
            img_resized,
            (pad_w, pad_h, w - new_w - pad_w, h - new_h - pad_h),
            fill=self.fill
        )
        return img_padded


class RandomResize:
    """Random resize augmentation."""

    def __init__(self, min_size: int = 180, interpolation=Image.BILINEAR):
        self.min_size = min_size
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = random.randint(self.min_size, w) if w > self.min_size else w
        new_h = random.randint(self.min_size, h) if h > self.min_size else h
        return img.resize((new_w, new_h), self.interpolation)


class SimpleColorBlend:
    """Simple color blending augmentation."""

    def __init__(self, p: float = 0.001):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        img_cv = np.array(img)[:, :, ::-1].copy()

        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        overlay_color = (b, g, r)

        bg_overlay = np.full_like(img_cv, overlay_color, dtype=np.uint8)

        alpha = random.uniform(0.6, 0.9)
        beta = 1 - alpha

        blended = cv2.addWeighted(img_cv, alpha, bg_overlay, beta, 0)

        return Image.fromarray(blended[:, :, ::-1])


class CnnDataset(Dataset):
    """Dataset for CNN-based deepfake classification."""

    # ImageNet normalization constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_config_path: str,
        data_class: str,
        img_size: int,
        printcheck: bool = True,
        enable_augmentation: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data_config_path: Path to data configuration file
            data_class: Data split ('train', 'validation', 'test')
            img_size: Target image size
            printcheck: Whether to print data statistics
            enable_augmentation: Whether to enable data augmentation (for training)
        """
        self.data_class = data_class
        self.img_size = img_size
        self.enable_augmentation = enable_augmentation and (data_class == "train")

        # Load data paths and labels
        data_infos = LoadDataInfo(
            data_config_path=data_config_path,
            data_class=data_class,
            printcheck=printcheck
        )
        self.img_path, self.labels = data_infos()

        if printcheck:
            n_fake = sum(1 for l in self.labels if l == 1)
            n_real = sum(1 for l in self.labels if l == 0)
            logger.info(f"{data_class}: {n_fake} fake, {n_real} real, total {len(self.labels)}")

        # Define transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup image transforms."""
        # Training augmentation for real images
        self.train_transform_real = transforms.Compose([
            RandomResize(min_size=180, interpolation=Image.BILINEAR),
            transforms.RandomApply(
                [RandomZoomOut(scale_range=(0.5, 0.90), fill=0, min_height=400)],
                p=0.5
            ),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        # Training augmentation for fake images
        self.train_transform_fake = transforms.Compose([
            SimpleColorBlend(p=0.2),
            RandomResize(min_size=180, interpolation=Image.BILINEAR),
            transforms.RandomApply(
                [RandomZoomOut(scale_range=(0.5, 0.90), fill=0, min_height=400)],
                p=0.5
            ),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        # Basic transform (no augmentation)
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.img_path)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'input', 'label', and 'file_path'
        """
        img_path = self.img_path[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transform based on split and label
        if self.enable_augmentation:
            if label == 0:
                img_array = self.train_transform_real(img)
            else:
                img_array = self.train_transform_fake(img)
        else:
            img_array = self.base_transform(img)

        return {
            'input': img_array,
            'label': label,
            'file_path': img_path
        }


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config")
    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    img_size = model_config["model"]["image-size"]

    dataset = CnnDataset(
        data_config_path=args.data_config,
        data_class="train",
        img_size=img_size,
        printcheck=True
    )

    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample['input'].shape}")
        print(f"Sample label: {sample['label']}")
