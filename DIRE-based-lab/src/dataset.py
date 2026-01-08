"""
Dataset classes for DIRE-based deepfake detection.

Supports both raw images and precomputed DIRE/SeDID features.
"""
import os
import glob
import logging
from typing import List, Tuple, Optional

import yaml
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


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
        """Search for files in directories and assign labels."""
        total_path_list = []
        if data_dir_list is None:
            data_dir_list = []

        for data_dir in data_dir_list:
            glob_dir = os.path.join(data_dir, "*")
            data_path_list = glob.glob(glob_dir, recursive=True)

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


class DIREDataset(Dataset):
    """
    Dataset for DIRE-based detection.

    Supports two modes:
    1. Raw images (for on-the-fly DIRE computation)
    2. Precomputed DIRE features (for efficient training)
    """

    def __init__(
        self,
        data_config_path: str,
        data_class: str,
        img_size: int = 512,
        use_precomputed: bool = False,
        cache_dir: Optional[str] = None,
        printcheck: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data_config_path: Path to data configuration file
            data_class: Data split ('train', 'validation', 'test')
            img_size: Target image size
            use_precomputed: Whether to use precomputed DIRE features
            cache_dir: Directory containing precomputed features
            printcheck: Whether to print data statistics
        """
        self.data_class = data_class
        self.img_size = img_size
        self.use_precomputed = use_precomputed
        self.cache_dir = cache_dir

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

        # Setup transforms for diffusion model input
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup image transforms for diffusion model."""
        # Transform to [-1, 1] range for diffusion model
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
        ])

    def _get_cache_path(self, img_path: str) -> str:
        """Get cache file path for precomputed features."""
        if self.cache_dir is None:
            raise ValueError("cache_dir must be set when use_precomputed=True")

        # Create unique filename from image path
        img_name = os.path.basename(img_path)
        cache_name = os.path.splitext(img_name)[0] + ".pt"
        return os.path.join(self.cache_dir, cache_name)

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

        if self.use_precomputed:
            # Load precomputed DIRE features
            cache_path = self._get_cache_path(img_path)
            if os.path.exists(cache_path):
                features = torch.load(cache_path)
            else:
                # Fallback to raw image if cache not found
                logger.warning(f"Cache not found: {cache_path}, loading raw image")
                img = Image.open(img_path).convert("RGB")
                features = self.transform(img)
        else:
            # Load raw image
            img = Image.open(img_path).convert("RGB")
            features = self.transform(img)

        return {
            'input': features,
            'label': label,
            'file_path': img_path
        }


class PrecomputedFeatureDataset(Dataset):
    """
    Dataset for precomputed DIRE or SeDID features.

    Use this when features have been precomputed and saved.
    """

    # ImageNet normalization for classifier
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        feature_dir: str,
        data_config_path: str,
        data_class: str,
        classifier_input_size: int = 224,
        printcheck: bool = True
    ):
        """
        Initialize dataset.

        Args:
            feature_dir: Directory containing precomputed features (.pt files)
            data_config_path: Path to data configuration file
            data_class: Data split ('train', 'validation', 'test')
            classifier_input_size: Input size for classifier
            printcheck: Whether to print data statistics
        """
        self.feature_dir = feature_dir
        self.classifier_input_size = classifier_input_size

        # Load data info for labels
        data_infos = LoadDataInfo(
            data_config_path=data_config_path,
            data_class=data_class,
            printcheck=False
        )
        img_paths, labels = data_infos()

        # Map image paths to feature paths
        self.feature_paths = []
        self.labels = []

        for img_path, label in zip(img_paths, labels):
            img_name = os.path.basename(img_path)
            feature_name = os.path.splitext(img_name)[0] + ".pt"
            feature_path = os.path.join(feature_dir, feature_name)

            if os.path.exists(feature_path):
                self.feature_paths.append(feature_path)
                self.labels.append(label)
            else:
                logger.warning(f"Feature not found: {feature_path}")

        if printcheck:
            n_fake = sum(1 for l in self.labels if l == 1)
            n_real = sum(1 for l in self.labels if l == 0)
            logger.info(f"{data_class} (precomputed): {n_fake} fake, {n_real} real, total {len(self.labels)}")

        # Normalization transform for classifier input
        self.normalize = transforms.Normalize(mean=self.MEAN, std=self.STD)

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.feature_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'input', 'label', and 'file_path'
        """
        feature_path = self.feature_paths[idx]
        label = self.labels[idx]

        # Load precomputed features
        features = torch.load(feature_path)

        # Resize if needed
        if features.shape[-1] != self.classifier_input_size:
            features = torch.nn.functional.interpolate(
                features.unsqueeze(0),
                size=(self.classifier_input_size, self.classifier_input_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Scale from [-1, 1] or [0, ?] to [0, 1] then normalize
        if features.min() < 0:
            features = (features + 1) / 2
        features = features.clamp(0, 1)
        features = self.normalize(features)

        return {
            'input': features,
            'label': label,
            'file_path': feature_path
        }


if __name__ == "__main__":
    # Test the dataset
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Test DIRE dataset loading")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config")
    args = parser.parse_args()

    dataset = DIREDataset(
        data_config_path=args.data_config,
        data_class="train",
        img_size=512,
        use_precomputed=False,
        printcheck=True
    )

    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample['input'].shape}")
        print(f"Sample label: {sample['label']}")
        print(f"Sample range: [{sample['input'].min():.2f}, {sample['input'].max():.2f}]")
