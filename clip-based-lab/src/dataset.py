"""
Dataset classes for CLIP-based deepfake detection (UnivFD).
"""
import os
import glob
import random
import logging
from typing import List, Tuple, Optional

import yaml
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


class ClipDataset(Dataset):
    """Dataset for CLIP-based deepfake detection."""

    # CLIP normalization constants (same as ImageNet)
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]

    # MEAN = [0.485, 0.456, 0.406]
    # STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_config_path: str,
        data_class: str,
        img_size: int = 224,
        printcheck: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data_config_path: Path to data configuration file
            data_class: Data split ('train', 'validation', 'test')
            img_size: Target image size (224 for CLIP)
            printcheck: Whether to print data statistics
        """
        self.data_class = data_class
        self.img_size = img_size

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

        # Define CLIP standard transforms (no augmentation for UnivFD)
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup CLIP standard image transforms."""
        # CLIP standard preprocessing: Resize -> CenterCrop -> ToTensor -> Normalize
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.img_size),
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

        # Apply CLIP transform
        img_tensor = self.transform(img)

        return {
            'input': img_tensor,
            'label': label,
            'file_path': img_path
        }


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser(description="Test CLIP dataset loading")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dataset = ClipDataset(
        data_config_path=args.data_config,
        data_class="train",
        img_size=224,
        printcheck=True
    )

    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample['input'].shape}")
        print(f"Sample label: {sample['label']}")
        print(f"Sample path: {sample['file_path']}")
