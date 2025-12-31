"""
Dataset classes for deepfake image classification.
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


# =============================================================================
# Utility Functions (선택적 사용)
# Note: 샘플링 전략은 sampler.py 사용 권장
# =============================================================================

def shuffle_and_balance(
    path_list: List[str],
    label_list: List[int]
) -> Tuple[List[str], List[int]]:
    """
    Shuffle paths and labels together.

    Note: 이 함수는 유틸리티 함수입니다.
          클래스 균형 샘플링은 sampler.py의 create_sampler() 사용을 권장합니다.

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
        printcheck: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data_config_path: Path to data configuration file
            data_class: Data split ('train', 'validation', 'test')
            img_size: Target image size
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

        # Define transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup image transforms."""
        # 기본 transform (augmentation 없음)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        # ----------------------------------------------------------
        # Augmentation 사용 시 아래 주석 해제
        # ----------------------------------------------------------
        # from augmentation import (
        #     TrainTransformReal, TrainTransformFake,
        #     ValTransformReal, ValTransformFake,
        #     TestTransform
        # )
        #
        # if self.data_class == "train":
        #     self.transform_train_real = TrainTransformReal(img_size=self.img_size)
        #     self.transform_train_fake = TrainTransformFake(img_size=self.img_size)
        # elif self.data_class == "validation":
        #     self.transform_val_real = ValTransformReal(img_size=self.img_size)
        #     self.transform_val_fake = ValTransformFake(img_size=self.img_size)
        # else:
        #     self.transform_test = TestTransform(img_size=self.img_size)
        # ----------------------------------------------------------

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

        # Apply transform
        img_array = self.transform(img)

        # ----------------------------------------------------------
        # Augmentation 사용 시 위 transform 라인 주석 처리 후 아래 주석 해제
        # ----------------------------------------------------------
        # if self.data_class == "train":
        #     if label == 0:  # Real
        #         img_array = self.transform_train_real(img)
        #     else:           # Fake
        #         img_array = self.transform_train_fake(img)
        # elif self.data_class == "validation":
        #     if label == 0:  # Real
        #         img_array = self.transform_val_real(img)
        #     else:           # Fake
        #         img_array = self.transform_val_fake(img)
        # else:  # test
        #     img_array = self.transform_test(img)
        # ----------------------------------------------------------

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
