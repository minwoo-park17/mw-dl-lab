"""Base dataset class for all datasets."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets.

    Provides common functionality for loading and processing images.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_images: bool = False,
    ):
        """Initialize base dataset.

        Args:
            root: Root directory of dataset
            split: Data split ('train', 'val', 'test')
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            cache_images: Whether to cache images in memory
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cache_images = cache_images

        # To be filled by subclasses
        self.samples: List[Dict[str, Any]] = []
        self.cache: Dict[int, Any] = {}

        # Validate
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

    @abstractmethod
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load sample list. Must be implemented by subclasses.

        Returns:
            List of sample dictionaries with at least 'image_path' key
        """
        pass

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a sample. Must be implemented by subclasses."""
        pass

    def _load_image(self, path: Union[str, Path]) -> Image.Image:
        """Load image from path.

        Args:
            path: Path to image file

        Returns:
            PIL Image in RGB format
        """
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Union[str, Path]) -> Image.Image:
        """Load mask from path.

        Args:
            path: Path to mask file

        Returns:
            PIL Image in grayscale format
        """
        return Image.open(path).convert("L")

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets.

        Returns:
            Tensor of class weights
        """
        labels = [s.get("label", 0) for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Calculate per-sample weights for weighted sampling.

        Returns:
            Tensor of sample weights
        """
        class_weights = self.get_class_weights()
        labels = [s.get("label", 0) for s in self.samples]
        sample_weights = class_weights[labels]
        return sample_weights

    def split_dataset(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple["BaseDataset", "BaseDataset", "BaseDataset"]:
        """Split dataset into train/val/test.

        Args:
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))

        n_test = int(len(indices) * test_ratio)
        n_val = int(len(indices) * val_ratio)

        test_indices = indices[:n_test]
        val_indices = indices[n_test : n_test + n_val]
        train_indices = indices[n_test + n_val :]

        # Create copies with different samples
        import copy

        train_ds = copy.copy(self)
        val_ds = copy.copy(self)
        test_ds = copy.copy(self)

        train_ds.samples = [self.samples[i] for i in train_indices]
        val_ds.samples = [self.samples[i] for i in val_indices]
        test_ds.samples = [self.samples[i] for i in test_indices]

        train_ds.split = "train"
        val_ds.split = "val"
        test_ds.split = "test"

        return train_ds, val_ds, test_ds


class ImageClassificationDataset(BaseDataset):
    """Base class for image classification datasets."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(root, split, transform, **kwargs)
        self.samples = self._load_samples()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        # Load from cache or disk
        if self.cache_images and index in self.cache:
            image = self.cache[index]
        else:
            image = self._load_image(sample["image_path"])
            if self.cache_images:
                self.cache[index] = image

        label = sample.get("label", 0)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return {
            "image": image,
            "label": label,
            "path": str(sample["image_path"]),
        }


class ImageSegmentationDataset(BaseDataset):
    """Base class for image segmentation datasets."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(root, split, transform, **kwargs)
        self.samples = self._load_samples()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        # Load image
        image = self._load_image(sample["image_path"])

        # Load mask
        if "mask_path" in sample and sample["mask_path"]:
            mask = self._load_mask(sample["mask_path"])
        else:
            # Create empty mask for authentic images
            mask = Image.new("L", image.size, 0)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(np.array(mask)).float() / 255.0

        return {
            "image": image,
            "mask": mask,
            "path": str(sample["image_path"]),
            "label": sample.get("label", 0),
        }
