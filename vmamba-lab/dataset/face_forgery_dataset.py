"""Face forgery detection datasets."""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .base_dataset import ImageClassificationDataset


class FaceForgeryDataset(ImageClassificationDataset):
    """Generic face forgery dataset for binary classification (Real vs Fake)."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        real_dirs: Optional[List[str]] = None,
        fake_dirs: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize face forgery dataset.

        Args:
            root: Root directory
            split: Data split
            transform: Image transforms
            real_dirs: Subdirectories containing real images
            fake_dirs: Subdirectories containing fake images
        """
        self.real_dirs = real_dirs or ["real"]
        self.fake_dirs = fake_dirs or ["fake"]
        super().__init__(root, split, transform, **kwargs)

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []

        # Load real images
        for real_dir in self.real_dirs:
            real_path = self.root / real_dir
            if real_path.exists():
                for img_path in real_path.rglob("*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        samples.append({
                            "image_path": img_path,
                            "label": 0,  # Real
                            "class_name": "real",
                        })

        # Load fake images
        for fake_dir in self.fake_dirs:
            fake_path = self.root / fake_dir
            if fake_path.exists():
                for img_path in fake_path.rglob("*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        samples.append({
                            "image_path": img_path,
                            "label": 1,  # Fake
                            "class_name": "fake",
                        })

        return samples


class FFPPDataset(ImageClassificationDataset):
    """FaceForensics++ Dataset.

    Supports multiple manipulation methods and compression levels.
    """

    METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    COMPRESSIONS = ["c0", "c23", "c40"]  # raw, HQ, LQ

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        compression: str = "c23",
        methods: Optional[List[str]] = None,
        split_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """Initialize FF++ dataset.

        Args:
            root: Root directory of FF++ dataset
            split: Data split ('train', 'val', 'test')
            transform: Image transforms
            compression: Compression level ('c0', 'c23', 'c40')
            methods: List of manipulation methods to include
            split_file: Path to split JSON file
        """
        self.compression = compression
        self.methods = methods or self.METHODS
        self.split_file = split_file
        self.video_ids: List[str] = []
        super().__init__(root, split, transform, **kwargs)

    def _load_split_ids(self) -> List[str]:
        """Load video IDs for the current split."""
        if self.split_file and Path(self.split_file).exists():
            with open(self.split_file, "r") as f:
                splits = json.load(f)
            return splits.get(self.split, [])

        # Default splits based on standard FF++ protocol
        split_path = self.root / "splits" / f"{self.split}.json"
        if split_path.exists():
            with open(split_path, "r") as f:
                return json.load(f)

        return []

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []
        self.video_ids = self._load_split_ids()

        # Real videos (original)
        real_dir = self.root / "original_sequences" / "youtube" / self.compression / "images"
        if real_dir.exists():
            for video_dir in real_dir.iterdir():
                if video_dir.is_dir():
                    video_id = video_dir.name
                    if not self.video_ids or video_id in self.video_ids:
                        for img_path in video_dir.glob("*.png"):
                            samples.append({
                                "image_path": img_path,
                                "label": 0,
                                "class_name": "real",
                                "method": "original",
                                "video_id": video_id,
                            })

        # Fake videos (manipulated)
        for method in self.methods:
            fake_dir = self.root / "manipulated_sequences" / method / self.compression / "images"
            if fake_dir.exists():
                for video_dir in fake_dir.iterdir():
                    if video_dir.is_dir():
                        video_id = video_dir.name.split("_")[0]  # Handle paired IDs
                        if not self.video_ids or video_id in self.video_ids:
                            for img_path in video_dir.glob("*.png"):
                                samples.append({
                                    "image_path": img_path,
                                    "label": 1,
                                    "class_name": "fake",
                                    "method": method,
                                    "video_id": video_id,
                                })

        return samples


class CelebDFDataset(ImageClassificationDataset):
    """Celeb-DF Dataset (v1 and v2)."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "test",  # Usually used for testing only
        transform: Optional[Callable] = None,
        version: str = "v2",
        include_youtube: bool = True,
        **kwargs,
    ):
        """Initialize Celeb-DF dataset.

        Args:
            root: Root directory
            split: Data split
            transform: Image transforms
            version: Dataset version ('v1' or 'v2')
            include_youtube: Include YouTube real videos (v2 only)
        """
        self.version = version
        self.include_youtube = include_youtube
        super().__init__(root, split, transform, **kwargs)

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []

        # Real images
        real_dir = self.root / "Celeb-real"
        if real_dir.exists():
            for img_path in real_dir.rglob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    samples.append({
                        "image_path": img_path,
                        "label": 0,
                        "class_name": "real",
                        "source": "celeb-real",
                    })

        # YouTube real (v2 only)
        if self.version == "v2" and self.include_youtube:
            yt_dir = self.root / "YouTube-real"
            if yt_dir.exists():
                for img_path in yt_dir.rglob("*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        samples.append({
                            "image_path": img_path,
                            "label": 0,
                            "class_name": "real",
                            "source": "youtube-real",
                        })

        # Fake images
        fake_dir = self.root / "Celeb-synthesis"
        if fake_dir.exists():
            for img_path in fake_dir.rglob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    samples.append({
                        "image_path": img_path,
                        "label": 1,
                        "class_name": "fake",
                        "source": "celeb-synthesis",
                    })

        return samples


class DFDCDataset(ImageClassificationDataset):
    """DeepFake Detection Challenge (DFDC) Dataset."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "test",
        transform: Optional[Callable] = None,
        metadata_file: Optional[str] = None,
        subset: str = "preview",  # 'preview' or 'full'
        **kwargs,
    ):
        """Initialize DFDC dataset.

        Args:
            root: Root directory
            split: Data split
            transform: Image transforms
            metadata_file: Path to metadata JSON file
            subset: Dataset subset ('preview' for smaller set)
        """
        self.metadata_file = metadata_file
        self.subset = subset
        self.metadata: Dict = {}
        super().__init__(root, split, transform, **kwargs)

    def _load_metadata(self) -> Dict:
        """Load DFDC metadata."""
        if self.metadata_file and Path(self.metadata_file).exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)

        # Try to find metadata in root
        meta_path = self.root / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)

        return {}

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []
        self.metadata = self._load_metadata()

        # If metadata available, use it
        if self.metadata:
            for video_name, info in self.metadata.items():
                label = 1 if info.get("label") == "FAKE" else 0
                video_dir = self.root / video_name.replace(".mp4", "")

                if video_dir.exists():
                    for img_path in video_dir.glob("*.png"):
                        samples.append({
                            "image_path": img_path,
                            "label": label,
                            "class_name": "fake" if label == 1 else "real",
                            "video_name": video_name,
                            "original": info.get("original", None),
                        })
        else:
            # Fallback: scan directories
            for subdir in ["real", "fake"]:
                label = 0 if subdir == "real" else 1
                dir_path = self.root / subdir
                if dir_path.exists():
                    for img_path in dir_path.rglob("*"):
                        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            samples.append({
                                "image_path": img_path,
                                "label": label,
                                "class_name": subdir,
                            })

        return samples


def create_face_forgery_dataloader(
    dataset: ImageClassificationDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    use_weighted_sampler: bool = False,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for face forgery dataset.

    Args:
        dataset: Face forgery dataset
        batch_size: Batch size
        shuffle: Shuffle data (ignored if use_weighted_sampler)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        use_weighted_sampler: Use weighted sampling for class balance

    Returns:
        DataLoader instance
    """
    sampler = None
    if use_weighted_sampler:
        weights = dataset.get_sample_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
