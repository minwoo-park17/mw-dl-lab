"""Image tampering localization datasets."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .base_dataset import ImageSegmentationDataset


class TamperingDataset(ImageSegmentationDataset):
    """Generic image tampering dataset for segmentation."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        image_dir: str = "images",
        mask_dir: str = "masks",
        authentic_dir: Optional[str] = "authentic",
        **kwargs,
    ):
        """Initialize tampering dataset.

        Args:
            root: Root directory
            split: Data split
            transform: Image and mask transforms
            image_dir: Subdirectory containing tampered images
            mask_dir: Subdirectory containing masks
            authentic_dir: Subdirectory containing authentic images
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.authentic_dir = authentic_dir
        super().__init__(root, split, transform, **kwargs)

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []

        # Tampered images with masks
        img_path = self.root / self.image_dir
        mask_path = self.root / self.mask_dir

        if img_path.exists():
            for img_file in img_path.rglob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
                    # Find corresponding mask
                    mask_file = self._find_mask(img_file, mask_path)

                    samples.append({
                        "image_path": img_file,
                        "mask_path": mask_file,
                        "label": 1,  # Tampered
                        "class_name": "tampered",
                    })

        # Authentic images (no tampering)
        if self.authentic_dir:
            auth_path = self.root / self.authentic_dir
            if auth_path.exists():
                for img_file in auth_path.rglob("*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
                        samples.append({
                            "image_path": img_file,
                            "mask_path": None,  # No mask for authentic
                            "label": 0,  # Authentic
                            "class_name": "authentic",
                        })

        return samples

    def _find_mask(self, img_path: Path, mask_dir: Path) -> Optional[Path]:
        """Find corresponding mask file for an image.

        Args:
            img_path: Path to image file
            mask_dir: Directory containing masks

        Returns:
            Path to mask file or None
        """
        stem = img_path.stem
        for ext in [".png", ".jpg", ".bmp", ".tif"]:
            mask_path = mask_dir / f"{stem}{ext}"
            if mask_path.exists():
                return mask_path
            # Try with _mask suffix
            mask_path = mask_dir / f"{stem}_mask{ext}"
            if mask_path.exists():
                return mask_path
        return None


class CASIADataset(ImageSegmentationDataset):
    """CASIA Image Tampering Detection Dataset (v1 and v2)."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        version: str = "v2",
        **kwargs,
    ):
        """Initialize CASIA dataset.

        Args:
            root: Root directory
            split: Data split
            transform: Transforms
            version: Dataset version ('v1' or 'v2')
        """
        self.version = version
        super().__init__(root, split, transform, **kwargs)

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []

        if self.version == "v1":
            # CASIA v1 structure
            au_dir = self.root / "Au"
            tp_dir = self.root / "Tp"

            # Authentic
            if au_dir.exists():
                for img_path in au_dir.glob("*"):
                    if img_path.suffix.lower() in [".jpg", ".bmp", ".tif"]:
                        samples.append({
                            "image_path": img_path,
                            "mask_path": None,
                            "label": 0,
                            "class_name": "authentic",
                        })

            # Tampered
            if tp_dir.exists():
                for img_path in tp_dir.glob("*"):
                    if img_path.suffix.lower() in [".jpg", ".bmp", ".tif"]:
                        samples.append({
                            "image_path": img_path,
                            "mask_path": None,  # CASIA v1 doesn't have masks
                            "label": 1,
                            "class_name": "tampered",
                        })

        else:  # v2
            # CASIA v2 structure
            au_dir = self.root / "Au"
            tp_dir = self.root / "Tp"
            mask_dir = self.root / "Mask" if (self.root / "Mask").exists() else self.root / "GT"

            # Authentic
            if au_dir.exists():
                for img_path in au_dir.rglob("*"):
                    if img_path.suffix.lower() in [".jpg", ".bmp", ".tif", ".png"]:
                        samples.append({
                            "image_path": img_path,
                            "mask_path": None,
                            "label": 0,
                            "class_name": "authentic",
                        })

            # Tampered with masks
            if tp_dir.exists():
                for img_path in tp_dir.rglob("*"):
                    if img_path.suffix.lower() in [".jpg", ".bmp", ".tif", ".png"]:
                        # Find mask
                        mask_path = self._find_casia_mask(img_path, mask_dir)
                        samples.append({
                            "image_path": img_path,
                            "mask_path": mask_path,
                            "label": 1,
                            "class_name": "tampered",
                        })

        return samples

    def _find_casia_mask(self, img_path: Path, mask_dir: Path) -> Optional[Path]:
        """Find CASIA mask file."""
        if not mask_dir.exists():
            return None

        stem = img_path.stem
        # CASIA naming: Tp_D_NRN_S_N_xxx -> mask
        for ext in [".png", ".bmp", ".tif", ".jpg"]:
            mask_path = mask_dir / f"{stem}{ext}"
            if mask_path.exists():
                return mask_path
            mask_path = mask_dir / f"{stem}_gt{ext}"
            if mask_path.exists():
                return mask_path
        return None


class ColumbiaDataset(ImageSegmentationDataset):
    """Columbia Uncompressed Image Splicing Detection Dataset."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "test",
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(root, split, transform, **kwargs)

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []

        # Authentic images (4cam_auth)
        auth_dir = self.root / "4cam_auth"
        if auth_dir.exists():
            for img_path in auth_dir.glob("*.tif"):
                samples.append({
                    "image_path": img_path,
                    "mask_path": None,
                    "label": 0,
                    "class_name": "authentic",
                })

        # Spliced images (4cam_splc)
        splc_dir = self.root / "4cam_splc"
        mask_dir = self.root / "masks" if (self.root / "masks").exists() else self.root / "edgemask"

        if splc_dir.exists():
            for img_path in splc_dir.glob("*.tif"):
                # Find corresponding mask
                mask_path = None
                if mask_dir.exists():
                    mask_name = img_path.stem + "_edgemask.jpg"
                    potential_mask = mask_dir / mask_name
                    if potential_mask.exists():
                        mask_path = potential_mask

                samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "label": 1,
                    "class_name": "spliced",
                })

        return samples


class CoverageDataset(ImageSegmentationDataset):
    """COVERAGE Copy-Move Forgery Detection Dataset."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "test",
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(root, split, transform, **kwargs)

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []

        img_dir = self.root / "image"
        mask_dir = self.root / "mask"

        if img_dir.exists():
            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() in [".jpg", ".png", ".tif"]:
                    # Find corresponding mask
                    mask_path = None
                    if mask_dir.exists():
                        # Coverage naming: image/1.tif -> mask/1forged.tif
                        mask_name = f"{img_path.stem}forged{img_path.suffix}"
                        potential_mask = mask_dir / mask_name
                        if potential_mask.exists():
                            mask_path = potential_mask
                        else:
                            # Try exact name match
                            potential_mask = mask_dir / img_path.name
                            if potential_mask.exists():
                                mask_path = potential_mask

                    samples.append({
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "label": 1,  # All images in Coverage are forged
                        "class_name": "copy-move",
                    })

        return samples


class NIST16Dataset(ImageSegmentationDataset):
    """NIST Nimble 2016 Media Forensics Challenge Dataset."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "test",
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(root, split, transform, **kwargs)

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []

        # Probe images
        probe_dir = self.root / "probe"
        mask_dir = self.root / "reference" / "manipulation" / "mask"

        if probe_dir.exists():
            for img_path in probe_dir.rglob("*"):
                if img_path.suffix.lower() in [".jpg", ".png", ".tif"]:
                    # Find corresponding mask
                    mask_path = self._find_nist_mask(img_path, mask_dir)

                    # Determine if tampered based on mask existence
                    label = 1 if mask_path else 0

                    samples.append({
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "label": label,
                        "class_name": "tampered" if label else "authentic",
                    })

        return samples

    def _find_nist_mask(self, img_path: Path, mask_dir: Path) -> Optional[Path]:
        """Find NIST mask file."""
        if not mask_dir.exists():
            return None

        stem = img_path.stem
        for ext in [".png", ".jpg", ".tif"]:
            mask_path = mask_dir / f"{stem}{ext}"
            if mask_path.exists():
                return mask_path
            # Try with manipulation suffix
            mask_path = mask_dir / f"{stem}_manipulation{ext}"
            if mask_path.exists():
                return mask_path
        return None


class CombinedTamperingDataset(torch.utils.data.Dataset):
    """Combine multiple tampering datasets."""

    def __init__(
        self,
        datasets: List[ImageSegmentationDataset],
        weights: Optional[List[float]] = None,
    ):
        """Initialize combined dataset.

        Args:
            datasets: List of datasets to combine
            weights: Optional sampling weights for each dataset
        """
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)

        # Calculate cumulative lengths
        self.cum_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cum_lengths.append(total)

    def __len__(self) -> int:
        return self.cum_lengths[-1] if self.cum_lengths else 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cum_lengths):
            if index < cum_len:
                prev_len = self.cum_lengths[i - 1] if i > 0 else 0
                local_idx = index - prev_len
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {index} out of range")


def create_tampering_dataloader(
    dataset: ImageSegmentationDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for tampering dataset.

    Args:
        dataset: Tampering dataset
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
