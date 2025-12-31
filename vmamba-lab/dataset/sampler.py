"""Custom samplers for handling class imbalance."""

from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


class BalancedBatchSampler(Sampler):
    """Sampler that ensures balanced classes in each batch.

    Each batch contains equal number of samples from each class.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
    ):
        """Initialize balanced batch sampler.

        Args:
            dataset: Dataset with 'label' in samples
            batch_size: Total batch size
            drop_last: Drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Get labels
        self.labels = np.array([s.get("label", 0) for s in dataset.samples])
        self.unique_labels = np.unique(self.labels)
        self.n_classes = len(self.unique_labels)

        # Samples per class per batch
        self.samples_per_class = batch_size // self.n_classes

        # Indices for each class
        self.class_indices = {
            label: np.where(self.labels == label)[0].tolist()
            for label in self.unique_labels
        }

        # Shuffle indices within each class
        for label in self.unique_labels:
            np.random.shuffle(self.class_indices[label])

    def __iter__(self) -> Iterator[List[int]]:
        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        n_batches = min_class_size // self.samples_per_class

        if self.drop_last:
            n_batches = n_batches
        else:
            n_batches = max(1, n_batches)

        # Generate batches
        class_pointers = {label: 0 for label in self.unique_labels}

        for _ in range(n_batches):
            batch = []
            for label in self.unique_labels:
                indices = self.class_indices[label]
                start = class_pointers[label]
                end = start + self.samples_per_class

                # Handle wrap-around
                if end > len(indices):
                    np.random.shuffle(self.class_indices[label])
                    start = 0
                    end = self.samples_per_class
                    class_pointers[label] = end
                else:
                    class_pointers[label] = end

                batch.extend(indices[start:end])

            # Shuffle within batch
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        return min_class_size // self.samples_per_class


def get_weighted_sampler(
    dataset: Dataset,
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Create weighted random sampler for class balance.

    Args:
        dataset: Dataset with samples containing 'label' key
        num_samples: Number of samples to draw (default: len(dataset))
        replacement: Sample with replacement

    Returns:
        WeightedRandomSampler instance
    """
    # Get labels
    labels = np.array([s.get("label", 0) for s in dataset.samples])

    # Calculate class weights
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = 1.0 / counts

    # Normalize
    class_weights = class_weights / class_weights.sum()

    # Create weight for each sample
    sample_weights = np.array([class_weights[label] for label in labels])

    if num_samples is None:
        num_samples = len(labels)

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=num_samples,
        replacement=replacement,
    )


class StratifiedBatchSampler(Sampler):
    """Stratified sampler that maintains class distribution in batches."""

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """Initialize stratified batch sampler.

        Args:
            labels: List of labels for each sample
            batch_size: Batch size
            shuffle: Shuffle samples
            drop_last: Drop last incomplete batch
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.n_samples = len(labels)
        self.n_batches = self.n_samples // batch_size
        if not drop_last and self.n_samples % batch_size != 0:
            self.n_batches += 1

    def __iter__(self) -> Iterator[List[int]]:
        indices = np.arange(self.n_samples)

        if self.shuffle:
            # Shuffle within each class first
            unique_labels = np.unique(self.labels)
            shuffled_indices = []

            for label in unique_labels:
                class_indices = indices[self.labels == label]
                np.random.shuffle(class_indices)
                shuffled_indices.append(class_indices)

            # Interleave class indices
            indices = self._interleave(shuffled_indices)

        # Create batches
        for i in range(self.n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.n_samples)
            if self.drop_last and end - start < self.batch_size:
                break
            yield indices[start:end].tolist()

    def _interleave(self, arrays: List[np.ndarray]) -> np.ndarray:
        """Interleave multiple arrays."""
        max_len = max(len(arr) for arr in arrays)
        result = []

        for i in range(max_len):
            for arr in arrays:
                if i < len(arr):
                    result.append(arr[i])

        return np.array(result)

    def __len__(self) -> int:
        return self.n_batches


class TwoStreamBatchSampler(Sampler):
    """Sampler for two-stream data (e.g., labeled and unlabeled)."""

    def __init__(
        self,
        primary_indices: List[int],
        secondary_indices: List[int],
        batch_size: int,
        secondary_batch_size: int,
    ):
        """Initialize two-stream sampler.

        Args:
            primary_indices: Indices for primary stream (e.g., labeled)
            secondary_indices: Indices for secondary stream (e.g., unlabeled)
            batch_size: Primary stream batch size
            secondary_batch_size: Secondary stream batch size
        """
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.batch_size = batch_size
        self.secondary_batch_size = secondary_batch_size

        self.primary_batches = len(primary_indices) // batch_size
        self.secondary_batches = len(secondary_indices) // secondary_batch_size

    def __iter__(self) -> Iterator[List[int]]:
        primary_iter = self._iterate_indices(self.primary_indices, self.batch_size)
        secondary_iter = self._iterate_indices(
            self.secondary_indices, self.secondary_batch_size
        )

        for _ in range(len(self)):
            primary_batch = next(primary_iter)
            secondary_batch = next(secondary_iter)
            yield primary_batch + secondary_batch

    def _iterate_indices(
        self, indices: List[int], batch_size: int
    ) -> Iterator[List[int]]:
        """Iterate over indices indefinitely."""
        indices = np.array(indices)
        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                yield indices[i : i + batch_size].tolist()

    def __len__(self) -> int:
        return min(self.primary_batches, self.secondary_batches)
