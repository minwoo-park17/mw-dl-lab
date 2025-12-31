"""
Data sampling strategies for handling class imbalance.

Supports multiple sampling strategies:
- weighted: WeightedRandomSampler (probability-based)
- balanced: BalancedBatchSampler (exact 1:1 ratio per batch)
- none: No sampling (use default shuffle)
"""
import random
import logging
from typing import List, Optional, Iterator

import torch
from torch.utils.data import Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


def create_sampler(
    strategy: str,
    labels: List[int],
    batch_size: int,
    epoch_mode: str = "full"
) -> Optional[Sampler]:
    """
    Factory function for creating data samplers.

    Args:
        strategy: Sampling strategy
            - "weighted": WeightedRandomSampler (probability-based balancing)
            - "balanced": BalancedBatchSampler (exact 1:1 per batch)
            - "none": No sampler (use DataLoader shuffle)
        labels: List of labels (0=real, 1=fake)
        batch_size: Batch size for training
        epoch_mode: How to count samples per epoch
            - "minority": Use minority class count * 2
            - "full": Use all samples

    Returns:
        Sampler instance or None
    """
    if strategy == "none":
        logger.info("Sampling strategy: none (using default shuffle)")
        return None

    elif strategy == "weighted":
        logger.info("Sampling strategy: WeightedRandomSampler")
        return create_weighted_sampler(labels, epoch_mode)

    elif strategy == "balanced":
        logger.info("Sampling strategy: BalancedBatchSampler")
        return BalancedBatchSampler(labels, batch_size, epoch_mode)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}. "
                        f"Available: 'weighted', 'balanced', 'none'")


def create_weighted_sampler(
    labels: List[int],
    epoch_mode: str = "full"
) -> WeightedRandomSampler:
    """
    Create WeightedRandomSampler for class-balanced sampling.

    Args:
        labels: List of labels (0=real, 1=fake)
        epoch_mode: "minority" or "full"

    Returns:
        WeightedRandomSampler instance
    """
    # Count classes
    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)

    logger.info(f"Class distribution - Real: {n_real}, Fake: {n_fake}")

    # Calculate class weights (inverse of frequency)
    class_counts = torch.tensor([n_real, n_fake], dtype=torch.float)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = torch.tensor([class_weights[label] for label in labels])

    # Determine number of samples per epoch
    if epoch_mode == "minority":
        num_samples = min(n_real, n_fake) * 2
        logger.info(f"Epoch mode: minority (num_samples={num_samples})")
    else:  # full
        num_samples = len(labels)
        logger.info(f"Epoch mode: full (num_samples={num_samples})")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )


class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures each batch has equal number of real and fake samples.

    For batch_size=8, each batch will have 4 real and 4 fake images.
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        epoch_mode: str = "full"
    ):
        """
        Initialize BalancedBatchSampler.

        Args:
            labels: List of labels (0=real, 1=fake)
            batch_size: Must be even number for 1:1 balance
            epoch_mode: "minority" or "full"
        """
        super().__init__(data_source=None)

        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even for balanced sampling, got {batch_size}")

        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.epoch_mode = epoch_mode

        # Separate indices by class
        self.real_indices = [i for i, l in enumerate(labels) if l == 0]
        self.fake_indices = [i for i, l in enumerate(labels) if l == 1]

        n_real = len(self.real_indices)
        n_fake = len(self.fake_indices)

        logger.info(f"BalancedBatchSampler - Real: {n_real}, Fake: {n_fake}")

        # Calculate number of batches per epoch
        if epoch_mode == "minority":
            min_class = min(n_real, n_fake)
            self.num_batches = min_class // self.half_batch
            logger.info(f"Epoch mode: minority (num_batches={self.num_batches})")
        else:  # full
            max_class = max(n_real, n_fake)
            self.num_batches = max_class // self.half_batch
            logger.info(f"Epoch mode: full (num_batches={self.num_batches})")

    def __iter__(self) -> Iterator[List[int]]:
        """Generate balanced batches."""
        # Shuffle indices
        real_indices = self.real_indices.copy()
        fake_indices = self.fake_indices.copy()
        random.shuffle(real_indices)
        random.shuffle(fake_indices)

        # Generate batches
        for i in range(self.num_batches):
            batch = []

            # Sample real images (with replacement if needed)
            for j in range(self.half_batch):
                idx = (i * self.half_batch + j) % len(real_indices)
                batch.append(real_indices[idx])

            # Sample fake images (with replacement if needed)
            for j in range(self.half_batch):
                idx = (i * self.half_batch + j) % len(fake_indices)
                batch.append(fake_indices[idx])

            # Shuffle within batch
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self.num_batches


# =============================================================================
# 확장 예시: 새로운 샘플러 추가 시 아래 패턴 따르기
# =============================================================================
#
# class NewCustomSampler(Sampler):
#     """새로운 샘플링 전략"""
#     def __init__(self, labels, batch_size, epoch_mode):
#         ...
#
#     def __iter__(self):
#         ...
#
#     def __len__(self):
#         ...
#
# 그 후 create_sampler() 함수에 elif 추가:
#     elif strategy == "new_custom":
#         return NewCustomSampler(labels, batch_size, epoch_mode)
# =============================================================================
