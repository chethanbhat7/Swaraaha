"""Class balancing: weights, pos_weight, oversampling, balanced sampler."""

from typing import Dict

import numpy as np


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    Useful for setting pos_weight in BCEWithLogitsLoss or class_weight in
    other loss functions. Higher weight for minority classes.

    Args:
        labels: 1-D array of binary labels (0 or 1) for a single class,
                or 2-D array (N, C) for multi-label.

    Returns:
        Dict mapping class index to weight. {0: weight_neg, 1: weight_pos}.
    """
    labels = np.asarray(labels).flatten()
    n_total = len(labels)
    n_pos = int(labels.sum())
    n_neg = n_total - n_pos

    if n_pos == 0 or n_neg == 0:
        return {0: 1.0, 1: 1.0}

    weight_neg = n_total / (2.0 * n_neg)
    weight_pos = n_total / (2.0 * n_pos)

    return {0: round(weight_neg, 4), 1: round(weight_pos, 4)}


def compute_pos_weight(labels: np.ndarray) -> float:
    """
    Compute pos_weight for BCEWithLogitsLoss from binary labels.

    pos_weight = n_neg / n_pos. This tells the loss to penalize
    false negatives more heavily when positives are rare.

    Args:
        labels: 1-D array of binary labels (0 or 1).

    Returns:
        pos_weight as float. Returns 1.0 if balanced.
    """
    labels = np.asarray(labels).flatten()
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    if n_pos == 0:
        return 1.0

    return round(n_neg / n_pos, 4)


def oversample_minority(
    indices: np.ndarray,
    labels: np.ndarray,
    target_ratio: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Oversample minority class indices to reach target_pos/neg ratio.

    Args:
        indices: Array of dataset indices to resample.
        labels: Binary labels aligned with indices (0 or 1).
        target_ratio: Desired ratio of positives to negatives. Default 1.0 (balanced).
        seed: Random seed for reproducibility.

    Returns:
        Resampled indices array with oversampled minority class.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    pos_idx = indices[labels == 1]
    neg_idx = indices[labels == 0]

    n_neg = len(neg_idx)
    target_pos = int(n_neg * target_ratio)

    if len(pos_idx) >= target_pos:
        return indices

    oversampled = rng.choice(pos_idx, size=target_pos, replace=True)
    return np.concatenate([neg_idx, oversampled])


def create_balanced_sampler(labels: np.ndarray) -> "torch.utils.data.WeightedRandomSampler":
    """
    Create a PyTorch WeightedRandomSampler for balanced mini-batches.

    Each sample gets weight = 1/class_frequency. Minority class samples
    are sampled more frequently to balance each batch.

    Args:
        labels: 1-D array of binary labels for the dataset.

    Returns:
        WeightedRandomSampler instance.

    Example:
        >>> sampler = create_balanced_sampler(train_labels)
        >>> loader = DataLoader(dataset, batch_size=8, sampler=sampler)
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(labels).flatten()
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    weights = np.where(labels == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    weights = weights / weights.sum()

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(labels),
        replacement=True,
    )
