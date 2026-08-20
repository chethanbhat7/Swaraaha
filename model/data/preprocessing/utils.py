"""General utilities: padding for arrays, audio, and labels."""

from typing import Tuple

import numpy as np


def pad_to_length(array: np.ndarray, target_length: int, axis: int = -1, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad a numpy array along a given axis to reach target_length.

    If the array is already longer than target_length, it is truncated.

    Args:
        array: Input array.
        target_length: Desired length along the given axis.
        axis: Axis to pad/truncate.
        pad_value: Value used for padding.

    Returns:
        Padded or truncated array.
    """
    current_length = array.shape[axis]
    if current_length >= target_length:
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(0, target_length)
        return array[tuple(slices)]
    else:
        pad_width = target_length - current_length
        pads = [(0, 0)] * array.ndim
        pads[axis] = (0, pad_width)
        return np.pad(array, pads, mode="constant", constant_values=pad_value)


def pad_audio_and_labels(
    audio: np.ndarray,
    labels: np.ndarray,
    max_length_samples: int,
    sr: int = 16000,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad audio and its corresponding frame labels to a fixed length.

    Used for batching variable-length samples in a DataLoader.

    Args:
        audio: 1-D audio array.
        labels: 1-D frame label array (must align with audio after padding).
        max_length_samples: Target audio length in samples.
        sr: Sample rate.
        hop_length: Hop length for label alignment.

    Returns:
        Tuple of (padded_audio, padded_labels).
    """
    audio = pad_to_length(audio, max_length_samples, axis=0, pad_value=0.0)
    max_frames = max_length_samples // hop_length
    labels = pad_to_length(labels, max_frames, axis=0, pad_value=0)
    return audio, labels
