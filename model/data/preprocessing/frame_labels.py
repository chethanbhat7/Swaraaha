"""Frame-level label creation for localization tasks."""

from typing import List, Tuple

import numpy as np


def create_frame_labels(
    dysfluency_intervals: List[Tuple[float, float]],
    num_frames: int,
    sr: int = 16000,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Create a binary frame-level label mask aligned to a spectrogram.

    Each spectrogram frame at index `i` corresponds to audio sample
    `i * hop_length`. This function converts (start_sec, end_sec) intervals
    into a binary mask over those frames.

    Args:
        dysfluency_intervals: List of (start_sec, end_sec) tuples marking
            dysfluent regions in the audio. Empty list = fully fluent.
        num_frames: Total number of spectrogram time frames (the mask length).
        sr: Sample rate used during spectrogram generation.
        hop_length: Hop length used during spectrogram generation.

    Returns:
        1-D numpy uint8 array of shape (num_frames,).
        1 = dysfluent frame, 0 = fluent frame.

    Example:
        >>> # Audio is 5 seconds, frames 50-100 are dysfluent
        >>> labels = create_frame_labels([(1.0, 2.0)], num_frames=156, sr=16000, hop_length=512)
        >>> print(labels.shape, labels.sum())  # (156,) — some 1s in the middle
    """
    labels = np.zeros(num_frames, dtype=np.uint8)

    samples_per_frame = hop_length

    for start_sec, end_sec in dysfluency_intervals:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        start_frame = start_sample // samples_per_frame
        end_frame = (end_sample + samples_per_frame - 1) // samples_per_frame
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        labels[start_frame:end_frame] = 1

    return labels


def create_frame_labels_from_samples(
    dysfluency_sample_ranges: List[Tuple[int, int]],
    num_frames: int,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Create frame labels when dysfluency boundaries are given as sample indices
    instead of seconds.

    Args:
        dysfluency_sample_ranges: List of (start_sample, end_sample) tuples.
        num_frames: Total spectrogram time frames.
        hop_length: Hop length used during spectrogram generation.

    Returns:
        1-D numpy uint8 array of shape (num_frames,).
    """
    labels = np.zeros(num_frames, dtype=np.uint8)

    for start_sample, end_sample in dysfluency_sample_ranges:
        start_frame = start_sample // hop_length
        end_frame = end_sample // hop_length
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        labels[start_frame:end_frame] = 1

    return labels
