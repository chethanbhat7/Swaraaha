"""Frame-level label creation for localization tasks."""

from typing import List, Tuple, Union

import numpy as np


def create_frame_labels(
    intervals: List[Tuple[Union[float, int], Union[float, int], ...]],
    num_frames: int,
    sr: int = 16000,
    hop_length: int = 512,
    units: str = "seconds",
    end_rounding: str = "ceil",
) -> np.ndarray:
    """
    Create binary frame-level labels.

    Args:
        intervals: List of (start, end) or (start, end, type) tuples.
            The third element (if present) is ignored.
        num_frames: Total number of output frames.
        sr: Sample rate (used when units='seconds').
        hop_length: Samples per frame.
        units: 'seconds' or 'samples'.
        end_rounding: 'ceil' to include partial final frames, 'floor' for strict division.

    Returns:
        uint8 array of shape (num_frames,).
    """
    labels = np.zeros(num_frames, dtype=np.uint8)

    for interval in intervals:
        start_val, end_val = float(interval[0]), float(interval[1])

        if units == "seconds":
            start_sample = int(start_val * sr)
            end_sample = int(end_val * sr)
        else:
            start_sample = int(start_val)
            end_sample = int(end_val)

        start_frame = start_sample // hop_length
        if end_rounding == "ceil":
            end_frame = (end_sample + hop_length - 1) // hop_length
        else:
            end_frame = end_sample // hop_length

        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        labels[start_frame:end_frame] = 1

    return labels


def create_frame_labels_from_samples(
    dysfluency_sample_ranges: List[Tuple[int, int]],
    num_frames: int,
    hop_length: int = 512,
) -> np.ndarray:
    """Deprecated: use create_frame_labels(..., units='samples') instead."""
    return create_frame_labels(
        dysfluency_sample_ranges, num_frames, hop_length=hop_length, units="samples"
    )
