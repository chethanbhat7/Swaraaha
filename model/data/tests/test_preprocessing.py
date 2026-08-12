"""Tests for frame-label generation (model/data/preprocessing.py)."""

import numpy as np
import pytest

from model.data.preprocessing import create_frame_labels

SR = 16000
HOP = 512


def test_create_frame_labels_includes_final_partial_frame():
    """An interval ending mid-frame must mark that final partial frame
    (previously end_frame was floored, systematically dropping it)."""
    labels = create_frame_labels([(1.0, 2.0)], num_frames=156, sr=SR, hop_length=HOP)
    start_frame = int(1.0 * SR) // HOP
    end_frame = (int(2.0 * SR) + HOP - 1) // HOP
    assert labels.sum() == end_frame - start_frame
    assert labels[start_frame:end_frame].sum() == end_frame - start_frame
    assert labels[start_frame - 1] == 0 if start_frame > 0 else True
    assert labels[end_frame] == 0
    assert labels[end_frame - 1] == 1


def test_create_frame_labels_sub_frame_interval_marks_one_frame():
    """An interval shorter than one hop must still mark a single frame rather
    than vanishing entirely."""
    labels = create_frame_labels([(0.0, 0.0005)], num_frames=156, sr=SR, hop_length=HOP)
    assert labels.sum() == 1
    assert labels[0] == 1


def test_create_frame_labels_end_clamped_at_num_frames():
    """An interval running to the end of the audio must not exceed num_frames."""
    labels = create_frame_labels([(0.0, 5.0)], num_frames=156, sr=SR, hop_length=HOP)
    assert labels.shape == (156,)
    assert labels.sum() == 156
