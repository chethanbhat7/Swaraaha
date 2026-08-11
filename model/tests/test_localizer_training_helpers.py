"""Tests for localizer training helpers (pos_weight + threshold-free metric)."""

import numpy as np
import torch

from model.training.utils import (
    build_localizer_criterion,
    compute_event_mean_iou,
    compute_frame_auroc,
    compute_frame_pos_weight,
    compute_mean_iou,
    extract_regions,
)


def _samples_with_labels(tmp_path, specs):
    """Build a samples list from (csv_name, [(start,end)]) specs."""
    samples = []
    for i, (name, intervals) in enumerate(specs):
        path = tmp_path / f"{name}.csv"
        lines = ["start_sec,end_sec,dysfluency_type"]
        for start, end in intervals:
            lines.append(f"{start},{end},Block")
        path.write_text("\n".join(lines) + "\n")
        samples.append({"clip_id": f"c{i}", "label_path": str(path)})
    return samples


def test_build_localizer_criterion_plain_by_default():
    criterion = build_localizer_criterion()
    assert isinstance(criterion, torch.nn.BCEWithLogitsLoss)
    assert criterion.pos_weight is None


def test_build_localizer_criterion_with_pos_weight():
    criterion = build_localizer_criterion(pos_weight=5.0)
    assert isinstance(criterion, torch.nn.BCEWithLogitsLoss)
    assert criterion.pos_weight is not None
    assert float(criterion.pos_weight) == 5.0


def test_build_localizer_criterion_pos_weight_on_device():
    criterion = build_localizer_criterion(pos_weight=5.0, device="cpu")
    assert criterion.pos_weight.device.type == "cpu"


def test_compute_frame_auroc_perfect_separation():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert compute_frame_auroc(y_true, y_score) == 1.0


def test_compute_frame_auroc_reversed():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    assert compute_frame_auroc(y_true, y_score) == 0.0


def test_compute_frame_auroc_degenerate_no_positives():
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    # Undefined for a single class -> neutral score so early stopping is not fooled
    assert compute_frame_auroc(y_true, y_score) == 0.5


def test_compute_frame_pos_weight_no_audio_load(tmp_path):
    samples = _samples_with_labels(tmp_path, [("clean", [])])
    assert compute_frame_pos_weight(samples, num_frames=100) == 1.0


def test_compute_frame_pos_weight_inverse_frequency(tmp_path):
    samples = _samples_with_labels(tmp_path, [("dys", [(1.0, 2.0)])])
    # sr=16000, hop=512: interval (1.0,2.0) -> frames 31..62 => 31 positive of 100
    assert compute_frame_pos_weight(samples, num_frames=100) == round(69 / 31, 4)


def test_compute_frame_pos_weight_aggregates_across_samples(tmp_path):
    samples = _samples_with_labels(
        tmp_path, [("clean", []), ("dys", [(1.0, 2.0)])]
    )
    # 200 total frames, 31 positive -> 169/31
    assert compute_frame_pos_weight(samples, num_frames=100) == round(169 / 31, 4)


def test_extract_regions_single_contiguous_run():
    mask = np.array([0, 0, 1, 1, 1, 0, 0])
    assert extract_regions(mask) == [(2, 5)]


def test_extract_regions_multiple_runs_and_open_end():
    mask = np.array([1, 0, 1, 1, 0, 1])
    assert extract_regions(mask) == [(0, 1), (2, 4), (5, 6)]


def test_extract_regions_all_negative():
    assert extract_regions(np.zeros(5, dtype=int)) == []


def test_compute_mean_iou_partial_overlap():
    true_regions = [(0, 10)]
    pred_regions = [(5, 10)]
    # intersection 5, union 10 -> 0.5
    assert compute_mean_iou(true_regions, pred_regions) == 0.5


def test_compute_mean_iou_empty_side_returns_zero():
    assert compute_mean_iou([], [(0, 5)]) == 0.0
    assert compute_mean_iou([(0, 5)], []) == 0.0


def test_compute_event_mean_iou_perfect_and_missing():
    y_true = np.array([0, 0, 1, 1, 1, 0, 0])
    assert compute_event_mean_iou(y_true, y_true) == 1.0
    assert compute_event_mean_iou(y_true, np.zeros_like(y_true)) == 0.0
