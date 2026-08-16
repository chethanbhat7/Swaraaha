import numpy as np
import pytest

from model.evaluation.probe_combiner import aggregate_mismatch, probe_from_regions


def test_aggregate_mismatch_empty():
    summary = aggregate_mismatch([])
    assert summary["mean_mismatch"] == 0.0
    assert summary["clips"] == 0


def test_aggregate_mismatch_averages():
    summary = aggregate_mismatch([0.5, 0.0, 1.0])
    assert summary["mean_mismatch"] == pytest.approx(0.5)
    assert summary["clips"] == 3
    assert summary["clips_with_mismatch"] == 2


def test_probe_from_regions_with_saliency():
    regions = [{"start": 0.0, "end": 0.5, "confidence": 0.8}]
    saliency = np.zeros((100, 5))
    saliency[10:20, 1] = 0.9  # covered by region (frames 0-25)
    saliency[80:90, 2] = 0.9  # NOT covered
    rate = probe_from_regions(regions, saliency)
    assert rate == pytest.approx(0.5)
