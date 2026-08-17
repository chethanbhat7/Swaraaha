"""Tests for the saliency-based localization synthesis fallback."""

import numpy as np
import pytest

from backend.services import localizer
from model.registry import DYSFLUENCY_CLASSES


def test_saliency_regions_finds_peak_spans():
    rng = np.random.default_rng(0)
    saliency = rng.uniform(0.2, 0.4, size=(500, 5)).astype(float)
    saliency[100:140, 3] = 0.95
    regions = localizer._saliency_regions(saliency, list(DYSFLUENCY_CLASSES), 3.0)
    wordrep = [r for r in regions if r["type"] == "wordrep"]
    assert len(wordrep) == 1
    r = wordrep[0]
    assert r["start"] == pytest.approx(2.0, abs=0.05)
    assert r["end"] == pytest.approx(2.8, abs=0.05)
    assert r["confidence"] == pytest.approx(0.95, abs=1e-4)


def test_saliency_regions_empty_when_no_peaks():
    rng = np.random.default_rng(1)
    saliency = rng.uniform(0.1, 0.3, size=(500, 5)).astype(float)
    regions = localizer._saliency_regions(saliency, list(DYSFLUENCY_CLASSES), 3.0)
    assert regions == []


def test_saliency_regions_short_spans_dropped():
    rng = np.random.default_rng(2)
    saliency = rng.uniform(0.2, 0.4, size=(500, 5)).astype(float)
    saliency[100:104, 0] = 0.95
    regions = localizer._saliency_regions(saliency, list(DYSFLUENCY_CLASSES), 3.0)
    assert regions == []


def test_saliency_regions_overlaps_deduped():
    rng = np.random.default_rng(3)
    saliency = rng.uniform(0.2, 0.4, size=(500, 5)).astype(float)
    saliency[100:200, 0] = 0.8
    saliency[120:180, 1] = 0.9
    regions = localizer._saliency_regions(saliency, list(DYSFLUENCY_CLASSES), 3.0)
    assert len(regions) == 1
    assert regions[0]["type"] == "block"


def test_saliency_regions_clamped_to_duration():
    rng = np.random.default_rng(4)
    saliency = rng.uniform(0.2, 0.4, size=(500, 5)).astype(float)
    saliency[450:480, 4] = 0.9
    regions = localizer._saliency_regions(saliency, list(DYSFLUENCY_CLASSES), 3.0)
    assert all(r["end"] <= 3.0 for r in regions)


def test_saliency_fallback_returns_well_formed(monkeypatch):
    class FakeMultiTask:
        def saliency(self, audio):
            arr = np.zeros((1, 500, 5), dtype=float)
            arr[0, 50:90, 2] = 0.99
            return arr

    monkeypatch.setattr(localizer, "_get_multitask", lambda: FakeMultiTask())
    result = localizer._saliency_fallback(np.zeros(160000, dtype=np.float32), 3.0)
    assert result["source"] == "saliency"
    assert len(result["regions"]) == 1
    assert result["regions"][0]["type"] == "soundrep"
    assert "error" not in result


def test_saliency_fallback_handles_failure(monkeypatch):
    class BrokenMultiTask:
        def saliency(self, audio):
            raise RuntimeError("boom")

    monkeypatch.setattr(localizer, "_get_multitask", lambda: BrokenMultiTask())
    result = localizer._saliency_fallback(np.zeros(160000, dtype=np.float32), 3.0)
    assert result["regions"] == []
    assert "boom" in result["error"]
