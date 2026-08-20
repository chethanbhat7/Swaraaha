"""Tests for the saliency-based localization synthesis fallback.

Now tests model.registry.saliency_regions directly (the logic was moved
from backend.services.localizer into model.registry).
"""

import numpy as np
import pytest

from model.registry import DYSFLUENCY_CLASSES, saliency_regions


def test_saliency_regions_finds_peak_spans():
    rng = np.random.default_rng(0)
    saliency = rng.uniform(0.2, 0.4, size=(500, 5)).astype(float)
    saliency[100:140, 3] = 0.95
    regions = saliency_regions(saliency, list(DYSFLUENCY_CLASSES), 3.0)
    wordrep = [r for r in regions if r["type"] == "wordrep"]
    assert len(wordrep) == 1
    r = wordrep[0]
    assert r["start"] == pytest.approx(2.0, abs=0.05)
    assert r["end"] == pytest.approx(2.8, abs=0.05)
    assert r["confidence"] == pytest.approx(0.95, abs=1e-4)


def test_saliency_regions_empty_when_no_peaks():
    rng = np.random.default_rng(1)
    saliency = rng.uniform(0.1, 0.3, size=(500, 5)).astype(float)
    regions = saliency_regions(saliency, list(DYSFLUENCY_CLASSES), 3.0)
    assert regions == []


def test_saliency_regions_short_spans_dropped():
    rng = np.random.default_rng(2)
    saliency = rng.uniform(0.2, 0.4, size=(500, 5)).astype(float)
    saliency[100:104, 0] = 0.95
    regions = saliency_regions(saliency, list(DYSFLUENCY_CLASSES), 3.0)
    assert regions == []


def test_saliency_regions_overlaps_deduped():
    rng = np.random.default_rng(3)
    saliency = rng.uniform(0.2, 0.4, size=(500, 5)).astype(float)
    saliency[100:200, 0] = 0.8
    saliency[120:180, 1] = 0.9
    regions = saliency_regions(saliency, list(DYSFLUENCY_CLASSES), 3.0)
    assert len(regions) == 1
    assert regions[0]["type"] == "block"


def test_saliency_regions_clamped_to_duration():
    rng = np.random.default_rng(4)
    saliency = rng.uniform(0.2, 0.4, size=(500, 5)).astype(float)
    saliency[450:480, 4] = 0.9
    regions = saliency_regions(saliency, list(DYSFLUENCY_CLASSES), 3.0)
    assert all(r["end"] <= 3.0 for r in regions)


def test_saliency_fallback_returns_well_formed(monkeypatch):
    """Test that localize_audio_bytes falls back to saliency when localizer fails."""
    import model.registry as reg

    class FakeLocalizer:
        def predict(self, spec):
            raise RuntimeError("no weights")

    class FakeMultiTask:
        def saliency(self, audio):
            arr = np.zeros((1, 500, 5), dtype=float)
            arr[0, 50:90, 2] = 0.99
            return arr

    monkeypatch.setattr(reg, "LocalizerRunner", lambda *a, **kw: FakeLocalizer())
    monkeypatch.setattr(reg, "MultiTaskRunner", lambda: FakeMultiTask())

    audio_bytes = _make_silence_wav()
    result = reg.localize_audio_bytes(audio_bytes)
    assert result["source"] == "saliency"
    assert len(result["regions"]) == 1
    assert result["regions"][0]["type"] == "soundrep"
    assert "error" not in result


def test_saliency_fallback_handles_failure(monkeypatch):
    """Test that localize_audio_bytes returns error when both localizer and saliency fail."""
    import model.registry as reg

    class BrokenLocalizer:
        def predict(self, spec):
            raise RuntimeError("no weights")

    class BrokenMultiTask:
        def saliency(self, audio):
            raise RuntimeError("boom")

    monkeypatch.setattr(reg, "LocalizerRunner", lambda *a, **kw: BrokenLocalizer())
    monkeypatch.setattr(reg, "MultiTaskRunner", lambda: BrokenMultiTask())

    audio_bytes = _make_silence_wav()
    result = reg.localize_audio_bytes(audio_bytes)
    assert result["regions"] == []
    assert "boom" in result["error"]


def _make_silence_wav() -> bytes:
    """Create a 3-second silence WAV in memory."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * (16000 * 3))
    return buf.getvalue()
