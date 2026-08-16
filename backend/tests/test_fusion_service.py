"""Tests for the backend fusion service (combiner + saliency)."""

import io

import numpy as np
import pytest
import soundfile as sf

from backend.services import fusion


def _tiny_wav(duration_s=0.5) -> bytes:
    buf = io.BytesIO()
    sr = 16000
    sf.write(buf, np.zeros(int(sr * duration_s), dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


class _FakeClassifier:
    def __init__(self):
        self._thresholds = {}

    def saliency(self, audio):
        sal = np.zeros((1, 500, 5))
        sal[:, :, 1] = 0.9  # block active every frame
        return sal


def test_combine_audio_bytes_returns_fused_regions(monkeypatch):
    monkeypatch.setattr(fusion.classifier, "get_model", lambda: _FakeClassifier())
    regions = [{"start": 0.0, "end": 0.5, "confidence": 0.9}]
    result = fusion.combine_audio_bytes(_tiny_wav(), regions)
    assert "error" not in result
    assert result["audio_duration"] == pytest.approx(0.5)
    assert result["total_stutters"] == 1
    region = result["regions"][0]
    assert region["primary_type"] == "block"
    assert set(region["classes"].keys()) == {
        "prolongation", "block", "soundrep", "wordrep", "interjection",
    }


def test_combine_audio_bytes_degrades_on_error(monkeypatch):
    class _BrokenClassifier:
        _thresholds = {}

        def saliency(self, audio):
            raise RuntimeError("no model")

    monkeypatch.setattr(fusion.classifier, "get_model", lambda: _BrokenClassifier())
    result = fusion.combine_audio_bytes(_tiny_wav(), [])
    assert result == {"error": "no model"}
