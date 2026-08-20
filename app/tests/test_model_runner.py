"""Tests for ModelRunner classification using the multitask classifier."""

import numpy as np
import pytest

from app.core.model_runner import ModelRunner


def test_classify_uses_multitask_classifier(monkeypatch):
    captured = {}

    class _FakeMultiTask:
        def __init__(self):
            captured["instance"] = self

        def analyze(self, audio, threshold=None):
            captured["audio"] = audio
            return {
                "prolongation": {"label": 1, "confidence": 0.87},
                "block": {"label": 0, "confidence": 0.91},
                "soundrep": {"label": 1, "confidence": 0.52},
                "wordrep": {"label": 0, "confidence": 0.66},
                "interjection": {"label": 0, "confidence": 0.99},
                "summary": {
                    "detected": ["prolongation", "soundrep"],
                    "primary": "prolongation",
                },
            }

    monkeypatch.setattr("model.registry.MultiTaskClassifier", _FakeMultiTask)
    runner = ModelRunner()
    audio = np.zeros(16000, dtype=np.float32)
    results = runner._classify(audio)

    assert isinstance(captured["instance"], _FakeMultiTask)
    # Audio is loaded from WAV bytes, so check values match (not identity)
    np.testing.assert_array_almost_equal(captured["audio"], audio)
    assert results == {
        "prolongation": (True, 0.87),
        "block": (False, 0.91),
        "soundrep": (True, 0.52),
        "wordrep": (False, 0.66),
        "interjection": (False, 0.99),
    }


def test_classify_propagates_model_errors(monkeypatch):
    class _FailingMultiTask:
        def analyze(self, audio, threshold=None):
            raise FileNotFoundError("weights not found")

    monkeypatch.setattr("model.registry.MultiTaskClassifier", _FailingMultiTask)
    runner = ModelRunner()

    with pytest.raises(FileNotFoundError):
        runner._classify(np.zeros(16000, dtype=np.float32))


def test_analyze_includes_combined(monkeypatch):
    class _FakeMultiTask:
        def __init__(self):
            self._thresholds = {}

        def analyze(self, audio, threshold=None):
            return {"block": {"label": 1, "confidence": 0.9}, "summary": {"detected": ["block"], "primary": "block"}}

        def saliency(self, audio):
            sal = np.zeros((1, 500, 5))
            sal[:, :, 1] = 0.9  # block active every frame
            return sal

    class _FakeLocalizer:
        def predict(self, spec, sr=16000, threshold=0.3, max_length_seconds=3.0):
            return [(0.0, 0.5, 0.9)]

    class _FakeTranscriber:
        def transcribe(self, audio, localizations=None, language="english"):
            return {"text": "hi", "words": [], "duration_sec": 0.5}

    monkeypatch.setattr("model.registry.MultiTaskClassifier", _FakeMultiTask)
    monkeypatch.setattr("model.registry.MultiTaskRunner", _FakeMultiTask)
    monkeypatch.setattr("model.registry.LocalizerRunner", lambda *a, **k: _FakeLocalizer())
    runner = ModelRunner()
    runner.transcriber = _FakeTranscriber()

    results = runner.analyze(np.zeros(16000, dtype=np.float32))
    assert set(results.keys()) == {"classifications", "localizations", "transcription", "combined"}
    assert results["combined"]["total_stutters"] == 1
    assert results["combined"]["regions"][0]["primary_type"] == "block"


def test_analyze_combined_degrades_on_saliency_error(monkeypatch):
    class _FailingMultiTask:
        def __init__(self):
            self._thresholds = {}

        def analyze(self, audio, threshold=None):
            return {"block": {"label": 0, "confidence": 0.5}, "summary": {"detected": [], "primary": None}}

        def saliency(self, audio):
            raise RuntimeError("no model")

    class _FakeLocalizer:
        def predict(self, spec, sr=16000, threshold=0.3, max_length_seconds=3.0):
            return [(0.0, 0.5, 0.9)]

    class _FakeTranscriber:
        def transcribe(self, audio, localizations=None, language="english"):
            return {"text": "hi", "words": [], "duration_sec": 0.5}

    monkeypatch.setattr("model.registry.MultiTaskClassifier", _FailingMultiTask)
    monkeypatch.setattr("model.registry.MultiTaskRunner", _FailingMultiTask)
    monkeypatch.setattr("model.registry.LocalizerRunner", lambda *a, **k: _FakeLocalizer())
    runner = ModelRunner()
    runner.transcriber = _FakeTranscriber()

    results = runner.analyze(np.zeros(16000, dtype=np.float32))
    assert "error" in results["combined"]
