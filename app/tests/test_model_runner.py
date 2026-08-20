"""Tests for ModelRunner using the model.init/analyze API."""

import numpy as np
import pytest

from app.core.model_runner import ModelRunner


def test_analyze_returns_all_sections(monkeypatch):
    """analyze() returns classifications, localizations, transcription, combined."""

    class _FakeClassifier:
        is_loaded = True

        def analyze(self, audio, threshold=None):
            return {
                "prolongation": {"label": 1, "confidence": 0.87},
                "block": {"label": 0, "confidence": 0.91},
                "summary": {"detected": ["prolongation"], "primary": "prolongation"},
            }

    class _FakeLocalizer:
        is_loaded = True

        def analyze(self, audio, threshold=0.3, text=None, language="en"):
            return {
                "regions": [{"start": 0.0, "end": 0.5, "confidence": 0.9}],
            }

    import model as _model
    monkeypatch.setattr(_model, "_classifier", _FakeClassifier())
    monkeypatch.setattr(_model, "_localizer", _FakeLocalizer())
    monkeypatch.setattr(_model, "_init_done", True)

    runner = ModelRunner()
    results = runner.analyze(np.zeros(16000, dtype=np.float32))

    assert "classifications" in results
    assert "localizations" in results
    assert "transcription" in results
    assert results["classifications"]["prolongation"] == (True, 0.87)
    assert results["classifications"]["block"] == (False, 0.91)
    assert len(results["localizations"]) == 1
    assert results["localizations"][0] == (0.0, 0.5, 0.9)


def test_analyze_gracefully_handles_classifier_error(monkeypatch):

    class _FailingClassifier:
        is_loaded = True

        def analyze(self, audio, threshold=None):
            raise FileNotFoundError("weights not found")

    class _FakeLocalizer:
        is_loaded = True

        def analyze(self, audio, threshold=0.3, text=None, language="en"):
            return {"regions": []}

    import model as _model
    monkeypatch.setattr(_model, "_classifier", _FailingClassifier())
    monkeypatch.setattr(_model, "_localizer", _FakeLocalizer())
    monkeypatch.setattr(_model, "_init_done", True)

    runner = ModelRunner()
    results = runner.analyze(np.zeros(16000, dtype=np.float32))
    assert isinstance(results["classifications"], dict)


def test_analyze_combined_degrades_on_saliency_error(monkeypatch):

    class _FakeClassifier:
        is_loaded = True

        def analyze(self, audio, threshold=None):
            return {"block": {"label": 1}, "summary": {"detected": ["block"], "primary": "block"}}

        def saliency(self, audio):
            raise RuntimeError("no model")

    class _FakeLocalizer:
        is_loaded = True

        def analyze(self, audio, threshold=0.3, text=None, language="en"):
            return {"regions": [{"start": 0.0, "end": 0.5, "confidence": 0.9}]}

    import model as _model
    monkeypatch.setattr(_model, "_classifier", _FakeClassifier())
    monkeypatch.setattr(_model, "_localizer", _FakeLocalizer())
    monkeypatch.setattr(_model, "_init_done", True)

    runner = ModelRunner()
    results = runner.analyze(np.zeros(16000, dtype=np.float32))
    assert "error" in results["combined"]
