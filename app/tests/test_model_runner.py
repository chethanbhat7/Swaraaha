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
    assert captured["audio"] is audio
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
