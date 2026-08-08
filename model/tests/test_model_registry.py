import numpy as np
import pytest

from model import ModelRegistry


def test_run_all_composes(monkeypatch):
    reg = ModelRegistry()

    monkeypatch.setattr(
        reg.classifier, "analyze",
        lambda audio, threshold=None: {"prolongation": {"label": 0}, "summary": {"detected": []}},
    )
    monkeypatch.setattr(
        reg.localizer, "analyze",
        lambda audio, text=None, language="en", threshold=0.3, max_length_seconds=10.0: {
            "regions": [{"start": 0.0, "end": 0.5, "confidence": 0.9}]
        },
    )
    monkeypatch.setattr(
        reg.transcriber, "transcribe",
        lambda audio, language="english", localizations=None, passage_text=None, sample_rate=16000: {
            "text": "hello world", "words": [], "duration_sec": 1.0
        },
    )

    result = reg.run_all(np.zeros(16000, dtype=np.float32), text="hello world")
    assert "classification" in result
    assert result["localization"]["regions"]
    assert result["transcription"]["text"] == "hello world"


def test_run_all_language_maps_to_iso(monkeypatch):
    reg = ModelRegistry()
    seen = {}

    def fake_analyze(audio, text=None, language="en", threshold=0.3, max_length_seconds=10.0):
        seen["language"] = language
        return {"regions": []}

    monkeypatch.setattr(reg.localizer, "analyze", fake_analyze)
    monkeypatch.setattr(
        reg.classifier, "analyze",
        lambda audio, threshold=None: {"summary": {"detected": []}},
    )
    monkeypatch.setattr(
        reg.transcriber, "transcribe",
        lambda audio, language="english", localizations=None, passage_text=None, sample_rate=16000: {
            "text": "", "words": [], "duration_sec": 0.0
        },
    )

    reg.run_all(np.zeros(16000, dtype=np.float32), language="english", text="hello")
    assert seen["language"] == "en"


def test_run_all_catches_missing_models(monkeypatch):
    reg = ModelRegistry()

    def _raise(*a, **k):
        raise FileNotFoundError("no")

    monkeypatch.setattr(reg.classifier, "analyze", _raise)
    monkeypatch.setattr(reg.localizer, "analyze", _raise)
    monkeypatch.setattr(
        reg.transcriber, "transcribe",
        lambda audio, **kwargs: {"text": "", "words": [], "duration_sec": 0.0},
    )
    result = reg.run_all(np.zeros(16000, dtype=np.float32))
    assert result["classification"]["error"]
    assert result["localization"]["error"]
