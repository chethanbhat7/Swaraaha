"""Tests for model.transcribe() — the unified transcription API."""

import numpy as np


def test_transcribe_empty_audio_returns_empty(monkeypatch):
    from model import transcribe as model_transcribe
    import model as _model

    class FakeTranscriber:
        def transcribe(self, audio, language="english", **kw):
            return {"text": "", "words": [], "duration_sec": 0.0}

    monkeypatch.setattr(_model, "_transcriber", FakeTranscriber())
    monkeypatch.setattr(_model, "_init_done", True)

    result = model_transcribe(np.zeros(1600, dtype=np.float32))
    assert result["text"] == ""
    assert result["words"] == []
    assert result["duration_sec"] == 0.0


def test_transcribe_returns_error_when_no_transcriber(monkeypatch):
    from model import transcribe as model_transcribe
    import model as _model

    monkeypatch.setattr(_model, "_transcriber", None)
    monkeypatch.setattr(_model, "_init_done", True)

    result = model_transcribe(np.zeros(1600, dtype=np.float32))
    assert "error" in result


def test_transcribe_delegates_to_transcriber(monkeypatch):
    from model import transcribe as model_transcribe
    import model as _model

    captured = {}

    class FakeTranscriber:
        def transcribe(self, audio, language="english", **kw):
            captured["audio"] = audio
            captured["language"] = language
            return {"text": "hello", "words": [], "duration_sec": 0.5}

    monkeypatch.setattr(_model, "_transcriber", FakeTranscriber())
    monkeypatch.setattr(_model, "_init_done", True)

    audio = np.ones(1600, dtype=np.float32) * 0.5
    result = model_transcribe(audio, language="kannada")
    assert result["text"] == "hello"
    assert captured["language"] == "kannada"
    assert np.array_equal(captured["audio"], audio)
