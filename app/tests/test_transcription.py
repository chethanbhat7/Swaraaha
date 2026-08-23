"""Tests for transcription panel UI and model constants."""

import numpy as np
import pytest


def test_whisper_model_mapping():
    from model.transcription import WHISPER_MODELS

    assert WHISPER_MODELS["english"] == "openai/whisper-tiny"
    assert WHISPER_MODELS["kannada"] == "vasista22/whisper-kannada-tiny"
    assert WHISPER_MODELS["hindi"] == "collabora/whisper-tiny-hindi"


def test_transcription_panel_ui(qapp, monkeypatch):
    from app.ui.transcription_panel import TranscriptionPanel

    monkeypatch.setattr(
        "model.transcribe",
        lambda audio, language="english", **kw: {
            "text": "hello", "words": [], "duration_sec": 0.0,
        },
    )

    panel = TranscriptionPanel()
    assert panel._text_edit.toPlainText() == ""
    audio = np.zeros(16000, dtype=np.float32)
    panel.set_audio(audio)
    assert panel._text_edit.toPlainText() == "hello"

    panel.clear()
    assert panel._text_edit.toPlainText() == ""
