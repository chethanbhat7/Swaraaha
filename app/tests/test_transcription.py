import numpy as np
import pytest

from app.core.transcription import WHISPER_MODELS, AudioTranscriber
from app.ui.transcription_panel import TranscriptionPanel


@pytest.fixture
def no_network(monkeypatch):
    """Make every pipeline load fail so transcription falls back to the aligner."""

    def _fail(language):
        raise RuntimeError("no network")

    monkeypatch.setattr("app.core.transcription.get_pipeline", _fail)


def test_audio_transcriber_fallback(no_network):
    transcriber = AudioTranscriber()
    audio = np.zeros(16000, dtype=np.float32)
    res = transcriber.transcribe(audio, sample_rate=16000)
    assert "text" in res
    assert "words" in res
    assert len(res["words"]) > 0


def test_audio_transcriber_stutter_alignment(no_network):
    transcriber = AudioTranscriber()
    audio = np.zeros(32000, dtype=np.float32)
    localizations = [(0.0, 2.0, 0.9)]
    res = transcriber.transcribe(audio, sample_rate=16000, localizations=localizations)
    stutter_words = [w for w in res["words"] if w["stutter"]]
    assert len(stutter_words) > 0


def test_transcription_panel_ui(qapp, no_network):
    panel = TranscriptionPanel()
    assert panel._text_edit.toPlainText() == ""
    audio = np.zeros(16000, dtype=np.float32)
    panel.set_audio(audio)
    assert panel._text_edit.toPlainText() != ""
    assert panel._table.rowCount() > 0

    panel.clear()
    assert panel._text_edit.toPlainText() == ""
    assert panel._table.rowCount() == 0


def test_whisper_model_mapping():
    assert WHISPER_MODELS["english"] == "openai/whisper-tiny"
    assert WHISPER_MODELS["kannada"] == "vasista22/whisper-kannada-tiny"
    assert WHISPER_MODELS["hindi"] == "collabora/whisper-tiny-hindi"


def test_transcribe_dedups_whisper_repeats(monkeypatch):
    captured = {}

    class FakePipe:
        def __call__(self, audio, return_timestamps="word"):
            captured["language"] = "kannada"
            return {
                "text": "ಹಲೋ ಹಲೋ ಜಗತ್ತು",
                "chunks": [
                    {"text": "ಹಲೋ", "timestamp": (0.0, 0.4), "confidence": 0.9},
                    {"text": "ಹಲೋ", "timestamp": (0.4, 0.8), "confidence": 0.9},
                    {"text": "ಜಗತ್ತು", "timestamp": (0.8, 1.3), "confidence": 0.8},
                ],
            }

    monkeypatch.setattr("app.core.transcription.get_pipeline", lambda language: FakePipe())
    transcriber = AudioTranscriber()
    res = transcriber.transcribe(np.zeros(16000, dtype=np.float32), language="kannada")

    assert captured["language"] == "kannada"
    assert res["text"] == "ಹಲೋ ಜಗತ್ತು"
    assert len(res["words"]) == 2
    assert res["words"][0]["word"] == "ಹಲೋ"
    assert res["words"][0]["start_sec"] == 0.0
