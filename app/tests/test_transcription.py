import numpy as np
from app.core.transcription import AudioTranscriber
from app.ui.transcription_panel import TranscriptionPanel


def test_audio_transcriber_fallback():
    transcriber = AudioTranscriber()
    audio = np.zeros(16000, dtype=np.float32)
    res = transcriber.transcribe(audio, sample_rate=16000)
    assert "text" in res
    assert "words" in res
    assert len(res["words"]) > 0


def test_audio_transcriber_stutter_alignment():
    transcriber = AudioTranscriber()
    audio = np.zeros(32000, dtype=np.float32)
    localizations = [(0.0, 2.0, 0.9)]
    res = transcriber.transcribe(audio, sample_rate=16000, localizations=localizations)
    stutter_words = [w for w in res["words"] if w["stutter"]]
    assert len(stutter_words) > 0


def test_transcription_panel_ui(qapp):
    panel = TranscriptionPanel()
    assert panel._text_edit.toPlainText() == ""
    audio = np.zeros(16000, dtype=np.float32)
    panel.set_audio(audio)
    assert panel._text_edit.toPlainText() != ""
    assert panel._table.rowCount() > 0

    panel.clear()
    assert panel._text_edit.toPlainText() == ""
    assert panel._table.rowCount() == 0
