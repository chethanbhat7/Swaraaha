import numpy as np
import pytest

from model.transcription import Transcriber, _is_repeated_fragment


def _fake_pipe(monkeypatch):
    from model import transcription as mod

    class _Pipe:
        def __call__(self, audio, return_timestamps="word"):
            return {
                "text": "hello hello world",
                "chunks": [
                    {"text": "hello", "timestamp": (0.0, 0.4), "confidence": 0.95},
                    {"text": "hello", "timestamp": (0.4, 0.8), "confidence": 0.9},
                    {"text": "world", "timestamp": (0.8, 1.2), "confidence": 0.9},
                ],
            }

    monkeypatch.setattr(mod, "get_pipeline", lambda language="english": _Pipe())


def test_is_repeated_fragment():
    assert _is_repeated_fragment("s-s")
    assert _is_repeated_fragment("ba-ba-ba")
    assert _is_repeated_fragment("sss")
    assert not _is_repeated_fragment("hello")


def test_transcribe_dedups_whisper_repeats(monkeypatch):
    _fake_pipe(monkeypatch)
    tr = Transcriber()
    audio = np.zeros(16000, dtype=np.float32)
    result = tr.transcribe(audio)
    assert result["text"] == "hello world"
    assert len(result["words"]) == 2
    assert result["words"][0]["word"] == "hello"
    assert result["words"][0]["start_sec"] == 0.0


def test_transcribe_overlays_localizations(monkeypatch):
    _fake_pipe(monkeypatch)
    tr = Transcriber()
    audio = np.zeros(16000, dtype=np.float32)
    result = tr.transcribe(audio, localizations=[(0.0, 1.2, 0.9)])
    flagged = [w for w in result["words"] if w["stutter"]]
    assert flagged and all(w["stutter_type"] == "dysfluency" for w in flagged)


def test_transcribe_empty_audio():
    tr = Transcriber()
    result = tr.transcribe(np.array([], dtype=np.float32))
    assert result == {"text": "", "words": [], "duration_sec": 0.0}


def test_transcribe_accepts_audio_bytes(monkeypatch):
    import io

    import soundfile as sf

    _fake_pipe(monkeypatch)
    tr = Transcriber()
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    result = tr.transcribe(buf.getvalue())
    assert result["text"] == "hello world"
