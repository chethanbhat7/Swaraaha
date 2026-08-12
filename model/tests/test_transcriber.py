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


def _capture_pipe(monkeypatch):
    """Fake Whisper pipeline that records the audio array it receives."""
    from model import transcription as mod

    seen = []

    class _Pipe:
        def __call__(self, audio, return_timestamps="word"):
            seen.append(np.array(audio))
            return {"text": "hello world", "chunks": []}

    monkeypatch.setattr(mod, "get_pipeline", lambda language="english": _Pipe())
    return seen


def _write_wav_44k(path):
    import soundfile as sf

    sf.write(path, np.zeros(44100, dtype=np.float32), 44100, format="WAV")


def test_transcribe_path_44k_resampled_to_16k_for_whisper(monkeypatch, tmp_path):
    """Whisper requires 16 kHz audio: a 44.1 kHz path input must be resampled
    before hitting the model and duration computed at the real rate."""
    seen = _capture_pipe(monkeypatch)
    wav = tmp_path / "in_44k.wav"
    _write_wav_44k(wav)

    result = Transcriber().transcribe(str(wav), sample_rate=44100)
    assert len(seen[0]) == 16000
    assert result["duration_sec"] == 1.0


def test_transcribe_bytes_44k_resampled_to_16k_for_whisper(monkeypatch):
    import io

    import soundfile as sf

    seen = _capture_pipe(monkeypatch)
    buf = io.BytesIO()
    sf.write(buf, np.zeros(44100, dtype=np.float32), 44100, format="WAV")

    result = Transcriber().transcribe(buf.getvalue(), sample_rate=44100)
    assert len(seen[0]) == 16000
    assert result["duration_sec"] == 1.0


def test_transcribe_array_44k_stays_16k(monkeypatch):
    """ndarray input is already resampled to 16 kHz by load_audio_input, so
    transcribe must not double-resample it or misreport duration."""
    seen = _capture_pipe(monkeypatch)
    audio = np.zeros(44100, dtype=np.float32)

    result = Transcriber().transcribe(audio, sample_rate=44100)
    assert len(seen[0]) == 16000
    assert result["duration_sec"] == 1.0


def test_get_pipeline_warns_when_generation_config_fails(monkeypatch, capsys):
    """Failing to set the Whisper generation config must log a warning instead
    of being silently swallowed (otherwise language/timestamp prompts silently
    default, which is hard to debug)."""
    from model.transcription import _configure_generation_config

    class _GenConfig:
        def __init__(self):
            self.no_timestamps_token_id = None

        @property
        def forced_decoder_ids(self):
            return None

        @forced_decoder_ids.setter
        def forced_decoder_ids(self, value):
            raise RuntimeError("decoder prompt ids unavailable")

    class _Tok:
        def get_decoder_prompt_ids(self, language=None, task=None):
            return [(1, language), (2, task)]

        def convert_tokens_to_ids(self, token):
            return 42

    pipe = type(
        "P",
        (),
        {
            "model": type("M", (), {"generation_config": _GenConfig()})(),
            "tokenizer": _Tok(),
        },
    )()

    _configure_generation_config(pipe, "en")
    captured = capsys.readouterr().out
    assert "WARNING" in captured
    assert "decoder prompt ids unavailable" in captured


def test_configure_generation_config_sets_prompts():
    from model.transcription import _configure_generation_config

    class _GenConfig:
        def __init__(self):
            self.forced_decoder_ids = None
            self.no_timestamps_token_id = None

    class _Tok:
        def get_decoder_prompt_ids(self, language=None, task=None):
            return [(1, language), (2, task)]

        def convert_tokens_to_ids(self, token):
            return 42

    pipe = type(
        "P",
        (),
        {
            "model": type("M", (), {"generation_config": _GenConfig()})(),
            "tokenizer": _Tok(),
        },
    )()

    _configure_generation_config(pipe, "en")
    assert pipe.model.generation_config.forced_decoder_ids == [
        (1, "en"),
        (2, "transcribe"),
    ]
    assert pipe.model.generation_config.no_timestamps_token_id == 42
