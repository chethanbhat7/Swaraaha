import numpy as np
import pytest

from model.registry import Localizer, _align_words_syllables


class _FakeCNN:
    def predict(self, spectrogram, sr=16000, hop_length=512, threshold=0.5):
        return [(0.0, 0.5, 0.9)]


class _FakeW2V2:
    def predict(self, audio, sr=16000, threshold=0.5, max_length_seconds=10.0):
        return [(0.5, 1.0, 0.8)]


@pytest.fixture
def audio():
    return np.random.rand(16000).astype(np.float32)


def test_align_words_syllables_uses_fallback(monkeypatch, audio):
    class _Raises:
        def __init__(self):
            raise RuntimeError("no CTC model")

    monkeypatch.setattr("model.localization.ctc_alignment.CTCTimeAligner", _Raises)
    words, syllables = _align_words_syllables(audio, "hello world", "en", sr=16000)
    assert len(words) == 2
    assert words[0]["word"] == "hello"
    assert "start" in words[0]
    assert syllables and syllables[0]["syllable"]


def test_align_words_syllables_empty_text(audio):
    words, syllables = _align_words_syllables(audio, "", "en", sr=16000)
    assert words == []
    assert syllables == []


def test_localizer_analyze_cnn(audio):
    loc = Localizer("cnn")
    loc._models = {"cnn": _FakeCNN()}
    result = loc.analyze(audio)
    assert result["regions"] == [{"start": 0.0, "end": 0.5, "confidence": 0.9}]
    assert "words" not in result


def test_localizer_analyze_cnn_with_text(monkeypatch, audio):
    class _Raises:
        def __init__(self):
            raise RuntimeError("no CTC model")

    monkeypatch.setattr("model.localization.ctc_alignment.CTCTimeAligner", _Raises)
    loc = Localizer("cnn")
    loc._models = {"cnn": _FakeCNN()}
    result = loc.analyze(audio, text="hello world", language="en")
    assert result["regions"]
    assert len(result["words"]) == 2
    assert len(result["syllables"]) >= 2


def test_localizer_analyze_wav2vec2(audio):
    loc = Localizer("wav2vec2")
    loc._models = {"wav2vec2": _FakeW2V2()}
    result = loc.analyze(audio)
    assert result["regions"] == [{"start": 0.5, "end": 1.0, "confidence": 0.8}]


def test_localizer_analyze_raises_when_not_loaded():
    loc = Localizer("cnn")
    with pytest.raises(FileNotFoundError):
        loc.analyze(np.random.rand(16000).astype(np.float32))
