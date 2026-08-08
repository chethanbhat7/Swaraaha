import numpy as np

from app.core.audio_handler import AudioHandler


def _recording_with_silence(silence_secs=0.5, speech_secs=1.0):
    rng = np.random.default_rng(0)
    silence = np.zeros(int(16000 * silence_secs), dtype=np.float32)
    speech = (0.5 * rng.standard_normal(int(16000 * speech_secs))).astype(np.float32)
    return np.concatenate([silence, speech, silence])


def test_trim_audio_removes_leading_silence():
    handler = AudioHandler(sample_rate=16000)
    audio = _recording_with_silence()
    trimmed = handler.trim_audio(audio)
    assert len(trimmed) < len(audio)
    first_speech_sample = int(np.argmax(np.abs(trimmed) > 0.01))
    assert first_speech_sample < 4096
    assert np.max(np.abs(trimmed)) > 0.1


def test_trim_audio_keeps_speech_content():
    handler = AudioHandler(sample_rate=16000)
    audio = _recording_with_silence()
    trimmed = handler.trim_audio(audio)
    speech_start = int(0.5 * 16000)
    speech_end = int(1.5 * 16000)
    speech = audio[speech_start:speech_end]
    assert len(trimmed) >= len(speech) - 2048
    assert len(trimmed) <= len(speech) + 2048


def test_trim_audio_removes_trailing_silence():
    handler = AudioHandler(sample_rate=16000)
    audio = _recording_with_silence()
    trimmed = handler.trim_audio(audio)
    assert len(trimmed) < len(audio)
    last_speech = len(trimmed) - 1 - int(np.argmax(np.abs(trimmed[::-1]) > 0.01))
    trailing = len(trimmed) - 1 - last_speech
    assert trailing < 4096
    assert np.max(np.abs(trimmed)) > 0.1


def test_trim_audio_all_silence_keeps_original():
    handler = AudioHandler(sample_rate=16000)
    audio = np.zeros(16000, dtype=np.float32)
    trimmed = handler.trim_audio(audio)
    assert len(trimmed) == len(audio)
    assert np.array_equal(trimmed, audio)


def test_trim_audio_empty_input():
    handler = AudioHandler(sample_rate=16000)
    audio = np.array([], dtype=np.float32)
    assert handler.trim_audio(audio).shape == (0,)
