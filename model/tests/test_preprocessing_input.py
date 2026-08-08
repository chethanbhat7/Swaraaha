import numpy as np
import pytest

from model.data.preprocessing import load_audio_input


def test_load_audio_input_from_array():
    audio = (np.random.rand(16000) * 2 - 1).astype(np.float32)
    out = load_audio_input(audio, sr=16000)
    assert out.shape == (16000,)
    assert out.dtype == np.float32
    assert abs(out).max() <= 1.0


def test_load_audio_input_unsupported_type():
    with pytest.raises(TypeError):
        load_audio_input(42, sr=16000)


def test_load_audio_input_array_requires_1d():
    with pytest.raises(ValueError):
        load_audio_input(np.zeros((2, 100), dtype=np.float32), sr=16000)
