"""
Tests for model/localization/wav2vec2_dataset.py.
"""

import wave

import numpy as np

from model.localization.wav2vec2_dataset import Wav2Vec2LocalizationDataset


def _write_wav_with_leading_silence(path, sr=16000):
    """0.5s silence followed by 0.5s of 440Hz sine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(sr // 2, dtype=np.float32)
    t = np.arange(sr // 2) / sr
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    samples = np.concatenate([silence, sine])
    pcm = (samples * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def _write_label(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("start_sec,end_sec,dysfluency_type\n0.500,1.000,block\n")


def _frame_energy(audio, sr=16000, hop=320):
    """Mean abs energy per 320-sample frame."""
    n = len(audio)
    n_frames = n // hop
    frames = audio[: n_frames * hop].reshape(n_frames, hop)
    return np.abs(frames).mean(axis=1)


def test_wav2vec2_frame_labels_align_with_leading_silence(tmp_path):
    """Silence-trim must not shift frame labels: positive frames must coincide
    with the voiced region of the returned audio."""
    data = tmp_path / "data"
    _write_wav_with_leading_silence(data / "audio" / "M_0002.wav")
    _write_label(data / "labels" / "M_0002.csv")
    with open(data / "sources.csv", "w") as f:
        f.write("clip_id,source\nM_0002,uclass\n")

    ds = Wav2Vec2LocalizationDataset(data_dir=str(data), max_length_seconds=1.0)
    audio, mask = ds[0]

    energy = _frame_energy(audio)
    voiced = np.where(energy > energy.max() * 0.1)[0]
    pos = np.where(mask > 0)[0]

    assert len(pos) > 0
    assert len(voiced) > 0
    assert np.array_equal(pos, voiced), f"pos {pos} vs voiced {voiced}"


def test_wav2vec2_dataset_skips_header_only_wav(tmp_path):
    """Header-only WAV stubs (44-byte, no audio data) must be excluded from the
    sample list; otherwise training would silently run on empty silence."""
    data = tmp_path / "data"
    (data / "audio").mkdir(parents=True)
    (data / "labels").mkdir(parents=True)
    (data / "audio" / "empty.wav").write_bytes(b"\x00" * 44)
    _write_label(data / "labels" / "empty.csv")
    _write_wav_with_leading_silence(data / "audio" / "valid.wav")
    _write_label(data / "labels" / "valid.csv")

    ds = Wav2Vec2LocalizationDataset(data_dir=str(data), max_length_seconds=1.0)
    assert len(ds) == 1
    assert ds.get_sample_info(0)["clip_id"] == "valid"


def test_wav2vec2_dataset_cache_reused_when_unchanged(tmp_path, monkeypatch):
    """Unchanged labels + audio + config must be served from the pickle cache,
    so the second access never reloads the waveform."""
    data = tmp_path / "data"
    _write_wav_with_leading_silence(data / "audio" / "M_0002.wav")
    _write_label(data / "labels" / "M_0002.csv")
    cache = tmp_path / "cache"

    import model.data.preprocessing as preprocessing

    orig_load = preprocessing.load_audio
    calls = {"n": 0}

    def counting_load(*a, **k):
        calls["n"] += 1
        return orig_load(*a, **k)

    monkeypatch.setattr(preprocessing, "load_audio", counting_load)

    ds = Wav2Vec2LocalizationDataset(data_dir=str(data), max_length_seconds=1.0,
                                     cache_dir=str(cache))
    audio, mask = ds[0]
    assert calls["n"] == 1

    ds2 = Wav2Vec2LocalizationDataset(data_dir=str(data), max_length_seconds=1.0,
                                      cache_dir=str(cache))
    audio2, mask2 = ds2[0]
    assert calls["n"] == 1  # second access must hit the cache
    assert np.array_equal(audio, audio2)
    assert np.array_equal(mask, mask2)


def test_wav2vec2_dataset_cache_invalidates_on_config_change(tmp_path):
    """A different max length changes the output shape; the config signature
    must invalidate stale pickles."""
    data = tmp_path / "data"
    _write_wav_with_leading_silence(data / "audio" / "M_0002.wav")
    _write_label(data / "labels" / "M_0002.csv")
    cache = tmp_path / "cache"

    ds = Wav2Vec2LocalizationDataset(data_dir=str(data), max_length_seconds=1.0,
                                     cache_dir=str(cache))
    audio, _ = ds[0]
    assert audio.shape == (16000,)

    ds2 = Wav2Vec2LocalizationDataset(data_dir=str(data), max_length_seconds=0.5,
                                      cache_dir=str(cache))
    audio2, _ = ds2[0]
    assert audio2.shape == (8000,)  # stale cache would return 16000-length audio
