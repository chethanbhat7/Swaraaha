"""
Tests for source filtering and cache invalidation in model/data/dataset.py.
"""

import os
import struct
import wave

import numpy as np
import pytest

from model.data.dataset import (
    CLASS_TO_IDX,
    ClassificationDataset,
    LocalizationDataset,
    SpectrogramClassificationDataset,
)


def _write_wav(path, seconds=1.0, sr=16000, freq=440.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    samples = (np.sin(2 * np.pi * freq * np.arange(n) / sr) * 0.5).astype(np.float32)
    pcm = (samples * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return path


def _write_label(path, intervals):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("start_sec,end_sec,dysfluency_type\n")
        for start, end, dtype in intervals:
            f.write(f"{start:.3f},{end:.3f},{dtype}\n")


def _make_data_dir(root):
    """Create a data dir with one UCLASS clip and one SEP-28K clip."""
    data = root / "data"
    for clip in ["M_0001_dysfluent_000", "FluencyBank_010_0"]:
        _write_wav(data / "audio" / f"{clip}.wav")
        _write_label(data / "labels" / f"{clip}.csv", [(0.1, 0.4, "block")])
    # sources.csv: clip_id -> source
    with open(data / "sources.csv", "w") as f:
        f.write("clip_id,source\n")
        f.write("M_0001_dysfluent_000,uclass\n")
        f.write("FluencyBank_010_0,sep28k\n")
    return data


def test_localization_dataset_sources_filter_keeps_only_matching(tmp_path):
    data = _make_data_dir(tmp_path)
    full = LocalizationDataset(data_dir=str(data), max_length_seconds=1.0)
    assert {s["clip_id"] for s in full.samples} == {
        "M_0001_dysfluent_000", "FluencyBank_010_0",
    }

    only_uclass = LocalizationDataset(
        data_dir=str(data), max_length_seconds=1.0, sources=["uclass"],
    )
    assert {s["clip_id"] for s in only_uclass.samples} == {"M_0001_dysfluent_000"}

    only_sep = LocalizationDataset(
        data_dir=str(data), max_length_seconds=1.0, sources=["sep28k"],
    )
    assert {s["clip_id"] for s in only_sep.samples} == {"FluencyBank_010_0"}


def test_localization_dataset_sources_filter_without_sources_csv(tmp_path):
    """If sources.csv is absent, sources filter must not be applied."""
    data = _make_data_dir(tmp_path)
    os.remove(data / "sources.csv")

    ds = LocalizationDataset(
        data_dir=str(data), max_length_seconds=1.0, sources=["uclass"],
    )
    assert len(ds.samples) == 2


def test_classification_cache_invalidates_on_label_change(tmp_path):
    """The pickle cache is keyed by clip_id only; changing the label CSV must
    invalidate the cached label vector."""
    data = _make_data_dir(tmp_path)
    clip = "M_0001_dysfluent_000"
    cache = tmp_path / "cache"
    ds = ClassificationDataset(
        data_dir=str(data), max_length_seconds=1.0, cache_dir=str(cache),
    )
    idx = next(i for i, s in enumerate(ds.samples) if s["clip_id"] == clip)

    # First access: cache created, block=1.
    _, labels = ds[idx]
    assert labels[1] == 1  # block

    # Change the label to interjection and re-access.
    _write_label(data / "labels" / f"{clip}.csv", [(0.1, 0.4, "interjection")])
    ds2 = ClassificationDataset(
        data_dir=str(data), max_length_seconds=1.0, cache_dir=str(cache),
    )
    idx2 = next(i for i, s in enumerate(ds2.samples) if s["clip_id"] == clip)
    _, labels2 = ds2[idx2]
    assert labels2[1] == 0  # block no longer present
    assert labels2[4] == 1  # interjection present


def test_classification_cache_invalidates_on_sr_change(tmp_path):
    """The cache key must include the dataset config (sr/max_samples): a run at
    a different sample rate must NOT reuse pickles built at another sr."""
    data = _make_data_dir(tmp_path)
    clip = "M_0001_dysfluent_000"
    cache = tmp_path / "cache"
    ds = ClassificationDataset(
        data_dir=str(data), sr=16000, max_length_seconds=1.0, cache_dir=str(cache),
    )
    idx = next(i for i, s in enumerate(ds.samples) if s["clip_id"] == clip)
    audio, _ = ds[idx]
    assert audio.shape[0] == 16000

    ds2 = ClassificationDataset(
        data_dir=str(data), sr=8000, max_length_seconds=1.0, cache_dir=str(cache),
    )
    idx2 = next(i for i, s in enumerate(ds2.samples) if s["clip_id"] == clip)
    audio2, _ = ds2[idx2]
    assert audio2.shape[0] == 8000  # stale cache would return 16000-length audio


def test_classification_cache_invalidates_on_audio_change(tmp_path):
    """The cache key must include the source audio identity: replacing the wav
    (same labels, same config) must NOT reuse the old preprocessed audio."""
    data = _make_data_dir(tmp_path)
    clip = "M_0001_dysfluent_000"
    cache = tmp_path / "cache"
    ds = ClassificationDataset(
        data_dir=str(data), max_length_seconds=1.0, cache_dir=str(cache),
    )
    idx = next(i for i, s in enumerate(ds.samples) if s["clip_id"] == clip)
    ds[idx]  # populates cache

    _write_wav(data / "audio" / f"{clip}.wav", seconds=0.5, sr=16000)

    ref = ClassificationDataset(data_dir=str(data), max_length_seconds=1.0)
    ref_audio, _ = ref[idx]
    ds2 = ClassificationDataset(
        data_dir=str(data), max_length_seconds=1.0, cache_dir=str(cache),
    )
    audio2, _ = ds2[idx]
    assert np.array_equal(audio2, ref_audio)  # stale cache would return old audio


def test_classification_cache_reused_when_unchanged(tmp_path, monkeypatch):
    """Unchanged labels + audio + config must still be served from cache."""
    data = _make_data_dir(tmp_path)
    cache = tmp_path / "cache"

    import model.data.preprocessing as preprocessing

    orig_load = preprocessing.load_audio
    calls = {"n": 0}

    def counting_load(*a, **k):
        calls["n"] += 1
        return orig_load(*a, **k)

    monkeypatch.setattr(preprocessing, "load_audio", counting_load)

    ds = ClassificationDataset(
        data_dir=str(data), max_length_seconds=1.0, cache_dir=str(cache),
    )
    audio, _ = ds[0]
    assert calls["n"] == 1

    ds2 = ClassificationDataset(
        data_dir=str(data), max_length_seconds=1.0, cache_dir=str(cache),
    )
    audio2, _ = ds2[0]
    assert calls["n"] == 1  # second access must hit the cache
    assert np.array_equal(audio, audio2)


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


def test_localization_frame_labels_align_with_leading_silence(tmp_path):
    """clean_audio's silence-trim must NOT shift frame labels off the original
    timeline: the returned mask must match labels computed on the untrimmed
    spectrogram."""
    data = tmp_path / "data"
    _write_wav_with_leading_silence(data / "audio" / "M_0002.wav")
    _write_label(data / "labels" / "M_0002.csv", [(0.5, 1.0, "block")])
    with open(data / "sources.csv", "w") as f:
        f.write("clip_id,source\nM_0002,uclass\n")

    ds = LocalizationDataset(data_dir=str(data), max_length_seconds=1.0)
    spec, mask = ds[0]

    from model.data.preprocessing import (
        create_frame_labels,
        generate_mel_spectrogram,
        load_audio,
    )
    audio, _ = load_audio(str(data / "audio" / "M_0002.wav"), sr=16000)
    full_spec = generate_mel_spectrogram(
        audio, sr=16000, n_mels=ds.n_mels, hop_length=ds.hop_length,
    )
    expected = create_frame_labels(
        [(0.5, 1.0)], num_frames=full_spec.shape[1], sr=16000,
        hop_length=ds.hop_length,
    )
    assert mask.shape[0] == ds.max_frames
    assert np.array_equal(mask, expected[: ds.max_frames])


def test_classification_dataset_sources_filter_keeps_only_matching(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    dataset = ClassificationDataset(str(data_dir), sr=16000,
                                    max_length_seconds=1.0, sources=['sep28k'])
    clip_ids = {s['clip_id'] for s in dataset.samples}
    assert clip_ids == {'FluencyBank_010_0'}


def test_spectrogram_classification_dataset_shape(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    dataset = SpectrogramClassificationDataset(str(data_dir), sr=16000,
                                               n_mels=8, hop_length=256,
                                               max_length_seconds=1.0)
    assert len(dataset) == 2
    spec, label_vector = dataset[0]
    assert spec.shape == (1, 8, 63)
    assert spec.dtype == np.float32
    assert label_vector.shape == (5,)
    assert label_vector.dtype == np.uint8
    assert label_vector[CLASS_TO_IDX['block']] == 1


def test_spectrogram_classification_dataset_sources_filter(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    dataset = SpectrogramClassificationDataset(str(data_dir), sr=16000,
                                               n_mels=8, hop_length=256,
                                               max_length_seconds=1.0, sources=['uclass'])
    clip_ids = {s['clip_id'] for s in dataset.samples}
    assert clip_ids == {'M_0001_dysfluent_000'}


def test_spectrogram_classification_label_vectors_skip_audio(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    dataset = SpectrogramClassificationDataset(str(data_dir), sr=16000,
                                               n_mels=8, hop_length=256,
                                               max_length_seconds=1.0)
    vectors = dataset.label_vectors
    assert len(vectors) == 2
    assert all(v.shape == (5,) for v in vectors)
    assert vectors[0][CLASS_TO_IDX['block']] == 1


def test_spectrogram_classification_cache_reused_when_unchanged(tmp_path, monkeypatch):
    """Unchanged labels + audio + config must be served from the pickle cache,
    so the second access never recomputes the mel spectrogram."""
    data_dir = _make_data_dir(tmp_path)
    cache = tmp_path / "cache"

    import model.data.preprocessing as preprocessing

    orig_load = preprocessing.load_audio
    calls = {"n": 0}

    def counting_load(*a, **k):
        calls["n"] += 1
        return orig_load(*a, **k)

    monkeypatch.setattr(preprocessing, "load_audio", counting_load)

    ds = SpectrogramClassificationDataset(str(data_dir), sr=16000, n_mels=8,
                                          hop_length=256, max_length_seconds=1.0,
                                          cache_dir=str(cache))
    spec, labels = ds[0]
    assert calls["n"] == 1

    ds2 = SpectrogramClassificationDataset(str(data_dir), sr=16000, n_mels=8,
                                           hop_length=256, max_length_seconds=1.0,
                                           cache_dir=str(cache))
    spec2, labels2 = ds2[0]
    assert calls["n"] == 1  # second access must hit the cache
    assert np.array_equal(spec, spec2)
    assert np.array_equal(labels, labels2)


def test_spectrogram_classification_cache_invalidates_on_config_change(tmp_path):
    """A different hop_length changes the output shape; the config signature
    must invalidate stale pickles."""
    data_dir = _make_data_dir(tmp_path)
    cache = tmp_path / "cache"
    ds = SpectrogramClassificationDataset(str(data_dir), sr=16000, n_mels=8,
                                          hop_length=256, max_length_seconds=1.0,
                                          cache_dir=str(cache))
    spec, _ = ds[0]
    assert spec.shape == (1, 8, 63)

    ds2 = SpectrogramClassificationDataset(str(data_dir), sr=16000, n_mels=8,
                                           hop_length=128, max_length_seconds=1.0,
                                           cache_dir=str(cache))
    spec2, _ = ds2[0]
    assert spec2.shape == (1, 8, 126)  # stale cache would return (1, 8, 63)


def test_spectrogram_classification_cache_keyed_per_config(tmp_path, monkeypatch):
    """Cache must be keyed per dataset config: two different configs sharing a
    cache_dir must not overwrite each other's pickles, and re-accessing each
    config must hit its own cache."""
    data_dir = _make_data_dir(tmp_path)
    cache = tmp_path / "cache"

    import model.data.preprocessing as preprocessing

    orig_load = preprocessing.load_audio
    calls = {"n": 0}

    def counting_load(*a, **k):
        calls["n"] += 1
        return orig_load(*a, **k)

    monkeypatch.setattr(preprocessing, "load_audio", counting_load)

    cfg_a = dict(n_mels=8, hop_length=256)
    cfg_b = dict(n_mels=16, hop_length=512)

    ds_a1 = SpectrogramClassificationDataset(str(data_dir), sr=16000,
                                             max_length_seconds=1.0,
                                             cache_dir=str(cache), **cfg_a)
    ds_b1 = SpectrogramClassificationDataset(str(data_dir), sr=16000,
                                             max_length_seconds=1.0,
                                             cache_dir=str(cache), **cfg_b)
    calls["n"] = 0
    ds_a1[0]
    ds_b1[0]
    assert calls["n"] == 2  # one miss each, distinct cache files

    ds_a2 = SpectrogramClassificationDataset(str(data_dir), sr=16000,
                                             max_length_seconds=1.0,
                                             cache_dir=str(cache), **cfg_a)
    ds_b2 = SpectrogramClassificationDataset(str(data_dir), sr=16000,
                                             max_length_seconds=1.0,
                                             cache_dir=str(cache), **cfg_b)
    calls["n"] = 0
    ds_a2[0]
    ds_b2[0]
    assert calls["n"] == 0  # each config hits its own cached pickle


def test_spectrogram_classification_cache_keyed_per_n_fft(tmp_path):
    """n_fft is part of the config key: a run at a different window size must
    not reuse pickles built at another n_fft."""
    data_dir = _make_data_dir(tmp_path)
    cache = tmp_path / "cache"
    ds = SpectrogramClassificationDataset(str(data_dir), sr=16000, n_mels=8,
                                          hop_length=256, max_length_seconds=1.0,
                                          n_fft=1024, cache_dir=str(cache))
    spec, _ = ds[0]
    assert spec.shape == (1, 8, 63)

    ds2 = SpectrogramClassificationDataset(str(data_dir), sr=16000, n_mels=8,
                                           hop_length=256, max_length_seconds=1.0,
                                           n_fft=2048, cache_dir=str(cache))
    spec2, _ = ds2[0]
    # Different n_fft yields different mel-filter responses.
    assert not np.array_equal(spec, spec2)


def test_localization_cache_reused_when_unchanged(tmp_path, monkeypatch):
    """Unchanged labels + audio + config must be served from the pickle cache."""
    data = _make_data_dir(tmp_path)
    cache = tmp_path / "cache"

    import model.data.preprocessing as preprocessing

    orig_load = preprocessing.load_audio
    calls = {"n": 0}

    def counting_load(*a, **k):
        calls["n"] += 1
        return orig_load(*a, **k)

    monkeypatch.setattr(preprocessing, "load_audio", counting_load)

    ds = LocalizationDataset(data_dir=str(data), max_length_seconds=1.0,
                             cache_dir=str(cache))
    spec, mask = ds[0]
    assert calls["n"] == 1

    ds2 = LocalizationDataset(data_dir=str(data), max_length_seconds=1.0,
                              cache_dir=str(cache))
    spec2, mask2 = ds2[0]
    assert calls["n"] == 1  # second access must hit the cache
    assert np.array_equal(spec, spec2)
    assert np.array_equal(mask, mask2)


def test_localization_cache_invalidates_on_label_change(tmp_path):
    data = _make_data_dir(tmp_path)
    clip = "M_0001_dysfluent_000"
    cache = tmp_path / "cache"
    ds = LocalizationDataset(data_dir=str(data), max_length_seconds=1.0,
                             cache_dir=str(cache))
    idx = next(i for i, s in enumerate(ds.samples) if s["clip_id"] == clip)
    spec, mask = ds[idx]
    assert mask.sum() > 0  # block interval present

    _write_label(data / "labels" / f"{clip}.csv", [])
    ds2 = LocalizationDataset(data_dir=str(data), max_length_seconds=1.0,
                              cache_dir=str(cache))
    idx2 = next(i for i, s in enumerate(ds2.samples) if s["clip_id"] == clip)
    _, mask2 = ds2[idx2]
    assert mask2.sum() == 0  # stale cache would keep the old frame mask


def test_localization_cache_auto_derives_from_data_dir(tmp_path):
    """With cache_dir omitted, the cache must default to the sibling
    ``<data_dir parent>/cache/<basename>/<config key>`` path and be enabled."""
    data = _make_data_dir(tmp_path)
    ds = LocalizationDataset(data_dir=str(data), max_length_seconds=1.0)
    expected = os.path.join(
        os.path.dirname(str(data)), "cache", os.path.basename(str(data)),
    )
    assert ds.use_cache
    assert ds.cache_dir.startswith(expected)
    assert os.path.isdir(ds.cache_dir)

    spec, mask = ds[0]
    assert spec.shape == (1, 128, 31)
