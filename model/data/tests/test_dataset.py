"""
Tests for source filtering and cache invalidation in model/data/dataset.py.
"""

import os
import struct
import wave

import numpy as np
import pytest

from model.data.dataset import ClassificationDataset, LocalizationDataset


def _write_wav(path, seconds=1.0, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    samples = (np.sin(2 * np.pi * 440 * np.arange(n) / sr) * 0.5).astype(np.float32)
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
