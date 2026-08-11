"""
Tests for CNNSpectrogramLocalizer checkpoint save/load round-trip.

The CNN localizer's checkpoints must carry their config (n_mels,
in_channels, dropout) so that ``from_pretrained`` can rebuild the model.
Without this, the registry loader crashes with KeyError on every CNN
localizer trained by the pipeline.
"""

import numpy as np
import pytest

from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer


def test_cnn_save_writes_config_and_state(tmp_path):
    loc = CNNSpectrogramLocalizer(n_mels=64, in_channels=1, dropout=0.5)
    path = str(tmp_path / "cnn.pt")
    loc.save(path)

    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["n_mels"] == 64
    assert ckpt["in_channels"] == 1
    assert ckpt["dropout"] == 0.5
    assert "model_state_dict" in ckpt


def test_cnn_from_pretrained_roundtrip(tmp_path):
    loc = CNNSpectrogramLocalizer(n_mels=64, in_channels=1, dropout=0.5)
    path = str(tmp_path / "cnn.pt")
    loc.save(path)

    loaded = CNNSpectrogramLocalizer.from_pretrained(path)
    assert loaded.n_mels == 64
    assert loaded.in_channels == 1
    assert loaded.dropout_rate == 0.5

    ours = loc.model.state_dict()
    theirs = loaded.model.state_dict()
    assert set(ours) == set(theirs)
    for key in ours:
        assert np.array_equal(ours[key].numpy(), theirs[key].numpy()), key


def _write_wav(path, seconds=1.0, sr=16000):
    import struct
    import wave

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


def test_cnn_trainer_checkpoint_loadable(tmp_path, monkeypatch):
    """The CNN trainer's best checkpoint must be loadable by from_pretrained."""
    from model.training import train_localizer

    data = tmp_path / "data"
    for clip in ["M_0001_dysfluent_000", "M_0002_dysfluent_001"]:
        _write_wav(data / "audio" / f"{clip}.wav")
        _write_label(data / "labels" / f"{clip}.csv", [(0.1, 0.4, "block")])
    with open(data / "sources.csv", "w") as f:
        f.write("clip_id,source\n")
        f.write("M_0001_dysfluent_000,uclass\n")
        f.write("M_0002_dysfluent_001,uclass\n")

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_localizer.py",
            "--data_dir", str(data),
            "--output_dir", str(tmp_path),
            "--epochs", "1",
            "--max_length_seconds", "1.0",
        ],
    )
    args = train_localizer.parse_args()
    train_localizer.train(args)

    best_path = next(tmp_path.glob("cnnloc_*_best.pt"))
    loaded = CNNSpectrogramLocalizer.from_pretrained(str(best_path))
    assert loaded.n_mels == args.n_mels
    assert loaded.dropout_rate == args.dropout
