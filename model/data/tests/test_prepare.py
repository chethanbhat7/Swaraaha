"""
Tests for training-data preparation (model/data/prepare.py).
"""

import json

import pandas as pd

from model.data.prepare import create_training_data


def _make_merged(tmp_path):
    """Create a minimal merged dataset (combined_labels.csv + labels/ + audio)."""
    merged = tmp_path / "merged"
    audio = merged / "audio"
    labels = merged / "labels"
    audio.mkdir(parents=True)
    labels.mkdir(parents=True)

    for stem in ["A_1", "B_1", "C_1", "D_1", "E_1"]:
        (audio / f"{stem}.wav").write_bytes(b"\x00" * 100)

    df = pd.DataFrame({
        "clip_file": [str(audio / f"{stem}.wav") for stem in ["A_1", "B_1", "C_1", "D_1", "E_1"]],
        "Prolongation": [1, 0, 0, 0, 0],
        "Block": [0, 1, 0, 0, 0],
        "SoundRep": [0, 0, 1, 0, 0],
        "WordRep": [0, 0, 0, 1, 0],
        "Interjection": [0, 0, 0, 0, 1],
        "source": ["sep28k", "sep28k", "uclass", "uclass", "uclass"],
    })
    df.to_csv(merged / "combined_labels.csv", index=False)

    for stem in ["A_1", "B_1", "C_1", "D_1", "E_1"]:
        (labels / f"{stem}.csv").write_text("start_sec,end_sec,dysfluency_type\n0.000,3.000,Block\n")

    sources = pd.DataFrame({
        "clip_id": ["A_1", "B_1", "C_1", "D_1", "E_1"],
        "source": ["sep28k", "sep28k", "uclass", "uclass", "uclass"],
    })
    sources.to_csv(merged / "sources.csv", index=False)

    return merged


def test_create_training_data_copies_sources_csv(tmp_path, monkeypatch):
    merged = _make_merged(tmp_path)
    monkeypatch.setattr("model.data.prepare.COMBINED_DATASET_PATH", str(merged / "combined_labels.csv"))

    out = create_training_data(
        output_dir=tmp_path / "out",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )

    for split in ["train", "val", "test"]:
        sources_path = out / split / "sources.csv"
        assert sources_path.exists(), f"{split}/sources.csv missing"
        copied = pd.read_csv(sources_path)
        assert set(copied["clip_id"]) <= {"A_1", "B_1", "C_1", "D_1", "E_1"}
        assert "source" in copied.columns


def _make_merged_with_boli(tmp_path):
    """_make_merged plus one extra boli clip."""
    merged = _make_merged(tmp_path)
    audio = merged / "audio"
    labels = merged / "labels"

    (audio / "F_1.wav").write_bytes(b"\x00" * 100)
    (labels / "F_1.csv").write_text("start_sec,end_sec,dysfluency_type\n0.000,3.000,SoundRep\n")

    df = pd.read_csv(merged / "combined_labels.csv")
    df = pd.concat([df, pd.DataFrame([{
        "clip_file": str(audio / "F_1.wav"),
        "Prolongation": 0, "Block": 0, "SoundRep": 1,
        "WordRep": 0, "Interjection": 0, "source": "boli",
    }])], ignore_index=True)
    df.to_csv(merged / "combined_labels.csv", index=False)

    sources = pd.concat([
        pd.read_csv(merged / "sources.csv"),
        pd.DataFrame([{"clip_id": "F_1", "source": "boli"}]),
    ], ignore_index=True)
    sources.to_csv(merged / "sources.csv", index=False)

    return merged


def _split_sources(out):
    return {
        split: set(pd.read_csv(out / split / "sources.csv")["clip_id"])
        for split in ["train", "val", "test"]
    }


def test_create_training_data_pins_boli_to_test_split(tmp_path, monkeypatch):
    merged = _make_merged_with_boli(tmp_path)
    monkeypatch.setattr("model.data.prepare.COMBINED_DATASET_PATH", str(merged / "combined_labels.csv"))

    out = create_training_data(
        output_dir=tmp_path / "out",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )

    splits = _split_sources(out)
    assert "F_1" in splits["test"]
    assert "F_1" not in splits["train"]
    assert "F_1" not in splits["val"]
    assert not (out / "train" / "audio" / "F_1.wav").exists()
    assert not (out / "val" / "audio" / "F_1.wav").exists()
    assert (out / "test" / "audio" / "F_1.wav").exists()


def test_create_training_data_test_only_sources_override(tmp_path, monkeypatch):
    merged = _make_merged_with_boli(tmp_path)
    monkeypatch.setattr("model.data.prepare.COMBINED_DATASET_PATH", str(merged / "combined_labels.csv"))

    out = create_training_data(
        output_dir=tmp_path / "out",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        test_only_sources=("uclass",),
    )

    splits = _split_sources(out)
    assert {"C_1", "D_1", "E_1"} <= splits["test"]
    assert splits["train"].isdisjoint({"C_1", "D_1", "E_1"})
    assert splits["val"].isdisjoint({"C_1", "D_1", "E_1"})


def test_create_training_data_overwrites_labels(tmp_path, monkeypatch):
    merged = _make_merged(tmp_path)
    monkeypatch.setattr("model.data.prepare.COMBINED_DATASET_PATH", str(merged / "combined_labels.csv"))

    out = create_training_data(
        output_dir=tmp_path / "out",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )

    # Stale label with wrong content; force=True must overwrite it.
    train_labels = out / "train" / "labels"
    first = next(train_labels.glob("*.csv"))
    first.write_text("start_sec,end_sec,dysfluency_type\n9.000,9.500,wordrep\n")

    create_training_data(
        output_dir=tmp_path / "out",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        force=True,
    )
    content = first.read_text()
    assert "9.000" not in content
    assert content.splitlines()[0] == "start_sec,end_sec,dysfluency_type"
