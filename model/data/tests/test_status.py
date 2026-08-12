"""Tests for the dataset status reporter (model/data/status.py)."""

import json

import pytest

from model.data import status as status_mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point status at a fake DATA_DIR/merged CSV on tmp_path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    merged = tmp_path / "combined_labels.csv"
    monkeypatch.setattr(status_mod, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(status_mod, "COMBINED_DATASET_PATH", str(merged))
    return {"data_dir": data_dir, "merged": merged}


def _write_merged(env, lines):
    env["merged"].write_text("\n".join(lines) + "\n")


HEADER = "clip_file,Prolongation,Block,SoundRep,WordRep,Interjection"


def test_report_handles_missing_merged_dataset(env, capsys):
    status_mod.report()
    out = capsys.readouterr().out
    assert "No merged dataset found" in out


def test_report_handles_empty_csv(env, capsys):
    _write_merged(env, [HEADER])
    status_mod.report()  # must not raise ZeroDivisionError
    out = capsys.readouterr().out
    assert "Total entries: 0" in out


def test_report_missing_label_column_no_crash(env, capsys):
    _write_merged(env, ["clip_file,Prolongation", "clip1,1"])
    status_mod.report()  # must not raise KeyError
    out = capsys.readouterr().out
    assert "Total entries: 1" in out


def test_report_missing_clip_file_column_no_crash(env, capsys):
    _write_merged(env, ["Prolongation,Block", "1,0"])
    status_mod.report()  # must not raise KeyError
    out = capsys.readouterr().out
    assert "Total entries: 1" in out


def test_report_normal_output_reports_label_distribution(env, capsys):
    _write_merged(
        env,
        [
            HEADER,
            "clip1,1,0,0,0,0",
            "clip2,0,1,0,0,0",
        ],
    )
    env["data_dir"].mkdir(parents=True, exist_ok=True)
    (env["data_dir"] / "splits.json").write_text(
        json.dumps({"train": ["clip1"], "test": ["clip2"]})
    )
    status_mod.report()
    out = capsys.readouterr().out
    assert "Total entries: 2" in out
    assert "Prolongation: 1 (50.0%)" in out
    assert "Per-split label counts" in out
