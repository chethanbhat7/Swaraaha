"""Tests for anonymous correction submissions."""

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.corrections import save_correction

client = TestClient(app)


@pytest.fixture
def corrections_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CORRECTIONS_DIR", str(tmp_path))
    return tmp_path


def test_save_correction_writes_audio_text_and_manifest(corrections_dir):
    result = save_correction(b"RIFFfake", filename="patient_name.wav", comment="missed the block", language="english")

    assert result["id"].startswith("corr_")
    assert result["audio_label"] == f"{result['id']}.wav"
    assert result["text_label"] == f"{result['id']}.txt"

    assert (corrections_dir / result["audio_label"]).read_bytes() == b"RIFFfake"
    assert (corrections_dir / result["text_label"]).read_text() == "missed the block"

    manifest = json.loads((corrections_dir / f"{result['id']}.json").read_text())
    assert manifest["anonymous"] is True
    assert manifest["audio_label"] == result["audio_label"]
    assert manifest["text_label"] == result["text_label"]
    assert manifest["audio_bytes"] == len(b"RIFFfake")
    assert "patient_name" not in json.dumps(manifest)


def test_save_correction_is_unique_per_submission(corrections_dir):
    first = save_correction(b"a", filename="a.wav", comment="one")
    second = save_correction(b"b", filename="a.wav", comment="two")

    assert first["id"] != second["id"]
    assert len(list(corrections_dir.glob("*.wav"))) == 2
    assert len(list(corrections_dir.glob("*.txt"))) == 2


def test_save_correction_rejects_empty_audio(corrections_dir):
    with pytest.raises(ValueError):
        save_correction(b"", comment="x")


def test_corrections_route_stores_pair(corrections_dir):
    response = client.post(
        "/api/corrections",
        files={"file": ("sample.wav", b"RIFFdata", "audio/wav")},
        data={"comment": "the word rep was mislabeled", "language": "english"},
    )

    assert response.status_code == 200
    body = response.json()
    assert (corrections_dir / body["audio_label"]).read_bytes() == b"RIFFdata"
    assert (corrections_dir / body["text_label"]).read_text() == "the word rep was mislabeled"


def test_corrections_route_rejects_empty_file(corrections_dir):
    response = client.post(
        "/api/corrections",
        files={"file": ("sample.wav", b"", "audio/wav")},
        data={"comment": "x"},
    )
    assert response.status_code == 422
