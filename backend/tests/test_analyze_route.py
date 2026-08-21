from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


def test_analyze_returns_localizer_regions_and_severity():
    fake_results = {
        "classification": {"prolongation": {"label": 0, "confidence": 0.9}},
        "localization": {"regions": [{"start": 1.0, "end": 2.5, "confidence": 0.8}], "duration_sec": 10.0},
        "transcription": {"text": "", "language": "English", "chunks": []},
        "combined": {"regions": [], "audio_duration": 10.0, "total_stutters": 0},
    }
    with TestClient(app) as client:
        with patch("backend.routes.localization._analyze", return_value=fake_results):
            response = client.post(
                "/api/analyze",
                files={"file": ("t.wav", b"RIFF", "audio/wav")},
                data={"language": "english"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["localization"]["regions"] == [{"start": 1.0, "end": 2.5, "confidence": 0.8}]
    assert body["severity"]["index_pct"] == 15.0
    assert body["severity"]["severity"] == "severe"


def test_analyze_with_no_localizer_regions_reports_fluent():
    fake_results = {
        "classification": {"prolongation": {"label": 0, "confidence": 0.9}},
        "localization": {"regions": [], "error": "no model", "duration_sec": 10.0},
        "transcription": {"text": "", "language": "English", "chunks": []},
        "combined": {"regions": [], "audio_duration": 10.0, "total_stutters": 0},
    }
    with TestClient(app) as client:
        with patch("backend.routes.localization._analyze", return_value=fake_results):
            response = client.post(
                "/api/analyze",
                files={"file": ("t.wav", b"RIFF", "audio/wav")},
                data={"language": "english"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["localization"]["regions"] == []
    assert body["severity"]["severity"] == "fluent"
    assert body["severity"]["index_pct"] == 0.0
