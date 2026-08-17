from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_analyze_returns_localizer_regions_and_severity():
    with (
        patch("backend.routes.localization.classify_audio_bytes",
              return_value={"prolongation": {"label": 0, "confidence": 0.9}}),
        patch("backend.routes.localization.localize_audio_bytes",
              return_value={"regions": [{"start": 1.0, "end": 2.5, "confidence": 0.8}], "duration_sec": 10.0}),
        patch("backend.routes.localization.transcribe_audio_bytes",
              return_value={"text": "", "language": "English", "chunks": []}),
        patch("backend.routes.localization.combine_audio_bytes",
              return_value={"error": None}),
    ):
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
    with (
        patch("backend.routes.localization.classify_audio_bytes",
              return_value={"prolongation": {"label": 0, "confidence": 0.9}}),
        patch("backend.routes.localization.localize_audio_bytes",
              return_value={"regions": [], "error": "no model", "duration_sec": 10.0}),
        patch("backend.routes.localization.transcribe_audio_bytes",
              return_value={"text": "", "language": "English", "chunks": []}),
        patch("backend.routes.localization.combine_audio_bytes",
              return_value={"error": None}),
    ):
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
