from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_analyze_falls_back_to_rule_based_and_returns_severity():
    transcription = {
        "text": "I I like tea",
        "language": "English",
        "chunks": [
            {"text": "I", "start": 0.0, "end": 0.2, "language": "English"},
            {"text": "I", "start": 0.2, "end": 0.4, "language": "English"},
            {"text": "like", "start": 0.4, "end": 0.8, "language": "English"},
            {"text": "tea", "start": 0.8, "end": 1.2, "language": "English"},
        ],
    }
    with (
        patch("backend.routes.localization.classify_audio_bytes",
              return_value={"prolongation": {"label": 0, "confidence": 0.9}}),
        patch("backend.routes.localization.localize_audio_bytes",
              return_value={"regions": [], "error": "no model", "duration_sec": 10.0}),
        patch("backend.routes.localization.transcribe_audio_bytes",
              return_value=transcription),
    ):
        response = client.post(
            "/api/analyze",
            files={"file": ("t.wav", b"RIFF", "audio/wav")},
            data={"language": "english"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["localization"]["source"] == "rule-based"
    assert len(body["localization"]["regions"]) == 1
    assert body["localization"]["regions"][0]["start"] == 0.0
    assert body["severity"]["severity"] == "mild"
    assert body["severity"]["index_pct"] == 4.0


def test_analyze_uses_model_regions_when_available():
    with (
        patch("backend.routes.localization.classify_audio_bytes",
              return_value={"prolongation": {"label": 0, "confidence": 0.9}}),
        patch("backend.routes.localization.localize_audio_bytes",
              return_value={"regions": [{"start": 1.0, "end": 2.5, "confidence": 0.8}], "duration_sec": 10.0}),
        patch("backend.routes.localization.transcribe_audio_bytes",
              return_value={"text": "", "language": "English", "chunks": []}),
    ):
        response = client.post(
            "/api/analyze",
            files={"file": ("t.wav", b"RIFF", "audio/wav")},
            data={"language": "english"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["localization"]["source"] == "model"
    assert len(body["localization"]["regions"]) == 1
    assert body["severity"]["severity"] == "severe"
