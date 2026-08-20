"""Tests for the backend classification service."""

from unittest.mock import patch

from backend.services import classifier as classifier_service


def test_classify_audio_bytes_returns_classification():
    fake_results = {
        "classification": {
            "prolongation": {"label": 1, "confidence": 0.9},
            "block": {"label": 0, "confidence": 0.7},
            "summary": {"detected": ["prolongation"], "primary": "prolongation"},
        },
    }
    with patch.object(classifier_service, "_analyze", return_value=fake_results):
        result = classifier_service.classify_audio_bytes(b"fake-wav-bytes")

    assert result["prolongation"] == {"label": 1, "confidence": 0.9}
    assert result["block"] == {"label": 0, "confidence": 0.7}
    assert result["summary"] == {"detected": ["prolongation"], "primary": "prolongation"}
