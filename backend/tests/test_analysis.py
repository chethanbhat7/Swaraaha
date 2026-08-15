from backend.services.analysis import finalize_localization


def test_uses_model_regions_when_present():
    localization = {
        "regions": [{"start": 0.0, "end": 0.5, "confidence": 0.9}],
        "error": None,
        "duration_sec": 5.0,
    }
    result = finalize_localization(localization, {"chunks": []})
    assert result["source"] == "model"
    assert len(result["regions"]) == 1
    assert result["duration_sec"] == 5.0


def test_falls_back_to_rule_based():
    localization = {"regions": [], "error": "no model", "duration_sec": 5.0}
    transcription = {
        "chunks": [
            {"text": "I", "start": 0.0, "end": 0.2},
            {"text": "I", "start": 0.2, "end": 0.4},
        ]
    }
    result = finalize_localization(localization, transcription)
    assert result["source"] == "rule-based"
    assert result["error"] is None
    assert len(result["regions"]) == 1
    assert result["regions"][0]["type"] == "wordrep"
