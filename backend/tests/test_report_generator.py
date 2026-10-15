from backend.services.report_generator import build_typ_source


def test_report_contains_classification_section():
    source = build_typ_source({
        "patient": {"name": "Jane", "phone": "123"},
        "audio": {"filename": "a.wav", "size": "1MB", "duration": "00:10"},
        "date": "Aug 15 2026",
        "classification": {"prolongation": {"label": 1, "confidence": 0.9}},
    })
    assert "Stuttering Classes" in source
    assert "prolongation" in source
    assert "90.0%" in source


def test_report_omits_severity_section():
    source = build_typ_source({
        "patient": {"name": "Jane", "phone": "123"},
        "audio": {"filename": "a.wav", "size": "1MB", "duration": "00:10"},
        "date": "Aug 15 2026",
        "classification": {},
        "severity": {"index_pct": 12.5, "severity": "moderate", "label": "Moderate"},
    })
    assert "Diagnostic Severity" not in source
    assert "Stutter Index" not in source
    assert "Severity Level" not in source
