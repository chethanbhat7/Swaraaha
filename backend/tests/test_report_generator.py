from backend.services.report_generator import build_typ_source


def test_report_includes_severity_section():
    source = build_typ_source({
        "patient": {"name": "Jane", "phone": "123"},
        "audio": {"filename": "a.wav", "size": "1MB", "duration": "00:10"},
        "date": "Aug 15 2026",
        "classification": {"prolongation": {"label": 1, "confidence": 0.9}},
        "severity": {"index_pct": 12.5, "severity": "moderate", "label": "Moderate"},
    })
    assert "Diagnostic Severity" in source
    assert "12.5%" in source
    assert "Moderate" in source


def test_report_omits_severity_when_absent():
    source = build_typ_source({
        "patient": {"name": "Jane", "phone": "123"},
        "audio": {"filename": "a.wav", "size": "1MB", "duration": "00:10"},
        "date": "Aug 15 2026",
        "classification": {},
    })
    assert "Diagnostic Severity" not in source
