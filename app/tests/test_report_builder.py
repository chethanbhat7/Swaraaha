from app.core.report_builder import build_report_html, severity_for

SAMPLE_RESULTS = {
    "classifications": {
        "prolongation": (False, 0.12),
        "block": (True, 0.87),
        "soundrep": (False, 0.08),
        "wordrep": (False, 0.05),
        "interjection": (True, 0.72),
    },
    "localizations": [(0.5, 1.2, 0.87), (3.4, 4.1, 0.72)],
    "transcription": {"text": "hello world", "words": [], "duration_sec": 4.0},
}


def _html(**overrides):
    params = {"filename": "test.wav", "language": "english", "duration_sec": 4.0}
    params.update(overrides)
    return build_report_html(SAMPLE_RESULTS, **params)


def test_build_report_html_contains_all_sections():
    html = _html()
    assert "SWARAAHA" in html
    assert "Speech Dysfluency Assessment Report" in html
    assert "Classification Results" in html
    assert "Localized Dysfluency Events" in html
    assert "Summary" in html
    assert "Transcript" in html
    assert "not a medical diagnosis" in html
    assert "test.wav" in html
    assert "English" in html
    assert "4.00" in html


def test_build_report_html_omits_patient_row_when_empty():
    assert "Patient Name" not in _html()


def test_build_report_html_shows_patient_name():
    assert "Aarav Sharma" in _html(patient_name="Aarav Sharma")


def test_build_report_html_empty_localizations():
    results = dict(SAMPLE_RESULTS, localizations=[])
    html = build_report_html(results, filename="a.wav", language="english", duration_sec=1.0)
    assert "No dysfluency events localized." in html


def test_build_report_html_empty_transcript():
    results = dict(SAMPLE_RESULTS)
    results["transcription"] = {"text": "", "words": [], "duration_sec": 0.0}
    html = build_report_html(results, filename="a.wav", language="english", duration_sec=1.0)
    assert "No transcription available." in html


def test_build_report_html_escapes_patient_name():
    html = _html(patient_name="Smith & Sons")
    assert "Smith &amp; Sons" in html
    assert "Smith & Sons" not in html


def test_build_report_html_uses_report_date():
    html = _html(report_date="2026-01-02")
    assert "2026-01-02" in html


def test_severity_for_fluent():
    results = {
        "classifications": {
            name: (False, 0.5)
            for name in ("prolongation", "block", "soundrep", "wordrep", "interjection")
        }
    }
    assert severity_for(results) == "Fluent"


def test_severity_for_mild():
    results = {"classifications": {"block": (True, 0.55), "prolongation": (False, 0.1)}}
    assert severity_for(results) == "Mild"


def test_severity_for_moderate():
    results = {"classifications": {"block": (True, 0.65)}}
    assert severity_for(results) == "Moderate"


def test_severity_for_severe():
    results = {"classifications": {"block": (True, 0.87)}}
    assert severity_for(results) == "Severe"
