"""Tests for the shared Typst report builder used by the desktop and web apps."""

import pytest

from shared.reporting.report_builder import (
    build_report_source,
    generate_report_pdf,
    severity_for,
)

SAMPLE_DATA = {
    "patient": {"name": "Aarav Sharma", "phone": "999-000-1111"},
    "audio": {"filename": "sample.wav", "size": "2.1 MB", "duration": "4.00 s"},
    "date": "2026-01-02",
    "classification": {
        "prolongation": {"label": 0, "confidence": 0.12},
        "block": {"label": 1, "confidence": 0.87},
        "soundrep": {"label": 0, "confidence": 0.08},
        "wordrep": {"label": 0, "confidence": 0.05},
        "interjection": {"label": 1, "confidence": 0.72},
    },
    "combined": {
        "regions": [
            {
                "start": 0.5,
                "end": 1.2,
                "confidence": 0.87,
                "primary_type": "block",
                "classes": {},
                "syllables": [],
            },
            {
                "start": 3.4,
                "end": 4.1,
                "confidence": 0.72,
                "primary_type": "interjection",
                "classes": {},
                "syllables": [],
            },
        ],
        "audio_duration": 4.0,
        "total_stutters": 2,
    },
    "transcription": {"text": "hello world"},
}


def test_severity_for_thresholds():
    assert severity_for(
        {"combined": {"regions": [], "audio_duration": 10.0, "total_stutters": 0}}
    ) == "Fluent"

    assert severity_for(
        {"combined": {"regions": [{"start": 0, "end": 0.2}], "audio_duration": 10.0, "total_stutters": 1}}
    ) == "Mild"

    assert severity_for(
        {"combined": {"regions": [{"start": 0, "end": 0.5}], "audio_duration": 10.0, "total_stutters": 1}}
    ) == "Moderate"

    assert severity_for(
        {"combined": {"regions": [{"start": 0, "end": 1.6}], "audio_duration": 10.0, "total_stutters": 1}}
    ) == "Severe"


def test_severity_for_missing_combined_returns_none():
    assert severity_for({}) is None
    assert severity_for({"combined": {"error": "boom"}}) is None
    assert severity_for({"combined": {"regions": [], "audio_duration": 0.0, "total_stutters": 0}}) is None


def test_build_report_source_contains_sections():
    src = build_report_source(SAMPLE_DATA)
    for section in (
        "Swaraaha Stutter Analysis Report",
        "Patient Details",
        "Audio Details",
        "Classification Results",
        "Localized Dysfluency Events",
        "Summary",
        "Transcript",
        "not a medical diagnosis",
    ):
        assert section in src
    assert "Aarav Sharma" in src
    assert "Block" in src
    assert "Total stuttering events: 2" in src


def test_build_report_source_escapes_patient_name():
    data = dict(SAMPLE_DATA, patient={"name": "Aarav [Sharma]", "phone": ""})
    src = build_report_source(data)
    assert "[Sharma]" not in src
    assert "\\[Sharma\\]" in src


def test_build_report_source_empty_localizations():
    data = dict(SAMPLE_DATA, combined={"regions": [], "audio_duration": 4.0, "total_stutters": 0})
    src = build_report_source(data)
    assert "No dysfluency events localized." in src


def test_build_report_source_falls_back_to_localization_regions():
    data = dict(SAMPLE_DATA)
    del data["combined"]
    data["localization"] = {"regions": [{"start": 0.5, "end": 1.2, "confidence": 0.87}]}
    src = build_report_source(data)
    assert "No dysfluency events localized." not in src
    assert "0.50" in src and "1.20" in src


def test_generate_report_pdf_returns_pdf_bytes():
    pdf = generate_report_pdf(SAMPLE_DATA)
    assert pdf[:4] == b"%PDF"
