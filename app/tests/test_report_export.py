"""Tests for desktop report export: data normalization + shared Typst builder."""

import pdfplumber

from app.core.report_data import build_report_data
from app.ui.report_export import PatientNameDialog
from shared.reporting.report_builder import generate_report_pdf

SAMPLE_RESULTS = {
    "classifications": {
        "prolongation": (False, 0.12),
        "block": (True, 0.87),
        "interjection": (True, 0.72),
    },
    "localizations": [(0.5, 1.2, 0.87), (3.4, 4.1, 0.72)],
    "transcription": {"text": "hello world", "words": [], "duration_sec": 4.0},
    "combined": {
        "regions": [
            {"start": 0.5, "end": 1.2, "confidence": 0.87, "primary_type": "block", "classes": {}, "syllables": []},
            {"start": 3.4, "end": 4.1, "confidence": 0.72, "primary_type": "interjection", "classes": {}, "syllables": []},
        ],
        "audio_duration": 4.0,
        "total_stutters": 2,
    },
}


def test_build_report_data_normalizes_contract():
    data = build_report_data(
        SAMPLE_RESULTS,
        patient_name="Aarav Sharma",
        filename="test.wav",
        duration_sec=4.0,
    )
    assert data["patient"] == {"name": "Aarav Sharma", "phone": ""}
    assert data["audio"] == {"filename": "test.wav", "size": "", "duration": "4.00 s"}
    assert data["classification"]["block"] == {"label": 1, "confidence": 0.87}
    assert data["classification"]["prolongation"] == {"label": 0, "confidence": 0.12}
    assert data["localization"]["regions"] == [
        {"start": 0.5, "end": 1.2, "confidence": 0.87},
        {"start": 3.4, "end": 4.1, "confidence": 0.72},
    ]
    assert data["combined"]["total_stutters"] == 2
    assert data["transcription"]["text"] == "hello world"
    assert data["date"]


def test_build_report_data_defaults_to_empty_patient():
    data = build_report_data(SAMPLE_RESULTS, duration_sec=2.0)
    assert data["patient"] == {"name": "", "phone": ""}


def test_generate_report_pdf_writes_to_path(qapp, tmp_path):
    data = build_report_data(SAMPLE_RESULTS, patient_name="Aarav Sharma", filename="test.wav", duration_sec=4.0)
    path = tmp_path / "report.pdf"
    path.write_bytes(generate_report_pdf(data))
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:4] == b"%PDF"


def test_report_pdf_text_is_legible(qapp, tmp_path):
    data = build_report_data(SAMPLE_RESULTS, patient_name="Aarav Sharma", filename="test.wav", duration_sec=4.0)
    path = tmp_path / "report.pdf"
    path.write_bytes(generate_report_pdf(data))
    with pdfplumber.open(str(path)) as pdf:
        page = pdf.pages[0]
        sizes = [round(ch["size"], 1) for ch in page.chars if ch["text"].strip()]
        assert sizes, "no text in PDF"
        table_word = next(w for w in page.extract_words() if w["text"].startswith("Block"))
        assert table_word["height"] >= 7.0, f"table text too small: {table_word['height']}pt"


def test_patient_name_dialog_selected_trimmed(qapp):
    dialog = PatientNameDialog()
    dialog._name_edit.setText("  Aarav Sharma  ")
    assert dialog.selected() == "Aarav Sharma"


def test_patient_name_dialog_default_empty(qapp):
    dialog = PatientNameDialog()
    assert dialog.selected() == ""
