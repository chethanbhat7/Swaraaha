import pdfplumber
import pytest

from app.core.report_builder import build_report_html
from app.ui.report_export import PatientNameDialog, export_report_to_pdf


def test_export_report_to_pdf_creates_pdf(qapp, tmp_path):
    path = tmp_path / "report.pdf"
    export_report_to_pdf("<html><body><h1>Hello</h1></body></html>", str(path))
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:4] == b"%PDF"


def test_export_report_to_pdf_bad_path_raises(qapp, tmp_path):
    with pytest.raises(RuntimeError):
        export_report_to_pdf(
            "<html><body><h1>Hello</h1></body></html>",
            str(tmp_path / "no-such-dir" / "report.pdf"),
        )


def test_patient_name_dialog_selected_trimmed(qapp):
    dialog = PatientNameDialog()
    dialog._name_edit.setText("  Aarav Sharma  ")
    assert dialog.selected() == "Aarav Sharma"


def test_patient_name_dialog_default_empty(qapp):
    dialog = PatientNameDialog()
    assert dialog.selected() == ""


def test_report_pdf_text_is_legible(qapp, tmp_path):
    results = {
        "classifications": {
            "prolongation": (False, 0.12),
            "block": (True, 0.87),
            "soundrep": (False, 0.08),
            "wordrep": (False, 0.05),
            "interjection": (True, 0.72),
        },
        "localizations": [(0.5, 1.2, 0.87)],
        "transcription": {"text": "hello world", "words": [], "duration_sec": 4.0},
    }
    html = build_report_html(results, filename="test.wav", language="english", duration_sec=4.0)
    path = tmp_path / "report.pdf"
    export_report_to_pdf(html, str(path))

    with pdfplumber.open(str(path)) as pdf:
        page = pdf.pages[0]
        sizes = [round(ch["size"], 1) for ch in page.chars if ch["text"].strip()]
        assert sizes, "no text in PDF"
        table_word = next(w for w in page.extract_words() if w["text"].startswith("Block"))
        assert table_word["height"] >= 7.0, f"table text too small: {table_word['height']}pt"
