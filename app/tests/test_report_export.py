import pytest

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
