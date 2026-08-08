"""Export the assessment report to PDF using Qt's QPrinter. Qt layer only."""

import os

from PySide6.QtCore import QMarginsF, Qt
from PySide6.QtGui import QPageLayout, QPageSize, QTextDocument
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


def export_report_to_pdf(html_text: str, path: str) -> None:
    """Render HTML to a PDF file at ``path``. Raises RuntimeError on failure."""
    printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
    printer.setOutputFileName(path)
    printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    printer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter)

    doc = QTextDocument()
    doc.setHtml(html_text)
    doc.print_(printer)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("PDF rendering failed")


class PatientNameDialog(QDialog):
    """Modal dialog prompting for an optional patient name before export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Report")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Export Report")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Patient name (optional)")
        layout.addWidget(self._name_edit)

        buttons = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setProperty("cssClass", "secondary")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.accept)
        buttons.addStretch()
        buttons.addWidget(cancel_btn)
        buttons.addWidget(export_btn)
        layout.addLayout(buttons)

    def selected(self) -> str:
        """Return the trimmed patient name (possibly empty)."""
        return self._name_edit.text().strip()
