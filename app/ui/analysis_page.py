"""Analysis page: fixed top bar + scrollable waveform, results, and timeline."""

from datetime import date

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.core.report_builder import build_report_html
from app.ui.report_export import PatientNameDialog, export_report_to_pdf
from app.ui.results_panel import ResultsPanel
from app.ui.waveform_view import WaveformView


class AnalysisPage(QWidget):
    back_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results_data = None
        self._audio = None
        self._sample_rate = 16000
        self._language = "english"
        self._filename = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(16)

        top_bar = QHBoxLayout()
        back_btn = QPushButton("← Back to Home")
        back_btn.setProperty("cssClass", "secondary")
        back_btn.clicked.connect(self.back_clicked.emit)
        top_bar.addWidget(back_btn)
        top_bar.addStretch()

        title = QLabel("Analysis Results")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        top_bar.addWidget(title)
        top_bar.addStretch()

        spacer = QWidget()
        spacer.setFixedWidth(back_btn.sizeHint().width())
        top_bar.addWidget(spacer)

        layout.addLayout(top_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        self._waveform = WaveformView()
        self._waveform.setMinimumHeight(150)
        content_layout.addWidget(self._waveform)

        self._results = ResultsPanel()
        content_layout.addWidget(self._results)

        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        footer = QHBoxLayout()
        footer.addStretch()
        self._export_btn = QPushButton("Export Report")
        self._export_btn.clicked.connect(self._export_report)
        footer.addWidget(self._export_btn)
        footer.addStretch()
        layout.addLayout(footer)

    def set_results(
        self,
        results: dict,
        audio: np.ndarray = None,
        sample_rate: int = 16000,
        language: str = "english",
        filename: str = "",
    ):
        """Update the page with analysis results and store metadata for export."""
        self._results_data = results
        self._audio = audio
        self._sample_rate = sample_rate
        self._language = language
        self._filename = filename

        self._results.set_results(results, audio, sample_rate, language=language)

        if audio is not None and len(audio) > 0:
            self._waveform.set_audio(audio, sample_rate)

    def _export_report(self):
        results = self._results_data
        if not results:
            return

        dialog = PatientNameDialog(self.window())
        dialog.exec()
        if dialog.result() != QDialog.DialogCode.Accepted:
            return
        patient_name = dialog.selected()

        default_name = f"swaraaha-report-{date.today().isoformat()}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self.window(), "Export Report", default_name, "PDF (*.pdf)"
        )
        if not path:
            return

        if self._audio is not None and len(self._audio) > 0:
            duration_sec = len(self._audio) / self._sample_rate
        else:
            duration_sec = float((results.get("transcription") or {}).get("duration_sec", 0.0))

        try:
            report_html = build_report_html(
                results,
                filename=self._filename,
                language=self._language,
                duration_sec=duration_sec,
                patient_name=patient_name,
            )
            export_report_to_pdf(report_html, path)
        except Exception as e:
            QMessageBox.critical(self.window(), "Export Failed", f"Could not export the report:\n{e}")
            return

        QMessageBox.information(self.window(), "Report Exported", f"Report saved to:\n{path}")

    def get_waveform(self) -> WaveformView:
        return self._waveform

    def get_results_panel(self) -> ResultsPanel:
        return self._results

    def get_export_button(self) -> QPushButton:
        return self._export_btn
