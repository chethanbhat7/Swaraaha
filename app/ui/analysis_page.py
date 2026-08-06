"""Analysis page: fixed top bar + scrollable waveform, results, and timeline."""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.results_panel import ResultsPanel
from app.ui.waveform_view import WaveformView


class AnalysisPage(QWidget):
    back_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
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

    def set_results(self, results: dict, audio: np.ndarray = None, sample_rate: int = 16000, language: str = "english"):
        """Update the page with analysis results."""
        self._results.set_results(results, audio, sample_rate, language=language)

        if audio is not None and len(audio) > 0:
            self._waveform.set_audio(audio, sample_rate)

    def get_waveform(self) -> WaveformView:
        return self._waveform

    def get_results_panel(self) -> ResultsPanel:
        return self._results
