"""Classification results table and localization timeline display."""

import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.table_utils import resize_table_to_contents
from app.ui.theme import COLORS
from app.ui.transcription_panel import TranscriptionPanel
from app.ui.waveform_view import WaveformView

CLASS_NAMES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]
DISPLAY_NAMES = ["Prolongation", "Block", "Sound Repetition", "Word Repetition", "Interjection"]


class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        class_label = QLabel("Classification Results")
        class_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(class_label)

        self._table = QTableWidget(5, 3)
        self._table.setHorizontalHeaderLabels(["Class", "Detected", "Confidence"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setAlternatingRowColors(True)

        for i, name in enumerate(DISPLAY_NAMES):
            self._table.setItem(i, 0, QTableWidgetItem(name))
            self._table.setItem(i, 1, QTableWidgetItem("—"))
            self._table.setItem(i, 2, QTableWidgetItem("—"))

        layout.addWidget(self._table)

        loc_label = QLabel("Localization Timeline")
        loc_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(loc_label)

        self._waveform = WaveformView()
        self._waveform.setMinimumHeight(100)
        layout.addWidget(self._waveform)

        legend_layout = QHBoxLayout()
        for class_name, display_name in zip(CLASS_NAMES, DISPLAY_NAMES):
            color = COLORS["dysfluency"][class_name]
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 12px;")
            label = QLabel(display_name)
            label.setStyleSheet(f"font-size: 11px; color: {COLORS['outline']};")
            legend_layout.addWidget(dot)
            legend_layout.addWidget(label)
            legend_layout.addSpacing(12)
        legend_layout.addStretch()
        layout.addLayout(legend_layout)

        self._transcription_panel = TranscriptionPanel()
        layout.addWidget(self._transcription_panel)

    def set_results(self, results: dict, audio: np.ndarray = None, sample_rate: int = 16000, language: str = "english"):
        """Update the panel with analysis results."""
        classifications = results.get("classifications", {})
        localizations = results.get("localizations", [])
        transcription = results.get("transcription", None)

        for i, class_name in enumerate(CLASS_NAMES):
            if class_name in classifications:
                detected, confidence = classifications[class_name]
                det_item = QTableWidgetItem("Yes" if detected else "No")
                conf_item = QTableWidgetItem(f"{confidence * 100:.0f}%")

                if detected:
                    det_item.setForeground(QColor(COLORS["dysfluency"][class_name]))
                    conf_item.setForeground(QColor(COLORS["dysfluency"][class_name]))
                    font = det_item.font()
                    font.setBold(True)
                    det_item.setFont(font)
                    conf_item.setFont(font)

                self._table.setItem(i, 1, det_item)
                self._table.setItem(i, 2, conf_item)

        resize_table_to_contents(self._table)

        if audio is not None and len(audio) > 0:
            self._waveform.set_audio(audio, sample_rate)
            overlays = []
            for start_sec, end_sec, conf in localizations:
                overlays.append((start_sec, end_sec, "#6A1B9A"))
            self._waveform.set_overlays(overlays)

        if transcription:
            self._transcription_panel.set_transcription(transcription)

    def clear_results(self):
        """Clear all results."""
        for i in range(5):
            self._table.setItem(i, 1, QTableWidgetItem("—"))
            self._table.setItem(i, 2, QTableWidgetItem("—"))
        self._waveform.clear_overlays()
        self._transcription_panel.clear()
