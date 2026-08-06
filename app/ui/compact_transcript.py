"""Compact transcript view: text area + word-level alignment table.

Used as the bottom half of the Home Page right column once audio is loaded.
Auto-runs transcription; intentionally has no header, status pill, or action buttons.
"""

import numpy as np
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.transcription import AudioTranscriber
from app.ui.table_utils import MAX_HEIGHT_UNCAP, cap_table_height, populate_transcript_table

MAX_ROWS = 8


class CompactTranscript(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._transcriber = AudioTranscriber()
        self._audio = None
        self._language = "english"
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QLabel("Transcription")
        header.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(header)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setPlaceholderText("Transcription will appear here after loading audio...")
        self._text_edit.setMinimumHeight(90)
        self._text_edit.setMaximumHeight(140)
        layout.addWidget(self._text_edit)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Word", "Start (s)", "End (s)", "Confidence", "Status"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

    def set_audio(self, audio: np.ndarray, sample_rate: int = 16000, localizations=None, language: str = "english"):
        """Set audio array and automatically run transcription."""
        self._audio = audio
        self._language = language
        if audio is not None and len(audio) > 0:
            self.run_transcription(localizations=localizations)
        else:
            self.clear()

    def run_transcription(self, localizations=None):
        """Run the transcription pipeline on the loaded audio."""
        if self._audio is None or len(self._audio) == 0:
            return
        data = self._transcriber.transcribe(self._audio, localizations=localizations, language=self._language)
        self.set_transcription(data)

    def set_transcription(self, data: dict):
        """Display transcription dictionary."""
        text = data.get("text", "")
        words = data.get("words", [])
        self._text_edit.setPlainText(text)
        populate_transcript_table(self._table, words)
        cap_table_height(self._table, MAX_ROWS)

    def clear(self):
        """Clear transcription display."""
        self._audio = None
        self._text_edit.clear()
        self._table.setRowCount(0)
        self._table.setMinimumHeight(0)
        self._table.setMaximumHeight(MAX_HEIGHT_UNCAP)
