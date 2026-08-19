"""Transcription window panel for displaying input audio speech-to-text transcription."""

import json

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.transcription import AudioTranscriber
from app.ui.theme import COLORS


class TranscriptionPanel(QWidget):
    """
    Transcription window displaying live/computed speech transcription
    along with word-level timestamps and stutter alignments.
    """
    transcribe_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._transcriber = AudioTranscriber()
        self._audio = None
        self._transcription_data = {"text": "", "words": []}
        self._language = "english"
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Header Row
        header_row = QHBoxLayout()
        header = QLabel("Audio Transcription")
        header.setStyleSheet("font-size: 16px; font-weight: 600; padding: 4px 0;")
        header_row.addWidget(header)

        header_row.addStretch()

        self._status_label = QLabel("No Audio Loaded")
        self._status_label.setStyleSheet(
            f"color: {COLORS['outline']}; font-size: 12px; font-weight: 500; "
            f"background-color: {COLORS['surface_variant']}; padding: 4px 10px; border-radius: 12px;"
        )
        header_row.addWidget(self._status_label)
        layout.addLayout(header_row)

        # Control Row
        ctrl_row = QHBoxLayout()

        self._transcribe_btn = QPushButton("Transcribe Audio")
        self._transcribe_btn.setProperty("cssClass", "secondary")
        self._transcribe_btn.clicked.connect(self._on_transcribe_click)
        ctrl_row.addWidget(self._transcribe_btn)

        self._copy_btn = QPushButton("Copy Text")
        self._copy_btn.setProperty("cssClass", "secondary")
        self._copy_btn.clicked.connect(self._on_copy_text)
        ctrl_row.addWidget(self._copy_btn)

        self._export_btn = QPushButton("Export")
        self._export_btn.setProperty("cssClass", "secondary")
        self._export_btn.clicked.connect(self._on_export)
        ctrl_row.addWidget(self._export_btn)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Transcript Box Label & Text Area
        txt_label = QLabel("Transcribed Speech")
        txt_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(txt_label)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setPlaceholderText("Transcription will appear here after loading audio or clicking Transcribe...")
        self._text_edit.setMinimumHeight(100)
        layout.addWidget(self._text_edit, stretch=1)

    def set_audio(self, audio: np.ndarray, sample_rate: int = 16000, localizations=None, language: str = "english"):
        """Set audio array and automatically run transcription."""
        self._audio = audio
        self._language = language
        if audio is not None and len(audio) > 0:
            self._status_label.setText(f"Audio Ready ({len(audio)/sample_rate:.1f}s)")
            self.run_transcription(localizations=localizations)
        else:
            self.clear()

    def run_transcription(self, localizations=None):
        """Run the transcription pipeline on the loaded audio."""
        if self._audio is None or len(self._audio) == 0:
            self._status_label.setText("No audio available")
            return

        self._status_label.setText("Transcribing...")
        data = self._transcriber.transcribe(self._audio, localizations=localizations, language=self._language)
        self.set_transcription(data)
        self._status_label.setText("Transcription Complete")

    def set_transcription(self, data: dict):
        """Display transcription dictionary."""
        self._transcription_data = data
        text = data.get("text", "")
        self._text_edit.setPlainText(text)

    def clear(self):
        """Clear transcription display."""
        self._audio = None
        self._transcription_data = {"text": "", "words": []}
        self._text_edit.clear()
        self._status_label.setText("No Audio Loaded")

    def _on_transcribe_click(self):
        if self._audio is not None and len(self._audio) > 0:
            self.run_transcription()
        else:
            self.transcribe_requested.emit()

    def _on_copy_text(self):
        text = self._text_edit.toPlainText()
        if text:
            from PySide6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(text)

    def _on_export(self):
        text = self._text_edit.toPlainText()
        if not text:
            return

        path, filter_used = QFileDialog.getSaveFileName(
            self, "Export Transcript", "transcript.txt", "Text Files (*.txt);;JSON Files (*.json)"
        )
        if path:
            if path.endswith(".json"):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self._transcription_data, f, indent=2)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
