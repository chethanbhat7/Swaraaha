"""Compact transcript view: text area only.

Used as the bottom half of the Home Page right column once audio is loaded.
Auto-runs transcription; intentionally has no header, status pill, or action buttons.
"""

from PySide6.QtWidgets import (
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class CompactTranscript(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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

    def set_transcription(self, data: dict):
        """Display transcription dictionary."""
        text = data.get("text", "")
        self._text_edit.setPlainText(text)

    def clear(self):
        """Clear transcription display."""
        self._text_edit.clear()
