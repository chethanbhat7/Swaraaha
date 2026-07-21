"""PDF file browser and text display widget."""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeView, QTextEdit, QLabel, QFileSystemModel,
)
from PySide6.QtCore import Signal

from app.core.pdf_handler import PdfHandler


class PdfViewer(QWidget):
    pdf_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_dir = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Rainbow Passage")
        header.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px 0;")
        layout.addWidget(header)

        self._tree = QTreeView()
        self._model = QFileSystemModel()
        self._model.setRootPath("")
        self._tree.setModel(self._model)
        self._model.setNameFilters(["*.pdf"])
        self._model.setNameFilterDisables(False)
        self._tree.setAnimated(True)
        self._tree.setHeaderHidden(True)

        for col in range(1, 4):
            self._tree.hideColumn(col)

        self._tree.clicked.connect(self._on_file_clicked)
        layout.addWidget(self._tree)

        self._text_display = QTextEdit()
        self._text_display.setReadOnly(True)
        self._text_display.setPlaceholderText("Select a PDF to view its content...")
        layout.addWidget(self._text_display)

    def set_directory(self, directory: str):
        """Set the root directory to browse for PDFs."""
        self._current_dir = directory
        if os.path.isdir(directory):
            self._tree.setRootIndex(self._model.index(directory))

    def _on_file_clicked(self, index):
        """Handle file click — extract and display text."""
        path = self._model.filePath(index)
        if path.lower().endswith(".pdf"):
            text = PdfHandler.extract_text(path)
            self._text_display.setPlainText(text)
            self.pdf_selected.emit(path)

    def get_text_display(self) -> QTextEdit:
        """Return the text display widget for external styling."""
        return self._text_display
