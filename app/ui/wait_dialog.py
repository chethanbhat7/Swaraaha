"""Modal wait dialog shown while a transcription is being generated."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout


class WaitDialog(QDialog):
    """Non-blocking modal dialog with an indeterminate spinner and a message."""

    def __init__(self, message: str = "Generating transcript, please wait", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Please wait")
        self.setModal(True)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        self._label = QLabel(message)
        self._label.setStyleSheet("font-size: 14px; font-weight: 500;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        spinner = QProgressBar()
        spinner.setRange(0, 0)
        spinner.setFixedSize(160, 8)
        layout.addWidget(spinner, alignment=Qt.AlignmentFlag.AlignCenter)

    def finish(self):
        """Close the dialog."""
        self.close()
