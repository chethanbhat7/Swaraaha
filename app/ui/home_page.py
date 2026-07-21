"""Home page: Rainbow Passage PDF viewer + audio controls."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QSplitter
from PySide6.QtCore import Signal, Qt

from app.ui.pdf_viewer import PdfViewer
from app.ui.audio_controls import AudioControls


class HomePage(QWidget):
    record_clicked = Signal()
    stop_clicked = Signal()
    load_clicked = Signal()
    play_clicked = Signal()
    analyze_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._pdf_viewer = PdfViewer()
        splitter.addWidget(self._pdf_viewer)

        self._audio_controls = AudioControls()
        splitter.addWidget(self._audio_controls)

        splitter.setSizes([400, 600])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        self._audio_controls.record_clicked.connect(self.record_clicked)
        self._audio_controls.stop_clicked.connect(self.stop_clicked)
        self._audio_controls.load_clicked.connect(self.load_clicked)
        self._audio_controls.play_clicked.connect(self.play_clicked)
        self._audio_controls.analyze_clicked.connect(self.analyze_clicked)

    def get_pdf_viewer(self) -> PdfViewer:
        return self._pdf_viewer

    def get_audio_controls(self) -> AudioControls:
        return self._audio_controls
