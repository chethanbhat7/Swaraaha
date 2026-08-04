"""Home page: Passage/Files tabs + audio controls."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QTabWidget, QWidget

from app.ui.audio_controls import AudioControls
from app.ui.file_panel import FilePanel
from app.ui.pdf_viewer import PdfViewer
from app.ui.transcription_panel import TranscriptionPanel


class HomePage(QWidget):
    record_clicked = Signal()
    stop_clicked = Signal()
    load_clicked = Signal()
    play_clicked = Signal()
    analyze_clicked = Signal()
    file_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._pdf_viewer = PdfViewer()
        self._tabs.addTab(self._pdf_viewer, "Passage")

        self._file_panel = FilePanel()
        self._tabs.addTab(self._file_panel, "Files")

        self._transcription_panel = TranscriptionPanel()
        self._tabs.addTab(self._transcription_panel, "Transcription")

        splitter.addWidget(self._tabs)

        self._audio_controls = AudioControls()
        splitter.addWidget(self._audio_controls)

        splitter.setSizes([500, 500])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        self._file_panel.file_selected.connect(self.file_selected)
        self._audio_controls.record_clicked.connect(self.record_clicked)
        self._audio_controls.stop_clicked.connect(self.stop_clicked)
        self._audio_controls.load_clicked.connect(self.load_clicked)
        self._audio_controls.play_clicked.connect(self.play_clicked)
        self._audio_controls.analyze_clicked.connect(self.analyze_clicked)
        self._transcription_panel.transcribe_requested.connect(self.load_clicked)

    def get_pdf_viewer(self) -> PdfViewer:
        return self._pdf_viewer

    def get_file_panel(self) -> FilePanel:
        return self._file_panel

    def get_transcription_panel(self) -> TranscriptionPanel:
        return self._transcription_panel

    def get_audio_controls(self) -> AudioControls:
        return self._audio_controls
