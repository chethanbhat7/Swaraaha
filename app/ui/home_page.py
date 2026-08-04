"""Home page: Passage/Files segmented nav + audio controls with compact transcript."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.ui.audio_controls import AudioControls
from app.ui.compact_transcript import CompactTranscript
from app.ui.file_panel import FilePanel
from app.ui.pdf_viewer import PdfViewer


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

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self._nav_bar = QFrame()
        self._nav_bar.setProperty("cssClass", "nav_bar")
        nav_layout = QHBoxLayout(self._nav_bar)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)

        self._passage_btn = QPushButton("Passage")
        self._passage_btn.setProperty("cssClass", "nav_btn_active")
        self._passage_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._passage_btn.clicked.connect(lambda: self._switch_page(0))
        nav_layout.addWidget(self._passage_btn)

        self._files_btn = QPushButton("Files")
        self._files_btn.setProperty("cssClass", "nav_btn")
        self._files_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._files_btn.clicked.connect(lambda: self._switch_page(1))
        nav_layout.addWidget(self._files_btn)

        left_layout.addWidget(self._nav_bar)

        self._stack = QStackedWidget()

        self._pdf_viewer = PdfViewer()
        self._stack.addWidget(self._pdf_viewer)

        self._file_panel = FilePanel()
        self._stack.addWidget(self._file_panel)

        left_layout.addWidget(self._stack, stretch=1)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setHandleWidth(1)

        self._audio_controls = AudioControls()

        audio_holder = QWidget()
        audio_layout = QVBoxLayout(audio_holder)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        audio_layout.addWidget(self._audio_controls)

        self._transcript = CompactTranscript()
        self._transcript.setVisible(False)

        self._right_splitter.addWidget(audio_holder)
        self._right_splitter.addWidget(self._transcript)
        self._right_splitter.setSizes([500, 400])

        right_layout.addWidget(self._right_splitter)

        splitter.addWidget(right)

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

    def _switch_page(self, index: int):
        self._stack.setCurrentIndex(index)
        self._passage_btn.setProperty("cssClass", "nav_btn_active" if index == 0 else "nav_btn")
        self._files_btn.setProperty("cssClass", "nav_btn_active" if index == 1 else "nav_btn")
        self._passage_btn.style().unpolish(self._passage_btn)
        self._passage_btn.style().polish(self._passage_btn)
        self._files_btn.style().unpolish(self._files_btn)
        self._files_btn.style().polish(self._files_btn)

    def set_transcript_visible(self, visible: bool):
        self._transcript.setVisible(visible)
        self._right_splitter.setSizes([500, 400] if visible else [900, 0])

    def get_pdf_viewer(self) -> PdfViewer:
        return self._pdf_viewer

    def get_file_panel(self) -> FilePanel:
        return self._file_panel

    def get_transcription_panel(self) -> CompactTranscript:
        return self._transcript

    def get_audio_controls(self) -> AudioControls:
        return self._audio_controls
