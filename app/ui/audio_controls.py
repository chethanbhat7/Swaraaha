"""Audio control buttons: Record, Stop, Load, Play, Analyze."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from app.ui.theme import COLORS


class AudioControls(QWidget):
    record_clicked = Signal()
    stop_clicked = Signal()
    load_clicked = Signal()
    play_clicked = Signal()
    analyze_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(f"font-size: 12px; color: {COLORS['outline']};")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._record_btn = QPushButton("Record Audio")
        self._record_btn.setProperty("cssClass", "record")
        self._record_btn.clicked.connect(self.record_clicked.emit)
        btn_row.addWidget(self._record_btn)

        self._load_btn = QPushButton("Load Audio")
        self._load_btn.clicked.connect(self.load_clicked.emit)
        btn_row.addWidget(self._load_btn)

        layout.addLayout(btn_row)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("cssClass", "secondary")
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self._stop_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self._play_btn = QPushButton("Play")
        self._play_btn.setProperty("cssClass", "secondary")
        self._play_btn.setVisible(False)
        self._play_btn.clicked.connect(self.play_clicked.emit)
        layout.addWidget(self._play_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setVisible(False)
        self._analyze_btn.clicked.connect(self.analyze_clicked.emit)
        layout.addWidget(self._analyze_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def set_recording(self, recording: bool):
        self._record_btn.setVisible(not recording)
        self._stop_btn.setVisible(recording)
        self._load_btn.setVisible(not recording)
        self._play_btn.setVisible(False)
        self._analyze_btn.setVisible(False)
        self._status_label.setText("Recording..." if recording else "Ready")

    def set_audio_loaded(self):
        self._record_btn.setVisible(True)
        self._stop_btn.setVisible(False)
        self._load_btn.setVisible(True)
        self._play_btn.setVisible(True)
        self._analyze_btn.setVisible(True)
        self._status_label.setText("Audio loaded — ready to analyze")

    def set_playing(self, playing: bool):
        self._play_btn.setText("Stop" if playing else "Play")
        self._status_label.setText("Playing..." if playing else "Ready")
