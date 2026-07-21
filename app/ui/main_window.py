"""Main window with QStackedWidget for page navigation."""

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QFileDialog,
)
from PySide6.QtCore import QThread, Signal

from app.ui.home_page import HomePage
from app.ui.analysis_page import AnalysisPage
from app.core.audio_handler import AudioHandler
from app.core.model_runner import ModelRunner


class AnalysisWorker(QThread):
    finished = Signal(dict)

    def __init__(self, model_runner: ModelRunner, audio: np.ndarray):
        super().__init__()
        self._model_runner = model_runner
        self._audio = audio

    def run(self):
        try:
            results = self._model_runner.analyze(self._audio)
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit({"error": str(e)})


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swaraaha — Speech Dysfluency Detection")
        self.setMinimumSize(1200, 800)

        self._audio_handler = AudioHandler()
        self._model_runner = ModelRunner()
        self._current_audio = None
        self._worker = None

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._home_page = HomePage()
        self._stack.addWidget(self._home_page)

        self._analysis_page = AnalysisPage()
        self._stack.addWidget(self._analysis_page)

        self._home_page.record_clicked.connect(self._on_record)
        self._home_page.stop_clicked.connect(self._on_stop)
        self._home_page.load_clicked.connect(self._on_load)
        self._home_page.play_clicked.connect(self._on_play)
        self._home_page.analyze_clicked.connect(self._on_analyze)

        self._analysis_page.back_clicked.connect(self._go_home)

    def _on_record(self):
        self._audio_handler.start_recording()
        self._home_page.get_audio_controls().set_recording(True)
        self.statusBar().showMessage("Recording...")

    def _on_stop(self):
        audio = self._audio_handler.stop_recording()
        if len(audio) > 0:
            self._current_audio = audio
            self._home_page.get_audio_controls().set_recording(False)
            self._home_page.get_audio_controls().set_audio_loaded()
            self.statusBar().showMessage(f"Recorded {len(audio) / self._audio_handler.sample_rate:.1f}s of audio")
        else:
            self._home_page.get_audio_controls().set_recording(False)
            self.statusBar().showMessage("Ready")

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Audio", "", "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        if path:
            try:
                audio = self._audio_handler.load_audio(path)
                self._current_audio = audio
                self._home_page.get_audio_controls().set_audio_loaded()
                self.statusBar().showMessage(f"Loaded: {path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error loading file: {e}")

    def _on_play(self):
        if self._current_audio is not None:
            self._audio_handler.play_audio(self._current_audio)
            self._home_page.get_audio_controls().set_playing(True)
            self.statusBar().showMessage("Playing...")

    def _on_analyze(self):
        if self._current_audio is None:
            self.statusBar().showMessage("No audio to analyze")
            return
        if self._worker and self._worker.isRunning():
            self.statusBar().showMessage("Analysis already in progress...")
            return

        self.statusBar().showMessage("Analyzing audio...")
        self._worker = AnalysisWorker(self._model_runner, self._current_audio)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.start()

    def _on_analysis_done(self, results: dict):
        if "error" in results:
            self.statusBar().showMessage(f"Analysis failed: {results['error']}")
            return
        self._analysis_page.set_results(results, self._current_audio)
        self._stack.setCurrentIndex(1)
        self.statusBar().showMessage("Analysis complete")

    def _go_home(self):
        self._stack.setCurrentIndex(0)
        self.statusBar().showMessage("Ready")
