"""Main window with QStackedWidget for page navigation."""

import os

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.audio_handler import AudioHandler
from app.core.model_runner import ModelRunner
from app.ui.analysis_page import AnalysisPage
from app.ui.home_page import HomePage
from app.ui.language_dialog import LanguageDialog
from app.ui.styles import build_stylesheet
from app.ui.theme import is_dark_mode, set_theme
from app.ui.transcription_worker import TranscriptionWorker
from app.ui.wait_dialog import WaitDialog

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


class AnalysisWorker(QThread):
    finished = Signal(dict)

    def __init__(self, model_runner: ModelRunner, audio: np.ndarray, language: str = "english"):
        super().__init__()
        self._model_runner = model_runner
        self._audio = audio
        self._language = language

    def run(self):
        try:
            results = self._model_runner.analyze(self._audio, language=self._language)
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
        self._current_filename = ""
        self._worker = None
        self._current_language = "english"
        self._transcription_worker = None
        self._wait_dialog = None

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
        self._home_page.file_selected.connect(self._on_file_selected)

        self._analysis_page.back_clicked.connect(self._go_home)

        theme_action = self.menuBar().addAction("Toggle Dark Mode")
        theme_action.triggered.connect(self._toggle_theme)

        self.setAcceptDrops(True)

    def _on_file_selected(self, path: str):
        self._load_path(path)

    def _load_path(self, path: str):
        try:
            audio = self._audio_handler.load_audio(path)
            self._current_audio = audio
            self._current_filename = os.path.basename(path)
            self._home_page.get_audio_controls().set_audio_loaded()
            self._home_page.get_transcription_panel().clear()
            self._home_page.set_transcript_visible(True)
            self._start_home_transcription(audio, self._current_language)
            self.statusBar().showMessage(f"Loaded: {path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading file: {e}")

    def _on_record(self):
        self._current_language = self._prompt_language()
        self._audio_handler.start_recording()
        self._home_page.get_audio_controls().set_recording(True)
        self.statusBar().showMessage("Recording...")

    def _on_stop(self):
        audio = self._audio_handler.stop_recording()
        if len(audio) > 0:
            self._current_audio = audio
            self._current_filename = "recording.wav"
            self._home_page.get_audio_controls().set_recording(False)
            self._home_page.get_audio_controls().set_audio_loaded()
            self._home_page.get_transcription_panel().clear()
            self._home_page.set_transcript_visible(True)
            self._start_home_transcription(audio, self._current_language)
            self.statusBar().showMessage(f"Recorded {len(audio) / self._audio_handler.sample_rate:.1f}s of audio")
        else:
            self._home_page.get_audio_controls().set_recording(False)
            self.statusBar().showMessage("Ready")

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Audio", "", "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        if path:
            self._current_language = self._prompt_language()
            self._load_path(path)

    def _prompt_language(self) -> str:
        """Ask the user which language to transcribe in; keeps the last choice on cancel."""
        dialog = LanguageDialog(self._current_language, self)
        dialog.exec()
        return dialog.selected()

    def _start_home_transcription(self, audio: np.ndarray, language: str):
        """Run home-page transcription in the background with a wait dialog."""
        if self._wait_dialog is not None:
            self._wait_dialog.finish()
            self._wait_dialog = None
        worker = TranscriptionWorker(self._model_runner.transcriber, audio, language)
        self._transcription_worker = worker
        self._wait_dialog = WaitDialog(self)
        self._wait_dialog.show()
        worker.finished.connect(lambda data, w=worker: self._on_home_transcription_done(data, w))
        worker.start()

    def _on_home_transcription_done(self, data: dict, worker):
        if worker is not self._transcription_worker:
            return
        self._transcription_worker = None
        worker.deleteLater()
        if self._wait_dialog is not None:
            self._wait_dialog.finish()
            self._wait_dialog = None
        if "error" in data:
            self.statusBar().showMessage(f"Transcription failed: {data['error']}")
            return
        self._home_page.get_transcription_panel().set_transcription(data)
        self.statusBar().showMessage("Transcript ready")

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
        self._worker = AnalysisWorker(self._model_runner, self._current_audio, self._current_language)
        self._worker.finished.connect(lambda results, w=self._worker: self._on_analysis_done(results, w))
        self._worker.start()

    def _on_analysis_done(self, results: dict, worker):
        if worker is not self._worker:
            return
        self._worker.deleteLater()
        self._worker = None
        if "error" in results:
            self.statusBar().showMessage(f"Analysis failed: {results['error']}")
            return
        self._analysis_page.set_results(
            results,
            self._current_audio,
            filename=self._current_filename,
            language=self._current_language,
        )
        self._stack.setCurrentIndex(1)
        self.statusBar().showMessage("Analysis complete")

    def _go_home(self):
        self._stack.setCurrentIndex(0)
        self.statusBar().showMessage("Ready")

    def _toggle_theme(self):
        set_theme(not is_dark_mode())
        self.setStyleSheet(build_stylesheet())

    def dragEnterEvent(self, event):
        if self._first_audio_path(event.mimeData()) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._first_audio_path(event.mimeData())
        if path:
            self._load_path(path)
            self._home_page.get_file_panel().add_recent(path)

    @staticmethod
    def _first_audio_path(mime) -> str | None:
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile():
                local = url.toLocalFile()
                if os.path.splitext(local)[1].lower() in _AUDIO_EXTENSIONS:
                    return local
        return None
