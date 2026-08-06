import threading

import numpy as np
import pytest
import soundfile as sf

from app.ui.main_window import MainWindow


@pytest.fixture
def app_name(qapp):
    qapp.setOrganizationName("SwaraahaTests")
    qapp.setApplicationName("SwaraahaTests")
    return qapp


def _make_wav(path, sr=16000):
    sf.write(str(path), np.zeros(sr, dtype=np.float32), sr)


def _no_background_transcription(win, monkeypatch):
    monkeypatch.setattr(win, "_start_home_transcription", lambda audio, language: None)


def test_load_path_sets_audio_and_state(qapp, app_name, tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    _make_wav(path)
    win = MainWindow()
    _no_background_transcription(win, monkeypatch)
    win._load_path(str(path))
    assert win._current_audio is not None
    assert "Loaded" in win.statusBar().currentMessage()
    assert not win._home_page.get_audio_controls()._analyze_btn.isHidden()
    assert not win._home_page.get_transcription_panel().isHidden()


def test_load_path_reports_error(qapp, app_name, tmp_path):
    win = MainWindow()
    win._load_path(str(tmp_path / "missing.wav"))
    assert win._current_audio is None
    assert "Error loading file" in win.statusBar().currentMessage()


def test_file_selected_wires_to_load(qapp, app_name, tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    _make_wav(path)
    win = MainWindow()
    _no_background_transcription(win, monkeypatch)
    win._home_page.file_selected.emit(str(path))
    assert win._current_audio is not None


def test_drop_accepts_only_audio(qapp, app_name, tmp_path, monkeypatch):
    from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent

    win = MainWindow()
    _no_background_transcription(win, monkeypatch)
    wav = tmp_path / "a.wav"
    _make_wav(wav)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(wav))])
    enter = QDragEnterEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
    )
    win.dragEnterEvent(enter)
    assert enter.isAccepted()

    bad = tmp_path / "notes.txt"
    bad.write_bytes(b"x")
    mime2 = QMimeData()
    mime2.setUrls([QUrl.fromLocalFile(str(bad))])
    enter2 = QDragEnterEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, mime2,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
    )
    win.dragEnterEvent(enter2)
    assert not enter2.isAccepted()


def test_analysis_worker_passes_language(qapp):
    from app.ui.main_window import AnalysisWorker

    captured = {}

    class FakeRunner:
        def analyze(self, audio, language="english"):
            captured["language"] = language
            return {"ok": True}

    results = []
    worker = AnalysisWorker(FakeRunner(), np.zeros(1600, dtype=np.float32), "hindi")
    worker.finished.connect(results.append)
    worker.run()

    assert captured["language"] == "hindi"
    assert results[0] == {"ok": True}


def test_stale_analysis_signal_does_not_clear_current_worker(qapp, app_name):
    from app.ui.main_window import AnalysisWorker

    class FakeRunner:
        def analyze(self, audio, language="english"):
            return {"ok": True}

    win = MainWindow()
    win._model_runner = FakeRunner()
    win._current_audio = np.zeros(1600, dtype=np.float32)
    win._current_language = "english"

    win._on_analyze()
    stale = win._worker
    stale.wait(5000)

    fresh = AnalysisWorker(FakeRunner(), np.zeros(1600, dtype=np.float32), "english")
    win._worker = fresh

    qapp.processEvents()

    assert win._worker is fresh
    assert win._stack.currentIndex() == 0


def test_start_home_transcription_populates_transcript(qapp, app_name):
    win = MainWindow()

    def fake_transcribe(audio, sample_rate=16000, localizations=None, passage_text=None, language="english"):
        return {
            "text": "hello world",
            "words": [
                {"word": "hello", "start_sec": 0.0, "end_sec": 0.4, "confidence": 0.9,
                 "stutter": False, "stutter_type": None},
                {"word": "world", "start_sec": 0.4, "end_sec": 0.8, "confidence": 0.8,
                 "stutter": False, "stutter_type": None},
            ],
        }

    win._model_runner.transcriber.transcribe = fake_transcribe
    win._start_home_transcription(np.zeros(1600, dtype=np.float32), "english")

    assert win._wait_dialog is not None
    worker = win._transcription_worker
    worker.wait(5000)
    qapp.processEvents()

    assert win._wait_dialog is None
    panel = win._home_page.get_transcription_panel()
    assert panel._text_edit.toPlainText() == "hello world"
    assert panel._table.rowCount() == 2


def test_home_transcription_error_is_handled(qapp, app_name):
    win = MainWindow()

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    win._model_runner.transcriber.transcribe = _raise
    win._start_home_transcription(np.zeros(1600, dtype=np.float32), "english")
    win._transcription_worker.wait(5000)
    qapp.processEvents()

    assert win._wait_dialog is None
    assert "Transcription failed" in win.statusBar().currentMessage()


def test_reload_dismisses_previous_wait_dialog_and_discards_stale_result(qapp, app_name):
    win = MainWindow()
    release_first = threading.Event()

    def fake_transcribe(audio, sample_rate=16000, localizations=None, passage_text=None, language="english"):
        if len(audio) == 1600:
            release_first.wait(5)
            return {
                "text": "first result",
                "words": [
                    {"word": "first", "start_sec": 0.0, "end_sec": 0.4, "confidence": 0.9,
                     "stutter": False, "stutter_type": None},
                ],
            }
        return {
            "text": "second result",
            "words": [
                {"word": "second", "start_sec": 0.0, "end_sec": 0.4, "confidence": 0.9,
                 "stutter": False, "stutter_type": None},
            ],
        }

    win._model_runner.transcriber.transcribe = fake_transcribe

    win._start_home_transcription(np.zeros(1600, dtype=np.float32), "english")
    first_dialog = win._wait_dialog
    assert first_dialog is not None
    assert first_dialog.isVisible()

    win._start_home_transcription(np.zeros(3200, dtype=np.float32), "english")
    second_dialog = win._wait_dialog
    assert second_dialog is not first_dialog
    assert not first_dialog.isVisible()
    assert second_dialog.isVisible()

    release_first.set()
    qapp.processEvents()
    qapp.processEvents()

    assert win._wait_dialog is None
    panel = win._home_page.get_transcription_panel()
    assert panel._text_edit.toPlainText() == "second result"


def test_prompt_language_uses_dialog(qapp, app_name, monkeypatch):
    win = MainWindow()

    class FakeDialog:
        def __init__(self, current, parent=None):
            self._current = current

        def exec(self):
            return 0

        def selected(self):
            return "hindi"

    monkeypatch.setattr("app.ui.main_window.LanguageDialog", FakeDialog)
    assert win._prompt_language() == "hindi"


def test_on_load_prompts_then_loads(qapp, app_name, tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    sf.write(str(path), np.zeros(1600, dtype=np.float32), 16000)
    win = MainWindow()
    monkeypatch.setattr("app.ui.main_window.QFileDialog.getOpenFileName", lambda *a, **k: (str(path), ""))
    monkeypatch.setattr(win, "_prompt_language", lambda: "kannada")
    _no_background_transcription(win, monkeypatch)
    win._on_load()
    assert win._current_language == "kannada"
    assert win._current_audio is not None
