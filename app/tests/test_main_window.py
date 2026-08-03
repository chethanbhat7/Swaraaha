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


def test_load_path_sets_audio_and_state(qapp, app_name, tmp_path):
    path = tmp_path / "a.wav"
    _make_wav(path)
    win = MainWindow()
    win._load_path(str(path))
    assert win._current_audio is not None
    assert "Loaded" in win.statusBar().currentMessage()
    assert not win._home_page.get_audio_controls()._analyze_btn.isHidden()


def test_load_path_reports_error(qapp, app_name, tmp_path):
    win = MainWindow()
    win._load_path(str(tmp_path / "missing.wav"))
    assert win._current_audio is None
    assert "Error loading file" in win.statusBar().currentMessage()


def test_file_selected_wires_to_load(qapp, app_name, tmp_path):
    path = tmp_path / "a.wav"
    _make_wav(path)
    win = MainWindow()
    win._home_page.file_selected.emit(str(path))
    assert win._current_audio is not None


def test_drop_accepts_only_audio(qapp, app_name, tmp_path):
    from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent

    win = MainWindow()
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
