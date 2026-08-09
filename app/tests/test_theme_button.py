import pytest
from PySide6.QtWidgets import QPushButton, QToolBar

from app.ui.main_window import MainWindow
from app.ui.theme import is_dark_mode, set_theme


@pytest.fixture(autouse=True)
def reset_theme():
    set_theme(False)
    yield
    set_theme(False)


def test_toolbar_and_action_removed(qapp):
    win = MainWindow()
    assert win.findChild(QToolBar) is None


def test_theme_button_exists_as_circular_pushbutton(qapp):
    win = MainWindow()
    btn = win._theme_btn
    assert isinstance(btn, QPushButton)
    assert btn.width() == 44 and btn.height() == 44


def test_theme_button_shows_moon_in_light(qapp):
    win = MainWindow()
    assert win._theme_btn.text() == "🌙"
    assert win._theme_btn.toolTip() == "Switch to Dark Mode"


def test_theme_button_toggles_theme_and_glyph(qapp):
    win = MainWindow()
    win._theme_btn.click()
    assert is_dark_mode() is True
    assert win._theme_btn.text() == "☀️"
    assert win._theme_btn.toolTip() == "Switch to Light Mode"
    win._theme_btn.click()
    assert is_dark_mode() is False
    assert win._theme_btn.text() == "🌙"


def test_theme_button_anchored_bottom_right(qapp):
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    btn = win._theme_btn
    cw = win.centralWidget()
    assert cw.x() + cw.width() - btn.geometry().right() - 1 == 24
    assert cw.y() + cw.height() - btn.geometry().bottom() - 1 == 24
