import pytest

from app.ui.styles import build_stylesheet
from app.ui.theme import COLORS, set_theme


@pytest.fixture(autouse=True)
def reset_theme():
    yield
    set_theme(False)


def test_stylesheet_has_table_scrollbar_inset():
    sheet = build_stylesheet()
    assert "QTableWidget QScrollBar:vertical" in sheet
    assert "margin: 2px 6px 2px 0" in sheet
    assert "QTableWidget QScrollBar:horizontal" in sheet


def test_stylesheet_has_theme_button_rule():
    assert 'QPushButton[cssClass="theme_btn"]' in build_stylesheet()


def test_record_button_uses_theme_token():
    sheet = build_stylesheet()
    assert "#8E24AA" not in sheet
    assert COLORS["record"] in sheet


def test_secondary_button_uses_secondary_token():
    sheet = build_stylesheet()
    assert COLORS["secondary"] in sheet


def test_stylesheet_reflects_dark_palette():
    set_theme(True)
    assert "#E23636" in build_stylesheet()
