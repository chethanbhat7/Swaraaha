from PySide6.QtWidgets import QDialog

from app.ui.language_dialog import LanguageDialog


def test_language_dialog_has_three_buttons(qapp):
    dialog = LanguageDialog()
    assert set(dialog._buttons.keys()) == {"english", "kannada", "hindi"}


def test_language_dialog_prehighlights_current(qapp):
    dialog = LanguageDialog("kannada")
    assert dialog._buttons["kannada"].property("cssClass") == "lang_btn_active"
    assert dialog._buttons["english"].property("cssClass") is None


def test_language_dialog_click_returns_code(qapp):
    dialog = LanguageDialog()
    dialog._buttons["hindi"].click()
    assert dialog.selected() == "hindi"
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_language_dialog_clamps_invalid_current(qapp):
    dialog = LanguageDialog("french")
    assert dialog.selected() == "english"
    assert dialog._buttons["english"].property("cssClass") is not None
    assert all(
        dialog._buttons[c].property("cssClass") is None
        for c in ("kannada", "hindi")
    )
