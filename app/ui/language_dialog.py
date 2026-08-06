"""Modal language selection dialog for transcription (English/Kannada/Hindi)."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

LANGUAGES = {
    "English": "english",
    "ಕನ್ನಡ (Kannada)": "kannada",
    "हिंदी (Hindi)": "hindi",
}


class LanguageDialog(QDialog):
    """Modal dialog asking the user to choose the transcription language."""

    def __init__(self, current: str = "english", parent=None):
        super().__init__(parent)
        self._selected = current if current in LANGUAGES.values() else "english"
        self.setWindowTitle("Choose Language")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Choose the language")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._buttons = {}
        for label, code in LANGUAGES.items():
            btn = QPushButton(label)
            if code == self._selected:
                btn.setProperty("cssClass", "lang_btn_active")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, c=code: self._choose(c))
            self._buttons[code] = btn
            layout.addWidget(btn)

    def _choose(self, code: str):
        self._selected = code
        self.accept()

    def selected(self) -> str:
        """Return the chosen language code ('english' | 'kannada' | 'hindi')."""
        return self._selected
