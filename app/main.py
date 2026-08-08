"""Swaraaha Desktop Application — Entry Point."""

import sys

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.ui.styles import build_stylesheet
from app.ui.theme import load_fonts


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Swaraaha")
    app.setOrganizationName("Swaraaha")

    loaded_fonts = load_fonts()
    if loaded_fonts:
        print(f"Loaded fonts: {', '.join(loaded_fonts)}")
    else:
        print("Warning: No custom fonts loaded. Using system defaults.")

    app.setStyleSheet(build_stylesheet())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
