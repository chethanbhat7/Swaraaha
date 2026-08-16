"""Export the assessment report to PDF via the shared Typst builder. Qt layer only."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class PatientNameDialog(QDialog):
    """Modal dialog prompting for an optional patient name before export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Report")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Export Report")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Patient name (optional)")
        layout.addWidget(self._name_edit)

        buttons = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setProperty("cssClass", "secondary")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.accept)
        buttons.addStretch()
        buttons.addWidget(cancel_btn)
        buttons.addWidget(export_btn)
        layout.addLayout(buttons)

    def selected(self) -> str:
        """Return the trimmed patient name (possibly empty)."""
        return self._name_edit.text().strip()
