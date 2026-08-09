"""PDF viewer widget that renders Rainbow Passage with upload and zoom support."""

import os
import shutil

import pypdfium2 as pdfium
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.theme import COLORS

PASSAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "passages")

_ZOOM_STEP = 0.25
_ZOOM_MIN = 0.5
_ZOOM_MAX = 4.0


def _ensure_passages_dir():
    os.makedirs(PASSAGES_DIR, exist_ok=True)


def _render_page_to_pixmap(pdf_doc, page_index, dpi=150):
    page = pdf_doc[page_index]
    bitmap = page.render(scale=dpi / 72)
    pil_image = bitmap.to_pil()
    data = pil_image.tobytes("raw", "RGB")
    q_image = QImage(data, pil_image.width, pil_image.height, 3 * pil_image.width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_image)


class PdfViewer(QWidget):
    pdf_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pdf_path = None
        self._pdf_doc = None
        self._zoom = 1.0
        self._page_pixmaps: list[QPixmap] = []
        self._setup_ui()
        self._load_default_passage()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        header = QLabel("Rainbow Passage")
        header.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px 0;")
        header_row.addWidget(header)
        header_row.addStretch()

        self._zoom_out_btn = QPushButton("-")
        self._zoom_out_btn.setProperty("cssClass", "zoom_btn")
        self._zoom_out_btn.setToolTip("Zoom Out")
        self._zoom_out_btn.clicked.connect(self._zoom_out)
        header_row.addWidget(self._zoom_out_btn)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(54)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zoom_label.setStyleSheet("font-size: 13px; font-weight: 500;")
        header_row.addWidget(self._zoom_label)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setProperty("cssClass", "zoom_btn")
        self._zoom_in_btn.setToolTip("Zoom In")
        self._zoom_in_btn.clicked.connect(self._zoom_in)
        header_row.addWidget(self._zoom_in_btn)

        self._zoom_reset_btn = QPushButton("Reset")
        self._zoom_reset_btn.setProperty("cssClass", "secondary")
        self._zoom_reset_btn.setToolTip("Reset Zoom")
        self._zoom_reset_btn.clicked.connect(self._zoom_reset)
        header_row.addWidget(self._zoom_reset_btn)

        layout.addLayout(header_row)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self._pdf_container = QWidget()
        self._pdf_layout = QVBoxLayout(self._pdf_container)
        self._pdf_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._pdf_layout.setContentsMargins(8, 8, 8, 8)

        self._pdf_label = QLabel("No PDF loaded")
        self._pdf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pdf_label.setStyleSheet(f"color: {COLORS['outline']}; font-size: 14px; padding: 40px;")
        self._pdf_layout.addWidget(self._pdf_label)

        self._scroll_area.setWidget(self._pdf_container)
        layout.addWidget(self._scroll_area, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._upload_btn = QPushButton("Upload PDF")
        self._upload_btn.setProperty("cssClass", "secondary")
        self._upload_btn.setFixedWidth(140)
        self._upload_btn.clicked.connect(self._on_upload)
        btn_row.addWidget(self._upload_btn)

        layout.addLayout(btn_row)

    def _load_default_passage(self):
        _ensure_passages_dir()
        pdfs = sorted(
            f for f in os.listdir(PASSAGES_DIR) if f.lower().endswith(".pdf")
        )
        if pdfs:
            self.load_pdf(os.path.join(PASSAGES_DIR, pdfs[0]))

    def load_pdf(self, path: str):
        if not os.path.isfile(path):
            return
        try:
            self._pdf_doc = pdfium.PdfDocument(path)
            self._current_pdf_path = path
            self._zoom = 1.0
            self._render_all_pages()
            self.pdf_selected.emit(path)
        except Exception as e:
            self._pdf_label.setText(f"Failed to load PDF: {e}")

    def _render_all_pages(self):
        while self._pdf_layout.count():
            item = self._pdf_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._page_pixmaps.clear()

        if self._pdf_doc is None:
            return

        base_dpi = 150
        for i in range(len(self._pdf_doc)):
            pixmap = _render_page_to_pixmap(self._pdf_doc, i, dpi=base_dpi)
            self._page_pixmaps.append(pixmap)

        self._apply_zoom()
        self._zoom_label.setText(f"{int(self._zoom * 100)}%")

    def _apply_zoom(self):
        while self._pdf_layout.count():
            item = self._pdf_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for pixmap in self._page_pixmaps:
            scaled = pixmap.scaled(
                int(pixmap.width() * self._zoom),
                int(pixmap.height() * self._zoom),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            page_label = QLabel()
            page_label.setPixmap(scaled)
            page_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            self._pdf_layout.addWidget(page_label)

        self._pdf_layout.addStretch()
        self._zoom_label.setText(f"{int(self._zoom * 100)}%")

    def _zoom_in(self):
        if self._zoom < _ZOOM_MAX:
            self._zoom = round(min(self._zoom + _ZOOM_STEP, _ZOOM_MAX), 2)
            self._apply_zoom()

    def _zoom_out(self):
        if self._zoom > _ZOOM_MIN:
            self._zoom = round(max(self._zoom - _ZOOM_STEP, _ZOOM_MIN), 2)
            self._apply_zoom()

    def _zoom_reset(self):
        self._zoom = 1.0
        self._apply_zoom()

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Upload Rainbow Passage", "", "PDF Files (*.pdf);;All Files (*)"
        )
        if path:
            _ensure_passages_dir()
            dest = os.path.join(PASSAGES_DIR, os.path.basename(path))
            shutil.copy2(path, dest)
            self.load_pdf(dest)

    def get_current_pdf_path(self) -> str | None:
        return self._current_pdf_path
