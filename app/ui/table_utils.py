"""Shared table sizing helpers."""

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from app.ui.theme import COLORS

MAX_HEIGHT_UNCAP = 16777215


def resize_table_to_contents(table: QTableWidget):
    """Set a table's minimum height so every row is visible without scrolling."""
    height = table.horizontalHeader().height()
    for row in range(table.rowCount()):
        height += table.rowHeight(row)
    table.setMinimumHeight(height + 4)


def cap_table_height(table: QTableWidget, max_rows: int):
    """Cap a table's maximum height so longer contents scroll inside the table.

    The minimum height is set too so the table claims space for the rows it
    currently shows (up to ``max_rows``) instead of collapsing in tight layouts.
    """
    header_height = table.horizontalHeader().height() or 30
    row_height = table.rowHeight(0) if table.rowCount() > 0 else 30
    row_count = table.rowCount()
    visible = min(row_count, max(1, max_rows)) if row_count > 0 else 0
    table.setMinimumHeight(header_height + visible * row_height + 4)
    table.setMaximumHeight(header_height + max(1, max_rows) * row_height + 4)


def populate_transcript_table(table: QTableWidget, words: list) -> None:
    """Populate a word-level transcript table with rows and stutter styling.

    words: list of dicts with keys word, start_sec, end_sec, confidence, stutter.
    """
    table.setUpdatesEnabled(False)
    try:
        table.setRowCount(len(words))
        for row, w in enumerate(words):
            word_item = QTableWidgetItem(str(w.get("word", "")))
            start_item = QTableWidgetItem(f"{w.get('start_sec', 0.0):.2f}")
            end_item = QTableWidgetItem(f"{w.get('end_sec', 0.0):.2f}")
            conf_item = QTableWidgetItem(f"{w.get('confidence', 0.0)*100:.0f}%")

            is_stutter = w.get("stutter", False)
            status_str = "Stutter Detected" if is_stutter else "Normal"
            status_item = QTableWidgetItem(status_str)

            if is_stutter:
                red_color = QColor(COLORS["dysfluency"]["prolongation"])
                status_item.setForeground(red_color)
                word_item.setForeground(red_color)
                font = word_item.font()
                font.setBold(True)
                word_item.setFont(font)
                status_item.setFont(font)

            table.setItem(row, 0, word_item)
            table.setItem(row, 1, start_item)
            table.setItem(row, 2, end_item)
            table.setItem(row, 3, conf_item)
            table.setItem(row, 4, status_item)
    finally:
        table.setUpdatesEnabled(True)
