"""Shared table sizing helpers."""

from PySide6.QtWidgets import QTableWidget

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
