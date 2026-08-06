"""Shared table sizing helpers."""

from PySide6.QtWidgets import QTableWidget


def resize_table_to_contents(table: QTableWidget):
    """Set a table's minimum height so every row is visible without scrolling."""
    height = table.horizontalHeader().height()
    for row in range(table.rowCount()):
        height += table.rowHeight(row)
    table.setMinimumHeight(height + 4)


def cap_table_height(table: QTableWidget, max_rows: int):
    """Cap a table's maximum height so longer contents scroll inside the table."""
    header_height = table.horizontalHeader().height() or 30
    row_height = table.rowHeight(0) if table.rowCount() > 0 else 30
    table.setMaximumHeight(header_height + max(1, max_rows) * row_height + 4)
