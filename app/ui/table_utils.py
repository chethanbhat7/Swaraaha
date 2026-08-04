"""Shared table sizing helpers."""

from PySide6.QtWidgets import QTableWidget


def resize_table_to_contents(table: QTableWidget):
    """Set a table's minimum height so every row is visible without scrolling."""
    height = table.horizontalHeader().height()
    for row in range(table.rowCount()):
        height += table.rowHeight(row)
    table.setMinimumHeight(height + 4)
