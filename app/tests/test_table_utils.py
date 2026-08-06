from PySide6.QtWidgets import QTableWidget

from app.ui.table_utils import cap_table_height


def test_cap_table_height_limits_max(qapp):
    table = QTableWidget(20, 3)
    table.setHorizontalHeaderLabels(["A", "B", "C"])
    table.resize(400, 800)
    table.show()
    qapp.processEvents()

    header = table.horizontalHeader().height()
    row = table.rowHeight(0)

    cap_table_height(table, 8)

    full = header + 20 * row
    assert table.maximumHeight() < full
    assert table.maximumHeight() > header


def test_cap_table_height_empty_table_uses_fallback(qapp):
    table = QTableWidget(0, 3)
    table.show()
    qapp.processEvents()
    cap_table_height(table, 5)
    header = table.horizontalHeader().height() or 30
    assert table.maximumHeight() == header + 5 * 30 + 4


def test_cap_table_height_clamps_max_rows(qapp):
    table = QTableWidget(20, 3)
    table.show()
    qapp.processEvents()
    header = table.horizontalHeader().height() or 30
    row = table.rowHeight(0)
    cap_table_height(table, 0)
    assert table.maximumHeight() == header + 1 * row + 4
