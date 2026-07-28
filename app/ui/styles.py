"""QSS stylesheet builder using design tokens from theme.py."""

from app.ui.theme import COLORS, RADIUS, SPACING


def build_stylesheet():
    """Build and return the complete QSS stylesheet string."""
    return f"""
    /* === Global === */
    QWidget {{
        font-family: "Google Sans Flex 9pt";
        font-size: 14px;
        color: {COLORS['on_surface']};
        background-color: {COLORS['surface']};
    }}

    /* === Main Window === */
    QMainWindow {{
        background-color: {COLORS['surface']};
    }}

    /* === Menu Bar === */
    QMenuBar {{
        background-color: {COLORS['surface']};
        border-bottom: 1px solid {COLORS['outline']}33;
        padding: {SPACING['sm']}px;
    }}
    QMenuBar::item {{
        padding: {SPACING['sm']}px {SPACING['md']}px;
        border-radius: {RADIUS['sm']}px;
    }}
    QMenuBar::item:selected {{
        background-color: {COLORS['primary_container']};
    }}

    /* === Buttons === */
    QPushButton {{
        background-color: {COLORS['primary']};
        color: {COLORS['on_primary']};
        border: none;
        border-radius: {RADIUS['lg']}px;
        padding: {SPACING['md']}px {SPACING['xl']}px;
        font-size: 16px;
        font-weight: 500;
        min-height: 48px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['primary']}DD;
    }}
    QPushButton:pressed {{
        background-color: {COLORS['primary']}BB;
    }}
    QPushButton:disabled {{
        background-color: {COLORS['outline']}44;
        color: {COLORS['outline']};
    }}

    /* === Secondary Button (e.g., Back) === */
    QPushButton[cssClass="secondary"] {{
        background-color: transparent;
        color: {COLORS['primary']};
        border: 2px solid {COLORS['outline']}44;
        border-radius: {RADIUS['lg']}px;
        padding: {SPACING['sm']}px {SPACING['md']}px;
        font-size: 14px;
        min-height: 36px;
    }}
    QPushButton[cssClass="secondary"]:hover {{
        background-color: {COLORS['primary_container']};
    }}

    /* === Record Button === */
    QPushButton[cssClass="record"] {{
        background-color: #2E7D32;
        border-radius: {RADIUS['lg']}px;
        padding: {SPACING['md']}px {SPACING['xl']}px;
        font-size: 16px;
        font-weight: 500;
        min-height: 48px;
    }}
    QPushButton[cssClass="record"]:hover {{
        background-color: #2E7D32DD;
    }}
    QPushButton[cssClass="record"]:pressed {{
        background-color: #2E7D32BB;
    }}

    /* === Panels === */
    QFrame[cssClass="panel"] {{
        background-color: {COLORS['surface_variant']};
        border-radius: {RADIUS['md']}px;
        padding: {SPACING['md']}px;
    }}

    /* === Tree View (PDF Browser) === */
    QTreeView {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['outline']}33;
        border-radius: {RADIUS['md']}px;
        padding: {SPACING['sm']}px;
        alternate-background-color: {COLORS['surface_variant']};
    }}
    QTreeView::item {{
        padding: {SPACING['sm']}px {SPACING['md']}px;
        border-radius: {RADIUS['sm']}px;
    }}
    QTreeView::item:selected {{
        background-color: {COLORS['primary_container']};
        color: {COLORS['on_surface']};
    }}

    /* === Text Edit (PDF Content) === */
    QTextEdit {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['outline']}33;
        border-radius: {RADIUS['md']}px;
        padding: {SPACING['md']}px;
        font-size: 14px;
        line-height: 1.6;
    }}

    /* === Status Bar === */
    QStatusBar {{
        background-color: {COLORS['surface_variant']};
        border-top: 1px solid {COLORS['outline']}33;
        padding: {SPACING['xs']}px {SPACING['md']}px;
        font-size: 12px;
    }}

    /* === Table Widget (Results) === */
    QTableWidget {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['outline']}33;
        border-radius: {RADIUS['md']}px;
        gridline-color: {COLORS['outline']}22;
    }}
    QTableWidget::item {{
        padding: {SPACING['sm']}px {SPACING['md']}px;
    }}
    QTableWidget::item:selected {{
        background-color: {COLORS['primary_container']};
    }}
    QHeaderView::section {{
        background-color: {COLORS['surface_variant']};
        border: none;
        border-bottom: 2px solid {COLORS['outline']}33;
        padding: {SPACING['sm']}px {SPACING['md']}px;
        font-weight: 600;
    }}

    /* === Scroll Bar === */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {COLORS['outline']}44;
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    """
