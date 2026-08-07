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

    /* === Sidebar Segmented Nav Buttons === */
    QFrame[cssClass="nav_bar"] {{
        background-color: {COLORS['surface_variant']};
        border-radius: {RADIUS['sm']}px;
        padding: 4px;
    }}
    QPushButton[cssClass="nav_btn"] {{
        background-color: transparent;
        color: {COLORS['on_surface']};
        border: none;
        border-radius: {RADIUS['sm']}px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 500;
        min-height: 36px;
    }}
    QPushButton[cssClass="nav_btn"]:hover {{
        background-color: {COLORS['primary_container']}88;
        color: {COLORS['primary']};
    }}
    QPushButton[cssClass="nav_btn_active"] {{
        background-color: {COLORS['surface']};
        color: {COLORS['primary']};
        border: 1px solid {COLORS['outline']}33;
        border-radius: {RADIUS['sm']}px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 600;
        min-height: 36px;
    }}
    QPushButton[cssClass="lang_btn_active"] {{
        background-color: {COLORS['surface_variant']};
        color: {COLORS['primary']};
        border: 2px solid {COLORS['primary']}66;
    }}
    QPushButton[cssClass="lang_btn_active"]:hover {{
        background-color: {COLORS['primary_container']};
    }}

    /* === Tab Widget & Tab Bar === */
    QTabWidget::pane {{
        border: 1px solid {COLORS['outline']}33;
        border-radius: {RADIUS['md']}px;
        background-color: {COLORS['surface']};
        top: -1px;
    }}

    QTabBar::tab {{
        background-color: {COLORS['surface_variant']};
        color: {COLORS['on_surface']};
        border: 1px solid {COLORS['outline']}33;
        border-bottom: none;
        border-top-left-radius: {RADIUS['sm']}px;
        border-top-right-radius: {RADIUS['sm']}px;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 500;
        margin-right: 4px;
    }}

    QTabBar::tab:selected {{
        background-color: {COLORS['surface']};
        color: {COLORS['primary']};
        font-weight: 600;
        border-top: 3px solid {COLORS['primary']};
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {COLORS['primary_container']}88;
        color: {COLORS['primary']};
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

    /* === Zoom Button === */
    QPushButton[cssClass="zoom_btn"] {{
        background-color: {COLORS['surface_variant']};
        color: {COLORS['on_surface']};
        border: 1px solid {COLORS['outline']}44;
        border-radius: {RADIUS['sm']}px;
        padding: 0px;
        font-size: 18px;
        font-weight: bold;
        min-width: 32px;
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
    }}
    QPushButton[cssClass="zoom_btn"]:hover {{
        background-color: {COLORS['primary_container']};
        color: {COLORS['primary']};
        border-color: {COLORS['primary']};
    }}
    QPushButton[cssClass="zoom_btn"]:pressed {{
        background-color: {COLORS['primary_container']}CC;
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

    /* === List Widget === */
    QListWidget {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['outline']}33;
        border-radius: {RADIUS['md']}px;
        padding: {SPACING['xs']}px;
    }}
    QListWidget::item {{
        padding: {SPACING['sm']}px {SPACING['md']}px;
        border-radius: {RADIUS['sm']}px;
    }}
    QListWidget::item:hover {{
        background-color: {COLORS['surface_variant']};
    }}
    QListWidget::item:selected {{
        background-color: {COLORS['primary_container']};
        color: {COLORS['primary']};
    }}

    /* === Status Bar === */
    QStatusBar {{
        background-color: {COLORS['surface_variant']};
        border-top: 1px solid {COLORS['outline']}33;
        padding: {SPACING['xs']}px {SPACING['md']}px;
        font-size: 12px;
    }}

    /* === Table Widget (Results & Transcription) === */
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

    /* === Progress Bar (Wait Dialog Spinner) === */
    QProgressBar {{
        background-color: {COLORS['surface_variant']};
        border: none;
        border-radius: 4px;
    }}
    QProgressBar::chunk {{
        background-color: {COLORS['primary']};
        border-radius: 4px;
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
