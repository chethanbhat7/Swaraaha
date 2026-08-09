"""Design tokens and QSS generation for Material 3 Expressive theme."""

import os

from PySide6.QtGui import QFont, QFontDatabase

LIGHT_COLORS = {
    "primary": "#8E6FB8",
    "on_primary": "#FFFFFF",
    "primary_container": "#E8DDF5",
    "surface": "#FBF7FE",
    "surface_variant": "#EDE4F8",
    "on_surface": "#3A2E45",
    "outline": "#9A8CA6",
    "secondary": "#8A9FD8",
    "record": "#E5738F",
    "dysfluency": {
        "prolongation": "#B39DDB",
        "block": "#9FA8DA",
        "soundrep": "#CE93D8",
        "wordrep": "#F48FB1",
        "interjection": "#80CBC4",
    },
}

DARK_COLORS = {
    "primary": "#E23636",
    "on_primary": "#FFFFFF",
    "primary_container": "#4A1E1E",
    "surface": "#141414",
    "surface_variant": "#242424",
    "on_surface": "#F2F2F2",
    "outline": "#6E6E6E",
    "secondary": "#447BBE",
    "record": "#E23636",
    "dysfluency": {
        "prolongation": "#FF5A4D",
        "block": "#C2261E",
        "soundrep": "#FF8A7A",
        "wordrep": "#4D8BD6",
        "interjection": "#7FB2F0",
    },
}

COLORS = dict(LIGHT_COLORS)

_dark_mode = False


def is_dark_mode() -> bool:
    return _dark_mode


def set_theme(dark: bool):
    global _dark_mode, COLORS
    _dark_mode = dark
    COLORS.clear()
    COLORS.update(DARK_COLORS if dark else LIGHT_COLORS)

SPACING = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}

RADIUS = {"sm": 8, "md": 16, "lg": 24, "full": 9999}

FONT_FAMILY = "Google Sans Flex 9pt"


def load_fonts():
    """Load Google Sans Flex fonts from assets/fonts/ directory."""
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts")
    font_files = [
        "GoogleSansFlex-Regular.ttf",
        "GoogleSansFlex-Medium.ttf",
        "GoogleSansFlex-Bold.ttf",
    ]
    loaded = []
    for fname in font_files:
        path = os.path.join(assets_dir, fname)
        if os.path.exists(path):
            font_id = QFontDatabase.addApplicationFont(path)
            if font_id != -1:
                loaded.append(fname)
    return loaded


def get_font(weight="regular"):
    """Return a QFont with the specified weight."""
    font = QFont(FONT_FAMILY)
    if weight == "bold":
        font.setBold(True)
    elif weight == "medium":
        font.setWeight(QFont.Weight.DemiBold)
    else:
        font.setWeight(QFont.Weight.Normal)
    return font
