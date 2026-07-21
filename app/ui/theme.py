"""Design tokens and QSS generation for Material 3 Expressive theme."""

import os
from PySide6.QtGui import QFontDatabase, QFont

COLORS = {
    "primary": "#6750A4",
    "on_primary": "#FFFFFF",
    "primary_container": "#EADDFF",
    "surface": "#FFFBFE",
    "surface_variant": "#F3EDF7",
    "on_surface": "#1C1B1F",
    "outline": "#79747E",
    "dysfluency": {
        "prolongation": "#B3261E",
        "block": "#7D5260",
        "soundrep": "#006D3F",
        "wordrep": "#0061A4",
        "interjection": "#984061",
    },
}

SPACING = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}

RADIUS = {"sm": 8, "md": 16, "lg": 24, "full": 9999}

FONT_FAMILY = "Google Sans Font"


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
