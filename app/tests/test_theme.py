import pytest

from app.ui.theme import COLORS, DARK_COLORS, LIGHT_COLORS, set_theme


@pytest.fixture(autouse=True)
def reset_theme():
    yield
    set_theme(False)


def test_light_default_is_pastel_palette():
    set_theme(False)
    assert COLORS["primary"] == "#8E6FB8"
    assert COLORS["surface"] == "#FBF7FE"
    assert COLORS["surface_variant"] == "#EDE4F8"
    assert COLORS["on_surface"] == "#3A2E45"


def test_dark_is_spiderman_palette():
    set_theme(True)
    assert COLORS["primary"] == "#E23636"
    assert COLORS["secondary"] == "#447BBE"
    assert COLORS["surface"] == "#141414"
    assert COLORS["on_surface"] == "#F2F2F2"


def test_secondary_token_present_in_both_themes():
    assert "secondary" in LIGHT_COLORS
    assert "record" in LIGHT_COLORS
    assert "secondary" in DARK_COLORS
    assert "record" in DARK_COLORS


def test_dysfluency_has_five_distinct_light_pastels():
    light = LIGHT_COLORS["dysfluency"]
    dark = DARK_COLORS["dysfluency"]
    assert len(light) == 5
    assert len(dark) == 5
    assert len(set(light.values())) == 5
    assert len(set(dark.values())) == 5


def test_set_theme_swaps_whole_dict():
    set_theme(True)
    dark_surface = COLORS["surface"]
    set_theme(False)
    assert COLORS["surface"] != dark_surface
