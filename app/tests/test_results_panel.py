import numpy as np

from app.ui.results_panel import ResultsPanel
from app.ui.theme import COLORS


def test_waveform_overlay_uses_primary_color(qapp):
    panel = ResultsPanel()
    panel._waveform.set_audio = lambda *args, **kwargs: None
    captured = []
    panel._waveform.set_overlays = captured.append
    results = {
        "classifications": {},
        "localizations": [(0.0, 1.0, 0.9)],
        "transcription": None,
    }
    panel.set_results(results, audio=np.zeros(1600, dtype=np.float32))
    assert captured[0][0][2] == COLORS["primary"]
