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


def test_waveform_overlay_colored_by_primary_type(qapp):
    panel = ResultsPanel()
    panel._waveform.set_audio = lambda *args, **kwargs: None
    captured = []
    panel._waveform.set_overlays = captured.append
    results = {
        "classifications": {},
        "localizations": [(0.0, 1.0, 0.9)],
        "transcription": None,
        "combined": {
            "regions": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "confidence": 0.9,
                    "primary_type": "block",
                    "classes": {},
                    "syllables": [],
                }
            ],
            "audio_duration": 2.0,
            "total_stutters": 1,
        },
    }
    panel.set_results(results, audio=np.zeros(1600, dtype=np.float32))
    assert captured[0][0][2] == COLORS["dysfluency"]["block"]


def test_total_events_label_uses_combined_count(qapp):
    panel = ResultsPanel()
    panel._waveform.set_audio = lambda *args, **kwargs: None
    panel._waveform.set_overlays = lambda *args, **kwargs: None
    results = {
        "classifications": {},
        "localizations": [(0.0, 1.0, 0.9)],
        "transcription": None,
        "combined": {"regions": [], "audio_duration": 2.0, "total_stutters": 3},
    }
    panel.set_results(results, audio=np.zeros(1600, dtype=np.float32))
    assert "Total Dysfluent Events: 3" in panel._total_label.text()


def test_total_events_label_falls_back_to_localization_count(qapp):
    panel = ResultsPanel()
    panel._waveform.set_audio = lambda *args, **kwargs: None
    panel._waveform.set_overlays = lambda *args, **kwargs: None
    results = {"classifications": {}, "localizations": [(0.0, 1.0, 0.9)], "transcription": None}
    panel.set_results(results, audio=np.zeros(1600, dtype=np.float32))
    assert "Total Dysfluent Events: 1" in panel._total_label.text()
