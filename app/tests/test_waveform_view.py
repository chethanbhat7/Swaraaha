import numpy as np

from app.ui.waveform_view import WaveformView


def test_waveform_set_audio_draws(qapp):
    view = WaveformView()
    view.show()
    view.resize(400, 200)
    view.set_audio(np.zeros(1600, dtype=np.float32), 16000)
    assert len(view.scene().items()) > 0


def test_waveform_resize_schedules_debounced_redraw(qapp):
    view = WaveformView()
    view.show()
    view.resize(400, 200)
    view.set_audio(np.zeros(1600, dtype=np.float32), 16000)
    qapp.processEvents()
    assert view._resize_timer is not None
    assert view._resize_timer.isActive()
