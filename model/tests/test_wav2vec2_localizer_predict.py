"""Tests for Wav2Vec2Localizer.predict region emission (M18)."""

import numpy as np
import pytest
import torch

from model.localization.wav2vec2_localizer import Wav2Vec2Localizer


def _loc_with_probs(probs):
    """Wav2Vec2Localizer whose predict_proba returns a fixed tensor."""
    loc = Wav2Vec2Localizer()
    loc.predict_proba = lambda tensor: probs
    return loc


def test_predict_does_not_emit_regions_from_zero_padded_tail():
    """2s clip zero-padded to 10s (500 frames @320 samples): a confident
    region entirely inside the padded tail must not be reported."""
    audio = np.zeros(32000, dtype=np.float32)
    probs = torch.zeros((1, 1, 500))
    probs[0, 0, 100:] = 0.9  # everything past the real 2s audio
    loc = _loc_with_probs(probs)

    regions = loc.predict(audio, sr=16000, threshold=0.5)
    assert regions == []


def test_predict_clips_tail_region_to_real_audio_end():
    """A region straddling the real-audio end must be clamped to it."""
    audio = np.zeros(32000, dtype=np.float32)  # 2s
    probs = torch.zeros((1, 1, 500))
    probs[0, 0, 90:121] = 0.9  # 1.8s .. 2.42s
    loc = _loc_with_probs(probs)

    regions = loc.predict(audio, sr=16000, threshold=0.5)
    start, end, conf = regions[0]
    assert (start, end) == (1.8, 2.0)
    assert conf == pytest.approx(0.9)


def test_predict_region_within_real_audio_unchanged():
    """Regions fully inside the real audio are reported unmodified."""
    audio = np.zeros(32000, dtype=np.float32)
    probs = torch.zeros((1, 1, 500))
    probs[0, 0, 10:30] = 0.9  # 0.2s .. 0.6s
    loc = _loc_with_probs(probs)

    regions = loc.predict(audio, sr=16000, threshold=0.5)
    start, end, conf = regions[0]
    assert (start, end) == (0.2, 0.6)
    assert conf == pytest.approx(0.9)


def test_predict_long_audio_truncated_to_max_length():
    """Audio longer than max_length_seconds is truncated, so the whole
    padded length is real audio and tail regions are legitimate."""
    audio = np.zeros(200000, dtype=np.float32)  # 12.5s
    probs = torch.zeros((1, 1, 500))
    probs[0, 0, 400:] = 0.9  # 8s..10s, all real (truncated audio)
    loc = _loc_with_probs(probs)

    regions = loc.predict(audio, sr=16000, threshold=0.5, max_length_seconds=10.0)
    start, end, conf = regions[0]
    assert (start, end) == (8.0, 10.0)
    assert conf == pytest.approx(0.9)


class _FakeW2V2(torch.nn.Module):
    """Fake backbone: no download, exposes config.hidden_size + last_hidden_state."""

    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"hidden_size": 768})()

    def forward(self, waveforms):
        B, L = waveforms.shape
        T = L // 320
        return type("O", (), {"last_hidden_state": torch.zeros(B, T, 768)})()


def _fake_backbone_factory():
    return type("F", (), {"from_pretrained": staticmethod(lambda name: _FakeW2V2())})


def _build_localizer(monkeypatch):
    from model.localization import wav2vec2_localizer as mod

    monkeypatch.setattr(mod, "_wav2vec2_model_class", _fake_backbone_factory)
    return mod.Wav2Vec2Localizer(model_name="fake", hidden_dim=32)


def test_wav2vec2_localizer_has_no_temporal_attention(monkeypatch):
    """temporal_attention is dead: computed but never used, so its params get
    no gradient and only bloat the checkpoint. It must be removed."""
    loc = _build_localizer(monkeypatch)
    model = loc.model
    assert not hasattr(model, "temporal_attention")
    keys = model.state_dict().keys()
    assert not any("temporal_attention" in k for k in keys)
    assert any(k.startswith("classifier.") for k in keys)


def test_wav2vec2_localizer_forward_shape_unchanged(monkeypatch):
    """Removing the dead head must not change the (B, 1, T) logit shape."""
    loc = _build_localizer(monkeypatch)
    logits = loc.forward(torch.zeros(2, 16000))
    assert logits.shape == (2, 1, 50)
