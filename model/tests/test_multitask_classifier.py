"""Tests for the shared-backbone multitask wav2vec2 classifier."""

import torch
import pytest

from model.classification import multitask as mt

CLASS_NAMES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]


class _FakeConfig:
    hidden_size = 8


class _FakeOut:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class _FakeW2V2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()

    def forward(self, input_values):
        B, L = input_values.shape
        return _FakeOut(torch.zeros(B, L // 320, self.config.hidden_size))


def _fake_model_factory():
    class _FakeFactory:
        @staticmethod
        def from_pretrained(model_name):
            return _FakeW2V2()

    return _FakeFactory


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(mt, "_wav2vec2_model_class", _fake_model_factory)
    return mt.MultiTaskWav2VecClassifier(
        model_name="fake", hidden_dim=8, class_names=list(CLASS_NAMES)
    )


class _FixedHead(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self._logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, pooled):
        return self._logits.unsqueeze(0)


def test_forward_returns_per_class_logits(model):
    logits = model.forward(torch.zeros(2, 3200))
    assert list(logits.keys()) == CLASS_NAMES
    for name in CLASS_NAMES:
        assert logits[name].shape == (2, 2)


def test_forward_head_returns_two_logits(model):
    out = model.forward_head("block", torch.zeros(1, 3200))
    assert out.shape == (1, 2)


def test_predict_honors_threshold_per_class(model):
    model.model.heads["block"] = _FixedHead([0.0, 0.5])  # softmax = [0.378, 0.622]
    audio = torch.zeros(1, 3200)

    label_hi, conf_hi = model.predict(audio, threshold=0.7)["block"]
    assert label_hi == 0
    assert conf_hi == pytest.approx(0.378, abs=1e-3)

    label_lo, conf_lo = model.predict(audio, threshold=0.6)["block"]
    assert label_lo == 1
    assert conf_lo == pytest.approx(0.622, abs=1e-3)

    assert model.predict(audio, threshold=0.5)["prolongation"][0] in (0, 1)


def test_predict_rejects_bad_threshold(model):
    with pytest.raises(ValueError):
        model.predict(torch.zeros(1, 3200), threshold=1.5)


def test_save_load_roundtrip(model, tmp_path):
    path = str(tmp_path / "mt.pt")
    model.save(path)

    loaded = mt.MultiTaskWav2VecClassifier.from_pretrained(path)
    assert loaded.class_names == CLASS_NAMES
    assert loaded.hidden_dim == 8
    loaded_sd = loaded.model.state_dict()
    orig_sd = model.model.state_dict()
    assert list(loaded_sd.keys()) == list(orig_sd.keys())
    for k in orig_sd:
        assert torch.equal(loaded_sd[k], orig_sd[k])


def test_saliency_returns_per_frame_per_class_probs(model):
    audio = torch.zeros(1, 6400)  # 20 frames at 320/frame
    sal = model.saliency(audio)
    assert sal.ndim == 3
    assert sal.shape[0] == 1
    assert sal.shape[1] == 6400 // 320
    assert sal.shape[2] == len(CLASS_NAMES)
    assert float(sal.min()) >= 0.0
    assert float(sal.max()) <= 1.0


def test_saliency_matches_pooled_forward_on_constant_input(model):
    audio = torch.zeros(1, 6400)
    sal = model.saliency(audio)              # (1, T, C)
    logits = model.forward(audio)            # {name: (1, 2)}
    for i, name in enumerate(CLASS_NAMES):
        p_pooled = torch.softmax(logits[name], dim=-1)[0, 1]
        # All frames are identical (zero input), so every frame's saliency
        # equals the head output on the pooled representation.
        assert sal[0, :, i].mean().item() == pytest.approx(p_pooled.item(), abs=1e-5)
