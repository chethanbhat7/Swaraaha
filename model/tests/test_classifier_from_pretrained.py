"""
Tests for BaseWav2VecClassifier.from_pretrained checkpoint loading.

Checkpoints saved from a torch.compile-wrapped model carry ``_orig_mod.``
key prefixes; loading must strip them and use strict=True so silent
weight-dropping is impossible.
"""

import pytest
import torch

from model.classification import BaseWav2VecClassifier


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._w = torch.nn.Parameter(torch.tensor([1.0]))

    def load_state_dict(self, state_dict, strict=True):
        missing, unexpected = super().load_state_dict(state_dict, strict=strict)
        return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)


def _patch_transformer(monkeypatch):
    def fake_from_pretrained(model_name, num_labels):
        return _FakeModel()

    monkeypatch.setattr(
        "transformers.Wav2Vec2ForSequenceClassification.from_pretrained",
        staticmethod(fake_from_pretrained),
    )


def test_from_pretrained_restores_compile_prefixed_weights(monkeypatch, tmp_path):
    _patch_transformer(monkeypatch)
    path = tmp_path / "ckpt.pt"
    torch.save({
        "model_state_dict": {"_orig_mod._w": torch.tensor([42.0])},
        "class_name": "block",
        "class_idx": 1,
        "model_name": "facebook/wav2vec2-base",
    }, path)

    loaded = BaseWav2VecClassifier.from_pretrained(path)

    assert loaded._model._w.item() == 42.0


def test_from_pretrained_rejects_mismatched_keys(monkeypatch, tmp_path):
    _patch_transformer(monkeypatch)
    path = tmp_path / "ckpt.pt"
    torch.save({
        "model_state_dict": {"_w": torch.tensor([2.0]), "bogus_key": torch.tensor([3.0])},
        "class_name": "block",
        "class_idx": 1,
        "model_name": "facebook/wav2vec2-base",
    }, path)

    with pytest.raises(RuntimeError):
        BaseWav2VecClassifier.from_pretrained(path)
