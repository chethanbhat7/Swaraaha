"""Tests for the lazy backbone model-class resolver."""

import pytest

from model.backbone import backbone_model_class


def test_wav2vec2_default():
    from transformers import Wav2Vec2Model
    assert backbone_model_class("facebook/wav2vec2-base") is Wav2Vec2Model


def test_wav2vec2_large():
    from transformers import Wav2Vec2Model
    assert backbone_model_class("facebook/wav2vec2-large-960h") is Wav2Vec2Model


def test_wavlm_maps_to_wavlm():
    from transformers import WavLMModel
    assert backbone_model_class("microsoft/wavlm-base-plus") is WavLMModel


def test_hubert_maps_to_hubert():
    from transformers import HubertModel
    assert backbone_model_class("facebook/hubert-base-ls960") is HubertModel


def test_unknown_name_defaults_to_wav2vec2():
    from transformers import Wav2Vec2Model
    assert backbone_model_class("totally/unknown-id") is Wav2Vec2Model