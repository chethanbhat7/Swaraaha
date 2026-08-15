import numpy as np
import pytest
import torch

from model.classification.cnn_multitask import CNNMultitaskClassifier

CLASS_NAMES = ['block', 'interjection']


@pytest.mark.parametrize('aggregator', ['pool', 'lstm', 'transformer'])
def test_forward_returns_per_class_logits(aggregator):
    model = CNNMultitaskClassifier(n_mels=16, hidden_dim=8, class_names=CLASS_NAMES,
                                   aggregator=aggregator)
    logits = model.forward(torch.randn(2, 1, 16, 8))
    assert set(logits.keys()) == set(CLASS_NAMES)
    assert all(v.shape == (2, 2) for v in logits.values())


@pytest.mark.parametrize('aggregator', ['pool', 'lstm', 'transformer'])
def test_forward_head(aggregator):
    model = CNNMultitaskClassifier(n_mels=16, hidden_dim=8, class_names=CLASS_NAMES,
                                   aggregator=aggregator)
    logits = model.forward_head('block', torch.randn(2, 1, 16, 8))
    assert logits.shape == (2, 2)


def test_predict_honors_threshold():
    model = CNNMultitaskClassifier(n_mels=16, hidden_dim=8, class_names=['block'],
                                   aggregator='pool')
    model._build_model()
    with torch.no_grad():
        final = model._model.heads['block'][-1]
        final.weight.zero_()
        final.bias.copy_(torch.tensor([0.0, 5.0]))
    spec = torch.randn(1, 1, 16, 8)
    label_hi, conf_hi = model.predict(spec, threshold=0.5)['block']
    assert label_hi == 1
    assert conf_hi == pytest.approx(0.993, abs=1e-3)
    label_lo, conf_lo = model.predict(spec, threshold=0.999)['block']
    assert label_lo == 0
    assert conf_lo == pytest.approx(0.0067, abs=1e-3)
    assert model.predict(spec[0], threshold=0.5)['block'][0] == 1


def test_predict_rejects_invalid_threshold():
    model = CNNMultitaskClassifier(n_mels=16, hidden_dim=8, class_names=['block'],
                                   aggregator='pool')
    model._build_model()
    with pytest.raises(ValueError):
        model.predict(torch.randn(1, 1, 16, 8), threshold=1.5)


def test_save_load_roundtrip(tmp_path):
    model = CNNMultitaskClassifier(n_mels=16, hop_length=256, n_fft=1024,
                                   hidden_dim=8, class_names=CLASS_NAMES,
                                   aggregator='lstm')
    model.forward(torch.randn(2, 1, 16, 8))
    path = str(tmp_path / 'cnn.pt')
    model.save(path)
    loaded = CNNMultitaskClassifier.from_pretrained(path)
    assert loaded.class_names == CLASS_NAMES
    assert loaded.hidden_dim == 8
    assert loaded.aggregator == 'lstm'
    assert loaded.n_mels == 16
    assert loaded.hop_length == 256
    assert loaded.n_fft == 1024
    for k, v in model.model.state_dict().items():
        assert torch.equal(v, loaded.model.state_dict()[k])


def test_from_pretrained_defaults_n_fft_to_2048(tmp_path):
    model = CNNMultitaskClassifier(n_mels=16, hop_length=512, hidden_dim=8,
                                   class_names=CLASS_NAMES, aggregator='pool')
    model.forward(torch.randn(2, 1, 16, 8))
    path = str(tmp_path / 'cnn.pt')
    model.save(path)
    loaded = CNNMultitaskClassifier.from_pretrained(path)
    assert loaded.n_fft == 2048


def test_count_parameters_positive():
    model = CNNMultitaskClassifier(n_mels=16, hidden_dim=8, class_names=['block'],
                                   aggregator='pool')
    assert model.count_parameters() > 0
