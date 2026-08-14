import torch

from model.classification.cnn_multitask import CNNMultitaskClassifier
from model.config.defaults import DYSFLUENCY_CLASSES
from model.evaluation.loader import load_localizer, load_multitask
from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer


def test_load_multitask_training_format_without_model_name(tmp_path, monkeypatch):
    """Training-format checkpoints (_best.pt/_final.pt saved by save_checkpoint)
    carry no model_name key; load_multitask must infer it from the fingerprint
    path instead of raising KeyError (regression)."""

    from model.classification import multitask as mt_module

    class _FakeMT:
        def __init__(self, model_name, hidden_dim, class_names):
            self.model_name = model_name
            self.hidden_dim = hidden_dim
            self.class_names = list(class_names or DYSFLUENCY_CLASSES)
            self._fake = torch.nn.Module()

        @property
        def model(self):
            return self._fake

    monkeypatch.setattr(mt_module, "MultiTaskWav2VecClassifier", _FakeMT)

    ckpt_path = tmp_path / (
        "multi_e20_b16_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml10"
        "_s42_train_w2v2base_best.pt"
    )
    torch.save(
        {"model_state_dict": {}, "epoch": 11, "metrics": {"macro_f1": 0.365}},
        ckpt_path,
    )

    instance = load_multitask(str(ckpt_path))
    assert instance.model_name == "facebook/wav2vec2-base"
    assert list(instance.class_names) == DYSFLUENCY_CLASSES


def test_load_localizer_cnn_training_format_uses_checkpoint_n_mels(tmp_path):
    model = CNNSpectrogramLocalizer(n_mels=256)
    ckpt_path = tmp_path / "cnn_resume.pt"
    torch.save(
        {"model_state_dict": model.model.state_dict(), "args": {"n_mels": 256}},
        ckpt_path,
    )

    instance = load_localizer("cnn", str(ckpt_path))
    assert instance.n_mels == 256


def test_load_multitask_cnn_own_format(tmp_path):
    model = CNNMultitaskClassifier(n_mels=16, hidden_dim=8, class_names=['block'],
                                   aggregator='pool')
    model.forward(torch.randn(2, 1, 16, 8))
    path = str(tmp_path / 'cnn_best.pt')
    model.save(path)
    loaded = load_multitask(path)
    assert isinstance(loaded, CNNMultitaskClassifier)
    assert loaded.aggregator == 'pool'
    assert loaded.n_mels == 16
    assert loaded.class_names == ['block']
    for k, v in model.model.state_dict().items():
        assert torch.equal(v, loaded.model.state_dict()[k])


def test_load_multitask_cnn_training_format(tmp_path):
    model = CNNMultitaskClassifier(n_mels=16, hidden_dim=8, class_names=['block'],
                                   aggregator='lstm')
    model.forward(torch.randn(2, 1, 16, 8))
    path = str(tmp_path / 'cnnclf_agglstm1_checkpoint.pt')
    torch.save({
        'epoch': 3,
        'model_state_dict': model.model.state_dict(),
        'best_f1': 0.5,
        'history': [],
        'backbone_frozen': False,
        'completed': False,
        'args': {'aggregator': 'lstm', 'n_mels': 16, 'hop_length': 256,
                 'hidden_dim': 8, 'dropout': 0.4,
                 'class_names': ['block'], 'num_lstm_layers': 1,
                 'num_transformer_layers': 1},
    }, path)
    loaded = load_multitask(path)
    assert isinstance(loaded, CNNMultitaskClassifier)
    assert loaded.aggregator == 'lstm'
    for k, v in model.model.state_dict().items():
        assert torch.equal(v, loaded.model.state_dict()[k])
