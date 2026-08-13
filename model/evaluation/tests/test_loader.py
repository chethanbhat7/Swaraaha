import torch

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
