import torch

from model.evaluation.loader import load_localizer
from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer


def test_load_localizer_cnn_training_format_uses_checkpoint_n_mels(tmp_path):
    model = CNNSpectrogramLocalizer(n_mels=256)
    ckpt_path = tmp_path / "cnn_resume.pt"
    torch.save(
        {"model_state_dict": model.model.state_dict(), "args": {"n_mels": 256}},
        ckpt_path,
    )

    instance = load_localizer("cnn", str(ckpt_path))
    assert instance.n_mels == 256
