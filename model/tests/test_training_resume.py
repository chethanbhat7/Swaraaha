"""Tests for shared training resume helpers and localizer fingerprints."""

from argparse import Namespace

import numpy as np
import torch

from model.training.utils import (
    maybe_skip_completed,
    save_resume_state,
    try_load_resume,
)


class StubModel:
    def __init__(self):
        self.model = torch.nn.Linear(2, 1)


def _cnn_args():
    return Namespace(
        data_dir="data/train", epochs=30, batch_size=8, lr=1e-3,
        n_mels=128, hop_length=512, max_length_seconds=10.0,
        dropout=0.4, patience=7, weight_decay=1e-4, seed=42, val_ratio=0.2,
    )


def _w2v2_args():
    return Namespace(
        data_dir="data/train", epochs=20, batch_size=4, lr=3e-5,
        max_length_seconds=10.0, dropout=0.3, hidden_dim=256,
        patience=5, weight_decay=0.01, freeze_backbone_epochs=5,
        model_name="facebook/wav2vec2-base", seed=42, val_ratio=0.2,
        warmup_steps=500,
    )


def test_localizer_fingerprint_roundtrip_cnn():
    from model.fingerprint import localizer_fingerprint, parse_localizer_fingerprint

    args = _cnn_args()
    fp = localizer_fingerprint(args, "loc")
    assert fp.startswith("cnnloc_e30_b8_lr0.001_n128_h512_ml10_d0.4_pa7_wd0.0001_vr0.2_s42_train")
    parsed = parse_localizer_fingerprint(fp)
    assert parsed["epochs"] == 30
    assert parsed["batch_size"] == 8
    assert parsed["lr"] == 1e-3
    assert parsed["n_mels"] == 128
    assert parsed["hop_length"] == 512
    assert parsed["dropout"] == 0.4
    assert parsed["val_ratio"] == 0.2


def test_localizer_fingerprint_roundtrip_w2v2():
    from model.fingerprint import localizer_fingerprint, parse_localizer_fingerprint

    args = _w2v2_args()
    fp = localizer_fingerprint(args, "wav2vec")
    assert fp.startswith("w2v2loc_e20_b4_lr3e-05_frz5_wu500_hd256_d0.3_wd0.01_ml10_pa5_vr0.2_s42_train_w2v2base")
    parsed = parse_localizer_fingerprint(fp)
    assert parsed["freeze_backbone_epochs"] == 5
    assert parsed["warmup_steps"] == 500
    assert parsed["hidden_dim"] == 256
    assert parsed["model_name"] == "facebook/wav2vec2-base"


def test_resume_state_roundtrip_and_skip(tmp_path):
    args = Namespace(output_dir=str(tmp_path), clean=False, epochs=30)
    model = StubModel()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    history = {"train_loss": [0.5], "val_frame_f1": [0.1]}

    save_resume_state(
        model, optimizer, scheduler, 5, 0.4, history, args,
        "cnnloc_testfp", resume_keys=["epochs"], completed=False,
    )
    ckpt = try_load_resume(args, "cpu", "cnnloc_testfp")
    assert ckpt is not None
    assert ckpt["epoch"] == 5
    assert ckpt["best_f1"] == 0.4
    assert ckpt["completed"] is False
    assert ckpt["args"] == {"epochs": 30}

    # Not completed -> no skip
    assert maybe_skip_completed(ckpt, 30) is None

    # Completed -> skip returns history
    ckpt["completed"] = True
    assert maybe_skip_completed(ckpt, 30) == history

    # --clean ignores the checkpoint
    args_clean = Namespace(output_dir=str(tmp_path), clean=True, epochs=30)
    assert try_load_resume(args_clean, "cpu", "cnnloc_testfp") is None


def test_generate_mel_spectrogram_short_audio_no_warning():
    import warnings

    from model.data.preprocessing import generate_mel_spectrogram

    audio = np.zeros(1727, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = generate_mel_spectrogram(audio, sr=16000)
    assert spec.shape[0] == 128
