"""Tests for the multitask classifier fingerprint helpers."""

import pytest

from model.fingerprint import (
    MULTITASK_RESUME_KEYS,
    multitask_fingerprint,
    parse_multitask_fingerprint,
)


class _Args:
    data_dir = "data/train"
    model_name = "facebook/wav2vec2-base"
    lr = 3e-5
    batch_size = 16
    max_length_seconds = 10.0
    warmup_steps = 500
    weight_decay = 0.01
    freeze_backbone_epochs = 3
    focal_gamma = 2.0
    seed = 42
    gradient_accumulation_steps = 1
    epochs = 20


def test_multitask_fingerprint_format():
    fp = multitask_fingerprint(_Args())
    assert fp == "multi_e20_b16_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml10_s42_train_w2v2base"


def test_parse_multitask_fingerprint_roundtrip():
    fp = multitask_fingerprint(_Args())
    parsed = parse_multitask_fingerprint(fp)
    assert parsed["epochs"] == 20
    assert parsed["batch_size"] == 16
    assert parsed["lr"] == 3e-5
    assert parsed["freeze_backbone_epochs"] == 3
    assert parsed["focal_gamma"] == 2.0
    assert parsed["model_name"] == "facebook/wav2vec2-base"


def test_multitask_resume_keys_match_args():
    for k in MULTITASK_RESUME_KEYS:
        assert hasattr(_Args(), k), f"{k} missing from _Args"
