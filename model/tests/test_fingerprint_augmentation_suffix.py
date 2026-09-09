"""Tests for the _naug augmentation fingerprint suffix."""

from argparse import Namespace

import pytest

from model.fingerprint import (
    parse_cnn_classifier_fingerprint,
    parse_fingerprint,
    parse_localizer_fingerprint,
    parse_multitask_fingerprint,
)


def test_single_class_fingerprint_appends_naug_when_off():
    from model.fingerprint import fingerprint

    args = Namespace(
        class_name="block", data_dir="data/train", model_name="facebook/wav2vec2-base",
        lr=3e-5, batch_size=8, max_length_seconds=3.0, warmup_steps=500,
        weight_decay=0.01, freeze_backbone_epochs=3, loss_type="focal",
        focal_gamma=2.0, seed=42, gradient_accumulation_steps=1, epochs=20,
        augmentation=False,
    )
    fp = fingerprint(args)
    assert fp.endswith("_naug")
    assert "_naug" not in fp[:-len("_naug")]


def test_single_class_fingerprint_parses_naug():
    from model.fingerprint import fingerprint

    args = Namespace(
        class_name="block", data_dir="data/train", model_name="facebook/wav2vec2-base",
        lr=3e-5, batch_size=8, max_length_seconds=3.0, warmup_steps=500,
        weight_decay=0.01, freeze_backbone_epochs=3, loss_type="focal",
        focal_gamma=2.0, seed=42, gradient_accumulation_steps=1, epochs=20,
        augmentation=False,
    )
    fp = fingerprint(args)
    parsed = parse_fingerprint(fp)
    assert parsed["augmentation"] is False
    assert parsed["class_name"] == "block"


def test_single_class_legacy_parse_implies_augmentation_on():
    fp = ("prolongation_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_"
          "wd0.01_ml3_s42_train_w2v2base")
    parsed = parse_fingerprint(fp)
    assert parsed["augmentation"] is True
    assert parsed["model_name"] == "facebook/wav2vec2-base"


def test_multitask_fingerprint_naug_roundtrip():
    from model.fingerprint import multitask_fingerprint

    args = Namespace(
        data_dir="data/train", model_name="microsoft/wavlm-base-plus", lr=3e-5,
        batch_size=16, max_length_seconds=3.0, warmup_steps=500, weight_decay=0.01,
        freeze_backbone_epochs=3, loss_type="focal", focal_gamma=2.0, seed=42,
        gradient_accumulation_steps=1, epochs=20, augmentation=False,
    )
    fp = multitask_fingerprint(args)
    assert fp.endswith("_train_wavlmbase_naug")
    parsed = parse_multitask_fingerprint(fp)
    assert parsed["augmentation"] is False
    assert parsed["model_name"] == "microsoft/wavlm-base-plus"
    assert parsed["loss_type"] == "focal"


def test_multitask_legacy_parse_implies_augmentation_on():
    fp = ("multi_e20_b16_lr3e-5_frz3_ltfocal_g2_ga1_wu500_"
          "wd0.01_ml3_s42_train_w2v2base")
    parsed = parse_multitask_fingerprint(fp)
    assert parsed["augmentation"] is True


def test_cnn_classifier_fingerprint_naug_roundtrip():
    from model.fingerprint import cnn_classifier_fingerprint

    args = Namespace(
        data_dir="data/train", epochs=20, batch_size=16, lr=3e-5, n_mels=128,
        hop_length=512, max_length_seconds=3.0, hidden_dim=128, dropout=0.4,
        patience=5, warmup_steps=500, weight_decay=0.01,
        gradient_accumulation_steps=1, seed=42, n_fft=2048, aggregator="pool",
        num_lstm_layers=1, num_transformer_layers=1, class_names=["block"],
        augmentation=False,
    )
    fp = cnn_classifier_fingerprint(args)
    assert fp.endswith("_train_block_naug")
    parsed = parse_cnn_classifier_fingerprint(fp)
    assert parsed["augmentation"] is False
    assert parsed["class_names"] == ["block"]


def test_cnn_classifier_legacy_parse_implies_augmentation_on():
    fp = ("cnnclf_aggpool_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_"
          "wu500_wd0.01_ga1_s42_train_all")
    parsed = parse_cnn_classifier_fingerprint(fp)
    assert parsed["augmentation"] is True


def test_localizer_fingerprints_naug_roundtrip():
    from model.fingerprint import localizer_fingerprint

    cnn = Namespace(
        data_dir="data/train", epochs=30, batch_size=8, lr=1e-3, n_mels=128,
        hop_length=512, max_length_seconds=3.0, dropout=0.4, patience=7,
        weight_decay=1e-4, seed=42, val_ratio=0.2, augmentation=False,
    )
    w2v2 = Namespace(
        data_dir="data/train", epochs=20, batch_size=4, lr=3e-5,
        max_length_seconds=3.0, dropout=0.3, hidden_dim=256, patience=5,
        weight_decay=0.01, freeze_backbone_epochs=5,
        model_name="facebook/wav2vec2-base", seed=42, val_ratio=0.2,
        warmup_steps=500, augmentation=False,
    )
    cnn_fp = localizer_fingerprint(cnn, "loc")
    assert cnn_fp.endswith("_train_naug")
    assert parse_localizer_fingerprint(cnn_fp)["augmentation"] is False

    w2v2_fp = localizer_fingerprint(w2v2, "wav2vec")
    assert w2v2_fp.endswith("_train_w2v2base_naug")
    parsed = parse_localizer_fingerprint(w2v2_fp)
    assert parsed["augmentation"] is False
    assert parsed["model_name"] == "facebook/wav2vec2-base"
