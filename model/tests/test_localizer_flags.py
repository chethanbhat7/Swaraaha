"""Tests for localizer CLI flags (--no-augmentation, --pos_weight)."""

import sys

import pytest

from model.training import train_localizer
from model.training import train_wav2vec2_localizer as w2v2


def _parse_with_args(module, argv):
    old = sys.argv
    sys.argv = [module.__file__ or "train.py"] + argv
    try:
        return module.parse_args()
    finally:
        sys.argv = old


def test_w2v2_parse_args_augmentation_default_on():
    args = _parse_with_args(w2v2, [])
    assert args.augmentation is True


def test_w2v2_parse_args_no_augmentation_flag():
    args = _parse_with_args(w2v2, ["--no-augmentation"])
    assert args.augmentation is False


def test_w2v2_parse_args_pos_weight_default():
    args = _parse_with_args(w2v2, [])
    assert args.pos_weight is None


def test_w2v2_parse_args_pos_weight_override():
    args = _parse_with_args(w2v2, ["--pos_weight", "3.0"])
    assert args.pos_weight == 3.0


def test_cnn_parse_args_augmentation_default_on():
    args = _parse_with_args(train_localizer, [])
    assert args.augmentation is True


def test_cnn_parse_args_no_augmentation_flag():
    args = _parse_with_args(train_localizer, ["--no-augmentation"])
    assert args.augmentation is False


def test_cnn_parse_args_pos_weight_default():
    args = _parse_with_args(train_localizer, [])
    assert args.pos_weight is None


def test_cnn_parse_args_pos_weight_override():
    args = _parse_with_args(train_localizer, ["--pos_weight", "3.0"])
    assert args.pos_weight == 3.0
