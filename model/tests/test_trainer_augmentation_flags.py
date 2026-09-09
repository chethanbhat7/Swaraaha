"""Tests for --no-augmentation CLI flags on the three classifier trainers."""

from model.training import train_multitask_classifier


def test_multitask_parse_args_augmentation_default_on():
    args = train_multitask_classifier.parse_args([])
    assert args.augmentation is True


def test_multitask_parse_args_no_augmentation_flag():
    args = train_multitask_classifier.parse_args(["--no-augmentation"])
    assert args.augmentation is False