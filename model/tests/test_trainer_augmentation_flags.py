"""Tests for --no-augmentation CLI flags on the three classifier trainers."""

from model.training import train_classifier
from model.training import train_multitask_classifier


def test_multitask_parse_args_augmentation_default_on():
    args = train_multitask_classifier.parse_args([])
    assert args.augmentation is True


def test_multitask_parse_args_no_augmentation_flag():
    args = train_multitask_classifier.parse_args(["--no-augmentation"])
    assert args.augmentation is False


def test_classifier_parse_args_default_on(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["train_classifier.py", "--class_name", "block"]
    )
    args = train_classifier.parse_args()
    assert args.augmentation is True


def test_classifier_parse_args_no_aug(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train_classifier.py", "--class_name", "block", "--no-augmentation"],
    )
    args = train_classifier.parse_args()
    assert args.augmentation is False
