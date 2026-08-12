"""Tests for the multitask classifier training script."""

import torch
import pytest

from model.training import train_multitask_classifier as tmc

CLASS_NAMES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]


class _FakeHead(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(*x.shape[:1], 2)


class _FakeModel:
    """Stands in for MultiTaskWav2VecClassifier (has .model + class_names)."""

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = torch.nn.ModuleDict({n: _FakeHead() for n in CLASS_NAMES})

        def forward(self, audio):
            return {n: self.heads[n](audio) for n in CLASS_NAMES}

    def __init__(self):
        self.model = self._Inner()
        self.class_names = list(CLASS_NAMES)

    def forward(self, audio):
        return self.model(audio)


@pytest.fixture
def fake_model():
    return _FakeModel()


def test_per_class_loss_aggregates_over_multi_hot_labels(fake_model):
    from model.training.utils import FocalLoss

    audio = torch.randn(4, 1600)
    labels = torch.zeros(4, 5, dtype=torch.uint8)
    labels[:, 0] = 1
    labels[1, 2] = 1

    criterion = FocalLoss(gamma=2.0)
    loss = tmc.multitask_loss(
        fake_model.forward(audio), labels, criterion, device="cpu"
    )
    assert loss.item() > 0.0
    # 5 heads, so the sum is >= any single head's loss
    single = criterion(fake_model.forward(audio)["prolongation"], labels[:, 0].long())
    assert loss.item() >= single.item()


def test_parse_args_has_multitask_defaults():
    args = tmc.parse_args([])
    assert args.focal_gamma == 2.0
    assert args.hidden_dim == 768
    assert args.class_names == CLASS_NAMES
    assert args.freeze_backbone_epochs == 3
