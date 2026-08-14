"""Tests for the multitask classifier training script."""

import numpy as np
import torch
import pytest

from model.config.defaults import DYSFLUENCY_CLASSES
from model.training import train_multitask_classifier as tmc
from model.training.utils import FocalLoss

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


def test_partition_finds_heads_after_torch_compile_prefix():
    """torch.compile renames params with an _orig_mod. prefix; heads must
    still be found (regression: startswith('heads.') matched nothing)."""

    class _FakeCompiled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            inner = torch.nn.Module()
            inner.heads = torch.nn.ModuleDict(
                {n: torch.nn.Linear(2, 2) for n in CLASS_NAMES}
            )
            inner.wav2vec2 = torch.nn.Linear(2, 2)
            self._orig_mod = inner

    model = _FakeModel()
    model.model = _FakeCompiled()

    backbone_params, head_params = tmc._partition_model_params(model)
    assert len(head_params) == 2 * len(CLASS_NAMES)  # weight + bias per head
    assert len(backbone_params) == 2  # weight + bias of the backbone
    assert head_params  # must be non-empty (the optimizer crash we fixed)


def test_multitask_loss_respects_class_names_subset():
    model = _FakeModel()
    audio = torch.zeros(2, 1600)
    labels = torch.zeros(2, 5)
    block_col = DYSFLUENCY_CLASSES.index('block')
    labels[:, block_col] = 1
    criterion = FocalLoss(gamma=2.0)
    logits = model.forward(audio)
    total = tmc.multitask_loss(logits, labels, criterion, device='cpu',
                               class_names=['block'])
    expected = criterion(logits['block'], labels[:, block_col].long())
    assert torch.allclose(total, expected)


class _ConfidentHead(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros(x.shape[0], 2)
        logits[:, 1] = 10.0
        return logits


class _ZeroHead(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 2)


def test_evaluate_multitask_respects_class_names_subset():
    heads = torch.nn.ModuleDict({
        name: (_ConfidentHead() if name == 'interjection' else _ZeroHead())
        for name in DYSFLUENCY_CLASSES
    })
    inner = torch.nn.Module()
    inner.heads = heads

    class _Model:
        class_names = list(DYSFLUENCY_CLASSES)
        model = inner

        def forward(self, audio):
            return {name: head(audio) for name, head in heads.items()}

    model = _Model()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.zeros(2, 1600),
            torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 0, 1]], dtype=torch.float32),
        ),
        batch_size=2)
    _, macro_f1, _ = tmc.evaluate_multitask(model, loader, torch.device('cpu'),
                                            class_names=['interjection'])
    assert macro_f1 == 1.0


def test_stratified_split_uses_label_vectors_fast_path():
    class _SlowDataset:
        calls = 0

        def __init__(self):
            self.label_vectors = [
                np.array([0, 1, 0, 0, 0]),
                np.array([1, 0, 0, 0, 0]),
            ]

        def __len__(self):
            return 2

        def __getitem__(self, i):
            _SlowDataset.calls += 1
            return (None, self.label_vectors[i])

    ds = _SlowDataset()
    train_idx, val_idx = tmc.stratified_split(ds, val_ratio=0.5, seed=42)
    assert _SlowDataset.calls == 0
    assert sorted(train_idx + val_idx) == [0, 1]
