"""Tests for localizer evaluate functions returning threshold-free AUROC."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model.training import train_localizer
from model.training import train_wav2vec2_localizer as w2v2


class _StubModule:
    def eval(self):
        return self


class _StubLocalizer:
    """Minimal stand-in: model.model.eval() + forward(waveforms)->logits (B,T)."""

    def __init__(self, logits: torch.Tensor):
        self.model = _StubModule()
        self._logits = logits

    def forward(self, waveforms):
        return self._logits


class _FrameDataset(Dataset):
    def __init__(self, item_shape, label_shape, n=2, pos_frames=None):
        self.items = []
        for i in range(n):
            label = np.zeros(label_shape, dtype=np.uint8)
            if pos_frames is not None and i == 1:
                label[pos_frames] = 1
            self.items.append((np.zeros(item_shape, dtype=np.float32), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_evaluate_model_returns_threshold_free_auroc():
    waveforms = torch.zeros(2, 1600)
    logits = torch.cat([torch.full((1, 500), -5.0), torch.full((1, 500), 5.0)])
    model = _StubLocalizer(logits)

    ds = _FrameDataset(item_shape=(1600,), label_shape=(500,), pos_frames=slice(0, 500))
    loader = DataLoader(ds, batch_size=2, collate_fn=w2v2.collate_wav2vec2)

    f1, mean_iou, auroc, avg_loss, all_true, all_pred = w2v2.evaluate_model(model, loader, "cpu")
    assert f1 == 1.0
    assert mean_iou == 1.0
    assert auroc == 1.0
    assert all_true.sum() == 500
    assert len(all_pred) == 1000


def test_evaluate_localizer_returns_threshold_free_auroc():
    spectrograms = torch.zeros(2, 128, 500)
    logits = torch.cat([torch.full((1, 500), -5.0), torch.full((1, 500), 5.0)])
    model = _StubLocalizer(logits)

    ds = _FrameDataset(item_shape=(128, 500), label_shape=(500,), pos_frames=slice(0, 500))
    loader = DataLoader(ds, batch_size=2)

    f1, mean_iou, auroc, avg_loss, all_true, all_pred = train_localizer.evaluate_localizer(
        model, loader, "cpu"
    )
    assert f1 == 1.0
    assert mean_iou == 1.0
    assert auroc == 1.0
    assert len(all_pred) == 1000
