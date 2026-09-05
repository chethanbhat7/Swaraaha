"""Tests for multitask training loss helpers."""

import numpy as np
import torch

from model.training.train_multitask_classifier import (
    MultiLabelBCEWithLogitsLoss,
    compute_class_pos_weights,
)


def _dataset_with_labels(label_vectors):
    class _D:
        def __init__(self):
            self.label_vectors = label_vectors

    return _D()


def test_compute_pos_weights_uses_neg_over_pos():
    labels = np.array([
        [1, 0, 0, 0, 0],   # prolongation pos
        [0, 1, 0, 0, 0],   # block pos
        [0, 0, 0, 0, 0],   # all neg
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    ds = _dataset_with_labels(labels)
    w = compute_class_pos_weights(ds)
    assert w["prolongation"] == 4.0   # 4 neg / 1 pos
    assert w["block"] == 4.0
    assert w["soundrep"] == 1.0       # no pos -> 1.0 (guard in compute_class_pos_weights)
    assert w["wordrep"] == 1.0
    assert w["interjection"] == 1.0


def test_multilabel_bce_loss_runs_on_cuda_or_cpu():
    from model.config.defaults import DYSFLUENCY_CLASSES

    pos_weights = {name: 2.0 for name in DYSFLUENCY_CLASSES}
    criterion = MultiLabelBCEWithLogitsLoss(pos_weights)
    logits = {name: torch.randn(4, 2) for name in DYSFLUENCY_CLASSES}
    labels = torch.zeros(4, len(DYSFLUENCY_CLASSES))
    labels[:, 0] = 1
    loss = criterion(logits, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)