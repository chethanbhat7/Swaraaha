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


def _write_mini_dataset(tmp_path, label_rows):
    """Create data_dir/{audio,labels} with N dummy clips (>44-byte audio)."""
    from model.config.defaults import DYSFLUENCY_CLASSES

    audio_dir = tmp_path / "audio"
    labels_dir = tmp_path / "labels"
    audio_dir.mkdir()
    labels_dir.mkdir()
    for i, row in enumerate(label_rows):
        (audio_dir / f"clip{i}.wav").write_bytes(b"\x00" * 1024)
        intervals = "\n".join(
            f"{idx},{idx + 0.5},{name}" for idx, name in enumerate(row)
        )
        (labels_dir / f"clip{i}.csv").write_text(
            f"start_sec,end_sec,dysfluency_type\n{intervals}\n"
        )
    return tmp_path


def test_classification_dataset_label_vectors_matches_getitem(tmp_path):
    from model.data.dataset import ClassificationDataset

    rows = [
        ["prolongation"],
        ["block"],
        [],
        [],
        ["prolongation", "block"],
    ]
    data_dir = _write_mini_dataset(tmp_path, rows)
    ds = ClassificationDataset(data_dir=str(data_dir), cache_dir=None)
    lv = np.asarray(ds.label_vectors, dtype=float)
    assert lv.shape == (5, 5)
    for i, row in enumerate(rows):
        assert lv[i].sum() == len(row)


def test_train_pos_weights_uses_train_subset_only(tmp_path):
    from model.data.dataset import ClassificationDataset
    from model.training.train_multitask_classifier import _train_pos_weights

    rows = [
        ["prolongation"],
        ["block"],
        [],
        [],
        ["prolongation", "block"],
    ]
    data_dir = _write_mini_dataset(tmp_path, rows)
    ds = ClassificationDataset(data_dir=str(data_dir), cache_dir=None)
    weights = _train_pos_weights(ds, train_idx=np.array([0, 2, 3]))
    assert weights["prolongation"] == 2.0   # train subset: 1 pos / 2 neg
    assert weights["block"] == 1.0          # 0 pos among (0,2,3) -> guard 1.0
    assert weights["soundrep"] == 1.0