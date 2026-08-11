"""
Tests for the --full (no-split) evaluation path.

The eval builders silently take a random ~20% of whatever data_dir is
given. --full must evaluate the ENTIRE prepared split (e.g. data/test)
instead of re-splitting it.
"""

import argparse
import struct
import wave

import numpy as np
import pytest

from model.evaluation.evaluate import (
    _build_classification_eval,
    _build_localization_eval,
)
from model.evaluation.metrics import THRESHOLD_SWEEP


def _write_wav(path, seconds=1.0, sr=16000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{len(audio)}h", *audio))


def _make_data(tmp_path, n=5):
    """data dir with n clips, alternating block/clean labels + sources.csv."""
    data = tmp_path / "data"
    (data / "audio").mkdir(parents=True)
    (data / "labels").mkdir(parents=True)

    sources = []
    for i in range(n):
        stem = f"clip_{i:02d}"
        _write_wav(data / "audio" / f"{stem}.wav")
        if i % 2 == 0:
            (data / "labels" / f"{stem}.csv").write_text(
                "start_sec,end_sec,dysfluency_type\n0.000,0.500,Block\n"
            )
        else:
            (data / "labels" / f"{stem}.csv").write_text(
                "start_sec,end_sec,dysfluency_type\n"
            )
        sources.append(f"{stem},sep28k")
    (data / "sources.csv").write_text("clip_id,source\n" + "\n".join(sources) + "\n")
    return data


def _args(**overrides):
    base = dict(
        data_dir=str(_data_dir := ""),
        batch_size=2,
        max_length_seconds=2.0,
        localizer_type="cnn",
        n_mels=128,
        hop_length=512,
        full=False,
        model_path="model/weights/localizer_best.pt",
        output_dir="model/evaluation/reports",
        threshold=0.5,
        save_misclassified=False,
        sweep_thresholds=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_localization_eval_full_uses_every_clip(tmp_path):
    data = _make_data(tmp_path, n=5)
    full_args = _args(data_dir=str(data), full=True)
    _, loader, val_idx = _build_localization_eval(full_args)
    assert len(loader.dataset) == 5
    assert len(val_idx) == 5


def test_localization_eval_default_still_splits(tmp_path):
    data = _make_data(tmp_path, n=5)
    args = _args(data_dir=str(data))
    _, loader, val_idx = _build_localization_eval(args)
    assert len(val_idx) == 1  # max(1, int(5*0.2))
    assert len(loader.dataset) == 1


def test_classification_eval_full_uses_every_clip(tmp_path):
    data = _make_data(tmp_path, n=5)
    full_args = _args(data_dir=str(data), full=True)
    _, loader, val_idx = _build_classification_eval(full_args)
    assert len(loader.dataset) == 5
    assert len(val_idx) == 5


def test_classification_eval_default_still_splits(tmp_path):
    data = _make_data(tmp_path, n=5)
    args = _args(data_dir=str(data))
    _, loader, val_idx = _build_classification_eval(args)
    assert len(loader.dataset) < 5
    assert len(val_idx) < 5


def _full_eval_args(tmp_path):
    """Build the Namespace full_evaluate passes to evaluate.py (H2)."""
    from model.evaluation.full_evaluate import _eval_args

    class _Args:
        data_dir = str(_make_data(tmp_path))
        output_dir = str(tmp_path / "out")
        registry = None
        batch_size = 2
        max_length_seconds = 2.0
        threshold = 0.5
        save_misclassified = False
        sweep_thresholds = False

    return _eval_args(_Args())


def test_full_evaluate_eval_args_sets_full_true(tmp_path):
    ea = _full_eval_args(tmp_path)
    assert ea.full is True


def test_full_evaluate_classification_eval_uses_every_clip(tmp_path):
    ea = _full_eval_args(tmp_path)
    _, loader, val_idx = _build_classification_eval(ea)
    assert len(loader.dataset) == 5
    assert len(val_idx) == 5


def test_full_evaluate_localization_eval_uses_every_clip(tmp_path):
    ea = _full_eval_args(tmp_path)
    _, loader, val_idx = _build_localization_eval(ea)
    assert len(loader.dataset) == 5
    assert len(val_idx) == 5


# ---------------------------------------------------------------------------
# --sweep_thresholds for the localizer path (M7)
# ---------------------------------------------------------------------------

def test_localizer_sweep_thresholds_included_in_metrics():
    """M7: --sweep_thresholds was a dead flag for the localizer path — it only
    produced a sweep in _finalize_classifier_report. The localizer report must
    include a threshold_sweep block with an optimal frame-F1 threshold."""
    from model.evaluation.evaluate import _run_localizer_sweep

    y_true = np.zeros(500)
    y_true[100:200] = 1
    y_pred = np.linspace(0.1, 0.9, 500)

    sweep = _run_localizer_sweep(y_true, y_pred, sr=16000, hop_length=512)

    assert "best_f1" in sweep
    assert "best_youden" in sweep
    assert 0.0 < sweep["best_f1"]["threshold"] < 1.0
    assert sweep["best_f1"]["f1"] > 0.0
    assert len(sweep["sweep"]) == len(THRESHOLD_SWEEP)


def test_localizer_sweep_flag_reaches_evaluate_localizer(tmp_path, monkeypatch):
    """The --sweep_thresholds flag must actually drive the localizer path."""
    import torch

    from model.evaluation import evaluate

    y_true = np.zeros(200)
    y_true[50:100] = 1

    class _FakeModel:
        def forward(self, x):
            return torch.full((x.shape[0], 1, 4), 0.5)

    class _FakeEval:
        def __len__(self):
            return 2

    data = _make_data(tmp_path, n=5)

    from model.evaluation import loader

    monkeypatch.setattr(loader, "load_localizer", lambda *a, **k: _FakeModel())

    def fake_build(args):
        import torch as _t

        audio = _t.randn(2, 1, 80, 10)
        labels = _t.tensor([y_true[:4], y_true[4:8]], dtype=_t.float32)
        return _FakeEval(), [(audio, labels)], [0, 1]

    monkeypatch.setattr(evaluate, "_build_localization_eval", fake_build)

    args = _args(data_dir=str(data), sweep_thresholds=True)
    metrics = evaluate.evaluate_localizer(args)

    assert "threshold_sweep" in metrics
    assert 0.0 < metrics["threshold_sweep"]["best_f1"]["f1"] <= 1.0


def test_localizer_sweep_all_negative_perfect_consistent():
    """All-negative frames must not crash the sweep; perfect agreement with
    zero events reports frame F1=1.0 (consistent with M4)."""
    from model.evaluation.evaluate import _run_localizer_sweep

    y_true = np.zeros(100)
    y_pred = np.zeros(100)

    sweep = _run_localizer_sweep(y_true, y_pred, sr=16000, hop_length=512)
    assert len(sweep["sweep"]) == len(THRESHOLD_SWEEP)
    assert sweep["best_f1"]["f1"] == 1.0
