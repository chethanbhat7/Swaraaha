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
        n_fft=2048,
        full=False,
        model_path="model/weights/localizer_best.pt",
        output_dir="model/evaluation/reports",
        threshold=0.5,
        save_misclassified=False,
        sweep_thresholds=False,
        thresholds_path=None,
        sources=None,
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


def test_evaluate_multitask_reports_per_class_and_macro(tmp_path, monkeypatch, capsys):
    """--multitask eval loads the multitask model and reports per-class F1."""
    import json

    import torch

    import model.evaluation.evaluate as ev

    CLASS_NAMES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]

    class _FakeEval:
        def __len__(self):
            return 2

    class _FakeHead(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, pooled):
            return torch.full((pooled.shape[0], 2), self.offset)

    class _FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = torch.nn.ModuleDict({
                n: _FakeHead(1.0 if i % 2 == 0 else -1.0)
                for i, n in enumerate(CLASS_NAMES)
            })
            self.config = type("C", (), {"hidden_size": 8})()

        def forward(self, audio):
            return {n: self.heads[n](audio) for n in CLASS_NAMES}

    class _FakeModel:
        def __init__(self):
            self.model = _FakeBackbone()
            self.class_names = list(CLASS_NAMES)

        def forward(self, audio):
            return self.model(audio)

    def fake_build(args):
        labels = torch.zeros(2, 5, dtype=torch.uint8)
        labels[0, 0] = 1
        labels[1, 1] = 1
        audio = torch.randn(2, 16000)
        loader = [(audio, labels)]
        return _FakeEval(), loader, [0, 1]

    def fake_load(path):
        return _FakeModel()

    monkeypatch.setattr(ev, "_build_classification_eval", fake_build)
    from model.evaluation import loader as eval_loader

    monkeypatch.setattr(eval_loader, "load_multitask", fake_load)

    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    args = _args(
        model_type="multitask",
        model_path="weights/mt.pt",
        output_dir=str(out_dir),
    )
    ev.evaluate_multitask(args)

    captured = capsys.readouterr().out
    assert "mac avg" in captured or "macro avg" in captured

    report_path = out_dir / "multitask_report.json"
    assert report_path.exists()
    with open(report_path) as f:
        report = json.load(f)
    assert set(report["per_class"].keys()) == set(CLASS_NAMES)
    assert 0.0 <= report["macro_f1"] <= 1.0


def test_evaluate_multitask_uses_model_spectrogram_config(tmp_path, monkeypatch, capsys):
    """CNN multitask eval must prefer the checkpoint's stored n_mels/
    hop_length/n_fft over CLI defaults, so ablation configs evaluate on
    the exact preprocessing they were trained with."""
    import torch

    import model.evaluation.evaluate as ev

    CLASS_NAMES = ["block"]

    class _FakeHead(torch.nn.Module):
        def forward(self, pooled):
            return torch.zeros(pooled.shape[0], 2)

    class _FakeEval:
        def __len__(self):
            return 2

    class _FakeModel:
        def __init__(self):
            self.model = torch.nn.ModuleDict({"heads": torch.nn.ModuleDict({
                n: _FakeHead() for n in CLASS_NAMES})})
            self.class_names = list(CLASS_NAMES)
            self.n_mels = 256
            self.hop_length = 256
            self.n_fft = 1024

        def forward(self, audio):
            return {n: self.model["heads"][n](audio) for n in CLASS_NAMES}

    seen = {}

    def fake_build(args):
        seen["n_mels"] = args.n_mels
        seen["hop_length"] = args.hop_length
        seen["n_fft"] = args.n_fft
        labels = torch.zeros(2, 5, dtype=torch.uint8)
        audio = torch.randn(2, 16000)
        loader = [(audio, labels)]
        return _FakeEval(), loader, [0, 1]

    def fake_load(path):
        return _FakeModel()

    monkeypatch.setattr(ev, "_build_spectrogram_classification_eval", fake_build)
    from model.evaluation import loader as eval_loader

    monkeypatch.setattr(eval_loader, "load_multitask", fake_load)

    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    args = _args(
        model_type="multitask",
        model_path="weights/mt.pt",
        output_dir=str(out_dir),
    )
    ev.evaluate_multitask(args)

    assert seen == {"n_mels": 256, "hop_length": 256, "n_fft": 1024}


def test_evaluate_multitask_sweep_reports_per_class_optimum(tmp_path, monkeypatch, capsys):
    """--sweep_thresholds must report per-class optimal F1/Youden thresholds
    on the multitask path and a macro_f1_at_optimal headline."""
    import json

    import torch

    import model.evaluation.evaluate as ev

    CLASS_NAMES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]

    class _FakeEval:
        def __len__(self):
            return 2

    class _FakeHead(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, pooled):
            return torch.full((pooled.shape[0], 2), self.offset)

    class _FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = torch.nn.ModuleDict({
                n: _FakeHead(1.0 if i % 2 == 0 else -1.0)
                for i, n in enumerate(CLASS_NAMES)
            })
            self.config = type("C", (), {"hidden_size": 8})()

        def forward(self, audio):
            return {n: self.heads[n](audio) for n in CLASS_NAMES}

    class _FakeModel:
        def __init__(self):
            self.model = _FakeBackbone()
            self.class_names = list(CLASS_NAMES)

        def forward(self, audio):
            return self.model(audio)

    def fake_build(args):
        labels = torch.zeros(2, 5, dtype=torch.uint8)
        labels[0, 0] = 1
        labels[1, 1] = 1
        audio = torch.randn(2, 16000)
        loader = [(audio, labels)]
        return _FakeEval(), loader, [0, 1]

    monkeypatch.setattr(ev, "_build_classification_eval", fake_build)
    from model.evaluation import loader as eval_loader

    monkeypatch.setattr(eval_loader, "load_multitask", lambda path: _FakeModel())

    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    args = _args(
        model_type="multitask",
        model_path="weights/mt.pt",
        output_dir=str(out_dir),
        sweep_thresholds=True,
    )
    result = ev.evaluate_multitask(args)

    for name in CLASS_NAMES:
        sweep = result["per_class"][name]["threshold_sweep"]
        assert "best_f1" in sweep
        assert "best_youden" in sweep
        assert 0.0 <= sweep["best_f1"]["f1"] <= 1.0
        assert 0.0 < sweep["best_f1"]["threshold"] < 1.0
        assert sweep["f1_at_default"] == result["per_class"][name]["binary"]["f1"]
        assert len(sweep["sweep"]) == len(THRESHOLD_SWEEP)

    for name in CLASS_NAMES:
        sweep = result["per_class"][name]["threshold_sweep"]
        row_f1s = [row["f1"] for row in sweep["sweep"]]
        assert sweep["best_f1"]["f1"] == max(row_f1s)

    assert result["macro_f1_at_optimal"] >= result["macro_f1"]

    with open(out_dir / "multitask_report.json") as f:
        report = json.load(f)
    assert "macro_f1_at_optimal" in report
    assert report["macro_f1_at_optimal"] >= report["macro_f1"]

    args_plain = _args(
        model_type="multitask",
        model_path="weights/mt.pt",
        output_dir=str(out_dir),
    )
    result_plain = ev.evaluate_multitask(args_plain)
    assert "macro_f1_at_optimal" not in result_plain
    for name in CLASS_NAMES:
        assert "threshold_sweep" not in result_plain["per_class"][name]


def test_evaluate_multitask_sweep_writes_thresholds_file(tmp_path, monkeypatch):
    """--sweep_thresholds must persist per-class optimal thresholds to
    multitask_thresholds.json with the documented schema."""
    import json

    import torch

    import model.evaluation.evaluate as ev

    CLASS_NAMES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]

    class _FakeEval:
        def __len__(self):
            return 2

    class _FakeHead(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, pooled):
            return torch.full((pooled.shape[0], 2), self.offset)

    class _FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = torch.nn.ModuleDict({
                n: _FakeHead(1.0 if i % 2 == 0 else -1.0)
                for i, n in enumerate(CLASS_NAMES)
            })
            self.config = type("C", (), {"hidden_size": 8})()

        def forward(self, audio):
            return {n: self.heads[n](audio) for n in CLASS_NAMES}

    class _FakeModel:
        def __init__(self):
            self.model = _FakeBackbone()
            self.class_names = list(CLASS_NAMES)

        def forward(self, audio):
            return self.model(audio)

    def fake_build(args):
        labels = torch.zeros(2, 5, dtype=torch.uint8)
        labels[0, 0] = 1
        labels[1, 1] = 1
        audio = torch.randn(2, 16000)
        return _FakeEval(), [(audio, labels)], [0, 1]

    monkeypatch.setattr(ev, "_build_classification_eval", fake_build)
    from model.evaluation import loader as eval_loader

    monkeypatch.setattr(eval_loader, "load_multitask", lambda path: _FakeModel())

    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    args = _args(
        model_type="multitask",
        model_path="weights/mt.pt",
        output_dir=str(out_dir),
        sweep_thresholds=True,
    )
    ev.evaluate_multitask(args)

    path = out_dir / "multitask_thresholds.json"
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["model_path"] == "weights/mt.pt"
    assert set(data["thresholds"].keys()) == set(CLASS_NAMES)
    for name in CLASS_NAMES:
        spec = data["thresholds"][name]
        assert set(spec.keys()) == {
            "f1_threshold", "youden_threshold", "f1_at_optimal", "f1_at_0_5"
        }
        assert 0.0 <= spec["f1_threshold"] <= 1.0
        assert 0.0 <= spec["youden_threshold"] <= 1.0
        assert round(spec["f1_threshold"], 2) == spec["f1_threshold"]
        assert round(spec["youden_threshold"], 2) == spec["youden_threshold"]
    assert data["macro_f1_at_optimal"] >= data["macro_f1_at_0_5"]


def test_parse_args_sources_and_thresholds_path():
    from model.evaluation import evaluate

    args = evaluate.parse_args([
        '--model_type', 'multitask', '--model_path', 'x.pt',
        '--sources', 'boli,sep28k', '--thresholds_path', 't.json',
    ])
    assert args.sources == ['boli', 'sep28k']
    assert args.thresholds_path == 't.json'


def test_build_spectrogram_classification_eval_full(tmp_path):
    from model.evaluation import evaluate

    data_dir = _make_data(tmp_path)
    args = _args(data_dir=str(data_dir), n_mels=8, hop_length=256,
                 max_length_seconds=1.0, full=True)
    dataset, loader, indices = evaluate._build_spectrogram_classification_eval(args)
    assert len(dataset) == 5
    spec, labels = next(iter(loader))
    assert spec.shape == (2, 1, 8, 63)
    assert labels.shape == (2, 5)


def test_build_classification_eval_sources_filter(tmp_path):
    from model.evaluation import evaluate

    data_dir = _make_data(tmp_path)
    sources_csv = data_dir / 'sources.csv'
    with open(sources_csv, 'w', encoding='utf-8') as f:
        f.write('clip_id,source\n')
        for i in range(5):
            f.write(f'clip_{i:02d},sep28k\n')
        f.write('clip_00,boli\n')
    args = _args(data_dir=str(data_dir), max_length_seconds=2.0,
                 full=True, sources=['boli'])
    dataset, loader, indices = evaluate._build_classification_eval(args)
    assert len(dataset) == 1


def test_classifier_report_honors_thresholds_path(tmp_path):
    import json

    from model.evaluation import evaluate

    thresholds_json = tmp_path / 'thr.json'
    thresholds_json.write_text(json.dumps({
        'thresholds': {'block': {'f1_threshold': 0.65}},
    }))
    args = _args(data_dir='data/train', output_dir=str(tmp_path),
                 thresholds_path=str(thresholds_json))
    y_true = np.array([0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.6, 0.7, 0.8, 0.9])
    metrics = evaluate._finalize_classifier_report(
        args, 'block', 'model.pt', y_true, y_scores, num_samples=5)
    assert metrics['threshold_tuned']['threshold'] == 0.65
    assert metrics['binary']['f1'] != metrics['threshold_tuned']['f1']
