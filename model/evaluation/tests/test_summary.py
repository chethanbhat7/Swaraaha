"""
Tests for the Swaraaha evaluation summary and metric aggregation logic.

These tests exercise the pure-numpy metric computations and the summary
generation helpers without requiring trained model checkpoints or audio data.
"""

import numpy as np
import pytest

from model.evaluation import summary
from model.evaluation.metrics import (
    compute_binary_metrics,
    compute_localization_metrics,
)

# ---------------------------------------------------------------------------
# Binary classifier metrics
# ---------------------------------------------------------------------------

def test_compute_binary_metrics_perfect_classifier():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])

    m = compute_binary_metrics(y_true, y_scores, threshold=0.5)

    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["specificity"] == 1.0
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert m["support"] == 3
    assert m["tp"] == 3 and m["fp"] == 0 and m["fn"] == 0 and m["tn"] == 3


def test_compute_binary_metrics_at_specific_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.6, 0.4, 0.9])

    m = compute_binary_metrics(y_true, y_scores, threshold=0.5)

    # Predicted positives: samples 2 (0.6) and 4 (0.9) → one true, one false.
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


# ---------------------------------------------------------------------------
# Localization metrics (frame-level + event-level)
# ---------------------------------------------------------------------------

def test_compute_localization_metrics_perfect():
    y_true = np.zeros(100, dtype=int)
    y_true[20:40] = 1
    y_pred = y_true.astype(float).copy()

    m = compute_localization_metrics(y_true, y_pred, threshold=0.5, sr=16000, hop_length=512)

    assert m["frame_level"]["precision"] == 1.0
    assert m["frame_level"]["recall"] == 1.0
    assert m["frame_level"]["f1"] == 1.0
    assert m["event_level"]["detection_accuracy"] == 1.0
    assert m["event_level"]["mean_iou"] == 1.0
    assert m["event_level"]["num_true_events"] == 1
    assert m["event_level"]["num_pred_events"] == 1
    assert m["event_level"]["num_false_alarms"] == 0


def test_compute_localization_metrics_partial_overlap():
    y_true = np.zeros(100, dtype=int)
    y_true[20:40] = 1
    y_pred = np.zeros(100)
    y_pred[30:45] = 1  # overlaps 20-40 partially, extends beyond

    m = compute_localization_metrics(y_true, y_pred, threshold=0.5, sr=16000, hop_length=512)

    # One true event is detected (IoU = 10/(20+15-10) = 0.4 < 0.5 → not matched).
    assert m["event_level"]["detection_accuracy"] == 0.0
    assert m["event_level"]["num_true_events"] == 1
    assert m["event_level"]["num_pred_events"] == 1
    assert m["event_level"]["num_false_alarms"] == 1
    assert m["frame_level"]["f1"] > 0.0


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _sample_per_class():
    return {
        "prolongation": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "support": 100,
                         "auroc": 0.8, "auprc": 0.7, "threshold": 0.5},
        "block": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 200,
                  "auroc": 0.9, "auprc": 0.85, "threshold": 0.5},
        "soundrep": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 50,
                     "auroc": 0.5, "auprc": 0.1, "threshold": 0.5},
        "wordrep": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 80,
                    "auroc": 0.85, "auprc": 0.75, "threshold": 0.5},
        "interjection": {"precision": 0.95, "recall": 0.9, "f1": 0.92, "support": 300,
                         "auroc": 0.95, "auprc": 0.93, "threshold": 0.5},
    }


def test_macro_f1():
    per_class = _sample_per_class()
    assert summary.macro_f1(per_class) == pytest.approx(0.634, abs=1e-3)


def test_macro_f1_empty():
    assert summary.macro_f1({}) == 0.0


def test_flag_underperforming():
    per_class = _sample_per_class()
    flagged = summary.flag_underperforming(per_class, threshold=0.7)

    assert [name for name, _ in flagged] == ["soundrep", "prolongation"]
    assert flagged[0][1] == pytest.approx(0.0)
    # Worst first
    assert flagged[0][0] == "soundrep"


def test_flag_boundary_exactly_at_threshold_not_flagged():
    per_class = {"block": {"f1": 0.7}}
    assert summary.flag_underperforming(per_class, threshold=0.7) == []


def test_build_classification_summary():
    per_class_results = {
        name: dict(metrics, status="evaluated")
        for name, metrics in _sample_per_class().items()
    }
    result = summary.build_classification_summary(per_class_results, threshold=0.7)

    assert result["macro_f1"] == pytest.approx(0.634, abs=1e-3)
    assert len(result["flagged"]) == 2
    assert result["flagged"][0]["class"] == "soundrep"
    assert result["status"]["block"] == "evaluated"
    assert result["per_class"]["interjection"]["f1"] == 0.92


def test_format_summary_markdown():
    per_class_results = {
        name: dict(metrics, status="evaluated")
        for name, metrics in _sample_per_class().items()
    }
    classification = summary.build_classification_summary(per_class_results, threshold=0.7)
    localization = summary.build_localization_summary({
        "cnn": {
            "frame_level": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
            "event_level": {"detection_accuracy": 0.9, "mean_iou": 0.6,
                            "false_alarm_rate_per_min": 1.2},
            "threshold": 0.5, "num_samples": 100,
        }
    })
    doc = summary.format_summary_markdown({
        "metadata": {"timestamp": "2026-01-01T00:00:00+00:00", "data_dir": "data",
                     "flag_threshold": 0.7},
        "classification": classification,
        "localization": localization,
        "missing": ["combiner — no checkpoint provided"],
    })

    assert doc.startswith("# Swaraaha — Full Model Evaluation Summary")
    assert "Macro-averaged F1 (all 5 classes): 0.634" in doc
    assert "## Classification" in doc
    assert "## Localization" in doc
    assert "| prolongation" in doc
    assert "soundrep" in doc and "0.000" in doc
    assert "## Models / data not evaluated" in doc
    assert "combiner — no checkpoint provided" in doc
