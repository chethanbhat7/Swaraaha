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
    compute_classification_metrics,
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


def test_compute_localization_metrics_all_negative_perfect():
    """All-negative truth with all-negative predictions is perfect agreement:
    frame F1 must be 1.0, consistent with the event-level 1.0/1.0."""
    y_true = np.zeros(100, dtype=int)
    y_pred = np.zeros(100)

    m = compute_localization_metrics(y_true, y_pred, threshold=0.5, sr=16000, hop_length=512)

    assert m["frame_level"]["precision"] == 1.0
    assert m["frame_level"]["recall"] == 1.0
    assert m["frame_level"]["f1"] == 1.0
    assert m["event_level"]["detection_accuracy"] == 1.0
    assert m["event_level"]["mean_iou"] == 1.0
    assert m["event_level"]["num_true_events"] == 0
    assert m["event_level"]["num_pred_events"] == 0
    assert m["event_level"]["num_false_alarms"] == 0


def test_compute_localization_metrics_all_negative_with_false_alarms():
    """All-negative truth with predicted positives: every prediction is a false
    alarm (precision 0.0), and recall is trivially 1.0 (nothing to miss)."""
    y_true = np.zeros(100, dtype=int)
    y_pred = np.zeros(100)
    y_pred[30:45] = 1

    m = compute_localization_metrics(y_true, y_pred, threshold=0.5, sr=16000, hop_length=512)

    assert m["frame_level"]["precision"] == 0.0
    assert m["frame_level"]["recall"] == 1.0
    assert m["frame_level"]["f1"] == 0.0
    assert m["event_level"]["detection_accuracy"] == 0.0
    assert m["event_level"]["num_pred_events"] == 1
    assert m["event_level"]["num_false_alarms"] == 1


def test_compute_localization_metrics_all_positive_missed():
    """All-positive truth with nothing predicted: recall 0.0, but precision is
    trivially 1.0 (no false predictions)."""
    y_true = np.ones(100, dtype=int)
    y_pred = np.zeros(100)

    m = compute_localization_metrics(y_true, y_pred, threshold=0.5, sr=16000, hop_length=512)

    assert m["frame_level"]["precision"] == 1.0
    assert m["frame_level"]["recall"] == 0.0
    assert m["frame_level"]["f1"] == 0.0
    assert m["event_level"]["detection_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Multi-class classification metrics
# ---------------------------------------------------------------------------

def test_compute_classification_metrics_degenerate_perfect():
    """All samples are one class (index 1) and the model predicts them
    perfectly: macro-F1 must be 1.0, not 0.5. The absent class 0 has no
    support and no predictions, so it is trivially perfect (zero_division=1)."""
    y_true = np.ones(50, dtype=int)
    y_pred = np.ones(50, dtype=int)

    m = compute_classification_metrics(y_true, y_pred, class_names=["a", "b"])

    assert m["per_class"]["a"]["precision"] == 1.0
    assert m["per_class"]["a"]["recall"] == 1.0
    assert m["per_class"]["a"]["f1"] == 1.0
    assert m["per_class"]["b"]["precision"] == 1.0
    assert m["per_class"]["b"]["recall"] == 1.0
    assert m["per_class"]["b"]["f1"] == 1.0
    assert m["macro"]["f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_compute_classification_metrics_degenerate_class_false_alarms():
    """A class with no true samples but predicted samples scores 0: every
    prediction is a false alarm, even with the zero_division=1 default."""
    y_true = np.zeros(50, dtype=int)
    y_pred = np.zeros(50, dtype=int)
    y_pred[:10] = 1  # 10 false alarms of class 1

    m = compute_classification_metrics(y_true, y_pred, class_names=["a", "b"])

    assert m["per_class"]["a"]["recall"] == pytest.approx(0.8)
    assert m["per_class"]["a"]["precision"] == 1.0
    assert m["per_class"]["b"]["precision"] == 0.0
    assert m["per_class"]["b"]["recall"] == 1.0
    assert m["per_class"]["b"]["f1"] == 0.0


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
        "missing": ["localization.cnn — checkpoint not found"],
    })

    assert doc.startswith("# Swaraaha — Full Model Evaluation Summary")
    assert "Macro-averaged F1 (all 5 classes): 0.634" in doc
    assert "## Classification" in doc
    assert "## Localization" in doc
    assert "| prolongation" in doc
    assert "soundrep" in doc and "0.000" in doc
    assert "## Models / data not evaluated" in doc
    assert "localization.cnn — checkpoint not found" in doc


def test_build_classification_summary_passes_through_model_info():
    results = {
        "prolongation": {
            "precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 100,
            "model": {"fingerprint": "prolongation_e20_b16_...", "lr": 3e-05,
                      "model_name": "facebook/wav2vec2-base"},
        }
    }
    summary_result = summary.build_classification_summary(results)
    entry = summary_result["per_class"]["prolongation"]
    assert entry["model"]["fingerprint"] == "prolongation_e20_b16_..."
    assert entry["model"]["lr"] == 3e-05


def _sample_nested_binary_result():
    return {
        "class_name": "prolongation",
        "model_path": "model/weights/prolongation_e20_b8_best.pt",
        "num_samples": 100,
        "threshold": 0.5,
        "binary": {"auroc": 0.8, "auprc": 0.7, "threshold": 0.5,
                   "precision": 0.7, "recall": 0.6, "f1": 0.65,
                   "specificity": 0.8, "accuracy": 0.75, "support": 100,
                   "tn": 60, "fp": 15, "fn": 10, "tp": 15},
        "threshold_sweep": {"best_f1": {"threshold": 0.4, "f1": 0.66}},
        "model": {"fingerprint": "prolongation_e20_b8_best", "lr": 3e-05,
                  "model_name": "facebook/wav2vec2-base"},
    }


def test_build_classification_summary_reads_nested_binary_metrics():
    """evaluate_classifier returns metrics nested under 'binary'; the summary
    must read them from there instead of flat keys."""
    results = {"prolongation": _sample_nested_binary_result()}
    result = summary.build_classification_summary(results, threshold=0.7)

    entry = result["per_class"]["prolongation"]
    assert entry["f1"] == 0.65
    assert entry["precision"] == 0.7
    assert entry["recall"] == 0.6
    assert entry["support"] == 100
    assert entry["auroc"] == 0.8
    assert entry["auprc"] == 0.7
    assert entry["specificity"] == 0.8
    assert entry["threshold"] == 0.5
    assert entry["model"]["fingerprint"] == "prolongation_e20_b8_best"
    assert result["macro_f1"] == pytest.approx(0.65)
    assert len(result["flagged"]) == 1


def test_format_summary_markdown_survives_missing_metrics():
    """The markdown renderer must not crash with 'Unknown format code f' when
    a per-class result is missing precision/recall (e.g. an errored run)."""
    classification = summary.build_classification_summary(
        {"prolongation": {"status": "error", "model": {"fingerprint": ""}}},
        threshold=0.7,
    )
    doc = summary.format_summary_markdown({"classification": classification})
    assert "## Classification" in doc
    assert "| prolongation |" in doc


def test_format_summary_markdown_renders_nested_binary_summary():
    results = {"prolongation": _sample_nested_binary_result()}
    classification = summary.build_classification_summary(results, threshold=0.7)
    doc = summary.format_summary_markdown({
        "metadata": {"timestamp": "2026-01-01T00:00:00+00:00", "data_dir": "data",
                     "flag_threshold": 0.7},
        "classification": classification,
    })

    assert "| prolongation | 0.700 | 0.600 | 0.650 | 0.800 | 0.700 | 100 |" in doc
    assert "Macro-averaged F1 (all 5 classes): 0.65" in doc
