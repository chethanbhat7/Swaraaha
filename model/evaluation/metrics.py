"""
Evaluation metrics for Swaraaha models.

Provides classification metrics (precision, recall, F1, confusion matrix)
and localization metrics (frame-level and event-level).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """
    Integrate ``y`` over ``x`` with the trapezoidal rule.

    numpy < 2.0 exposes ``np.trapz``; numpy >= 2.0 renamed it to
    ``np.trapezoid`` (and removed the old alias). This helper keeps the
    metrics working across both.
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Compute per-class and macro-averaged classification metrics.

    Args:
        y_true: Ground truth labels, shape (N,) with integer class indices,
                or (N, C) multi-hot encoded.
        y_pred: Predicted labels, shape (N,) with integer class indices,
                or (N, C) with probabilities/thresholded values.
        class_names: Optional list of class names for the report.
                     If None, returns numeric keys.

    Returns:
        Dict with keys:
            "per_class": dict mapping class_name -> {precision, recall, f1, support}
            "macro": {precision, recall, f1}
            "accuracy": float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle multi-hot encoding → convert to class indices
    if y_true.ndim == 2:
        y_true_idx = _multi_hot_to_indices(y_true)
    else:
        y_true_idx = y_true.astype(int)

    if y_pred.ndim == 2:
        y_pred_idx = _multi_hot_to_indices(y_pred)
    else:
        y_pred_idx = y_pred.astype(int)

    num_classes = max(y_true_idx.max(), y_pred_idx.max()) + 1

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    elif len(class_names) < num_classes:
        class_names = list(class_names) + [
            str(i) for i in range(len(class_names), num_classes)
        ]

    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for c in range(num_classes):
        tp = int(np.sum((y_true_idx == c) & (y_pred_idx == c)))
        fp = int(np.sum((y_true_idx != c) & (y_pred_idx == c)))
        fn = int(np.sum((y_true_idx == c) & (y_pred_idx != c)))
        support = int(np.sum(y_true_idx == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        name = class_names[c] if c < len(class_names) else str(c)
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    total = sum(supports)
    accuracy = float(np.sum(y_true_idx == y_pred_idx)) / total if total > 0 else 0.0

    macro = {
        "precision": round(float(np.mean(precisions)), 4),
        "recall": round(float(np.mean(recalls)), 4),
        "f1": round(float(np.mean(f1s)), 4),
    }

    return {
        "per_class": per_class,
        "macro": macro,
        "accuracy": round(accuracy, 4),
    }


def _multi_hot_to_indices(multi_hot: np.ndarray) -> np.ndarray:
    """Convert multi-hot encoded labels to class indices (first positive class)."""
    if multi_hot.ndim == 1:
        return multi_hot.astype(int)
    indices = np.argmax(multi_hot, axis=1)
    # For samples with no positives, mark as -1 (will be handled as no-prediction)
    no_pos = multi_hot.sum(axis=1) == 0
    indices[no_pos] = -1
    return indices


# ---------------------------------------------------------------------------
# Binary Classifier Metrics (AUROC, AUPRC, Specificity)
# ---------------------------------------------------------------------------

def compute_binary_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics including
    AUROC, AUPRC, specificity, and threshold-dependent metrics.

    This is the recommended metric function for evaluating individual
    dysfluency classifiers (prolongation, block, soundrep, etc.).

    Args:
        y_true: Ground truth binary labels (0 or 1), shape (N,).
        y_scores: Predicted probabilities (0-1), shape (N,).
        threshold: Decision threshold for P/R/F1 computation.

    Returns:
        Dict with:
            - "auroc": Area under ROC curve (threshold-independent)
            - "auprc": Area under PR curve (threshold-independent, better for imbalanced)
            - "threshold": the threshold used
            - "precision": at given threshold
            - "recall": at given threshold
            - "f1": at given threshold
            - "specificity": true negative rate (critical for screening)
            - "accuracy": at given threshold
            - "support": number of positive samples
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    auroc = _compute_auroc(y_true, y_scores)
    auprc = _compute_auprc(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "support": int(y_true.sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under ROC curve using the trapezoidal rule (no sklearn needed)."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    sorted_indices = np.argsort(-y_scores)
    sorted_true = y_true[sorted_indices]

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for label in sorted_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    return abs(_trapezoid(tpr_list, fpr_list))


def _compute_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under PR Curve (more informative than AUROC for imbalanced data)."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0

    sorted_indices = np.argsort(-y_scores)
    sorted_true = y_true[sorted_indices]

    tp = 0
    fp = 0
    prec_list = [1.0]
    rec_list = [0.0]

    for label in sorted_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        prec_list.append(tp / (tp + fp))
        rec_list.append(tp / n_pos)

    return abs(_trapezoid(prec_list, rec_list))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes a given metric.

    Useful for selecting operating point after training.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Predicted probabilities.
        metric: One of "f1", "specificity", "recall", "youden".

    Returns:
        Tuple of (best_threshold, best_metric_value).
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    thresholds = np.arange(0.05, 0.96, 0.05)
    best_thresh = 0.5
    best_val = 0.0

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        youden = recall + specificity - 1.0

        val_map = {"f1": f1, "specificity": specificity, "recall": recall, "youden": youden}
        val = val_map.get(metric, f1)

        if val > best_val:
            best_val = val
            best_thresh = thresh

    return float(best_thresh), float(best_val)


# ---------------------------------------------------------------------------
# Localization Metrics (Enhanced)
# ---------------------------------------------------------------------------

def compute_localization_metrics(
    y_true_frames: np.ndarray,
    y_pred_frames: np.ndarray,
    threshold: float = 0.5,
    iou_threshold: float = 0.5,
    sr: int = 16000,
    hop_length: int = 512,
) -> Dict[str, object]:
    """
    Compute frame-level and event-level localization metrics.

    Args:
        y_true_frames: Ground truth binary frame labels, shape (N,) or (N, T).
        y_pred_frames: Predicted probabilities per frame, shape (N,) or (N, T).
        threshold: Probability threshold for positive detection.
        iou_threshold: IoU threshold for event-level matching.
        sr: Sample rate (for converting frames to seconds).
        hop_length: Hop length (for converting frames to seconds).

    Returns:
        Dict with keys:
            "frame_level": {precision, recall, f1}
            "event_level": {detection_accuracy, mean_iou, num_true_events, num_pred_events}
    """
    y_true = np.asarray(y_true_frames).flatten()
    y_pred = np.asarray(y_pred_frames).flatten()

    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Binarize predictions
    y_pred_bin = (y_pred >= threshold).astype(int)

    # Frame-level metrics
    tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))

    # With no predicted positives, precision is trivially 1.0 (no false
    # alarms); with no true positives, recall is trivially 1.0 (nothing
    # missed). Keeps all-negative perfect predictions at F1=1.0, consistent
    # with _compute_event_metrics returning (1.0, 1.0) for empty region lists
    # (matches sklearn's zero_division=1 convention).
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    frame_level = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
    }

    # Event-level metrics
    true_regions = _extract_regions(y_true)
    pred_regions = _extract_regions(y_pred_bin)

    detection_accuracy, mean_iou = _compute_event_metrics(
        true_regions, pred_regions, iou_threshold
    )

    # False alarm rate (false positive events per minute of audio)
    total_duration_sec = min_len * hop_length / sr
    total_duration_min = total_duration_sec / 60.0
    n_false_alarms = len([r for r in pred_regions
                          if not any(_compute_iou(r[0], r[1], t[0], t[1]) >= iou_threshold
                                     for t in true_regions)])
    false_alarm_rate = n_false_alarms / total_duration_min if total_duration_min > 0 else 0.0

    # False alarm rate per minute (events that don't overlap any true event)
    false_alarm_events_per_min = round(false_alarm_rate, 4)

    event_level = {
        "detection_accuracy": round(detection_accuracy, 4),
        "mean_iou": round(mean_iou, 4),
        "num_true_events": len(true_regions),
        "num_pred_events": len(pred_regions),
        "num_false_alarms": n_false_alarms,
        "false_alarm_rate_per_min": false_alarm_events_per_min,
    }

    return {
        "frame_level": frame_level,
        "event_level": event_level,
    }


def _extract_regions(binary_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Extract contiguous positive regions from a binary mask. Returns (start, end) index pairs."""
    regions = []
    in_region = False
    start = 0

    for i, v in enumerate(binary_mask):
        if v == 1 and not in_region:
            in_region = True
            start = i
        elif v == 0 and in_region:
            in_region = False
            regions.append((start, i))
    if in_region:
        regions.append((start, len(binary_mask)))

    return regions


def _compute_event_metrics(
    true_regions: List[Tuple[int, int]],
    pred_regions: List[Tuple[int, int]],
    iou_threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute event-level detection accuracy and mean IoU.

    Detection accuracy = fraction of true events matched by at least one prediction.
    Mean IoU = average IoU of matched event pairs.
    """
    if not true_regions and not pred_regions:
        return 1.0, 1.0
    if not true_regions or not pred_regions:
        return 0.0, 0.0

    matched_true = set()
    matched_pred = set()
    ious = []

    for ti, (ts, te) in enumerate(true_regions):
        best_iou = 0.0
        best_pi = -1
        for pi, (ps, pe) in enumerate(pred_regions):
            if pi in matched_pred:
                continue
            iou = _compute_iou(ts, te, ps, pe)
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_iou >= iou_threshold and best_pi >= 0:
            matched_true.add(ti)
            matched_pred.add(best_pi)
            ious.append(best_iou)

    detection_accuracy = len(matched_true) / len(true_regions) if true_regions else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return detection_accuracy, mean_iou


def _compute_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Compute Intersection over Union between two intervals."""
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute a confusion matrix.

    Args:
        y_true: Ground truth class indices, shape (N,).
        y_pred: Predicted class indices, shape (N,).
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes),
        where entry [i, j] = count of true class i predicted as class j.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Save a confusion matrix as a PNG image.

    Args:
        cm: Confusion matrix of shape (C, C).
        class_names: List of class names.
        output_path: Path to save the PNG.
        title: Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report Saving
# ---------------------------------------------------------------------------

def save_report(report: dict, output_path: str) -> None:
    """Save an evaluation report as a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def print_classification_report(metrics: Dict) -> None:
    """Print a human-readable classification report to stdout."""
    print("\n  Classification Report")
    print("  " + "=" * 60)
    per_class = metrics.get("per_class", {})
    for cls_name, vals in per_class.items():
        print(
            f"  {cls_name:>20s}  "
            f"P={vals['precision']:.3f}  "
            f"R={vals['recall']:.3f}  "
            f"F1={vals['f1']:.3f}  "
            f"(n={vals['support']})"
        )
    print("  " + "-" * 60)
    macro = metrics["macro"]
    print(
        f"  {'macro avg':>20s}  "
        f"P={macro['precision']:.3f}  "
        f"R={macro['recall']:.3f}  "
        f"F1={macro['f1']:.3f}"
    )
    print(f"  Accuracy: {metrics['accuracy']:.3f}")


def print_binary_report(metrics: Dict, class_name: str = "") -> None:
    """Print a human-readable binary classifier report with AUROC/AUPRC/specificity."""
    header = f"  Binary Report — {class_name}" if class_name else "  Binary Report"
    print(f"\n{header}")
    print("  " + "=" * 60)
    print(f"  AUROC:       {metrics['auroc']:.3f}")
    print(f"  AUPRC:       {metrics['auprc']:.3f}")
    print(f"  Threshold:   {metrics['threshold']:.2f}")
    print("  " + "-" * 60)
    print(f"  Precision:   {metrics['precision']:.3f}")
    print(f"  Recall:      {metrics['recall']:.3f}")
    print(f"  F1:          {metrics['f1']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Support:     {metrics['support']}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")


def print_localization_report(metrics: Dict) -> None:
    """Print a human-readable localization report to stdout."""
    print("\n  Localization Report")
    print("  " + "=" * 60)
    fl = metrics["frame_level"]
    print(f"  Frame-level:  P={fl['precision']:.3f}  R={fl['recall']:.3f}  F1={fl['f1']:.3f}  Spec={fl.get('specificity', 0):.3f}")
    el = metrics["event_level"]
    print(
        f"  Event-level:  acc={el['detection_accuracy']:.3f}  "
        f"mean_IoU={el['mean_iou']:.3f}  "
        f"(true={el['num_true_events']}, pred={el['num_pred_events']})"
    )
    print(f"  False alarms: {el['num_false_alarms']} events, rate={el['false_alarm_rate_per_min']:.2f}/min")
