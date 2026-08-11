"""
Comprehensive evaluation summary generation for Swaraaha.

Aggregates per-model evaluation results into a single machine-readable
JSON report and a human-readable Markdown summary. Provides the
macro-averaged F1 across the five dysfluency classes and flags classes
whose F1 falls below a configurable target (default 0.7, per Task 4.6).
"""

import os
from typing import Dict, List, Tuple

import numpy as np

FLAG_F1_THRESHOLD = 0.7

_CLASSIFICATION_FIELDS = ("precision", "recall", "f1", "support")


def macro_f1(per_class: Dict[str, Dict]) -> float:
    """
    Compute macro-averaged F1 across classes.

    Args:
        per_class: Dict mapping class name → metrics containing "f1".

    Returns:
        Mean F1 over all classes (0.0 if empty).
    """
    f1s = [float(v["f1"]) for v in per_class.values() if v.get("f1") is not None]
    return round(float(np.mean(f1s)), 4) if f1s else 0.0


def flag_underperforming(
    per_class: Dict[str, Dict], threshold: float = FLAG_F1_THRESHOLD
) -> List[Tuple[str, float]]:
    """
    Flag classes whose F1 is below the given threshold.

    Args:
        per_class: Dict mapping class name → metrics containing "f1".
        threshold: Minimum acceptable F1.

    Returns:
        Sorted list of (class_name, f1) for classes with f1 < threshold,
        worst first.
    """
    flagged = [(name, float(v["f1"])) for name, v in per_class.items()
               if v.get("f1") is not None and v["f1"] < threshold]
    return sorted(flagged, key=lambda item: item[1])


def build_classification_summary(
    per_class_results: Dict[str, Dict],
    threshold: float = FLAG_F1_THRESHOLD,
) -> Dict[str, object]:
    """
    Aggregate per-class binary classifier results into a summary block.

    Args:
        per_class_results: Dict mapping class name → result dict produced by
            ``compute_binary_metrics`` (may also carry model_path/status).
        threshold: F1 flag threshold.

    Returns:
        Summary dict with per_class metrics, macro_f1, flag_threshold and
        the list of flagged classes.
    """
    per_class: Dict[str, Dict] = {}
    status: Dict[str, str] = {}

    for name, result in per_class_results.items():
        status[name] = result.get("status", "evaluated")
        # evaluate_classifier nests the binary metrics under result['binary'];
        # fall back to flat keys for results that are already flat.
        metrics = result.get("binary") if isinstance(result.get("binary"), dict) else result
        per_class[name] = {k: metrics[k] for k in _CLASSIFICATION_FIELDS if k in metrics}
        for k in ("auroc", "auprc", "specificity"):
            if k in metrics:
                per_class[name][k] = metrics[k]
        per_class[name]["threshold"] = metrics.get("threshold", result.get("threshold", 0.5))
        if "model" in result:
            per_class[name]["model"] = result["model"]

    return {
        "per_class": per_class,
        "macro_f1": macro_f1(per_class),
        "flag_threshold": threshold,
        "flagged": [{"class": name, "f1": f1}
                    for name, f1 in flag_underperforming(per_class, threshold)],
        "status": status,
    }


def build_localization_summary(localizer_results: Dict[str, Dict]) -> Dict[str, object]:
    """
    Aggregate per-model localization results into a summary block.

    Args:
        localizer_results: Dict mapping localizer type (e.g. "cnn") → result
            dict produced by ``compute_localization_metrics``.

    Returns:
        Summary dict with frame-level and event-level metrics per model.
    """
    per_model: Dict[str, Dict] = {}
    for name, result in localizer_results.items():
        per_model[name] = {
            "frame_level": result.get("frame_level", {}),
            "event_level": result.get("event_level", {}),
            "threshold": result.get("threshold", 0.5),
            "num_samples": result.get("num_samples", 0),
        }
    return {"per_model": per_model}


def _f1_str(f1: float) -> str:
    return "—" if f1 is None else f"{f1:.3f}"


def _fmt(value, fmt: str = ".3f") -> str:
    """Format a metric value as a string, tolerating missing/None/non-numeric."""
    if value is None:
        return "—"
    try:
        return f"{value:{fmt}}"
    except (ValueError, TypeError):
        return str(value)


def format_summary_markdown(summary: Dict[str, object]) -> str:
    """
    Render a comprehensive summary as human-readable Markdown.

    Args:
        summary: The dict produced by the full-evaluation runner.

    Returns:
        Markdown string.
    """
    lines: List[str] = []
    lines.append("# Swaraaha — Full Model Evaluation Summary")
    lines.append("")

    metadata = summary.get("metadata", {})
    if metadata:
        lines.append(f"- Generated: {metadata.get('timestamp', '')}")
        lines.append(f"- Data directory: `{metadata.get('data_dir', '')}`")
        lines.append(f"- F1 flag threshold: {metadata.get('flag_threshold', FLAG_F1_THRESHOLD)}")
        lines.append("")

    missing = summary.get("missing", [])
    if missing:
        lines.append("## Models / data not evaluated")
        lines.append("")
        lines.append("The following could not be evaluated in this run:")
        for item in missing:
            lines.append(f"- {item}")
        lines.append("")

    classification = summary.get("classification")
    if classification:
        lines.append("## Classification")
        lines.append("")
        lines.append("Five parallel Wav2Vec2 binary classifiers — per-class F1 and macro F1.")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | AUROC | AUPRC | Support |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, metrics in classification.get("per_class", {}).items():
            f1 = metrics.get("f1")
            model_cell = metrics.get("model", {}).get("fingerprint", "")
            lines.append(
                f"| {name} | {_fmt(metrics.get('precision'))} | "
                f"{_fmt(metrics.get('recall'))} | {_f1_str(f1)} | "
                f"{_fmt(metrics.get('auroc'))} | {_fmt(metrics.get('auprc'))} | "
                f"{_fmt(metrics.get('support'), 'd')} |"
            )
            if model_cell:
                lines.append(f"| &nbsp; | model: `{model_cell}` | | | | | |")
        lines.append("")
        lines.append(
            f"**Macro-averaged F1 (all 5 classes): {classification.get('macro_f1', '—')}**"
        )
        lines.append("")

        flagged = classification.get("flagged", [])
        if flagged:
            lines.append(f"### Classes with F1 < {classification.get('flag_threshold')}")
            lines.append("")
            lines.append("| Class | F1 |")
            lines.append("|---|---|")
            for item in flagged:
                lines.append(f"| {item['class']} | {item['f1']:.3f} |")
            lines.append("")

    localization = summary.get("localization")
    if localization:
        lines.append("## Localization")
        lines.append("")
        lines.append("| Model | Frame P | Frame R | Frame F1 | Event acc | Mean IoU | False alarms/min |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, metrics in localization.get("per_model", {}).items():
            fl = metrics.get("frame_level", {})
            el = metrics.get("event_level", {})
            lines.append(
                f"| {name} | {fl.get('precision', '—')} | {fl.get('recall', '—')} | "
                f"{fl.get('f1', '—')} | {el.get('detection_accuracy', '—')} | "
                f"{el.get('mean_iou', '—')} | {el.get('false_alarm_rate_per_min', '—')} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("Generated by `model/evaluation/full_evaluate.py`.")
    lines.append("")
    return "\n".join(lines)


def write_summary(summary: Dict[str, object], output_dir: str) -> Tuple[str, str]:
    """
    Write the comprehensive summary as JSON and Markdown.

    Args:
        summary: Summary dict.
        output_dir: Directory where the report files are written.

    Returns:
        Tuple of (json_path, markdown_path).
    """
    import json
    import sys

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "evaluation_summary.json")
    md_path = os.path.join(output_dir, "SUMMARY.md")

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    markdown = format_summary_markdown(summary)
    if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
        pass  # allow non-ascii characters (e.g. en-dashes) in markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return json_path, md_path
