"""Generate presentation deliverables (Markdown + PNG) for the selected models.

Selected models (Part I decision):
  - Classification: wav2vec2 shared backbone + 5 heads (multitask w2v2, arm 2)
  - Localization:   wav2vec2 localizer (arm L2)

Reads the clean test/Boli reports already produced by evaluate.py and renders
per-class F1 bar charts plus a summary Markdown report.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.classification import DYSFLUENCY_CLASSES

ARM02_TEST = "model/evaluation/reports/arms/arm02_mt_w2v2_frz3/test/multitask_report.json"
ARM02_BOLI = "model/evaluation/reports/arms/arm02_mt_w2v2_frz3/boli/multitask_report.json"
ARML2_TEST = "model/evaluation/reports/arms/armL2_w2v2_loc/test/wav2vec2_localizer_report.json"
ARML2_BOLI = "model/evaluation/reports/arms/armL2_w2v2_loc/boli/wav2vec2_localizer_report.json"


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bar_chart(labels, series, path, title, ylabel, colors):
    n = len(labels)
    x = list(range(n))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, values) in enumerate(series.items()):
        ax.bar([xi + i * width for xi in x], values, width, label=name,
               color=colors[i % len(colors)])
    ax.set_xticks([xi + width * (len(series) - 1) / 2 for xi in x])
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=str, default="model/evaluation/reports/presentation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    test = _load(ARM02_TEST)
    boli = _load(ARM02_BOLI)
    loc_test = _load(ARML2_TEST)
    loc_boli = _load(ARML2_BOLI)

    labels = list(DYSFLUENCY_CLASSES)
    f1_05 = [test["per_class"][c]["binary"]["f1"] for c in labels]
    f1_tuned = [test["per_class"][c]["threshold_tuned"]["f1"] for c in labels]
    _bar_chart(
        labels,
        {"F1@0.5": f1_05, "F1@tuned": f1_tuned},
        os.path.join(args.output_dir, "classifier_test_f1.png"),
        "Multitask Wav2Vec2 — Test F1 by class (same-speaker held-out)",
        "F1",
        ["#4472c4", "#ed7d31"],
    )

    b_f1_05 = [boli["per_class"][c]["binary"]["f1"] for c in labels]
    b_f1_tuned = [boli["per_class"][c]["threshold_tuned"]["f1"] for c in labels]
    _bar_chart(
        labels,
        {"F1@0.5": b_f1_05, "F1@tuned": b_f1_tuned},
        os.path.join(args.output_dir, "classifier_boli_f1.png"),
        "Multitask Wav2Vec2 — Boli F1 by class (cross-corpus held-out)",
        "F1",
        ["#4472c4", "#ed7d31"],
    )

    auroc = [test["per_class"][c]["binary"]["auroc"] for c in labels]
    auprc = [test["per_class"][c]["binary"]["auprc"] for c in labels]
    _bar_chart(
        labels,
        {"AUROC": auroc, "AUPRC": auprc},
        os.path.join(args.output_dir, "classifier_test_auc.png"),
        "Multitask Wav2Vec2 — Test AUROC/AUPRC by class",
        "Score",
        ["#70ad47", "#7030a0"],
    )

    loc_rows = [
        ("Frame F1", loc_test["frame_level"]["f1"], loc_boli["frame_level"]["f1"]),
        ("Frame precision", loc_test["frame_level"]["precision"], loc_boli["frame_level"]["precision"]),
        ("Frame recall", loc_test["frame_level"]["recall"], loc_boli["frame_level"]["recall"]),
        ("Detection accuracy", loc_test["event_level"]["detection_accuracy"], loc_boli["event_level"]["detection_accuracy"]),
        ("Mean IoU", loc_test["event_level"]["mean_iou"], loc_boli["event_level"]["mean_iou"]),
    ]
    _bar_chart(
        [r[0] for r in loc_rows],
        {"Test": [r[1] for r in loc_rows], "Boli": [r[2] for r in loc_rows]},
        os.path.join(args.output_dir, "localizer_metrics.png"),
        "Wav2Vec2 Localizer — Test vs Boli",
        "Score",
        ["#4472c4", "#ed7d31"],
    )

    lines = ["# Swaraaha — Selected Model Presentation", ""]
    lines.append("Selected models (best-performing):")
    lines.append("- **Classification:** Wav2Vec2 shared backbone + 5 heads "
                 "(multitask), frozen backbone 3 layers, 3 s clips.")
    lines.append("- **Localization:** Wav2Vec2-based event localizer, 3 s clips.")
    lines.append("")
    lines.append("Protocol: thresholds tuned on internal val only (seed 42); "
                 "test = in-distribution held-out (same-speaker overlap); "
                 "Boli = cross-corpus held-out.")
    lines.append("")
    lines.append("## Classification — per-class F1 (multitask Wav2Vec2)")
    lines.append("")
    lines.append("| Class | Test F1@0.5 | Test F1@tuned | Test AUROC | Test AUPRC | Boli F1@0.5 | Boli F1@tuned |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in labels:
        t = test["per_class"][c]
        b = boli["per_class"][c]
        lines.append(
            f"| {c} | {t['binary']['f1']:.3f} | {t['threshold_tuned']['f1']:.3f} | "
            f"{t['binary']['auroc']:.3f} | {t['binary']['auprc']:.3f} | "
            f"{b['binary']['f1']:.3f} | {b['threshold_tuned']['f1']:.3f} |"
        )
    lines.append("")
    lines.append(f"**Macro F1:** test @0.5 = {test['macro_f1']:.4f}, "
                 f"test @tuned = {test['macro_f1_tuned']:.4f}, "
                 f"Boli @0.5 = {boli['macro_f1']:.4f}, "
                 f"Boli @tuned = {boli['macro_f1_tuned']:.4f}")
    lines.append("")
    lines.append("![Test F1 by class](classifier_test_f1.png)")
    lines.append("")
    lines.append("![Boli F1 by class](classifier_boli_f1.png)")
    lines.append("")
    lines.append("![Test AUROC/AUPRC by class](classifier_test_auc.png)")
    lines.append("")
    lines.append("## Localization — Wav2Vec2 localizer")
    lines.append("")
    lines.append("| Metric | Test | Boli |")
    lines.append("|---|---|---|")
    ft = loc_test["frame_level"]
    fb = loc_boli["frame_level"]
    lines.append(f"| Frame F1 | {ft['f1']:.3f} | {fb['f1']:.3f} |")
    lines.append(f"| Frame precision | {ft['precision']:.3f} | {fb['precision']:.3f} |")
    lines.append(f"| Frame recall | {ft['recall']:.3f} | {fb['recall']:.3f} |")
    lines.append(
        f"| Detection accuracy | {loc_test['event_level']['detection_accuracy']:.3f} "
        f"| {loc_boli['event_level']['detection_accuracy']:.3f} |"
    )
    lines.append(
        f"| Mean IoU | {loc_test['event_level']['mean_iou']:.3f} "
        f"| {loc_boli['event_level']['mean_iou']:.3f} |"
    )
    lines.append(
        f"| False alarms/min | {loc_test['event_level']['false_alarm_rate_per_min']:.2f} "
        f"| {loc_boli['event_level']['false_alarm_rate_per_min']:.2f} |"
    )
    lines.append("")
    lines.append("![Localizer metrics](localizer_metrics.png)")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Test set has known same-speaker overlap (in-distribution "
                 "held-out is optimistic).")
    lines.append("- Boli is the only cross-corpus held-out set (53 clips) — "
                 "numbers are noisy but honest.")
    lines.append("- Single seed 42; thresholds tuned on val only, no test or "
                 "Boli threshold fitting.")
    lines.append("")

    md_path = os.path.join(args.output_dir, "presentation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {md_path}")
    print(f"Wrote charts to {args.output_dir}")


if __name__ == "__main__":
    main()
