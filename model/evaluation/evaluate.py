#!/usr/bin/env python3
"""
Unified evaluation script for Swaraaha models.

Evaluates trained classifiers or localizers and
produces per-model reports plus (optionally) an aggregate report.

Usage:
    # Evaluate a single classifier:
    python -m model.evaluation.evaluate \
        --model_type classifier \
        --class_name prolongation \
        --model_path model/weights/prolongation_best.pt \
        --data_dir data

    # Evaluate all five classifiers (paths from the model registry):
    python -m model.evaluation.evaluate \
        --model_type classifier --all \
        --data_dir data

    # Evaluate the localization model (CNN or Wav2Vec2):
    python -m model.evaluation.evaluate \
        --model_type localizer \
        --model_path model/weights/localizer_best.pt \
        --data_dir data --localizer_type cnn

    # Comprehensive run over every registered model:
    python -m model.evaluation.full_evaluate --data_dir data
"""

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Swaraaha models.")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["classifier", "localizer", "multitask"],
                        help="Type of model to evaluate.")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Dysfluency class (required for classifier type).")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all five classifiers using paths from the registry.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory.")
    parser.add_argument("--output_dir", type=str, default="model/evaluation/reports",
                        help="Report output directory.")
    parser.add_argument("--registry", type=str, default=None,
                        help="Path to registry.json (default: model/registry.json).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation.")
    parser.add_argument("--max_length_seconds", type=float, default=10.0,
                        help="Max audio length.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification/detection threshold.")
    parser.add_argument("--localizer_type", type=str, default="cnn",
                        choices=["cnn", "wav2vec2"],
                        help="Localizer model type.")
    parser.add_argument("--save_misclassified", action="store_true",
                        help="Save misclassified sample paths.")
    parser.add_argument("--sweep_thresholds", action="store_true",
                        help="Run threshold sweep and report optimal thresholds.")
    parser.add_argument("--n_mels", type=int, default=128, help="Mel bins (for localizer).")
    parser.add_argument("--hop_length", type=int, default=512,
                        help="Hop length (for localizer).")
    parser.add_argument("--full", action="store_true",
                        help="Evaluate the ENTIRE data_dir (no 20% split). "
                             "Use with a prepared test split, e.g. --data_dir data/test.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_classification_eval(args):
    """Build the classification eval dataset/loader (20% stratified split)."""
    from torch.utils.data import DataLoader

    from model.data.dataset import ClassificationDataset
    from model.training.train_classifier import SubsetDataset, stratified_split

    dataset = ClassificationDataset(
        data_dir=args.data_dir, sr=16000, max_length_seconds=args.max_length_seconds,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {args.data_dir}. "
                           "Prepare the dataset first (see model/data/setup.py).")

    _, val_idx = stratified_split(dataset, val_ratio=0.2, seed=42)
    if args.full:
        indices = list(range(len(dataset)))
    else:
        indices = val_idx
    eval_dataset = SubsetDataset(dataset, indices)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    return eval_dataset, eval_loader, indices


def _build_localization_eval(args):
    """Build the localization eval dataset/loader (20% split)."""
    from torch.utils.data import DataLoader

    if args.localizer_type == "cnn":
        from model.data.dataset import LocalizationDataset
        from model.training.train_localizer import SubsetDataset, split_dataset

        dataset = LocalizationDataset(
            data_dir=args.data_dir, sr=16000, n_mels=args.n_mels,
            hop_length=args.hop_length, max_length_seconds=args.max_length_seconds,
        )
    else:
        from model.localization.wav2vec2_dataset import Wav2Vec2LocalizationDataset
        from model.training.train_classifier import SubsetDataset, stratified_split

        dataset = Wav2Vec2LocalizationDataset(
            data_dir=args.data_dir, sr=16000, max_length_seconds=args.max_length_seconds,
        )
        # Reuse stratified split; localization split helper shares the same shape.
        def split_dataset(ds, val_ratio=0.2, seed=42):
            return stratified_split(ds, val_ratio, seed)

    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {args.data_dir}. "
                           "Prepare the dataset first (see model/data/setup.py).")

    _, val_idx = split_dataset(dataset, val_ratio=0.2, seed=42)
    if args.full:
        indices = list(range(len(dataset)))
    else:
        indices = val_idx
    eval_dataset = SubsetDataset(dataset, indices)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    return eval_dataset, eval_loader, indices


def _move_model(model, device):
    """Move a model wrapper (and its internal nn.Module) to device and set eval mode."""
    if hasattr(model, "_model") and model._model is not None:
        model._model.to(device)
        model._model.eval()
    elif hasattr(model, "model"):
        model.model.to(device)
        model.model.eval()
    return model


def _run_binary_classifier(eval_loader, model, class_idx, device, threshold,
                           save_misclassified=False, val_idx=None, eval_dataset=None):
    """Run one binary classifier over the loader, returning y_true/y_scores."""
    import torch

    all_true, all_scores = [], []
    misclassified = []

    with torch.no_grad():
        for batch_idx, (audio, labels) in enumerate(eval_loader):
            audio = audio.to(device)
            binary_labels = labels[:, class_idx].cpu().numpy()

            logits = model.forward(audio)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            scores = probs[:, 1]
            preds = (scores >= threshold).astype(int)

            all_scores.extend(scores.tolist())
            all_true.extend(binary_labels.tolist())

            if save_misclassified:
                for i, (p, t) in enumerate(zip(preds, binary_labels)):
                    if p != t and val_idx is not None:
                        sample_idx = val_idx[batch_idx * eval_loader.batch_size + i]
                        info = eval_dataset.dataset.samples[sample_idx]
                        misclassified.append({
                            "path": info["audio_path"],
                            "predicted": int(p),
                            "true": int(t),
                        })

    return np.array(all_true), np.array(all_scores), misclassified


def _save_misclassified(args, misclassified, class_name):
    if misclassified:
        path = os.path.join(args.output_dir, f"{class_name}_misclassified.json")
        with open(path, "w") as f:
            json.dump(misclassified, f, indent=2)
        print(f"  Misclassified samples saved: {path} ({len(misclassified)} samples)")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _finalize_classifier_report(args, class_name, model_path, y_true, y_scores,
                                num_samples) -> Dict:
    from model.evaluation.metrics import (
        compute_binary_metrics,
        compute_classification_metrics,
        confusion_matrix,
        print_binary_report,
        print_classification_report,
        save_confusion_matrix_plot,
        save_report,
    )

    binary_metrics = compute_binary_metrics(y_true, y_scores, threshold=args.threshold)
    print_binary_report(binary_metrics, class_name)

    metrics = compute_classification_metrics(y_true, (y_scores >= args.threshold).astype(int),
                                             class_names=["not_present", "present"])
    metrics["binary"] = binary_metrics
    metrics["class_name"] = class_name
    metrics["model_path"] = model_path
    metrics["num_samples"] = num_samples
    metrics["threshold"] = args.threshold

    from model.evaluation.loader import model_info_from_path

    metrics["model"] = model_info_from_path(model_path)

    if args.sweep_thresholds:
        from model.evaluation.metrics import THRESHOLD_SWEEP, find_optimal_threshold

        print("\n  --- Threshold Sweep ---")
        sweep_results = []
        for t in THRESHOLD_SWEEP:
            m = compute_binary_metrics(y_true, y_scores, threshold=t)
            sweep_results.append(m)
            print(f"  t={t:.2f}  F1={m['f1']:.3f}  P={m['precision']:.3f}  "
                  f"R={m['recall']:.3f}  Spec={m['specificity']:.3f}")
        best_f1_t, best_f1_val = find_optimal_threshold(y_true, y_scores, metric="f1")
        best_spec_t, best_spec_val = find_optimal_threshold(y_true, y_scores, metric="specificity")
        best_youden_t, best_youden_val = find_optimal_threshold(y_true, y_scores, metric="youden")
        print("\n  Optimal thresholds:")
        print(f"    Best F1:          t={best_f1_t:.2f}  (F1={best_f1_val:.3f})")
        print(f"    Best Specificity: t={best_spec_t:.2f}  (Spec={best_spec_val:.3f})")
        print(f"    Best Youden's J:  t={best_youden_t:.2f}  (J={best_youden_val:.3f})")
        metrics["threshold_sweep"] = {
            "best_f1": {"threshold": best_f1_t, "f1": best_f1_val},
            "best_specificity": {"threshold": best_spec_t, "specificity": best_spec_val},
            "best_youden": {"threshold": best_youden_t, "youden": best_youden_val},
        }

    cm = confusion_matrix(y_true, (y_scores >= args.threshold).astype(int), num_classes=2)
    print_classification_report(metrics)
    print("\n  Confusion Matrix (not_present / present):")
    print("                    pred_neg  pred_pos")
    print(f"  true_neg          {cm[0, 0]:>6d}    {cm[0, 1]:>6d}")
    print(f"  true_pos          {cm[1, 0]:>6d}    {cm[1, 1]:>6d}")

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"{class_name}_report.json")
    save_report(metrics, report_path)
    print(f"  Report saved: {report_path}")

    cm_path = os.path.join(args.output_dir, f"{class_name}_confusion_matrix.png")
    save_confusion_matrix_plot(cm, ["Not Present", "Present"], cm_path,
                               title=f"{class_name} Confusion Matrix")
    print(f"  Confusion matrix saved: {cm_path}")

    return metrics


def evaluate_classifier(args) -> Dict:
    """Evaluate a single trained binary classifier."""
    import torch

    from model.classification import DYSFLUENCY_CLASSES
    from model.evaluation import loader

    if args.class_name is None:
        print("ERROR: --class_name is required for classifier evaluation.")
        sys.exit(1)

    class_idx = DYSFLUENCY_CLASSES.index(args.class_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Evaluating classifier: {args.class_name.upper()}")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_dir}")

    eval_dataset, eval_loader, val_idx = _build_classification_eval(args)
    print(f"  Eval samples: {len(eval_dataset)}")

    model = loader.load_classifier(args.class_name, args.model_path)
    _move_model(model, device)

    y_true, y_scores, misclassified = _run_binary_classifier(
        eval_loader, model, class_idx, device, args.threshold,
        save_misclassified=args.save_misclassified, val_idx=val_idx,
        eval_dataset=eval_dataset,
    )

    metrics = _finalize_classifier_report(args, args.class_name, args.model_path,
                                          y_true, y_scores, len(eval_dataset))
    if args.save_misclassified:
        _save_misclassified(args, misclassified, args.class_name)

    return metrics


def evaluate_all_classifiers(args) -> Dict[str, Dict]:
    """Evaluate all five classifiers using model paths from the registry."""
    from model.classification import DYSFLUENCY_CLASSES
    from model.evaluation import loader
    from model.evaluation.metrics import save_report
    from model.evaluation.summary import build_classification_summary

    paths = loader.registry_paths(args.registry)["classification"]
    results: Dict[str, Dict] = {}

    print("\n" + "=" * 60)
    print("  EVALUATING ALL FIVE CLASSIFIERS")
    print("=" * 60)

    for class_name in DYSFLUENCY_CLASSES:
        path = paths.get(class_name)
        if path is None or not os.path.isfile(path):
            print(f"\n  SKIPPING {class_name}: checkpoint not found "
                  f"({path or 'no registry entry'})")
            results[class_name] = {
                "status": "missing_weights",
                "class_name": class_name,
                "model_path": path,
            }
            continue

        args.class_name = class_name
        args.model_path = path
        try:
            results[class_name] = evaluate_classifier(args)
            results[class_name]["status"] = "evaluated"
        except Exception as e:  # noqa: BLE001
            print(f"\n  FAILED {class_name}: {e}")
            results[class_name] = {"status": "error", "error": str(e), "class_name": class_name}

    summary = build_classification_summary(results)

    print("\n" + "=" * 60)
    print("  AGGREGATE CLASSIFICATION SUMMARY")
    print("=" * 60)
    for name, metrics in summary["per_class"].items():
        print(f"  {name:>15s}  F1={metrics['f1']:.3f}  AUROC={metrics.get('auroc', '—')}  "
              f"support={metrics['support']}")
    print(f"  {'macro avg':>15s}  F1={summary['macro_f1']}")
    if summary["flagged"]:
        print(f"  WARNING — classes below F1 {summary['flag_threshold']}: "
              f"{', '.join(i['class'] for i in summary['flagged'])}")

    aggregate_path = os.path.join(args.output_dir, "all_classifiers_report.json")
    save_report({"per_class_results": results, "summary": summary}, aggregate_path)
    print(f"\n  Aggregate report saved: {aggregate_path}")

    return results


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------

def _run_localizer_sweep(y_true, y_pred, sr=16000, hop_length=512):
    """Run a frame-level threshold sweep for a localizer.

    Returns dict with:
        "sweep": list of per-threshold frame-F1 rows
        "best_f1": {"threshold", "f1"} maximizing frame F1
        "best_youden": {"threshold", "youden"} maximizing Youden's J

    Frame F1 (and Youden) are threshold-dependent for localizers, so the
    optimal operating point is not fixed at 0.5.
    """
    from model.evaluation.metrics import THRESHOLD_SWEEP, compute_localization_metrics

    sweep = []
    best_f1 = {"threshold": 0.5, "f1": 0.0}
    best_youden = {"threshold": 0.5, "youden": -float("inf")}

    for t in THRESHOLD_SWEEP:
        m = compute_localization_metrics(
            y_true, y_pred, threshold=t, sr=sr, hop_length=hop_length,
        )
        fl = m["frame_level"]
        row = {
            "threshold": float(t),
            "frame_f1": fl["f1"],
            "precision": fl["precision"],
            "recall": fl["recall"],
            "specificity": fl["specificity"],
        }
        sweep.append(row)
        if fl["f1"] > best_f1["f1"]:
            best_f1 = {"threshold": float(t), "f1": fl["f1"]}
        youden = fl["recall"] + fl["specificity"] - 1.0
        if youden > best_youden["youden"]:
            best_youden = {"threshold": float(t), "youden": youden}

    return {"sweep": sweep, "best_f1": best_f1, "best_youden": best_youden}


def evaluate_localizer(args) -> Dict:
    """Evaluate a trained localization model (CNN or Wav2Vec2)."""
    import torch

    from model.evaluation import loader
    from model.evaluation.metrics import (
        compute_localization_metrics,
        print_localization_report,
        save_report,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Evaluating localization model ({args.localizer_type})")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_dir}")

    eval_dataset, eval_loader, val_idx = _build_localization_eval(args)
    print(f"  Eval samples: {len(eval_dataset)}")

    model = loader.load_localizer(args.localizer_type, args.model_path)
    _move_model(model, device)

    all_true, all_pred = [], []
    with torch.no_grad():
        for inputs, frame_labels in eval_loader:
            inputs = inputs.to(device)
            logits = model.forward(inputs).squeeze(1)  # (B, T)
            from model.training.utils import align_frame_labels
            frame_labels = align_frame_labels(frame_labels, logits)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_true.extend(frame_labels.numpy())
            all_pred.extend(probs)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    metrics = compute_localization_metrics(
        y_true, y_pred, threshold=args.threshold, sr=16000, hop_length=args.hop_length,
    )
    metrics["model_type"] = args.localizer_type
    metrics["model_path"] = args.model_path
    metrics["num_samples"] = len(eval_dataset)
    metrics["threshold"] = args.threshold

    if args.sweep_thresholds:
        print("\n  --- Threshold Sweep (frame-level) ---")
        sweep = _run_localizer_sweep(y_true, y_pred, sr=16000, hop_length=args.hop_length)
        for row in sweep["sweep"]:
            print(f"  t={row['threshold']:.2f}  F1={row['frame_f1']:.3f}  "
                  f"P={row['precision']:.3f}  R={row['recall']:.3f}  "
                  f"Spec={row['specificity']:.3f}")
        print("\n  Optimal thresholds:")
        print(f"    Best frame F1:   t={sweep['best_f1']['threshold']:.2f}  "
              f"(F1={sweep['best_f1']['f1']:.3f})")
        print(f"    Best Youden's J: t={sweep['best_youden']['threshold']:.2f}  "
              f"(J={sweep['best_youden']['youden']:.3f})")
        metrics["threshold_sweep"] = {
            "best_f1": sweep["best_f1"],
            "best_youden": sweep["best_youden"],
            "sweep": sweep["sweep"],
        }

    from model.evaluation.loader import model_info_from_path

    metrics["model"] = model_info_from_path(args.model_path)

    print_localization_report(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"{args.localizer_type}_localizer_report.json")
    save_report(metrics, report_path)
    print(f"\n  Report saved: {report_path}")

    return metrics


def evaluate_multitask(args) -> Dict:
    """Evaluate the shared-backbone multitask classifier on all five heads."""
    import torch

    from model.classification import DYSFLUENCY_CLASSES
    from model.evaluation import loader
    from model.evaluation.metrics import (
        compute_binary_metrics,
        print_binary_report,
        save_report,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Evaluating multitask classifier")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_dir}")

    eval_dataset, eval_loader, _ = _build_classification_eval(args)
    print(f"  Eval samples: {len(eval_dataset)}")

    model = loader.load_multitask(args.model_path)
    _move_model(model, device)

    all_true = {name: [] for name in DYSFLUENCY_CLASSES}
    all_scores = {name: [] for name in DYSFLUENCY_CLASSES}

    with torch.no_grad():
        for audio, labels in eval_loader:
            audio = audio.to(device)
            logits = model.forward(audio)
            for i, name in enumerate(DYSFLUENCY_CLASSES):
                y_true = labels[:, i].cpu().numpy()
                probs = torch.softmax(logits[name], dim=-1).cpu().numpy()
                all_true[name].extend(y_true.tolist())
                all_scores[name].extend(probs[:, 1].tolist())

    results: Dict[str, Dict] = {}
    for name in DYSFLUENCY_CLASSES:
        y_true = np.array(all_true[name])
        y_scores = np.array(all_scores[name])
        binary = compute_binary_metrics(y_true, y_scores, threshold=args.threshold)
        print_binary_report(binary, name)
        results[name] = {
            "class_name": name,
            "model_path": args.model_path,
            "num_samples": len(eval_dataset),
            "threshold": args.threshold,
            "binary": binary,
            "support": int(y_true.sum()),
        }

    macro_f1 = float(np.mean([results[n]["binary"]["f1"] for n in DYSFLUENCY_CLASSES]))

    print("\n  Aggregate multitask summary:")
    for name in DYSFLUENCY_CLASSES:
        print(f"  {name:>15s}  F1={results[name]['binary']['f1']:.3f}  "
              f"AUROC={results[name]['binary']['auroc']:.3f}")
    print(f"  {'macro avg':>15s}  F1={macro_f1:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "multitask_report.json")
    save_report(
        {"per_class": results, "macro_f1": round(macro_f1, 4),
         "model_path": args.model_path},
        report_path,
    )
    print(f"\n  Report saved: {report_path}")

    return {"per_class": results, "macro_f1": macro_f1}


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == "classifier":
        if args.all:
            evaluate_all_classifiers(args)
        else:
            evaluate_classifier(args)
    elif args.model_type == "localizer":
        evaluate_localizer(args)
    elif args.model_type == "multitask":
        evaluate_multitask(args)
