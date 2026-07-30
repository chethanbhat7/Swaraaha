#!/usr/bin/env python3
"""
Unified evaluation script for Swaraaha models.

Evaluates trained classifiers or localizers and produces a comprehensive report.

Usage:
    # Evaluate a single classifier:
    python -m model.evaluation.evaluate \
        --model_type classifier \
        --class_name prolongation \
        --model_path model/weights/prolongation_best.pt \
        --data_dir data

    # Evaluate the localization model:
    python -m model.evaluation.evaluate \
        --model_type localizer \
        --model_path model/weights/localizer_best.pt \
        --data_dir data

    # Evaluate all classifiers:
    for cls in prolongation block soundrep wordrep interjection; do
        python -m model.evaluation.evaluate \
            --model_type classifier --class_name $cls \
            --model_path model/weights/${cls}_best.pt --data_dir data
    done
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Swaraaha models.")
    parser.add_argument("--model_type", type=str, required=True, choices=["classifier", "localizer"],
                        help="Type of model to evaluate.")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Dysfluency class (required for classifier type).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory.")
    parser.add_argument("--output_dir", type=str, default="model/evaluation/reports", help="Report output directory.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--max_length_seconds", type=float, default=10.0, help="Max audio length.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification/detection threshold.")
    parser.add_argument("--save_misclassified", action="store_true", help="Save misclassified sample paths.")
    parser.add_argument("--sweep_thresholds", action="store_true",
                        help="Run threshold sweep and report optimal thresholds.")
    parser.add_argument("--n_mels", type=int, default=128, help="Mel bins (for localizer).")
    parser.add_argument("--hop_length", type=int, default=512, help="Hop length (for localizer).")
    return parser.parse_args()


def evaluate_classifier(args) -> Dict:
    """Evaluate a trained binary classifier."""
    import torch
    from torch.utils.data import DataLoader

    from model.classification import DYSFLUENCY_CLASSES
    from model.data.dataset import ClassificationDataset
    from model.evaluation.metrics import (
        compute_binary_metrics,
        compute_classification_metrics,
        confusion_matrix,
        print_binary_report,
        print_classification_report,
        save_confusion_matrix_plot,
        save_report,
    )
    from model.training.train_classifier import SubsetDataset, stratified_split
    from model.training.utils import load_checkpoint

    if args.class_name is None:
        print("ERROR: --class_name is required for classifier evaluation.")
        sys.exit(1)

    class_idx = DYSFLUENCY_CLASSES.index(args.class_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Evaluating classifier: {args.class_name.upper()}")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_dir}")

    # Load dataset
    dataset = ClassificationDataset(
        data_dir=args.data_dir, sr=16000, max_length_seconds=args.max_length_seconds,
    )
    print(f"  Total samples: {len(dataset)}")

    if len(dataset) == 0:
        print("  ERROR: No samples found.")
        sys.exit(1)

    # Use full dataset for evaluation (or could use a held-out test set)
    _, val_idx = stratified_split(dataset, val_ratio=0.2, seed=42)
    eval_dataset = SubsetDataset(dataset, val_idx)
    print(f"  Eval samples: {len(eval_dataset)}")

    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    cls_map = {
        "prolongation": "model.classification.prolongation.ProlongationClassifier",
        "block": "model.classification.block.BlockClassifier",
        "soundrep": "model.classification.soundrep.SoundRepClassifier",
        "wordrep": "model.classification.wordrep.WordRepClassifier",
        "interjection": "model.classification.interjection.InterjectionClassifier",
    }
    module_path, cls_name_str = cls_map[args.class_name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    ClassifierCls = getattr(mod, cls_name_str)

    model = ClassifierCls()
    load_checkpoint(args.model_path, model=model)
    model.model.to(device)
    model.model.eval()

    # Run inference
    all_preds, all_labels, all_scores = [], [], []
    misclassified = []

    with torch.no_grad():
        for batch_idx, (audio, labels) in enumerate(eval_loader):
            audio = audio.to(device)
            binary_labels = labels[:, class_idx]

            logits = model.forward(audio)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = (probs[:, 1] >= args.threshold).astype(int)

            all_scores.extend(probs[:, 1].tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(binary_labels.numpy().tolist())

            if args.save_misclassified:
                for i, (pred, true) in enumerate(zip(preds, binary_labels.numpy())):
                    if pred != true:
                        sample_idx = val_idx[batch_idx * args.batch_size + i]
                        info = eval_dataset.dataset.samples[sample_idx]
                        misclassified.append({
                            "path": info["audio_path"],
                            "predicted": int(pred),
                            "true": int(true),
                        })

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_scores)

    # Compute enhanced binary metrics (AUROC, AUPRC, specificity)
    binary_metrics = compute_binary_metrics(y_true, y_scores, threshold=args.threshold)
    print_binary_report(binary_metrics, args.class_name)

    # Also compute basic classification metrics
    metrics = compute_classification_metrics(y_true, y_pred, class_names=["not_present", "present"])

    # Merge binary metrics into report
    metrics["binary"] = binary_metrics

    # Add class info
    metrics["class_name"] = args.class_name
    metrics["model_path"] = args.model_path
    metrics["num_samples"] = len(eval_dataset)
    metrics["threshold"] = args.threshold

    # Threshold sweep
    if args.sweep_thresholds:
        print("\n  --- Threshold Sweep ---")
        sweep_results = []
        for t in np.arange(0.1, 0.91, 0.05):
            m = compute_binary_metrics(y_true, y_scores, threshold=t)
            sweep_results.append(m)
            print(f"  t={t:.2f}  F1={m['f1']:.3f}  P={m['precision']:.3f}  "
                  f"R={m['recall']:.3f}  Spec={m['specificity']:.3f}")

        # Find optimal thresholds
        from model.evaluation.metrics import find_optimal_threshold
        best_f1_t, best_f1_val = find_optimal_threshold(y_true, y_scores, metric="f1")
        best_spec_t, best_spec_val = find_optimal_threshold(y_true, y_scores, metric="specificity")
        best_youden_t, best_youden_val = find_optimal_threshold(y_true, y_scores, metric="youden")

        print(f"\n  Optimal thresholds:")
        print(f"    Best F1:         t={best_f1_t:.2f}  (F1={best_f1_val:.3f})")
        print(f"    Best Specificity: t={best_spec_t:.2f}  (Spec={best_spec_val:.3f})")
        print(f"    Best Youden's J:  t={best_youden_t:.2f}  (J={best_youden_val:.3f})")

        metrics["threshold_sweep"] = {
            "best_f1": {"threshold": best_f1_t, "f1": best_f1_val},
            "best_specificity": {"threshold": best_spec_t, "specificity": best_spec_val},
            "best_youden": {"threshold": best_youden_t, "youden": best_youden_val},
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, num_classes=2)

    # Print report
    print_classification_report(metrics)
    print(f"\n  Confusion Matrix (not_present=pred, present=pred):")
    print(f"                    pred_neg  pred_pos")
    print(f"  true_neg          {cm[0,0]:>6d}    {cm[0,1]:>6d}")
    print(f"  true_pos          {cm[1,0]:>6d}    {cm[1,1]:>6d}")

    # Save report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"{args.class_name}_report.json")
    save_report(metrics, report_path)
    print(f"\n  Report saved: {report_path}")

    # Save confusion matrix plot
    cm_path = os.path.join(args.output_dir, f"{args.class_name}_confusion_matrix.png")
    save_confusion_matrix_plot(cm, ["Not Present", "Present"], cm_path, title=f"{args.class_name} Confusion Matrix")
    print(f"  Confusion matrix saved: {cm_path}")

    # Save misclassified
    if args.save_misclassified and misclassified:
        mc_path = os.path.join(args.output_dir, f"{args.class_name}_misclassified.json")
        with open(mc_path, "w") as f:
            json.dump(misclassified, f, indent=2)
        print(f"  Misclassified samples saved: {mc_path} ({len(misclassified)} samples)")

    return metrics


def evaluate_localizer(args) -> Dict:
    """Evaluate a trained localization model."""
    import torch
    from torch.utils.data import DataLoader

    from model.data.dataset import LocalizationDataset
    from model.evaluation.metrics import (
        compute_localization_metrics,
        print_localization_report,
        save_report,
    )
    from model.training.train_localizer import SubsetDataset, split_dataset
    from model.training.utils import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Evaluating localization model")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_dir}")

    # Load dataset
    dataset = LocalizationDataset(
        data_dir=args.data_dir, sr=16000, n_mels=args.n_mels,
        hop_length=args.hop_length, max_length_seconds=args.max_length_seconds,
    )
    print(f"  Total samples: {len(dataset)}")

    if len(dataset) == 0:
        print("  ERROR: No samples found.")
        sys.exit(1)

    _, val_idx = split_dataset(dataset, val_ratio=0.2, seed=42)
    eval_dataset = SubsetDataset(dataset, val_idx)
    print(f"  Eval samples: {len(eval_dataset)}")

    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer
    model = CNNSpectrogramLocalizer(n_mels=args.n_mels)

    load_checkpoint(args.model_path, model=model)
    model.model.to(device)
    model.model.eval()

    # Run inference
    all_true, all_pred = [], []

    with torch.no_grad():
        for spectrograms, frame_labels in eval_loader:
            spectrograms = spectrograms.to(device)
            logits = model.forward(spectrograms).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_true.extend(frame_labels.numpy())
            all_pred.extend(probs)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # Compute metrics
    metrics = compute_localization_metrics(
        y_true, y_pred, threshold=args.threshold,
        sr=16000, hop_length=args.hop_length,
    )
    metrics["model_path"] = args.model_path
    metrics["num_samples"] = len(eval_dataset)
    metrics["threshold"] = args.threshold

    # Print report
    print_localization_report(metrics)

    # Save report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "localizer_report.json")
    save_report(metrics, report_path)
    print(f"\n  Report saved: {report_path}")

    return metrics


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == "classifier":
        evaluate_classifier(args)
    elif args.model_type == "localizer":
        evaluate_localizer(args)
