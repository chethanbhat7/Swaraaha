#!/usr/bin/env python3
"""
Comprehensive evaluation runner for Swaraaha (Task 4.6: Full Evaluation).

Evaluates every trained model referenced by the model registry — the five
Wav2Vec2 binary classifiers and the registered localization models
(CNN / Wav2Vec2) — and writes a single machine-readable JSON report plus a
human-readable Markdown summary to ``model/evaluation/reports/``.

The summary reports per-class F1 and macro-averaged F1 for classification,
frame-level and event-level metrics for localization, and flags every class
whose F1 falls below the configured threshold (default 0.7).

Models or datasets that are not present are reported as such instead of
aborting the run, so the tool can be run on the training machine where the
checkpoints and data live.

Usage:
    python -m model.evaluation.full_evaluate --data_dir data
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Dict

from model.evaluation import evaluate, loader
from model.evaluation import summary as summary_mod

_DYSFLUENCY_CLASSES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive Swaraaha evaluation.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory (audio/ + labels/).")
    parser.add_argument("--output_dir", type=str, default="model/evaluation/reports",
                        help="Where per-model reports and the summary are written.")
    parser.add_argument("--registry", type=str, default=None,
                        help="Path to registry.json (default: model/registry.json).")
    parser.add_argument("--localizer_type", type=str, default=None,
                        choices=["cnn", "wav2vec2"],
                        help="Which localizer type(s) to evaluate. Default: all registered.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification/detection threshold.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation.")
    parser.add_argument("--max_length_seconds", type=float, default=10.0,
                        help="Max audio length.")
    parser.add_argument("--flag_f1", type=float, default=0.7,
                        help="Flag classes with F1 below this value.")
    parser.add_argument("--save_misclassified", action="store_true",
                        help="Save misclassified sample paths for each classifier.")
    parser.add_argument("--sweep_thresholds", action="store_true",
                        help="Run threshold sweeps and report optimal thresholds.")
    return parser.parse_args()


def _data_available(data_dir: str) -> bool:
    audio_dir = os.path.join(data_dir, "audio")
    return os.path.isdir(audio_dir) and any(
        f.endswith((".wav", ".flac", ".mp3")) for f in os.listdir(audio_dir)
    )


def _eval_args(args, **overrides):
    """Build an argparse.Namespace compatible with evaluate.py functions."""
    base = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "registry": args.registry,
        "batch_size": args.batch_size,
        "max_length_seconds": args.max_length_seconds,
        "threshold": args.threshold,
        "localizer_type": "cnn",
        "save_misclassified": args.save_misclassified,
        "sweep_thresholds": args.sweep_thresholds,
        "n_mels": 128,
        "hop_length": 512,
        "class_name": None,
        "model_path": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def main() -> int:
    args = parse_args()
    data_ok = _data_available(args.data_dir)
    missing: list = []

    registry = loader.registry_paths(args.registry)
    classification_paths = registry["classification"]
    localization_paths = registry["localization"]

    print("=" * 70)
    print("  SWARAAHA — FULL MODEL EVALUATION (Task 4.6)")
    print("=" * 70)
    print(f"  Data directory: {args.data_dir}  "
          f"({'OK' if data_ok else 'NOT AVAILABLE'})")

    # --- Classification: five binary classifiers ---------------------------
    per_class_results: Dict[str, Dict] = {}
    if not data_ok:
        missing.append(f"classification — data directory `{args.data_dir}` not available")
    else:
        for class_name in _DYSFLUENCY_CLASSES:
            path = classification_paths.get(class_name)
            if path is None or not os.path.isfile(path):
                per_class_results[class_name] = {
                    "status": "missing_weights", "class_name": class_name, "model_path": path,
                }
                missing.append(f"classification.{class_name} — checkpoint not found "
                               f"({path or 'no registry entry'})")
                continue
            print(f"\n  --- Evaluating classifier: {class_name} ---")
            ea = _eval_args(args, class_name=class_name, model_path=path)
            try:
                result = evaluate.evaluate_classifier(ea)
                result["status"] = "evaluated"
                per_class_results[class_name] = result
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {class_name}: {e}")
                per_class_results[class_name] = {"status": "error", "error": str(e),
                                                 "class_name": class_name}
                missing.append(f"classification.{class_name} — evaluation error: {e}")

    classification_summary = summary_mod.build_classification_summary(
        per_class_results, threshold=args.flag_f1
    )

    # --- Localization --------------------------------------------------------
    localizer_results: Dict[str, Dict] = {}
    if not data_ok:
        missing.append("localization — data directory not available")
    else:
        types_to_eval = (
            [args.localizer_type] if args.localizer_type else list(localization_paths.keys())
        )
        for lt in types_to_eval:
            path = localization_paths.get(lt)
            if path is None or not os.path.isfile(path):
                missing.append(f"localization.{lt} — checkpoint not found "
                               f"({path or 'no registry entry'})")
                continue
            print(f"\n  --- Evaluating localizer: {lt} ---")
            ea = _eval_args(args, localizer_type=lt, model_path=path)
            try:
                result = evaluate.evaluate_localizer(ea)
                result["status"] = "evaluated"
                localizer_results[lt] = result
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED localizer {lt}: {e}")
                localizer_results[lt] = {"status": "error", "error": str(e)}
                missing.append(f"localization.{lt} — evaluation error: {e}")

    localization_summary = summary_mod.build_localization_summary(localizer_results)

    # --- Assemble and write the comprehensive summary ------------------------
    summary = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_dir": args.data_dir,
            "data_available": data_ok,
            "threshold": args.threshold,
            "flag_threshold": args.flag_f1,
        },
        "classification": classification_summary,
        "localization": localization_summary,
        "missing": sorted(set(missing)),
    }

    json_path, md_path = summary_mod.write_summary(summary, args.output_dir)
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE SUMMARY")
    print("=" * 70)
    print(f"  Classification macro F1: {classification_summary['macro_f1']}")
    if classification_summary["flagged"]:
        print(f"  Flagged (F1 < {args.flag_f1}): "
              f"{', '.join(i['class'] for i in classification_summary['flagged'])}")
    else:
        print(f"  Flagged (F1 < {args.flag_f1}): none")
    print(f"  Localization models evaluated: {list(localizer_results)}")
    if missing:
        print(f"\n  NOT evaluated ({len(missing)}):")
        for item in missing:
            print(f"    - {item}")
    print(f"\n  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")

    return 0 if not classification_summary["flagged"] else 1


if __name__ == "__main__":
    sys.exit(main())
