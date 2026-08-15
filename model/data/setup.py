# Swaraaha - Data Setup Orchestrator
# Runs the full data pipeline: download -> merge.
# Adapted from Major-Project/Scripts/complete_setup.py.
#
# Usage:
#   python -m model.data.setup          (from project root)

import argparse
import subprocess
import sys
import time
from pathlib import Path

from model.data.config import WORKFLOW_TIMEOUT_SECONDS


def run_step(script_module: str, description: str, extra_args: list[str] | None = None) -> bool:
    """
    Run a pipeline step as a subprocess.

    Args:
        script_module: Python module path (e.g. "model.data.download").
        description: Human-readable step description.
        extra_args: Additional CLI arguments forwarded to the step script.

    Returns:
        True if step succeeded.
    """
    print()
    print("=" * 70)
    print(f"  {description}")
    print("=" * 70)
    print()

    start = time.time()
    try:
        cmd = [sys.executable, "-m", script_module]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, timeout=WORKFLOW_TIMEOUT_SECONDS)
        elapsed = time.time() - start
        print(f"\n  Completed in {elapsed:.1f}s")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        timeout_min = WORKFLOW_TIMEOUT_SECONDS // 60
        print(f"  Timed out after {timeout_min} minutes")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Swaraaha Data Setup Orchestrator — download, merge, and prepare all datasets."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate per-clip label CSVs and overwrite existing split labels "
             "(forwarded to the merge and prepare steps).",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional args forwarded to every step (download, merge, prepare).",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    steps = [
        ("model.data.download", "STEP 1: DOWNLOADING ALL DATASETS"),
        ("model.data.merge", "STEP 2: COMBINING DATASETS"),
        ("model.data.prepare", "STEP 3: PREPARING TRAINING DATA"),
    ]

    print()
    print("+" + "-" * 68 + "+")
    print("|" + " SWARAAHA: COMPLETE DATA SETUP".center(68) + "|")
    print("+" + "-" * 68 + "+")

    results = {}
    for module, description in steps:
        step_extra = list(args.extra_args)
        # --force is meaningful for merge (regenerate interval CSVs) and
        # prepare (overwrite existing split labels). download has no
        # argparse and harmlessly ignores unknown argv, so leave it out.
        if args.force and module in ("model.data.merge", "model.data.prepare"):
            step_extra.insert(0, "--force")
        success = run_step(module, description, step_extra)
        results[description] = success
        if not success:
            print(f"\n  {description} failed. Run manually to retry:")
            print(f"    python -m {module}")
            break

    # Summary
    print()
    print("=" * 70)
    print("  WORKFLOW SUMMARY")
    print("=" * 70)
    for desc, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {desc}")

    if all(results.values()):
        from model.data.config import COMBINED_DATASET_PATH

        print()
        print("  Setup complete!")
        print(f"  Dataset ready at: {COMBINED_DATASET_PATH}")
        print()
        print("  Next steps:")
        print("    1. Run training: python -m model.training.train")
        print("    2. Or train individually: python -m model.training.train_classifier --class_name prolongation")
        return 0
    else:
        print()
        print("  Setup incomplete. Check errors above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
