# Swaraaha - Data Setup Orchestrator
# Runs the full data pipeline: download -> merge.
# Adapted from Major-Project/Scripts/complete_setup.py.
#
# Usage:
#   python -m model.data.setup          (from project root)

import subprocess
import sys
import time
from pathlib import Path

from model.data.config import WORKFLOW_TIMEOUT_SECONDS


def run_step(script_module: str, description: str) -> bool:
    """
    Run a pipeline step as a subprocess.

    Args:
        script_module: Python module path (e.g. "model.data.download").
        description: Human-readable step description.

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
        result = subprocess.run(
            [sys.executable, "-m", script_module],
            timeout=WORKFLOW_TIMEOUT_SECONDS,
        )
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


def main():
    steps = [
        ("model.data.download", "STEP 1: DOWNLOADING ALL DATASETS"),
        ("model.data.merge", "STEP 2: COMBINING DATASETS"),
    ]

    print()
    print("+" + "-" * 68 + "+")
    print("|" + " SWARAAHA: COMPLETE DATA SETUP".center(68) + "|")
    print("+" + "-" * 68 + "+")

    results = {}
    for module, description in steps:
        success = run_step(module, description)
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
        print("    1. Use Dataset class in model.data.dataset to load data")
        print("    2. Run training: python -m model.training.train")
    else:
        print()
        print("  Setup incomplete. Check errors above.")


if __name__ == "__main__":
    main()
