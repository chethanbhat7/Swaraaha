# Swaraaha - Dataset Status Reporter
# Prints summary of merged data, splits, and data health without re-running setup.

import json
import os
from pathlib import Path

import pandas as pd

from model.data.config import COMBINED_DATASET_PATH, DYSFLUENCY_LABELS, DATA_DIR


def report():
    merged_path = Path(COMBINED_DATASET_PATH)
    data_dir = Path(DATA_DIR)
    splits_path = data_dir / "splits.json"

    print("=" * 60)
    print("  SWARAAHA: DATASET STATUS")
    print("=" * 60)

    # Merged dataset
    if not merged_path.exists():
        print("\n  No merged dataset found. Run: python -m model.data.setup")
        return

    df = pd.read_csv(merged_path)
    print(f"\n  Merged dataset: {merged_path}")
    print(f"  Total entries: {len(df)}")

    if len(df) == 0:
        print("  Merged dataset is empty; nothing to report.")
        return

    # Label distribution (only over columns actually present)
    label_cols = [c for c in DYSFLUENCY_LABELS if c in df.columns]
    if label_cols:
        print(f"\n  Label distribution:")
        for label in label_cols:
            count = df[label].sum()
            print(f"    {label}: {count} ({100 * count / len(df):.1f}%)")
    else:
        print(f"\n  Label distribution: no dysfluency label columns present")

    # Data health
    if "clip_file" in df.columns:
        missing = sum(1 for p in df["clip_file"] if not os.path.exists(p))
        empty = sum(1 for p in df["clip_file"] if os.path.exists(p) and os.path.getsize(p) == 0)
        print(f"\n  Data health:")
        print(f"    Missing files: {missing}")
        print(f"    Empty files: {empty}")
        print(f"    Valid: {len(df) - missing - empty}")
    else:
        print(f"\n  Data health: clip_file column missing, skipped")

    # Interval CSVs
    labels_dir = merged_path.parent / "labels"
    if labels_dir.exists():
        interval_count = len(list(labels_dir.glob("*.csv")))
        print(f"\n  Interval CSVs: {interval_count}")
    else:
        print(f"\n  Interval CSVs: 0")

    # Splits
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        print(f"\n  Splits ({splits_path}):")
        for split_name, clip_files in splits.items():
            broken = sum(1 for p in clip_files if not os.path.exists(p))
            valid = len(clip_files) - broken
            print(f"    {split_name}: {len(clip_files)} entries, {broken} broken, {valid} valid")

        # Per-split label dist
        if "clip_file" in df.columns:
            print(f"\n  Per-split label counts (sample > 0):")
            for split_name, clip_files in splits.items():
                subset = df[df["clip_file"].isin(clip_files)]
                counts = {label: int(subset[label].sum()) for label in label_cols}
                print(f"    {split_name}: {counts}")
        else:
            print(f"\n  Per-split label counts: clip_file column missing, skipped")
    else:
        print(f"\n  Splits: not yet created")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    report()
