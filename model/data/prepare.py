# Swaraaha - Training Data Preparation
# Reads Dataset/combined_labels.csv + Dataset/labels/*.csv and creates
# a training-ready directory structure with train/val/test splits.
#
# Output structure:
#   data/
#   ├── audio/        (symlinks to actual audio files)
#   ├── labels/       (per-clip interval CSVs)
#   └── splits.json   (split assignments)

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.data.config import (
    COMBINED_DATASET_PATH,
    DYSFLUENCY_LABELS,
    PROJECT_ROOT,
)


def create_training_data(
    output_dir: str | Path | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    """
    Create training-ready directory structure from merged dataset.

    Args:
        output_dir: Where to create data/train, data/val, data/test.
                    Defaults to PROJECT_DIR / 'data'.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducible splits.

    Returns:
        Path to output directory.
    """
    out = Path(output_dir) if output_dir else Path(PROJECT_ROOT) / "data"
    merged_path = Path(COMBINED_DATASET_PATH)

    if not merged_path.exists():
        print(f"Error: Merged dataset not found at {merged_path}")
        print("Run 'python -m model.data.setup' first to download and merge datasets.")
        sys.exit(1)

    labels_dir = merged_path.parent / "labels"

    df = pd.read_csv(merged_path)
    print(f"Loaded {len(df)} clips from {merged_path}")

    # Filter out clips with missing audio files
    exists = df["clip_file"].apply(lambda p: Path(p).exists())
    missing = (~exists).sum()
    if missing > 0:
        print(f"  Skipping {missing} clips with missing audio files")
        df = df[exists].reset_index(drop=True)
    print(f"  {len(df)} clips with valid audio")

    # Filter out header-only WAV files (just 44-byte header, no audio data)
    has_data = df["clip_file"].apply(lambda p: Path(p).stat().st_size > 44)
    empty_hdr = (~has_data).sum()
    if empty_hdr > 0:
        print(f"  Skipping {empty_hdr} clips with empty audio (header-only)")
        df = df[has_data].reset_index(drop=True)
    print(f"  {len(df)} clips with usable audio")

    # Count interval CSVs available
    labels_dir = merged_path.parent / "labels"
    total_intervals = sum(1 for _ in labels_dir.glob("*.csv")) if labels_dir.exists() else 0
    print(f"  Interval CSVs available: {total_intervals}")

    # Create splits
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)

    split_indices = {
        "train": indices[:n_train].tolist(),
        "val": indices[n_train:n_train + n_val].tolist(),
        "test": indices[n_train + n_val:].tolist(),
    }

    # Build splits.json and create directories
    splits = {}
    all_label_found = 0
    all_label_missing = 0
    for split_name, idx_list in split_indices.items():
        split_dir = out / split_name
        audio_dir = split_dir / "audio"
        split_labels_dir = split_dir / "labels"
        audio_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        clip_files = df.iloc[idx_list]["clip_file"].tolist()
        splits[split_name] = clip_files

        symlinked = 0
        label_found = 0
        label_missing = 0
        for clip_file in clip_files:
            clip_path = Path(clip_file)
            clip_stem = clip_path.stem

            # Symlink audio
            target_audio = audio_dir / clip_path.name
            if not target_audio.exists():
                try:
                    os.symlink(clip_path, target_audio)
                except OSError:
                    shutil.copy2(clip_path, target_audio)
            symlinked += 1

            # Copy label CSV
            src_label = labels_dir / f"{clip_stem}.csv"
            dst_label = split_labels_dir / f"{clip_stem}.csv"
            if src_label.exists() and not dst_label.exists():
                shutil.copy2(src_label, dst_label)
                label_found += 1
            elif not src_label.exists():
                label_missing += 1

        all_label_found += label_found
        all_label_missing += label_missing
        print(f"  {split_name}: {len(clip_files)} clips, labels: {label_found} found, {label_missing} missing")

    # Save splits.json
    splits_path = out / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n  === Prepare Report ===")
    print(f"  Audio symlinks: {sum(len(v) for v in splits.values())}")
    print(f"  Labels copied: {all_label_found}, missing: {all_label_missing}")
    print(f"  Training data created at: {out}")
    print(f"  Splits saved to: {splits_path}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create training-ready data splits from merged dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: PROJECT_DIR/data)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train split ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    create_training_data(
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
