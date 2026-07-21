# Swaraaha - Dataset Merger
# Normalizes SEP-28K, UCLASS, and Project Boli into a single combined CSV.
# Adapted from Major-Project/Scripts/merge_datasets.py.
#
# Output format: CSV with columns
#   - clip_file: path to .wav file
#   - Prolongation, Block, SoundRep, WordRep, Interjection: binary labels (0/1)
#
# Project Boli is fully implemented: parses .txt transcript files and maps
# annotation codes (B, PR, SR, WR, IN) to dysfluency labels.

import os
import re
from pathlib import Path

import pandas as pd

from model.data.config import (
    COMBINED_DATASET_PATH,
    DYSFLUENCY_LABELS,
    PROJECT_BOLI_LABEL_MAP,
    RAW_DATA_DIR,
)


def _find_matching_audio(audios_dir: Path, transcript_stem: str) -> Path | None:
    """Find the audio file matching a transcript filename."""
    candidates = [audios_dir / f"{transcript_stem}.wav"]
    candidates.extend(sorted(audios_dir.rglob(f"{transcript_stem}.wav")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_project_boli_transcript(transcript_path: Path) -> set:
    """Parse a Project Boli .txt transcript and return set of label names."""
    labels = set()

    with transcript_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line, maxsplit=2)
            if len(parts) < 3:
                continue

            code = parts[2].strip().upper()
            mapped = PROJECT_BOLI_LABEL_MAP.get(code)
            if mapped:
                labels.add(mapped)

    return labels


def normalize_project_boli() -> pd.DataFrame | None:
    """
    Normalize the Project Boli Dataset.

    Reads .txt transcript files, extracts dysfluency annotation codes,
    maps them to label names, and pairs with matching .wav audio files.

    Transcript format (per line):
        <start_ms> <end_ms> <annotation_code>
        e.g.: 1200 1800 PR    (prolongation)

    Returns:
        DataFrame with columns: clip_file, Prolongation, Block, SoundRep,
        WordRep, Interjection. Or None if dataset not found.
    """
    base_dir = Path(RAW_DATA_DIR) / "Project Boli Dataset"
    transcripts_dir = base_dir / "Transcripts"
    audios_dir = base_dir / "Audios"

    if not transcripts_dir.exists() or not audios_dir.exists():
        print("  Project Boli: Transcripts/ or Audios/ directory not found, skipping.")
        return None

    rows = []
    for transcript_path in sorted(transcripts_dir.glob("*.txt")):
        audio_path = _find_matching_audio(audios_dir, transcript_path.stem)
        if audio_path is None:
            continue

        labels_found = _parse_project_boli_transcript(transcript_path)
        if not labels_found:
            continue

        row = {"clip_file": str(audio_path)}
        for label in DYSFLUENCY_LABELS:
            row[label] = 1 if label in labels_found else 0
        rows.append(row)

    if not rows:
        print("  Project Boli: No valid examples found.")
        return None

    df = pd.DataFrame(rows)
    print(f"  Project Boli: {len(df)} examples")
    return df


def normalize_sep28k() -> pd.DataFrame | None:
    """
    Normalize the SEP-28K Dataset.

    Reads two label CSVs (fluencybank_labels.csv + SEP-28k_labels.csv),
    concatenates them, and constructs clip_file paths from Show/EpId/ClipId.

    Returns:
        DataFrame with clip_file + label columns. Or None if dataset not found.
    """
    base_dir = Path(RAW_DATA_DIR) / "SEP-28K Dataset"
    clips_dir = base_dir / "clips" / "stuttering-clips" / "clips"
    metadata_file1 = base_dir / "fluencybank_labels.csv"
    metadata_file2 = base_dir / "SEP-28k_labels.csv"

    if not metadata_file1.exists() and not metadata_file2.exists():
        print("  SEP-28K: Label CSVs not found, skipping.")
        return None

    dfs = []
    if metadata_file1.exists():
        dfs.append(pd.read_csv(metadata_file1))
    if metadata_file2.exists():
        dfs.append(pd.read_csv(metadata_file2))

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    df["clip_file"] = df.apply(
        lambda x: os.path.join(
            clips_dir,
            f"{x['Show']}_{int(x['EpId']):03d}_{int(x['ClipId']):01d}.wav",
        ),
        axis=1,
    )

    # Keep only the columns we need
    label_cols = [c for c in DYSFLUENCY_LABELS if c in df.columns]
    keep_cols = ["clip_file"] + label_cols
    df = df[keep_cols].copy()

    # Convert label values to binary (0/1)
    for col in label_cols:
        df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).astype(int)

    print(f"  SEP-28K: {len(df)} examples")
    return df


def normalize_uclass() -> pd.DataFrame | None:
    """
    Normalize the UCLASS SEP-28K Format dataset.

    Reads metadata.json, expands nested label dicts, and constructs
    clip_file paths from clip_id.

    Returns:
        DataFrame with clip_file + label columns. Or None if dataset not found.
    """
    base_dir = Path(RAW_DATA_DIR) / "UCLASS SEP-28K Format"
    clips_dir = base_dir / "clips" / "clips"
    metadata_file = base_dir / "clips" / "metadata.json"

    if not metadata_file.exists():
        print("  UCLASS: metadata.json not found, skipping.")
        return None

    df = pd.read_json(metadata_file)

    df["clip_file"] = df.apply(
        lambda x: os.path.join(clips_dir, x["clip_id"]), axis=1
    )

    # Expand nested labels dict into columns
    if "labels" in df.columns:
        expand_labels = pd.json_normalize(df["labels"])
        df = pd.concat([df, expand_labels], axis=1)

    # Keep only the columns we need
    label_cols = [c for c in DYSFLUENCY_LABELS if c in df.columns]
    keep_cols = ["clip_file"] + label_cols
    df = df[keep_cols].copy()

    # Convert label values to binary (0/1)
    for col in label_cols:
        df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).astype(int)

    print(f"  UCLASS: {len(df)} examples")
    return df


def merge_datasets(output_path: str | None = None) -> pd.DataFrame | None:
    """
    Merge all datasets into a single combined CSV.

    Runs each normalizer, concatenates valid results, and writes output.

    Args:
        output_path: Override output path. Defaults to config.COMBINED_DATASET_PATH.

    Returns:
        Combined DataFrame, or None if no data was produced.
    """
    print("Merging datasets...")

    print("  Normalizing Project Boli...")
    df_boli = normalize_project_boli()

    print("  Normalizing SEP-28K...")
    df_sep28k = normalize_sep28k()

    print("  Normalizing UCLASS...")
    df_uclass = normalize_uclass()

    dfs = [df for df in [df_boli, df_sep28k, df_uclass] if df is not None]

    if not dfs:
        print("  No datasets available to merge!")
        return None

    combined = pd.concat(dfs, ignore_index=True, ignore_statistics=False)

    # Ensure all label columns exist
    for label in DYSFLUENCY_LABELS:
        if label not in combined.columns:
            combined[label] = 0
    combined = combined[["clip_file"] + DYSFLUENCY_LABELS].copy()

    # Drop rows with missing clip_file
    combined = combined.dropna(subset=["clip_file"])
    combined = combined[combined["clip_file"] != ""]

    out = Path(output_path or COMBINED_DATASET_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    print(f"\n  Combined dataset: {len(combined)} examples")
    print(f"  Label distribution:")
    for label in DYSFLUENCY_LABELS:
        count = combined[label].sum()
        print(f"    {label}: {count} ({100 * count / len(combined):.1f}%)")
    print(f"  Saved to: {out}")

    return combined


if __name__ == "__main__":
    merge_datasets()
