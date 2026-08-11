# Swaraaha - Dataset Merger
# Normalizes SEP-28K, UCLASS, and Project Boli into a single combined CSV.
# Adapted from Major-Project/Scripts/merge_datasets.py.
#
# Output format:
#   1. combined_labels.csv: clip_file + binary label columns (0/1)
#   2. labels/<clip_stem>.csv: per-clip interval CSVs (start_sec, end_sec, dysfluency_type)
#
# Project Boli is fully implemented: parses .txt transcript files and maps
# annotation codes (B, PR, SR, WR, IN) to dysfluency labels with intervals.

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from model.data.config import (
    COMBINED_DATASET_PATH,
    DYSFLUENCY_LABELS,
    PROJECT_BOLI_LABEL_MAP,
    RAW_DATA_DIR,
    SEP28K_CLIP_DURATION_SEC,
    UCLASS_EVENT_CENTER_SEC,
    UCLASS_INTERVAL_HALF_WIDTH_SEC,
)

# Source identifiers recorded in combined_labels.csv and sources.csv.
SOURCE_SEP28K = "sep28k"
SOURCE_UCLASS = "uclass"
SOURCE_BOLI = "boli"

# SEP-28K label columns hold aggregate 'yes' vote counts across ~3 annotators
# per clip. A clip is treated as positive only when a majority (>=2) of
# annotators marked it; counting any single vote inflated positives ~3.5x
# (e.g. Block 11,970 -> 3,370 clips) on this mirror.
SEP28K_MIN_VOTES = 2

# Boli transcript task codes -> audio filename task token.
# Transcript stems look like '{nEvents}_{speaker}_{task}' (e.g. 10_727253_EI)
# while audio files are '{speaker}_english_{task}_blob.wav'.
BOLI_TASK_CODE_TO_AUDIO = {
    "E1": "1_paragraph",
    "E2": "2_paragraph",
    "E3": "3_paragraph",
    "EI": "image",
}
_BOLI_STEM_RE = re.compile(r"^\d+_(\d+)_([A-Za-z]+\d*)(?:_\d+)?$")


def _boli_audio_name(transcript_stem: str) -> str | None:
    """Map a Boli transcript stem to its audio filename, or None if unmapped.

    Handles the leading event-count prefix and the optional trailing segment
    index (e.g. ``1_14446_E1_1`` -> ``14446_english_1_paragraph_blob.wav``).
    """
    match = _BOLI_STEM_RE.fullmatch(transcript_stem)
    if not match:
        return None
    speaker, task_code = match.group(1), match.group(2)
    audio_task = BOLI_TASK_CODE_TO_AUDIO.get(task_code.upper())
    if audio_task is None:
        return None
    return f"{speaker}_english_{audio_task}_blob.wav"


def _find_matching_audio(audios_dir: Path, transcript_stem: str) -> Path | None:
    """Find the audio file matching a transcript filename.

    Boli transcripts are named ``{nEvents}_{speaker}_{task}`` but the audio
    files are ``{speaker}_english_{task}_blob.wav``, so a mapped-name lookup
    is tried when the direct filename does not exist.
    """
    candidates = [audios_dir / f"{transcript_stem}.wav"]
    mapped = _boli_audio_name(transcript_stem)
    if mapped is not None:
        candidates.append(audios_dir / mapped)
    candidates.extend(sorted(audios_dir.rglob(f"{transcript_stem}.wav")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_project_boli_transcript(transcript_path: Path) -> Tuple[set, List[Tuple[float, float, str]]]:
    """Parse a Project Boli .txt transcript and return labels + intervals.

    Returns:
        Tuple of (label_set, intervals_list) where intervals_list contains
        (start_sec, end_sec, label_name) tuples.
    """
    labels = set()
    intervals = []

    with transcript_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line, maxsplit=2)
            if len(parts) < 3:
                continue

            start_sec = float(parts[0])
            end_sec = float(parts[1])
            code = parts[2].strip().upper()
            mapped = PROJECT_BOLI_LABEL_MAP.get(code)
            if mapped:
                labels.add(mapped)
                intervals.append((start_sec, end_sec, mapped))

    return labels, intervals


def normalize_project_boli() -> Tuple[Optional[pd.DataFrame], Dict[str, List[Tuple[float, float, str]]]]:
    """
    Normalize the Project Boli Dataset.

    Returns:
        Tuple of (DataFrame with binary labels, dict mapping clip_file to intervals).
        Or (None, {}) if dataset not found.
    """
    base_dir = Path(RAW_DATA_DIR) / "Project Boli Dataset"
    transcripts_dir = base_dir / "Transcripts"
    audios_dir = base_dir / "Audios"

    if not transcripts_dir.exists() or not audios_dir.exists():
        print("  Project Boli: Transcripts/ or Audios/ directory not found, skipping.")
        return None, {}

    rows = []
    all_intervals = {}
    for transcript_path in sorted(transcripts_dir.glob("*.txt")):
        audio_path = _find_matching_audio(audios_dir, transcript_path.stem)
        if audio_path is None:
            continue

        labels_found, intervals = _parse_project_boli_transcript(transcript_path)
        if not labels_found:
            continue

        row = {"clip_file": str(audio_path)}
        for label in DYSFLUENCY_LABELS:
            row[label] = 1 if label in labels_found else 0
        row["source"] = SOURCE_BOLI
        rows.append(row)
        all_intervals[str(audio_path)] = intervals

    if not rows:
        print("  Project Boli: No valid examples found.")
        return None, {}

    df = pd.DataFrame(rows)
    print(f"  Project Boli: {len(df)} examples")
    return df, all_intervals


def normalize_sep28k() -> Tuple[Optional[pd.DataFrame], Dict[str, List[Tuple[float, float, str]]]]:
    """
    Normalize the SEP-28K Dataset.

    Returns:
        Tuple of (DataFrame with binary labels, dict mapping clip_file to intervals).
        Or (None, {}) if dataset not found.
    """
    base_dir = Path(RAW_DATA_DIR) / "SEP-28K Dataset"
    clips_dir = base_dir / "clips" / "stuttering-clips" / "clips"
    metadata_file1 = base_dir / "fluencybank_labels.csv"
    metadata_file2 = base_dir / "SEP-28k_labels.csv"

    if not metadata_file1.exists() and not metadata_file2.exists():
        print("  SEP-28K: Label CSVs not found, skipping.")
        return None, {}

    dfs = []
    if metadata_file1.exists():
        dfs.append(pd.read_csv(metadata_file1))
    if metadata_file2.exists():
        dfs.append(pd.read_csv(metadata_file2))

    if not dfs:
        return None, {}

    df = pd.concat(dfs, ignore_index=True)

    # Build a lookup of actual filenames on disk (show -> {(ep, clip) -> filename})
    existing_files = {}
    for f in clips_dir.glob("*.wav"):
        parts = f.stem.split("_")
        if len(parts) >= 3:
            show = "_".join(parts[:-2])
            try:
                ep = int(parts[-2])
                clip = int(parts[-1])
                existing_files.setdefault(show, {})[(ep, clip)] = f.name
            except ValueError:
                continue

    def _safe_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _find_clip_file(row):
        show = row["Show"]
        ep = _safe_int(row["EpId"])
        clip = _safe_int(row["ClipId"])
        if ep is None or clip is None:
            return None
        show_files = existing_files.get(show, {})
        filename = show_files.get((ep, clip))
        if filename:
            return os.path.join(clips_dir, filename)
        # Fallback: try constructed name
        return os.path.join(
            clips_dir,
            f"{show}_{ep}_{clip}.wav",
        )

    df["clip_file"] = df.apply(_find_clip_file, axis=1)
    n_dropped = int(df["clip_file"].isna().sum())
    if n_dropped:
        print(f"  SEP-28K: skipping {n_dropped} row(s) with malformed Episode/Clip IDs")
        df = df[df["clip_file"].notna()].copy()

    # SEP-28K Start/Stop columns are episode-relative SAMPLE boundaries (the
    # clip edges Apple's extract_clips.py used to cut the 3s clips), NOT
    # event intervals. There is no event-level ground truth, so each present
    # clip-level label is materialized as a full-clip weak interval.
    all_intervals = {}
    for clip_file, group in df.groupby("clip_file"):
        clip_intervals = []
        for label in DYSFLUENCY_LABELS:
            if label in group.columns:
                val = pd.to_numeric(group[label], errors="coerce")
                if val.fillna(0).ge(SEP28K_MIN_VOTES).any():
                    clip_intervals.append(
                        (0.0, SEP28K_CLIP_DURATION_SEC, label)
                    )
        if clip_intervals:
            all_intervals[clip_file] = clip_intervals

    # Keep only the columns we need
    label_cols = [c for c in DYSFLUENCY_LABELS if c in df.columns]
    keep_cols = ["clip_file"] + label_cols
    df = df[keep_cols].copy()
    df["source"] = SOURCE_SEP28K

    # Convert label values to binary (0/1): positive requires a majority
    # (>= SEP28K_MIN_VOTES) of annotators to have marked the clip.
    for col in label_cols:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce").fillna(0) >= SEP28K_MIN_VOTES
        ).astype(int)

    print(f"  SEP-28K: {len(df)} examples")
    return df, all_intervals


def normalize_uclass() -> Tuple[Optional[pd.DataFrame], Dict[str, List[Tuple[float, float, str]]]]:
    """
    Normalize the UCLASS SEP-28K Format dataset.

    Returns:
        Tuple of (DataFrame with binary labels, dict mapping clip_file to intervals).
        Or (None, {}) if dataset not found.
    """
    base_dir = Path(RAW_DATA_DIR) / "UCLASS SEP-28K Format"
    clips_dir = base_dir / "clips" / "clips"
    metadata_file = base_dir / "clips" / "metadata.json"

    if not metadata_file.exists():
        print("  UCLASS: metadata.json not found, skipping.")
        return None, {}

    df = pd.read_json(metadata_file)

    # Build a lookup of actual filenames on disk
    existing_clips = {}
    for f in clips_dir.glob("*.wav"):
        existing_clips[f.stem] = f.name

    def _find_uclass_clip(row):
        clip_id = row["clip_id"]
        stem = Path(clip_id).stem
        filename = existing_clips.get(stem)
        if filename:
            return os.path.join(clips_dir, filename)
        return os.path.join(clips_dir, clip_id)

    df["clip_file"] = df.apply(_find_uclass_clip, axis=1)

    # Expand nested labels dict into columns
    if "labels" in df.columns:
        expand_labels = pd.json_normalize(df["labels"])
        df = pd.concat([df, expand_labels], axis=1)

    # UCLASS clips are 3s windows centered on the stutter event, so the event
    # sits ~1.5s into every dysfluent clip. Metadata carries clip-level labels
    # only (no intervals), so each present label becomes a centered interval.
    all_intervals = {}
    for _, row in df.iterrows():
        clip_file = row["clip_file"]
        is_fluent = bool(row.get("is_fluent", True))
        if is_fluent:
            continue
        clip_intervals = []
        for label in DYSFLUENCY_LABELS:
            if label in df.columns:
                val = pd.to_numeric(row.get(label), errors="coerce")
                if not pd.isna(val) and val > 0:
                    clip_intervals.append(
                        (
                            UCLASS_EVENT_CENTER_SEC - UCLASS_INTERVAL_HALF_WIDTH_SEC,
                            UCLASS_EVENT_CENTER_SEC + UCLASS_INTERVAL_HALF_WIDTH_SEC,
                            label,
                        )
                    )
        if clip_intervals:
            all_intervals[clip_file] = clip_intervals

    # Keep only the columns we need
    label_cols = [c for c in DYSFLUENCY_LABELS if c in df.columns]
    keep_cols = ["clip_file"] + label_cols
    df = df[keep_cols].copy()
    df["source"] = SOURCE_UCLASS

    # Convert label values to binary (0/1)
    for col in label_cols:
        df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).astype(int)

    print(f"  UCLASS: {len(df)} examples")
    return df, all_intervals


def merge_datasets(output_path: str | None = None, force: bool = False) -> pd.DataFrame | None:
    """
    Merge all datasets into a single combined CSV and per-clip interval CSVs.

    Runs each normalizer, concatenates valid results, writes:
    - combined_labels.csv (binary multi-label + source column)
    - labels/<clip_stem>.csv (per-clip interval CSVs)

    Args:
        output_path: Override output path. Defaults to config.COMBINED_DATASET_PATH.
        force: Rewrite per-clip interval CSVs even when their content is
            unchanged. Stale CSVs (content differs from the merge output) are
            always regenerated.

    Returns:
        Combined DataFrame, or None if no data was produced.
    """
    print("Merging datasets...")

    print("  Normalizing Project Boli...")
    df_boli, intervals_boli = normalize_project_boli()

    print("  Normalizing SEP-28K...")
    df_sep28k, intervals_sep28k = normalize_sep28k()

    print("  Normalizing UCLASS...")
    df_uclass, intervals_uclass = normalize_uclass()

    dfs = [df for df in [df_boli, df_sep28k, df_uclass] if df is not None]

    if not dfs:
        print("  No datasets available to merge!")
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Ensure all label columns exist
    for label in DYSFLUENCY_LABELS:
        if label not in combined.columns:
            combined[label] = 0
    if "source" not in combined.columns:
        combined["source"] = SOURCE_SEP28K
    combined = combined[["clip_file"] + DYSFLUENCY_LABELS + ["source"]].copy()

    # Drop rows with missing clip_file
    before_drop = len(combined)
    combined = combined.dropna(subset=["clip_file"])
    combined = combined[combined["clip_file"] != ""]
    dropped = before_drop - len(combined)

    out = Path(output_path or COMBINED_DATASET_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    # Validate audio files exist
    import os as _os
    valid_mask = combined["clip_file"].apply(lambda p: _os.path.exists(p))
    missing_count = (~valid_mask).sum()

    # Write per-clip interval CSVs
    all_intervals = {}
    all_intervals.update(intervals_boli)
    all_intervals.update(intervals_sep28k)
    all_intervals.update(intervals_uclass)

    labels_dir = out.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    no_intervals = 0
    skipped = 0
    for clip_file in combined["clip_file"]:
        clip_stem = Path(clip_file).stem
        label_path = labels_dir / f"{clip_stem}.csv"

        intervals = all_intervals.get(clip_file, [])
        if not intervals:
            no_intervals += 1
        content = "start_sec,end_sec,dysfluency_type\n" + "".join(
            f"{start:.3f},{end:.3f},{dtype}\n" for start, end, dtype in intervals
        )

        # Always keep per-clip CSVs in sync with combined_labels.csv: rewrite
        # when content changed. Only skip when identical and not forced, so
        # re-merges don't gratuitously churn mtimes (which would invalidate
        # the audio caches keyed on label mtime/content).
        if label_path.exists() and not force and label_path.read_text() == content:
            skipped += 1
            continue

        with open(label_path, "w") as f:
            f.write(content)
        written += 1

    # Write sources.csv mapping clip_id -> dataset source
    sources_path = out.parent / "sources.csv"
    combined[["clip_file"]].assign(
        source=combined["source"],
        clip_id=combined["clip_file"].apply(lambda p: Path(p).stem),
    )[["clip_id", "source"]].to_csv(sources_path, index=False)

    # Summary report
    print(f"\n  === Merge Report ===")
    print(f"  Total entries: {len(combined)}")
    if dropped > 0:
        print(f"  Dropped (missing clip_file): {dropped}")
    if missing_count > 0:
        print(f"  Missing audio files: {missing_count} ({100 * missing_count / len(combined):.1f}%)")
    print(f"  Interval CSVs written: {written}")
    if skipped > 0:
        print(f"  Interval CSVs skipped (up to date): {skipped}")
    if no_intervals > 0:
        print(f"  Clips without interval data: {no_intervals}")
    print(f"  Sources CSV: {sources_path}")
    print(f"\n  Label distribution:")
    for label in DYSFLUENCY_LABELS:
        count = combined[label].sum()
        print(f"    {label}: {count} ({100 * count / len(combined):.1f}%)")
    print(f"  Saved to: {out}")

    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge datasets into combined labels.")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing per-clip interval CSVs.",
    )
    args = parser.parse_args()
    merge_datasets(force=args.force)
