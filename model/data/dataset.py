"""
PyTorch Dataset classes for Swaraaha.

Provides data loading for both classification and localization tasks.
Handles variable-length audio with padding for batched training.

Expected directory layout for localization:
    data_dir/
    ├── audio/
    │   ├── clip_001.wav
    │   ├── clip_002.wav
    │   └── ...
    └── labels/
        ├── clip_001.csv    # columns: start_sec, end_sec, dysfluency_type
        ├── clip_002.csv
        └── ...

Label CSV format:
    start_sec,end_sec,dysfluency_type
    0.5,1.2,prolongation
    2.1,2.8,soundrep
"""

import csv
import hashlib
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from model.config.defaults import DYSFLUENCY_CLASSES

NUM_CLASSES = len(DYSFLUENCY_CLASSES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(DYSFLUENCY_CLASSES)}


def load_label_csv(csv_path: str) -> List[Tuple[float, float, str]]:
    """
    Load a label CSV file containing dysfluency intervals.

    Args:
        csv_path: Path to CSV with columns: start_sec, end_sec, dysfluency_type.

    Returns:
        List of (start_sec, end_sec, dysfluency_type_str) tuples.
    """
    intervals = []
    with open(csv_path, "r") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            start_sec = float(parts[0])
            end_sec = float(parts[1])
            dtype = parts[2].strip().lower() if len(parts) > 2 else "unknown"
            intervals.append((start_sec, end_sec, dtype))
    return intervals


def intervals_to_frame_mask(
    intervals: List[Tuple[float, float]],
    num_frames: int,
    sr: int = 16000,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Convert a list of (start_sec, end_sec) intervals to a binary frame mask.

    Args:
        intervals: List of (start_sec, end_sec) tuples.
        num_frames: Number of spectrogram time frames.
        sr: Sample rate.
        hop_length: Hop length.

    Returns:
        1-D uint8 array of shape (num_frames,). 1 = dysfluent.
    """
    from model.data.preprocessing import create_frame_labels

    return create_frame_labels(
        [(s, e) for s, e, _ in intervals] if intervals and len(intervals[0]) == 3 else intervals,
        num_frames=num_frames,
        sr=sr,
        hop_length=hop_length,
    )


class LocalizationDataset:
    """
    PyTorch Dataset for dysfluency localization.

    Each item returns a (spectrogram, frame_label) pair where:
        - spectrogram: float32 tensor of shape (1, n_mels, time_frames)
        - frame_label: uint8 tensor of shape (time_frames,) — binary mask

    Variable-length audio is handled by padding to a max length
    (set via max_length_seconds in __init__).

    Usage:
        dataset = LocalizationDataset(data_dir="data/train", max_length_seconds=10)
        spec, labels = dataset[0]
    """

    def __init__(
        self,
        data_dir: str,
        sr: int = 16000,
        n_mels: int = 128,
        hop_length: int = 512,
        max_length_seconds: float = 10.0,
        sources: Optional[List[str]] = None,
    ):
        """
        Args:
            data_dir: Root directory containing audio/ and labels/ subdirs.
            sr: Target sample rate.
            n_mels: Number of mel bins for spectrogram.
            hop_length: Hop length for spectrogram.
            max_length_seconds: Pad/truncate all audio to this length.
            sources: If given, only include clips whose source (from
                sources.csv) is in this list. Ignored when sources.csv is
                missing.
        """
        self.data_dir = data_dir
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.max_samples = int(max_length_seconds * sr)
        self.max_frames = self.max_samples // hop_length
        self.sources = set(sources) if sources else None
        self._source_map = self._load_source_map()

        self.audio_dir = os.path.join(data_dir, "audio")
        self.labels_dir = os.path.join(data_dir, "labels")

        self.samples = self._scan_samples()

    def _load_source_map(self) -> Dict[str, str]:
        """Load clip_id -> source mapping from sources.csv (if present)."""
        path = os.path.join(self.data_dir, "sources.csv")
        if not os.path.isfile(path):
            return {}
        source_map = {}
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                clip_id = row.get("clip_id", "").strip()
                source = row.get("source", "").strip()
                if clip_id and source:
                    source_map[clip_id] = source
        return source_map

    def _scan_samples(self) -> List[Dict]:
        """Scan the data directory and build a list of available samples."""
        samples = []
        if not os.path.isdir(self.audio_dir):
            return samples

        for fname in sorted(os.listdir(self.audio_dir)):
            if not fname.endswith((".wav", ".flac", ".mp3")):
                continue
            clip_id = os.path.splitext(fname)[0]
            audio_path = os.path.join(self.audio_dir, fname)
            label_path = os.path.join(self.labels_dir, f"{clip_id}.csv")

            if not os.path.isfile(label_path):
                continue  # skip clips without labels

            # Skip header-only WAV files (no audio data)
            if os.path.getsize(audio_path) <= 44:
                continue

            # Apply source filter if configured
            if self.sources is not None and self._source_map:
                if self._source_map.get(clip_id) not in self.sources:
                    continue

            samples.append({
                "clip_id": clip_id,
                "audio_path": audio_path,
                "label_path": label_path,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple["np.ndarray", "np.ndarray"]:
        """
        Get a single (spectrogram, frame_label) pair.

        Returns:
            spectrogram: float32 ndarray, shape (1, n_mels, max_frames).
            frame_label: uint8 ndarray, shape (max_frames,).
        """
        from model.data.preprocessing import (
            clean_audio,
            generate_mel_spectrogram,
            load_audio,
            normalize_spectrogram,
            pad_to_length,
            spectrogram_to_image_array,
        )

        sample = self.samples[idx]

        # Load audio
        audio, _ = load_audio(sample["audio_path"], sr=self.sr)

        # Clean audio: DC removal, peak normalization, silence trimming
        audio = clean_audio(audio, sr=self.sr)

        # Load labels
        intervals = load_label_csv(sample["label_path"])

        # Generate spectrogram (before padding, so we get the true frame count)
        spec = generate_mel_spectrogram(
            audio, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length,
        )
        num_frames = spec.shape[1]  # librosa returns (n_mels, time)

        # Create frame labels aligned to this spectrogram
        frame_mask = intervals_to_frame_mask(
            intervals,
            num_frames=num_frames,
            sr=self.sr,
            hop_length=self.hop_length,
        )

        # Pad spectrogram to max_frames
        spec = pad_to_length(spec, self.max_frames, axis=1, pad_value=spec.min())
        frame_mask = pad_to_length(frame_mask, self.max_frames, axis=0, pad_value=0)

        # Add channel dimension: (1, n_mels, max_frames)
        spec = spectrogram_to_image_array(spec)

        return spec.astype(np.float32), frame_mask.astype(np.uint8)

    def get_sample_info(self, idx: int) -> Dict:
        """Return metadata for a sample without loading audio."""
        return self.samples[idx].copy()


class ClassificationDataset:
    """
    PyTorch Dataset for dysfluency classification (multi-label).

    Each item returns a (audio, label_vector) pair where:
        - audio: float32 ndarray of shape (max_samples,) — padded audio
        - label_vector: uint8 ndarray of shape (5,) — multi-hot per class

    Label CSV format for classification:
        start_sec,end_sec,dysfluency_type
        (same as localization — we aggregate to multi-label here)
    """

    def __init__(
        self,
        data_dir: str,
        sr: int = 16000,
        max_length_seconds: float = 10.0,
        cache_dir: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.sr = sr
        self.max_samples = int(max_length_seconds * sr)
        self.use_cache = cache_dir is not None

        if self.use_cache:
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)
            self._cache_index: Optional[Dict[str, str]] = None
        else:
            self.cache_dir = None
            self._cache_index = None

        self.audio_dir = os.path.join(data_dir, "audio")
        self.labels_dir = os.path.join(data_dir, "labels")

        self.samples = self._scan_samples()

    def _scan_samples(self) -> List[Dict]:
        samples = []
        if not os.path.isdir(self.audio_dir):
            return samples

        for fname in sorted(os.listdir(self.audio_dir)):
            if not fname.endswith((".wav", ".flac", ".mp3")):
                continue
            clip_id = os.path.splitext(fname)[0]
            audio_path = os.path.join(self.audio_dir, fname)
            label_path = os.path.join(self.labels_dir, f"{clip_id}.csv")

            if not os.path.isfile(label_path):
                continue

            # Skip header-only WAV files (no audio data)
            if os.path.getsize(audio_path) <= 44:
                continue

            samples.append({
                "clip_id": clip_id,
                "audio_path": audio_path,
                "label_path": label_path,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple["np.ndarray", "np.ndarray"]:
        """
        Get a single (audio, label_vector) pair.

        Returns:
            audio: float32 ndarray, shape (max_samples,).
            label_vector: uint8 ndarray, shape (5,) — multi-hot.
        """
        sample = self.samples[idx]

        label_signature = self._label_signature(sample["label_path"])

        if self.use_cache:
            cached = self._load_from_cache(sample["clip_id"], label_signature)
            if cached is not None:
                return cached

        from model.data.preprocessing import clean_audio, load_audio, pad_to_length

        audio, _ = load_audio(sample["audio_path"], sr=self.sr)

        # Clean audio: DC removal, peak normalization, silence trimming
        audio = clean_audio(audio, sr=self.sr)

        intervals = load_label_csv(sample["label_path"])

        # Multi-hot encoding: 1 if any interval of that class exists
        label_vector = np.zeros(NUM_CLASSES, dtype=np.uint8)
        for _, _, dtype in intervals:
            if dtype in CLASS_TO_IDX:
                label_vector[CLASS_TO_IDX[dtype]] = 1

        audio = pad_to_length(audio, self.max_samples, axis=0, pad_value=0.0)

        result = (audio.astype(np.float32), label_vector)

        if self.use_cache:
            self._save_to_cache(sample["clip_id"], label_signature, result)

        return result

    def _cache_path(self, clip_id: str) -> str:
        return os.path.join(self.cache_dir, f"{clip_id}.pkl")

    def _label_signature(self, label_path: str) -> str:
        """Hash of the label CSV contents, so regenerated labels invalidate
        the pickle cache (which is keyed by clip_id only)."""
        try:
            with open(label_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:12]
        except OSError:
            return "missing"

    def _load_from_cache(
        self, clip_id: str, label_signature: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        path = self._cache_path(clip_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            return None
        if isinstance(payload, tuple) and len(payload) == 3 and payload[0] == label_signature:
            return payload[1], payload[2]
        return None

    def _save_to_cache(
        self, clip_id: str, label_signature: str, data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        path = self._cache_path(clip_id)
        with open(path, "wb") as f:
            pickle.dump((label_signature, data[0], data[1]), f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_sample_info(self, idx: int) -> Dict:
        return self.samples[idx].copy()
