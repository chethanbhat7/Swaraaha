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

import os
from typing import Dict, List, Optional, Tuple

import numpy as np


DYSFLUENCY_CLASSES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]
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
            dtype = parts[2].strip() if len(parts) > 2 else "unknown"
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
    ):
        """
        Args:
            data_dir: Root directory containing audio/ and labels/ subdirs.
            sr: Target sample rate.
            n_mels: Number of mel bins for spectrogram.
            hop_length: Hop length for spectrogram.
            max_length_seconds: Pad/truncate all audio to this length.
        """
        self.data_dir = data_dir
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.max_samples = int(max_length_seconds * sr)
        self.max_frames = self.max_samples // hop_length

        self.audio_dir = os.path.join(data_dir, "audio")
        self.labels_dir = os.path.join(data_dir, "labels")

        self.samples = self._scan_samples()

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
            generate_mel_spectrogram,
            load_audio,
            normalize_spectrogram,
            pad_to_length,
            spectrogram_to_image_array,
        )

        sample = self.samples[idx]

        # Load audio
        audio, _ = load_audio(sample["audio_path"], sr=self.sr)

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
    ):
        self.data_dir = data_dir
        self.sr = sr
        self.max_samples = int(max_length_seconds * sr)

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
        from model.data.preprocessing import load_audio, pad_to_length

        sample = self.samples[idx]
        audio, _ = load_audio(sample["audio_path"], sr=self.sr)
        intervals = load_label_csv(sample["label_path"])

        # Multi-hot encoding: 1 if any interval of that class exists
        label_vector = np.zeros(NUM_CLASSES, dtype=np.uint8)
        for _, _, dtype in intervals:
            if dtype in CLASS_TO_IDX:
                label_vector[CLASS_TO_IDX[dtype]] = 1

        audio = pad_to_length(audio, self.max_samples, axis=0, pad_value=0.0)

        return audio.astype(np.float32), label_vector

    def get_sample_info(self, idx: int) -> Dict:
        return self.samples[idx].copy()
