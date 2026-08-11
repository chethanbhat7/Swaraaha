"""
PyTorch Dataset for Wav2Vec 2.0 localization training.

Unlike the spectrogram-based LocalizationDataset, this dataset returns
raw waveforms suitable for Wav2Vec 2.0 input (16kHz, float32).

Expected directory layout:
    data_dir/
    ├── audio/
    │   ├── clip_001.wav
    │   └── ...
    └── labels/
        ├── clip_001.csv    # columns: start_sec, end_sec, dysfluency_type
        └── ...

Frame resolution: 20ms per frame (320 samples at 16kHz)
"""

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class Wav2Vec2LocalizationDataset:
    """
    Dataset for Wav2Vec2-based localization.

    Returns (waveform, frame_label) pairs where:
        - waveform: float32 ndarray of shape (max_samples,) — raw audio
        - frame_label: uint8 ndarray of shape (max_frames,) — binary mask

    Frame resolution: 20ms per frame (320 samples at 16kHz)
    """

    def __init__(
        self,
        data_dir: str,
        sr: int = 16000,
        max_length_seconds: float = 10.0,
        hop_samples: int = 320,  # Wav2Vec2 subsampling factor
        sources: Optional[List[str]] = None,
    ):
        """
        Args:
            data_dir: Root directory containing audio/ and labels/ subdirs.
            sr: Target sample rate.
            max_length_seconds: Pad/truncate all audio to this length.
            hop_samples: Samples per frame for Wav2Vec2 (320 = 20ms).
            sources: If given, only include clips whose source (from
                sources.csv) is in this list. Ignored when sources.csv is
                missing.
        """
        self.data_dir = data_dir
        self.sr = sr
        self.max_samples = int(max_length_seconds * sr)
        self.hop_samples = hop_samples
        self.max_frames = self.max_samples // hop_samples
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

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single (waveform, frame_label) pair.

        Returns:
            waveform: float32 ndarray, shape (max_samples,).
            frame_label: uint8 ndarray, shape (max_frames,).
        """
        from model.data.preprocessing import clean_audio, load_audio, pad_to_length
        from model.data.dataset import load_label_csv

        sample = self.samples[idx]

        # Load and clean audio. Silence trimming is disabled — labels use the
        # original timeline and trimming would shift frame labels off the audio.
        audio, _ = load_audio(sample["audio_path"], sr=self.sr)
        audio = clean_audio(audio, sr=self.sr, trim=False)

        # Load labels
        intervals = load_label_csv(sample["label_path"])

        # Create frame labels aligned to Wav2Vec2 frames (20ms = 320 samples)
        frame_mask = self._create_w2v2_frame_labels(
            intervals, num_frames=self.max_frames
        )

        # Pad or truncate audio
        audio = pad_to_length(audio, self.max_samples, axis=0, pad_value=0.0)

        return audio.astype(np.float32), frame_mask.astype(np.uint8)

    def _create_w2v2_frame_labels(
        self,
        intervals: List[Tuple[float, float, str]],
        num_frames: int,
    ) -> np.ndarray:
        """
        Create frame labels at Wav2Vec2 resolution (20ms = 320 samples).

        Args:
            intervals: List of (start_sec, end_sec, dysfluency_type).
            num_frames: Number of output frames.

        Returns:
            Binary mask of shape (num_frames,).
        """
        mask = np.zeros(num_frames, dtype=np.uint8)
        frame_duration = self.hop_samples / self.sr  # 0.02 seconds

        for start_sec, end_sec, _ in intervals:
            start_frame = int(start_sec / frame_duration)
            end_frame = int(end_sec / frame_duration)
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)
            mask[start_frame:end_frame] = 1

        return mask

    def get_sample_info(self, idx: int) -> Dict:
        """Return metadata for a sample without loading audio."""
        return self.samples[idx].copy()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Wav2Vec2 Localization Dataset — Self Test ===")

    dataset = Wav2Vec2LocalizationDataset(data_dir="data")
    print(f"Samples: {len(dataset)}")

    if len(dataset) > 0:
        waveform, frame_label = dataset[0]
        print(f"Waveform shape: {waveform.shape}")
        print(f"Frame label shape: {frame_label.shape}")
        print(f"Dysfluent frames: {frame_label.sum()}/{len(frame_label)}")

        # Verify shapes
        assert waveform.shape == (160000,), f"Wrong waveform shape: {waveform.shape}"
        assert frame_label.shape == (500,), f"Wrong frame shape: {frame_label.shape}"
        assert waveform.dtype == np.float32, f"Wrong dtype: {waveform.dtype}"

        print("=== Self test passed ===")
    else:
        print("No data available for testing — skipping")
        print("=== Self test passed (no data) ===")
