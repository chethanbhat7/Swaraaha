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

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from model.config.defaults import SAMPLE_RATE
from model.data.dataset import _PickleCacheMixin, _load_source_map, _scan_samples, load_label_csv


class Wav2Vec2LocalizationDataset(_PickleCacheMixin):
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
        sr: int = SAMPLE_RATE,
        max_length_seconds: float = 3.0,
        hop_samples: int = 320,  # Wav2Vec2 subsampling factor
        sources: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
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
            cache_dir: Directory for the pickle cache of preprocessed items.
                None (default) auto-derives <data_dir parent>/cache/<basename>.
        """
        self.data_dir = data_dir
        self.sr = sr
        self.max_samples = int(max_length_seconds * sr)
        self.hop_samples = hop_samples
        self.max_frames = self.max_samples // hop_samples
        self.sources = set(sources) if sources else None
        self._init_cache(cache_dir)
        self._source_map = self._load_source_map()

        self.audio_dir = os.path.join(data_dir, "audio")
        self.labels_dir = os.path.join(data_dir, "labels")

        self.samples = _scan_samples(self.audio_dir, self.labels_dir, self.sources, self._source_map)

    def _load_source_map(self) -> Dict[str, str]:
        return _load_source_map(self.data_dir)

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

        sample = self.samples[idx]

        label_signature = self._label_signature(sample["label_path"])

        if self.use_cache:
            cached = self._load_from_cache(
                sample["clip_id"],
                label_signature,
                self._config_signature(),
                self._audio_signature(sample["audio_path"]),
            )
            if cached is not None:
                return cached

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

        result = (audio.astype(np.float32), frame_mask.astype(np.uint8))

        if self.use_cache:
            self._save_to_cache(
                sample["clip_id"],
                label_signature,
                self._config_signature(),
                self._audio_signature(sample["audio_path"]),
                result,
            )

        return result

    def _config_signature(self) -> str:
        return (f"sr={self.sr};hop_samples={self.hop_samples};"
                f"max_frames={self.max_frames}")

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
