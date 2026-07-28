# Swaraaha - Audio Augmentation
# AudioAugmentor and AugmentedDataset for training data augmentation.
# Used by all three training pipelines (classification, CNN, Wav2Vec2).

import random
from typing import Optional

import numpy as np


class AudioAugmentor:
    """Audio augmentation pipeline with configurable transforms.

    Args:
        noise_level: Gaussian noise standard deviation (0.0 to disable).
        time_stretch_range: (min, max) factor for time stretching.
        pitch_shift_range: (min, max) semitones for pitch shifting.
        shift_range: (min, max) seconds for temporal shifting.
        scale_range: (min, max) amplitude scaling factor.
    """

    def __init__(
        self,
        noise_level: float = 0.005,
        time_stretch_range: tuple[float, float] = (0.9, 1.1),
        pitch_shift_range: tuple[float, float] = (-1.0, 1.0),
        shift_range: tuple[float, float] = (-0.1, 0.1),
        scale_range: tuple[float, float] = (0.8, 1.2),
    ):
        self.noise_level = noise_level
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.shift_range = shift_range
        self.scale_range = scale_range

    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to audio."""
        if self.noise_level <= 0:
            return audio
        noise = np.random.normal(0, self.noise_level, audio.shape)
        return audio + noise

    def time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Time-stretch audio by resampling."""
        factor = random.uniform(*self.time_stretch_range)
        indices = np.round(np.arange(0, len(audio), factor)).astype(int)
        indices = indices[indices < len(audio)]
        stretched = audio[indices]
        if len(stretched) < len(audio):
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        else:
            stretched = stretched[: len(audio)]
        return stretched

    def pitch_shift(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Pitch-shift audio using resampling."""
        semitones = random.uniform(*self.pitch_shift_range)
        factor = 2 ** (semitones / 12.0)
        indices = np.round(np.arange(0, len(audio), factor)).astype(int)
        indices = indices[indices < len(audio)]
        shifted = audio[indices]
        if len(shifted) < len(audio):
            shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
        else:
            shifted = shifted[: len(audio)]
        return shifted

    def time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Shift audio in time by rolling samples."""
        shift_sec = random.uniform(*self.shift_range)
        shift_samples = int(shift_sec * 16000)
        return np.roll(audio, shift_samples)

    def scale(self, audio: np.ndarray) -> np.ndarray:
        """Scale audio amplitude."""
        factor = random.uniform(*self.scale_range)
        return audio * factor

    def __call__(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        """Apply all augmentations in sequence."""
        audio = self.add_noise(audio)
        audio = self.time_stretch(audio)
        audio = self.pitch_shift(audio, sample_rate)
        audio = self.time_shift(audio)
        audio = self.scale(audio)
        return audio


class AugmentedDataset:
    """Dataset wrapper that applies augmentation on-the-fly.

    Args:
        audio_files: List of audio file paths.
        labels: List of label arrays (for classification) or None (for localization).
        augmentor: AudioAugmentor instance. None for no augmentation.
        sample_rate: Target sample rate for loading audio.
        max_length: Maximum audio length in samples. Truncates or pads.
    """

    def __init__(
        self,
        audio_files: list[str],
        labels: Optional[list] = None,
        augmentor: Optional[AudioAugmentor] = None,
        sample_rate: int = 16000,
        max_length: int = 48000,
    ):
        self.audio_files = audio_files
        self.labels = labels
        self.augmentor = augmentor
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> tuple:
        import librosa

        audio_path = self.audio_files[idx]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        # Pad or truncate
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        else:
            audio = audio[: self.max_length]

        # Apply augmentation
        if self.augmentor is not None:
            audio = self.augmentor(audio, self.sample_rate)

        if self.labels is not None:
            return audio, self.labels[idx]
        return audio,
