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
        noise = np.random.normal(0, self.noise_level, audio.shape).astype(np.float32)
        return audio + noise

    def time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Time-stretch audio by resampling."""
        factor = random.uniform(*self.time_stretch_range)
        indices = np.round(np.arange(0, len(audio), factor)).astype(np.int64)
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
        indices = np.round(np.arange(0, len(audio), factor)).astype(np.int64)
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
        return audio.astype(np.float32)


class AugmentedDataset:
    """Wrapper around a torch Dataset that applies augmentation on-the-fly.

    Args:
        dataset: A torch-compatible Dataset (must have __getitem__ and __len__).
        augmentor: AudioAugmentor instance for waveform augmentation.
        augment_spectrogram: If True, also augment spectrograms (for localizer).
        sample_rate: Sample rate for audio augmentations.
    """

    def __init__(
        self,
        dataset,
        augmentor: Optional[AudioAugmentor] = None,
        augment_spectrogram: bool = False,
        sample_rate: int = 16000,
    ):
        self.dataset = dataset
        self.augmentor = augmentor
        self.augment_spectrogram = augment_spectrogram
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        item = self.dataset[idx]

        if self.augmentor is None:
            return item

        # Unpack based on what the dataset returns
        if isinstance(item, tuple):
            audio = item[0]
            rest = item[1:]
        else:
            return item

        # Apply waveform augmentation to numpy arrays
        if isinstance(audio, np.ndarray):
            audio = self.augmentor(audio, self.sample_rate)
        elif type(audio).__module__.startswith("torch"):
            import torch

            audio = audio.cpu().numpy()
            audio = self.augmentor(audio, self.sample_rate)
            audio = torch.from_numpy(audio)

        if len(rest) == 0:
            return (audio,)
        return (audio,) + rest
