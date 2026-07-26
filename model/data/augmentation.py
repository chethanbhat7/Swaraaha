"""
Audio data augmentation pipeline for Swaraaha.

Provides on-the-fly augmentations during training to improve model
generalization with small stuttering datasets. Each augmentation has
configurable probability and parameters.

Usage:
    from model.data.augmentation import AudioAugmentor, AugmentedDataset

    # Basic usage
    augmentor = AudioAugmentor()
    augmented_audio = augmentor(audio, sr=16000)

    # Wrap a dataset
    dataset = ClassificationDataset(data_dir="data")
    augmented_dataset = AugmentedDataset(dataset, augmentor)
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


class AudioAugmentor:
    """
    Applies random audio augmentations with configurable probabilities.

    Each augmentation is applied independently with its own probability.
    All augmentations are designed to preserve speech intelligibility while
    increasing diversity in training data.
    """

    def __init__(
        self,
        time_stretch_prob: float = 0.3,
        pitch_shift_prob: float = 0.3,
        noise_inject_prob: float = 0.4,
        time_mask_prob: float = 0.2,
        freq_mask_prob: float = 0.2,
        time_stretch_range: Tuple[float, float] = (0.8, 1.2),
        pitch_shift_semitones: float = 2.0,
        noise_snr_range: Tuple[float, float] = (20.0, 40.0),
        time_mask_max_frames: int = 10,
        freq_mask_max_bins: int = 8,
    ):
        """
        Args:
            time_stretch_prob: Probability of applying time stretching.
            pitch_shift_prob: Probability of applying pitch shifting.
            noise_inject_prob: Probability of adding noise.
            time_mask_prob: Probability of applying time masking.
            freq_mask_prob: Probability of applying frequency masking.
            time_stretch_range: Min/max stretch factor (0.8 = slower, 1.2 = faster).
            pitch_shift_semitones: Max pitch shift in semitones (±).
            noise_snr_range: Min/max signal-to-noise ratio in dB.
            time_mask_max_frames: Maximum number of consecutive frames to mask.
            freq_mask_max_bins: Maximum number of consecutive mel bins to mask.
        """
        self.time_stretch_prob = time_stretch_prob
        self.pitch_shift_prob = pitch_shift_prob
        self.noise_inject_prob = noise_inject_prob
        self.time_mask_prob = time_mask_prob
        self.freq_mask_prob = freq_mask_prob
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_semitones = pitch_shift_semitones
        self.noise_snr_range = noise_snr_range
        self.time_mask_max_frames = time_mask_max_frames
        self.freq_mask_max_bins = freq_mask_max_bins

    def __call__(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Apply random augmentations to an audio waveform.

        Args:
            audio: 1-D float32 array, values in [-1.0, 1.0].
            sr: Sample rate in Hz.

        Returns:
            Augmented audio array (same length as input).
        """
        audio = audio.copy()

        if random.random() < self.time_stretch_prob:
            audio = self._time_stretch(audio)

        if random.random() < self.pitch_shift_prob:
            audio = self._pitch_shift(audio, sr)

        if random.random() < self.noise_inject_prob:
            audio = self._noise_inject(audio, sr)

        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def augment_spectrogram(
        self,
        spectrogram: np.ndarray,
    ) -> np.ndarray:
        """
        Apply spectrogram-level augmentations (time/frequency masking).

        Args:
            spectrogram: 2-D array of shape (n_mels, time_frames) or
                         3-D array of shape (1, n_mels, time_frames).

        Returns:
            Augmented spectrogram (same shape as input).
        """
        spec = spectrogram.copy()

        # Handle 3-D input (1, n_mels, T)
        squeeze = False
        if spec.ndim == 3:
            spec = spec[0]
            squeeze = True

        if random.random() < self.time_mask_prob:
            spec = self._time_mask(spec)

        if random.random() < self.freq_mask_prob:
            spec = self._freq_mask(spec)

        if squeeze:
            spec = spec[np.newaxis, ...]

        return spec

    def _time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """
        Randomly speed up or slow down audio.

        Uses librosa.effects.time_stretch which preserves pitch.
        """
        import librosa

        rate = random.uniform(*self.time_stretch_range)
        stretched = librosa.effects.time_stretch(audio, rate=rate)

        # Maintain original length by padding or truncating
        if len(stretched) < len(audio):
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        else:
            stretched = stretched[: len(audio)]

        return stretched.astype(np.float32)

    def _pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Randomly shift pitch by ±semitones.

        Uses librosa.effects.pitch_shift which preserves duration.
        """
        import librosa

        n_steps = random.uniform(-self.pitch_shift_semitones, self.pitch_shift_semitones)
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        return shifted.astype(np.float32)

    def _noise_inject(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Add white noise at a random SNR level.

        SNR range: 20-40 dB (higher = less noise).
        """
        snr_db = random.uniform(*self.noise_snr_range)
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return (audio + noise).astype(np.float32)

    def _time_mask(self, spec: np.ndarray) -> np.ndarray:
        """
        Randomly mask consecutive time frames in spectrogram.

        SpecAugment-style time masking. Masked frames are set to 0.
        """
        n_frames = spec.shape[1]
        max_mask = min(self.time_mask_max_frames, n_frames // 4)

        if max_mask <= 0:
            return spec

        mask_len = random.randint(1, max_mask)
        start = random.randint(0, n_frames - mask_len)
        spec[:, start : start + mask_len] = 0.0
        return spec

    def _freq_mask(self, spec: np.ndarray) -> np.ndarray:
        """
        Randomly mask consecutive frequency bins in spectrogram.

        SpecAugment-style frequency masking. Masked bins are set to 0.
        """
        n_mels = spec.shape[0]
        max_mask = min(self.freq_mask_max_bins, n_mels // 4)

        if max_mask <= 0:
            return spec

        mask_len = random.randint(1, max_mask)
        start = random.randint(0, n_mels - mask_len)
        spec[start : start + mask_len, :] = 0.0
        return spec


class AugmentedDataset:
    """
    Wrapper that applies on-the-fly augmentations to a dataset.

    Wraps either ClassificationDataset or LocalizationDataset.
    Augmentations are applied only during training (when training=True).
    """

    def __init__(
        self,
        dataset,
        augmentor: Optional[AudioAugmentor] = None,
        augment_spectrogram: bool = False,
    ):
        """
        Args:
            dataset: ClassificationDataset or LocalizationDataset instance.
            augmentor: AudioAugmentor instance. If None, uses default config.
            augment_spectrogram: If True, also apply spectrogram augmentations
                                 (time/frequency masking) to localization data.
        """
        self.dataset = dataset
        self.augmentor = augmentor or AudioAugmentor()
        self.augment_spectrogram = augment_spectrogram

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get an augmented sample.

        Returns same format as the wrapped dataset:
            - ClassificationDataset: (audio, label_vector)
            - LocalizationDataset: (spectrogram, frame_label)
        """
        sample = self.dataset[idx]

        # Check if this is localization data (spectrogram) or classification (audio)
        data, labels = sample

        # ClassificationDataset returns (audio, label_vector)
        if data.ndim == 1:
            data = self.augmentor(data, sr=getattr(self.dataset, "sr", 16000))
        # LocalizationDataset returns (spectrogram, frame_label)
        elif self.augment_spectrogram and data.ndim >= 2:
            data = self.augmentor.augment_spectrogram(data)

        return data, labels

    def get_sample_info(self, idx: int) -> Dict:
        """Pass through to wrapped dataset."""
        return self.dataset.get_sample_info(idx)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    p: float = 0.5,
    augmentor: Optional[AudioAugmentor] = None,
) -> np.ndarray:
    """
    Apply random augmentation to audio with a given probability.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.
        p: Probability of applying augmentation.
        augmentor: AudioAugmentor instance. If None, uses default.

    Returns:
        Augmented (or original) audio array.
    """
    if random.random() > p:
        return audio

    aug = augmentor or AudioAugmentor()
    return aug(audio, sr=sr)


def create_augmented_dataloader(
    dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    augmentor: Optional[AudioAugmentor] = None,
    augment_spectrogram: bool = False,
    num_workers: int = 0,
):
    """
    Create a DataLoader with augmentation applied.

    Args:
        dataset: ClassificationDataset or LocalizationDataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        augmentor: AudioAugmentor instance.
        augment_spectrogram: Apply spectrogram augmentations for localization.
        num_workers: DataLoader workers.

    Returns:
        DataLoader with augmented samples.
    """
    from torch.utils.data import DataLoader

    augmented = AugmentedDataset(
        dataset,
        augmentor=augmentor,
        augment_spectrogram=augment_spectrogram,
    )

    return DataLoader(
        augmented,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Audio Augmentation Pipeline — Self Test ===")

    # Generate test audio
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print(f"Original audio shape: {audio.shape}, range: [{audio.min():.3f}, {audio.max():.3f}]")

    # Test augmentor
    augmentor = AudioAugmentor()

    # Time stretch
    stretched = augmentor._time_stretch(audio)
    print(f"Time stretched: {stretched.shape}")

    # Pitch shift
    shifted = augmentor._pitch_shift(audio, sr)
    print(f"Pitch shifted: {shifted.shape}")

    # Noise inject
    noisy = augmentor._noise_inject(audio, sr)
    print(f"Noise injected: {noisy.shape}, range: [{noisy.min():.3f}, {noisy.max():.3f}]")

    # Full augmentation
    augmented = augmentor(audio, sr=sr)
    print(f"Full augmented: {augmented.shape}, range: [{augmented.min():.3f}, {augmented.max():.3f}]")

    # Spectrogram augmentation
    spec = np.random.randn(128, 100).astype(np.float32)
    aug_spec = augmentor.augment_spectrogram(spec)
    print(f"Spectrogram augmented: {aug_spec.shape}")

    # Test 3-D spectrogram
    spec_3d = np.random.randn(1, 128, 100).astype(np.float32)
    aug_spec_3d = augmentor.augment_spectrogram(spec_3d)
    print(f"3-D spectrogram augmented: {aug_spec_3d.shape}")

    print("=== Self test passed ===")
