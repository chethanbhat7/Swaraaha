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

    @staticmethod
    def _resample(arr: np.ndarray, factor: float) -> np.ndarray:
        """Resample a 1-D array by a time factor, keeping the same length."""
        indices = np.round(np.arange(0, len(arr), factor)).astype(np.int64)
        indices = indices[indices < len(arr)]
        out = arr[indices]
        if len(out) < len(arr):
            out = np.pad(out, (0, len(arr) - len(out)))
        else:
            out = out[: len(arr)]
        return out

    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to audio."""
        if self.noise_level <= 0:
            return audio
        noise = np.random.normal(0, self.noise_level, audio.shape).astype(np.float32)
        return audio + noise

    def time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Time-stretch audio by resampling."""
        factor = random.uniform(*self.time_stretch_range)
        return self._resample(audio, factor)

    def pitch_shift(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Pitch-shift audio using resampling."""
        semitones = random.uniform(*self.pitch_shift_range)
        factor = 2 ** (semitones / 12.0)
        return self._resample(audio, factor)

    def time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Shift audio in time by rolling samples."""
        shift_sec = random.uniform(*self.shift_range)
        shift_samples = int(shift_sec * 16000)
        return np.roll(audio, shift_samples)

    def scale(self, audio: np.ndarray) -> np.ndarray:
        """Scale audio amplitude."""
        factor = random.uniform(*self.scale_range)
        return audio * factor

    def apply_with_labels(
        self,
        audio: np.ndarray,
        frame_labels: np.ndarray,
        frame_hop_samples: int,
        sample_rate: int = 16000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply augmentation while keeping frame labels aligned with audio.

        Samples each transform parameter once and applies the same temporal
        transform to both the waveform and the frame labels, so an event stays
        at the same position in both. Noise and scaling have no temporal
        effect on labels.

        Args:
            audio: 1-D waveform array.
            frame_labels: 1-D frame-level label array (one per hop).
            frame_hop_samples: Number of samples per frame.
            sample_rate: Audio sample rate.

        Returns:
            Tuple of (augmented_audio, aligned_frame_labels).
        """
        audio = audio.astype(np.float32)
        frame_labels = frame_labels.astype(np.float32)

        if self.noise_level > 0:
            audio = self.add_noise(audio)

        stretch_factor = random.uniform(*self.time_stretch_range)
        audio = self._resample(audio, stretch_factor)
        frame_labels = self._resample(frame_labels, stretch_factor)

        semitones = random.uniform(*self.pitch_shift_range)
        pitch_factor = 2 ** (semitones / 12.0)
        audio = self._resample(audio, pitch_factor)
        frame_labels = self._resample(frame_labels, pitch_factor)

        shift_sec = random.uniform(*self.shift_range)
        shift_samples = int(shift_sec * sample_rate)
        audio = np.roll(audio, shift_samples)
        shift_frames = int(round(shift_samples / frame_hop_samples))
        frame_labels = np.roll(frame_labels, shift_frames)

        scale_factor = random.uniform(*self.scale_range)
        audio = audio * scale_factor

        return audio.astype(np.float32), frame_labels.astype(np.uint8)

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


class SpectrogramAugmentor:
    """Label-safe spectrogram augmentation (SpecAugment-style masking).

    Applies frequency and time masking plus optional Gaussian noise. Masking
    only zeroes regions in place — it never moves events in time — so frame
    labels remain valid without any alignment step.

    Args:
        freq_mask_max: Maximum width (in mel bands) of each frequency mask.
        time_mask_max: Maximum width (in frames) of each time mask.
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.
        noise_level: Gaussian noise standard deviation (0.0 to disable).
        mask_fill: Value used for masked regions. Defaults to spec minimum.
    """

    def __init__(
        self,
        freq_mask_max: int = 20,
        time_mask_max: int = 40,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        noise_level: float = 0.005,
        mask_fill: Optional[float] = None,
    ):
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.noise_level = noise_level
        self.mask_fill = mask_fill

    def _freq_mask(self, spec: np.ndarray, fill: float) -> np.ndarray:
        n_mels = spec.shape[-2]
        for _ in range(self.num_freq_masks):
            width = random.randint(1, min(self.freq_mask_max, n_mels - 1))
            start = random.randint(0, n_mels - width - 1)
            spec[..., start : start + width, :] = fill
        return spec

    def _time_mask(self, spec: np.ndarray, fill: float) -> np.ndarray:
        n_frames = spec.shape[-1]
        for _ in range(self.num_time_masks):
            width = random.randint(1, min(self.time_mask_max, n_frames - 1))
            start = random.randint(0, n_frames - width - 1)
            spec[..., :, start : start + width] = fill
        return spec

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """Augment a spectrogram, preserving shape and dtype."""
        spec = np.array(spec, dtype=np.float32).copy()
        fill = float(spec.min()) if self.mask_fill is None else float(self.mask_fill)
        if self.num_freq_masks > 0:
            spec = self._freq_mask(spec, fill)
        if self.num_time_masks > 0:
            spec = self._time_mask(spec, fill)
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, spec.shape).astype(np.float32)
            spec = spec + noise
        return spec.astype(np.float32)


class AugmentedDataset:
    """Wrapper around a torch Dataset that applies augmentation on-the-fly.

    Args:
        dataset: A torch-compatible Dataset (must have __getitem__ and __len__).
        augmentor: AudioAugmentor instance for waveform augmentation.
        augment_spectrogram: If True, route multi-dimensional inputs through
            the spectrogram augmentor instead of the waveform augmentor.
        spectrogram_augmentor: SpectrogramAugmentor used for spectrogram
            inputs. Defaults to SpectrogramAugmentor().
        label_aligned: If True and the input is 1-D waveform, use
            AudioAugmentor.apply_with_labels so frame labels move with the
            audio.
        frame_hop_samples: Samples per frame; required when label_aligned.
        sample_rate: Sample rate for audio augmentations.
    """

    def __init__(
        self,
        dataset,
        augmentor: Optional[AudioAugmentor] = None,
        augment_spectrogram: bool = False,
        spectrogram_augmentor: Optional[SpectrogramAugmentor] = None,
        label_aligned: bool = False,
        frame_hop_samples: Optional[int] = None,
        sample_rate: int = 16000,
    ):
        self.dataset = dataset
        self.augmentor = augmentor
        self.augment_spectrogram = augment_spectrogram
        self.spectrogram_augmentor = (
            spectrogram_augmentor
            if spectrogram_augmentor is not None
            else SpectrogramAugmentor()
        )
        self.label_aligned = label_aligned
        self.frame_hop_samples = frame_hop_samples
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.dataset)

    def _to_numpy(self, arr):
        if isinstance(arr, np.ndarray):
            return arr
        if type(arr).__module__.startswith("torch"):
            import torch

            return arr.cpu().numpy()
        return np.asarray(arr)

    def _is_torch(self, arr) -> bool:
        return type(arr).__module__.startswith("torch")

    def __getitem__(self, idx: int) -> tuple:
        item = self.dataset[idx]

        # Unpack based on what the dataset returns
        if isinstance(item, tuple):
            audio = item[0]
            rest = item[1:]
        else:
            return item

        arr = self._to_numpy(audio)

        # Spectrogram path: multi-dimensional input, mask-only augmentation.
        if arr.ndim >= 2:
            if self.augment_spectrogram:
                aug = self.spectrogram_augmentor(arr)
                if self._is_torch(audio):
                    import torch

                    audio = torch.from_numpy(aug)
                else:
                    audio = aug
            if len(rest) == 0:
                return (audio,)
            return (audio,) + rest

        # Waveform path: 1-D audio.
        if self.augmentor is None:
            return item

        if self.label_aligned and self.frame_hop_samples and len(rest) > 0:
            frame_labels = rest[0]
            lab_is_torch = self._is_torch(frame_labels)
            lab_arr = self._to_numpy(frame_labels)
            aug_audio, aug_labels = self.augmentor.apply_with_labels(
                arr, lab_arr, self.frame_hop_samples, self.sample_rate
            )
            if self._is_torch(audio):
                import torch

                audio = torch.from_numpy(aug_audio)
            else:
                audio = aug_audio
            if lab_is_torch:
                import torch

                new_labels = torch.from_numpy(aug_labels)
            else:
                new_labels = aug_labels
            if len(rest) == 1:
                return (audio, new_labels)
            return (audio, new_labels) + rest[1:]

        # Default waveform augmentation.
        if self._is_torch(audio):
            import torch

            audio = torch.from_numpy(self.augmentor(arr, self.sample_rate))
        else:
            audio = self.augmentor(arr, self.sample_rate)

        if len(rest) == 0:
            return (audio,)
        return (audio,) + rest
