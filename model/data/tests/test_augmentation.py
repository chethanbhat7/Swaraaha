"""
Tests for augmentation fixes:
- SpectrogramAugmentor (label-safe masking for the CNN localizer).
- AugmentedDataset routing: spectrogram vs waveform paths.
- Label-aligned AudioAugmentor for the wav2vec2 localizer.
"""

import numpy as np
import pytest

from model.data.augmentation import (
    AudioAugmentor,
    AugmentedDataset,
    SpectrogramAugmentor,
)


def test_time_shift_uses_sample_rate():
    """time_shift must convert seconds to samples using the given sample rate,
    not a hardcoded 16000."""
    aug = AudioAugmentor(shift_range=(1.0, 1.0))
    audio = np.arange(20000, dtype=np.float32)
    shifted_8k = aug.time_shift(audio, sample_rate=8000)
    assert np.array_equal(shifted_8k, np.roll(audio, 8000))
    shifted_16k = aug.time_shift(audio, sample_rate=16000)
    assert np.array_equal(shifted_16k, np.roll(audio, 16000))


def test_call_passes_sample_rate_to_time_shift(monkeypatch):
    """AudioAugmentor.__call__ must forward its sample_rate to time_shift."""
    aug = AudioAugmentor()
    seen = {}

    def fake_time_shift(audio, sample_rate=16000):
        seen["sr"] = sample_rate
        return audio

    for name in ("add_noise", "time_stretch", "scale"):
        monkeypatch.setattr(aug, name, lambda audio: audio)
    monkeypatch.setattr(aug, "pitch_shift", lambda audio, sample_rate: audio)
    monkeypatch.setattr(aug, "time_shift", fake_time_shift)
    aug(np.zeros(100, dtype=np.float32), sample_rate=8000)
    assert seen["sr"] == 8000


class _FakeSpectrogramDataset:
    """Mimics LocalizationDataset: (spec (1, n_mels, T), frame_labels (T,))."""

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        spec = np.random.randn(1, 128, 312).astype(np.float32)
        labels = np.zeros(312, dtype=np.uint8)
        labels[50:80] = 1
        return spec, labels


class _FakeWaveformDataset:
    """Mimics Wav2Vec2LocalizationDataset: (audio (N,), frame_labels (T,))."""

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        labels = np.zeros(500, dtype=np.uint8)
        labels[100:150] = 1
        return audio, labels


class _FakeClassDataset:
    """Mimics ClassificationDataset: (audio (N,), class_vector (5,))."""

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        audio = np.random.randn(16000).astype(np.float32)
        return audio, np.array([1, 0, 0, 0, 0], dtype=np.uint8)


class _ImpulseDataset:
    """Spectrogram with a single energy impulse at time frame 100."""

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        spec = np.zeros((1, 128, 312), dtype=np.float32)
        spec[:, :, 100] = 1.0
        return spec, np.zeros(312, dtype=np.uint8)


# --- SpectrogramAugmentor -------------------------------------------------


def test_spectrogram_augmentor_preserves_shape_and_dtype():
    spec = np.random.randn(1, 128, 312).astype(np.float32)
    aug = SpectrogramAugmentor(noise_level=0.0)
    out = aug(spec)
    assert out.shape == spec.shape
    assert out.dtype == spec.dtype


def test_spectrogram_augmentor_masks_frequency_bands():
    spec = np.ones((1, 128, 312), dtype=np.float32) * 5.0
    aug = SpectrogramAugmentor(
        noise_level=0.0,
        num_freq_masks=2,
        freq_mask_max=20,
        num_time_masks=0,
        mask_fill=0.0,
    )
    out = aug(spec)
    masked_bands = (out[0] == 0.0).all(axis=1).sum()
    assert masked_bands >= 1


def test_spectrogram_augmentor_masks_time_frames():
    spec = np.ones((1, 128, 312), dtype=np.float32) * 5.0
    aug = SpectrogramAugmentor(
        noise_level=0.0,
        num_time_masks=2,
        time_mask_max=40,
        num_freq_masks=0,
        mask_fill=0.0,
    )
    out = aug(spec)
    masked_frames = (out[0] == 0.0).all(axis=0).sum()
    assert masked_frames >= 1


def test_spectrogram_augmentor_does_not_move_energy_in_time():
    spec = np.zeros((1, 128, 312), dtype=np.float32)
    spec[:, :, 100] = 1.0
    aug = SpectrogramAugmentor(
        noise_level=0.0,
        num_time_masks=3,
        time_mask_max=10,
        num_freq_masks=0,
    )
    out = aug(spec)
    energy = out.sum(axis=1).sum(axis=0)
    other_cols = np.delete(energy, 100)
    assert np.all(other_cols == 0)


def test_spectrogram_augmentor_default_fill_is_spec_min():
    spec = np.ones((1, 128, 312), dtype=np.float32) * 5.0
    spec[0, 0, 0] = -3.0
    aug = SpectrogramAugmentor(
        noise_level=0.0,
        num_freq_masks=1,
        freq_mask_max=10,
        num_time_masks=0,
    )
    out = aug(spec)
    assert (out == -3.0).any()


# --- AugmentedDataset routing ---------------------------------------------


def test_augmented_dataset_routes_spectrogram_to_spectrogram_augmentor():
    ds = _FakeSpectrogramDataset()
    aug = AugmentedDataset(
        ds,
        augmentor=AudioAugmentor(),
        augment_spectrogram=True,
        spectrogram_augmentor=SpectrogramAugmentor(noise_level=0.0, mask_fill=0.0),
    )
    spec, labels = aug[0]
    assert spec.shape == (1, 128, 312)
    assert spec.dtype == np.float32
    assert np.array_equal(labels, ds[0][1])


def test_augmented_dataset_default_spectrogram_augmentor_used():
    ds = _FakeSpectrogramDataset()
    aug = AugmentedDataset(ds, augmentor=AudioAugmentor(), augment_spectrogram=True)
    spec, labels = aug[0]
    assert spec.shape == (1, 128, 312)
    assert np.array_equal(labels, ds[0][1])


def test_augmented_dataset_spectrogram_not_waveform_shifted():
    ds = _ImpulseDataset()
    aug = AugmentedDataset(
        ds,
        augmentor=AudioAugmentor(),
        augment_spectrogram=True,
        spectrogram_augmentor=SpectrogramAugmentor(
            noise_level=0.0, num_time_masks=0, num_freq_masks=0
        ),
    )
    spec, labels = aug[0]
    energy = spec.sum(axis=(0, 1))
    assert np.all(np.delete(energy, 100) == 0)
    assert np.array_equal(labels, np.zeros(312, dtype=np.uint8))


# --- _resample direction ---------------------------------------------------


def test_resample_stretch_factor_above_one_repeats_samples():
    arr = np.arange(10, dtype=np.float32)
    out = AudioAugmentor._resample(arr, 2.0)
    assert out.shape == arr.shape
    assert np.array_equal(out, np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))


def test_resample_compress_factor_below_one_skips_samples():
    arr = np.arange(10, dtype=np.float32)
    out = AudioAugmentor._resample(arr, 0.5)
    assert out.shape == arr.shape
    assert np.array_equal(out, np.array([0, 2, 4, 6, 8, 9, 9, 9, 9, 9]))


def test_resample_stretch_does_not_zero_pad_tail():
    arr = np.ones(100, dtype=np.float32)
    out = AudioAugmentor._resample(arr, 1.1)
    assert out.shape == arr.shape
    assert np.all(out == 1.0)


def test_resample_factor_one_is_identity():
    arr = np.random.randn(50).astype(np.float32)
    out = AudioAugmentor._resample(arr, 1.0)
    assert np.array_equal(out, arr)


def _dominant_freq(x, sr=16000):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spec) * sr / len(x))


def test_time_stretch_above_one_slows_content():
    sr = 16000
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    augmentor = AudioAugmentor(
        noise_level=0.0,
        time_stretch_range=(1.0, 1.0),
        pitch_shift_range=(0.0, 0.0),
        shift_range=(0.0, 0.0),
        scale_range=(1.0, 1.0),
    )
    augmentor.time_stretch_range = (2.0, 2.0)
    out = augmentor.time_stretch(tone)
    assert out.shape == tone.shape
    assert _dominant_freq(out) < _dominant_freq(tone)


def test_pitch_shift_up_raises_pitch():
    sr = 16000
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    augmentor = AudioAugmentor(
        noise_level=0.0,
        time_stretch_range=(1.0, 1.0),
        pitch_shift_range=(0.0, 0.0),
        shift_range=(0.0, 0.0),
        scale_range=(1.0, 1.0),
    )
    augmentor.pitch_shift_range = (12.0, 12.0)
    out = augmentor.pitch_shift(tone, sample_rate=sr)
    assert out.shape == tone.shape
    assert _dominant_freq(out) > _dominant_freq(tone)


def test_pitch_shift_down_lowers_pitch():
    sr = 16000
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    augmentor = AudioAugmentor(
        noise_level=0.0,
        time_stretch_range=(1.0, 1.0),
        pitch_shift_range=(0.0, 0.0),
        shift_range=(0.0, 0.0),
        scale_range=(1.0, 1.0),
    )
    augmentor.pitch_shift_range = (-12.0, -12.0)
    out = augmentor.pitch_shift(tone, sample_rate=sr)
    assert out.shape == tone.shape
    assert _dominant_freq(out) < _dominant_freq(tone)


# --- Label-aligned waveform augmentation ----------------------------------


def test_apply_with_labels_rolls_labels_with_audio():
    augmentor = AudioAugmentor(
        noise_level=0.0,
        time_stretch_range=(1.0, 1.0),
        pitch_shift_range=(0.0, 0.0),
        shift_range=(0.1, 0.1),
        scale_range=(1.0, 1.0),
    )
    audio = np.random.randn(16000).astype(np.float32)
    labels = np.zeros(500, dtype=np.uint8)
    labels[10:20] = 1
    out_audio, out_labels = augmentor.apply_with_labels(
        audio, labels, frame_hop_samples=320, sample_rate=16000
    )
    assert out_audio.shape == audio.shape
    assert out_labels.shape == labels.shape
    expected = np.zeros(500, dtype=np.uint8)
    expected[15:25] = 1
    assert np.array_equal(out_labels, expected)


def test_augmented_dataset_label_aligned_shift_moves_mask_in_lockstep():
    ds = _FakeWaveformDataset()
    augmentor = AudioAugmentor(
        noise_level=0.0,
        time_stretch_range=(1.0, 1.0),
        pitch_shift_range=(0.0, 0.0),
        shift_range=(0.1, 0.1),
        scale_range=(1.0, 1.0),
    )
    aug = AugmentedDataset(
        ds, augmentor=augmentor, label_aligned=True, frame_hop_samples=320
    )
    audio, labels = aug[0]
    assert audio.shape == (16000,)
    assert labels.shape == (500,)
    expected = np.zeros(500, dtype=np.uint8)
    expected[105:155] = 1
    assert np.array_equal(labels, expected)


def test_augmented_dataset_label_aligned_stretch_moves_mask_with_audio():
    ds = _FakeWaveformDataset()
    augmentor = AudioAugmentor(
        noise_level=0.0,
        time_stretch_range=(1.1, 1.1),
        pitch_shift_range=(0.0, 0.0),
        shift_range=(0.0, 0.0),
        scale_range=(1.0, 1.0),
    )
    aug = AugmentedDataset(
        ds, augmentor=augmentor, label_aligned=True, frame_hop_samples=320
    )
    audio, labels = aug[0]
    assert audio.shape == (16000,)
    assert labels.shape == (500,)
    assert labels[110] == 0
    assert labels[111] == 1
    assert labels[164] == 1
    assert labels[165] == 0


def test_augmented_dataset_waveform_default_path_keeps_class_labels():
    ds = _FakeClassDataset()
    aug = AugmentedDataset(ds, augmentor=AudioAugmentor())
    audio, labels = aug[0]
    assert audio.ndim == 1
    assert labels.shape == (5,)
    assert np.array_equal(labels, ds[0][1])
