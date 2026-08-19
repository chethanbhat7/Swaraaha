"""Spectrogram generation, normalization, saving, and visualization."""

import os
from typing import Optional, Tuple

import numpy as np

from model.data.preprocessing.io import load_audio


def generate_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a mel-spectrogram from a raw audio waveform.

    Args:
        audio: 1-D float32 array, values in [-1.0, 1.0].
        sr: Sample rate in Hz.
        n_mels: Number of mel frequency bins (height of spectrogram).
        hop_length: Number of samples between successive frames (controls width).
        n_fft: FFT window size.
        fmin: Minimum frequency for mel filter bank.
        fmax: Maximum frequency. Defaults to sr / 2.

    Returns:
        2-D numpy array of shape (n_mels, time_frames) with mel-spectrogram
        values in dB scale (log-mel). dtype: float32.

    Example:
        >>> audio, sr = load_audio("speech.wav")
        >>> spec = generate_mel_spectrogram(audio, sr)
        >>> print(f"Spectrogram shape: {spec.shape}")  # e.g. (128, 313)
    """
    import librosa

    if fmax is None:
        fmax = sr / 2.0

    n = len(audio)
    if 0 < n < n_fft:
        pad = n_fft - n
        audio = np.pad(audio, (pad // 2, pad - pad // 2), mode="constant")

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec.astype(np.float32)


def normalize_spectrogram(
    spec: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """
    Normalize a spectrogram array.

    Args:
        spec: Spectrogram array of any shape (n_mels, time) or (1, n_mels, time).
        method: Normalization method.
            - "zscore": Zero-mean, unit-variance per spectrogram.
            - "minmax": Scale to [0, 1] range.

    Returns:
        Normalized spectrogram, same shape as input.

    Example:
        >>> spec = generate_mel_spectrogram(audio)
        >>> spec_norm = normalize_spectrogram(spec, method="zscore")
    """
    if method == "zscore":
        mean = spec.mean()
        std = spec.std()
        if std == 0:
            return np.zeros_like(spec)
        return ((spec - mean) / std).astype(np.float32)

    elif method == "minmax":
        smin = spec.min()
        smax = spec.max()
        if smax == smin:
            return np.zeros_like(spec)
        return ((spec - smin) / (smax - smin)).astype(np.float32)

    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'zscore' or 'minmax'.")


def spectrogram_to_image_array(spec: np.ndarray) -> np.ndarray:
    """
    Convert a 2-D spectrogram to a 3-D image-like array for CNN input.

    Adds a channel dimension so the output shape is (1, n_mels, time_frames),
    matching PyTorch's (C, H, W) format for single-channel images.

    Args:
        spec: 2-D array of shape (n_mels, time_frames).

    Returns:
        3-D array of shape (1, n_mels, time_frames), dtype float32.
    """
    if spec.ndim != 2:
        raise ValueError(f"Expected 2-D spectrogram, got shape {spec.shape}")
    return spec[np.newaxis, ...]


def save_spectrogram(spec: np.ndarray, path: str) -> None:
    """
    Save a spectrogram as a .npy file for later loading.

    Args:
        spec: Spectrogram array (2-D or 3-D).
        path: Output file path (.npy).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, spec)


def save_spectrogram_image(
    spec: np.ndarray,
    path: str,
    sr: int = 16000,
    hop_length: int = 512,
    title: str = "Mel Spectrogram",
) -> None:
    """
    Save a spectrogram as a PNG image for visual debugging.

    Args:
        spec: 2-D spectrogram array (n_mels, time_frames).
        path: Output image path (.png).
        sr: Sample rate (used to compute time axis).
        hop_length: Hop length used during spectrogram generation.
        title: Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    times = np.arange(spec.shape[1]) * hop_length / sr
    img = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        extent=[0, times[-1], 0, spec.shape[0]],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Filter Index")
    ax.set_title(title)
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_waveform(
    audio: np.ndarray,
    sr: int,
    path: str,
    title: str = "Waveform",
) -> None:
    """
    Save a waveform plot as PNG for debugging.

    Args:
        audio: 1-D audio array.
        sr: Sample rate.
        path: Output image path (.png).
        title: Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    times = np.arange(len(audio)) / sr
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(times, audio, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def audio_to_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    hop_length: int = 512,
    normalize: bool = True,
    as_image: bool = True,
) -> np.ndarray:
    """
    End-to-end conversion from raw audio to a CNN-ready spectrogram tensor.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.
        n_mels: Number of mel bins.
        hop_length: Hop length for STFT.
        normalize: Apply z-score normalization.
        as_image: Add channel dimension (1, H, W) for CNN.

    Returns:
        Spectrogram array. Shape:
            - (n_mels, T) if as_image=False
            - (1, n_mels, T) if as_image=True

    Example:
        >>> audio, sr = load_audio("speech.wav")
        >>> tensor = audio_to_spectrogram(audio, sr)
        >>> print(tensor.shape)  # (1, 128, 313)
    """
    spec = generate_mel_spectrogram(audio, sr=sr, n_mels=n_mels, hop_length=hop_length)

    if normalize:
        spec = normalize_spectrogram(spec, method="zscore")

    if as_image:
        spec = spectrogram_to_image_array(spec)

    return spec


def file_to_spectrogram(
    path: str,
    sr: int = 16000,
    n_mels: int = 128,
    hop_length: int = 512,
    normalize: bool = True,
    as_image: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and convert directly to spectrogram.

    Returns:
        Tuple of (spectrogram, sample_rate).
    """
    audio, sr = load_audio(path, sr=sr)
    spec = audio_to_spectrogram(
        audio, sr=sr, n_mels=n_mels, hop_length=hop_length,
        normalize=normalize, as_image=as_image,
    )
    return spec, sr
