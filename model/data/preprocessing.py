"""
Audio preprocessing pipeline for Swaraaha.

Converts raw audio waveforms into spectrogram images suitable for CNN input.
All functions operate on numpy arrays for compatibility with both PyTorch and
desktop app (PyQt5) without requiring GPU dependencies.

Expected flow:
    audio = load_audio("speech.wav")
    spec = generate_mel_spectrogram(audio)
    spec_norm = normalize_spectrogram(spec)
    # spec_norm shape: [1, n_mels, time_frames] — ready for CNN input

Localization-specific flow:
    spec, sr = file_to_spectrogram("speech.wav")
    labels = create_frame_labels(dysfluency_intervals, spec.shape[2], sr, hop_length=512)
    # labels shape: (time_frames,) — binary mask aligned to spectrogram
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sample rate.

    Args:
        path: Path to audio file (.wav, .flac, .mp3, etc.).
        sr: Target sample rate in Hz. Default 16000 (matches Wav2Vec 2.0).

    Returns:
        Tuple of (audio_array, sample_rate).
        audio_array is 1-D numpy float32 array with values in [-1.0, 1.0].

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio file cannot be decoded.

    Example:
        >>> audio, sr = load_audio("recording.wav")
        >>> print(f"Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}")
    """
    import librosa

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32), sr


def load_audio_from_array(audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Accept a raw audio numpy array (e.g. from desktop app recording) and
    ensure it matches the expected format.

    Args:
        audio: 1-D numpy array of audio samples.
        sr: Sample rate of the input audio.

    Returns:
        Tuple of (resampled_audio, target_sr).
    """
    import librosa

    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D audio array, got shape {audio.shape}")

    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    return audio, sr


# ---------------------------------------------------------------------------
# Spectrogram Generation
# ---------------------------------------------------------------------------

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

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
    )

    # Convert to log scale (dB) — more suitable for CNN input
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec.astype(np.float32)


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


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Saving / Visualization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Full Pipeline (convenience)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Debug Utilities
# ---------------------------------------------------------------------------

def debug_save_spectrogram(
    audio: np.ndarray,
    sr: int,
    output_dir: str = "model/data/debug",
    prefix: str = "sample",
    n_mels: int = 128,
    hop_length: int = 512,
) -> dict:
    """
    Generate and save debug artifacts: waveform PNG + spectrogram PNG + .npy.

    Useful for visually verifying the preprocessing pipeline during development.

    Args:
        audio: 1-D audio array.
        sr: Sample rate.
        output_dir: Directory to save debug files.
        prefix: Filename prefix for saved files.
        n_mels: Number of mel bins.
        hop_length: Hop length.

    Returns:
        Dict with paths to saved files:
        {"waveform": path, "spectrogram_png": path, "spectrogram_npy": path}
    """
    os.makedirs(output_dir, exist_ok=True)

    waveform_path = os.path.join(output_dir, f"{prefix}_waveform.png")
    spec_png_path = os.path.join(output_dir, f"{prefix}_spectrogram.png")
    spec_npy_path = os.path.join(output_dir, f"{prefix}_spectrogram.npy")

    # Save waveform
    plot_waveform(audio, sr, waveform_path, title=f"Waveform — {prefix}")

    # Generate and save spectrogram
    spec = generate_mel_spectrogram(audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    save_spectrogram_image(spec, spec_png_path, sr=sr, hop_length=hop_length, title=f"Mel Spectrogram — {prefix}")
    save_spectrogram(spec, spec_npy_path)

    return {
        "waveform": waveform_path,
        "spectrogram_png": spec_png_path,
        "spectrogram_npy": spec_npy_path,
    }


def debug_generate_test_audio(duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """
    Generate a synthetic sine wave for pipeline testing when no real audio is available.

    Args:
        duration: Duration in seconds.
        sr: Sample rate.

    Returns:
        1-D float32 array of a 440 Hz sine wave.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Localization — Frame-Level Labels
# ---------------------------------------------------------------------------

def create_frame_labels(
    dysfluency_intervals: List[Tuple[float, float]],
    num_frames: int,
    sr: int = 16000,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Create a binary frame-level label mask aligned to a spectrogram.

    Each spectrogram frame at index `i` corresponds to audio sample
    `i * hop_length`. This function converts (start_sec, end_sec) intervals
    into a binary mask over those frames.

    Args:
        dysfluency_intervals: List of (start_sec, end_sec) tuples marking
            dysfluent regions in the audio. Empty list = fully fluent.
        num_frames: Total number of spectrogram time frames (the mask length).
        sr: Sample rate used during spectrogram generation.
        hop_length: Hop length used during spectrogram generation.

    Returns:
        1-D numpy uint8 array of shape (num_frames,).
        1 = dysfluent frame, 0 = fluent frame.

    Example:
        >>> # Audio is 5 seconds, frames 50-100 are dysfluent
        >>> labels = create_frame_labels([(1.0, 2.0)], num_frames=156, sr=16000, hop_length=512)
        >>> print(labels.shape, labels.sum())  # (156,) — some 1s in the middle
    """
    labels = np.zeros(num_frames, dtype=np.uint8)

    samples_per_frame = hop_length

    for start_sec, end_sec in dysfluency_intervals:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        start_frame = start_sample // samples_per_frame
        end_frame = end_sample // samples_per_frame
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        labels[start_frame:end_frame] = 1

    return labels


def create_frame_labels_from_samples(
    dysfluency_sample_ranges: List[Tuple[int, int]],
    num_frames: int,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Create frame labels when dysfluency boundaries are given as sample indices
    instead of seconds.

    Args:
        dysfluency_sample_ranges: List of (start_sample, end_sample) tuples.
        num_frames: Total spectrogram time frames.
        hop_length: Hop length used during spectrogram generation.

    Returns:
        1-D numpy uint8 array of shape (num_frames,).
    """
    labels = np.zeros(num_frames, dtype=np.uint8)

    for start_sample, end_sample in dysfluency_sample_ranges:
        start_frame = start_sample // hop_length
        end_frame = end_sample // hop_length
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        labels[start_frame:end_frame] = 1

    return labels


def pad_to_length(array: np.ndarray, target_length: int, axis: int = -1, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad a numpy array along a given axis to reach target_length.

    If the array is already longer than target_length, it is truncated.

    Args:
        array: Input array.
        target_length: Desired length along the given axis.
        axis: Axis to pad/truncate.
        pad_value: Value used for padding.

    Returns:
        Padded or truncated array.
    """
    current_length = array.shape[axis]
    if current_length >= target_length:
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(0, target_length)
        return array[tuple(slices)]
    else:
        pad_width = target_length - current_length
        pads = [(0, 0)] * array.ndim
        pads[axis] = (0, pad_width)
        return np.pad(array, pads, mode="constant", constant_values=pad_value)


def pad_audio_and_labels(
    audio: np.ndarray,
    labels: np.ndarray,
    max_length_samples: int,
    sr: int = 16000,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad audio and its corresponding frame labels to a fixed length.

    Used for batching variable-length samples in a DataLoader.

    Args:
        audio: 1-D audio array.
        labels: 1-D frame label array (must align with audio after padding).
        max_length_samples: Target audio length in samples.
        sr: Sample rate.
        hop_length: Hop length for label alignment.

    Returns:
        Tuple of (padded_audio, padded_labels).
    """
    audio = pad_to_length(audio, max_length_samples, axis=0, pad_value=0.0)
    max_frames = max_length_samples // hop_length
    labels = pad_to_length(labels, max_frames, axis=0, pad_value=0)
    return audio, labels


if __name__ == "__main__":
    # Quick self-test: generate synthetic audio and run the full pipeline
    print("=== Swaraaha Preprocessing Pipeline — Self Test ===")

    test_audio = debug_generate_test_audio(duration=3.0, sr=16000)
    print(f"Test audio shape: {test_audio.shape}, dtype: {test_audio.dtype}")

    spec = audio_to_spectrogram(test_audio, sr=16000)
    print(f"Spectrogram shape: {spec.shape}, dtype: {spec.dtype}")
    print(f"Spectrogram range: [{spec.min():.2f}, {spec.max():.2f}]")

    # Localization test: frame labels
    print("\n--- Localization Frame Label Test ---")
    # Mark 1.0s–2.0s as dysfluent in a 3-second audio
    intervals = [(1.0, 2.0)]
    num_frames = spec.shape[2]  # time dimension of spectrogram
    labels = create_frame_labels(intervals, num_frames=num_frames, sr=16000, hop_length=512)
    print(f"Labels shape: {labels.shape}, sum (dysfluent frames): {labels.sum()}")
    print(f"Labels: {labels}")

    # Verify alignment: label shape matches spectrogram time dimension
    assert labels.shape[0] == spec.shape[2], \
        f"Mismatch: labels {labels.shape[0]} frames != spec {spec.shape[2]} frames"
    print("Alignment check passed: labels match spectrogram time frames")

    # Pad test
    padded_audio, padded_labels = pad_audio_and_labels(test_audio, labels, max_length_samples=32000)
    print(f"Padded audio: {padded_audio.shape}, padded labels: {padded_labels.shape}")

    results = debug_save_spectrogram(test_audio, sr=16000, prefix="test")
    for key, path in results.items():
        print(f"  Saved {key}: {path}")

    print("=== Self test passed ===")
