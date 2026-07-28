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
from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Augmentation Integration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Audio Cleaning / Normalization
# ---------------------------------------------------------------------------

def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from audio signal.

    Subtracts the mean amplitude so the signal is centered around zero.
    This prevents bias in downstream feature extraction.

    Args:
        audio: 1-D float32 audio array.

    Returns:
        DC-corrected audio array.
    """
    return (audio - np.mean(audio)).astype(np.float32)


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio to a target peak amplitude.

    Scales the signal so its absolute maximum equals target_peak.
    Prevents clipping while maximizing dynamic range.

    Args:
        audio: 1-D float32 audio array.
        target_peak: Target peak amplitude (0-1). Default 0.95.

    Returns:
        Peak-normalized audio array.
    """
    if audio.size == 0:
        return audio
    peak = np.abs(audio).max()
    if peak == 0:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


def normalize_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    Normalize audio to a target RMS level.

    Scales the signal so its RMS equals target_rms.
    More perceptually uniform than peak normalization.

    Args:
        audio: 1-D float32 audio array.
        target_rms: Target RMS amplitude. Default 0.1.

    Returns:
        RMS-normalized audio array.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    return (audio * (target_rms / rms)).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sr: int = 16000,
    top_db: int = 25,
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Uses librosa's trim which detects non-silent intervals based on
    a dB threshold relative to the peak.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate (used for frame length).
        top_db: Threshold in dB below peak to consider as silence. Default 25.

    Returns:
        Trimmed audio array.
    """
    import librosa
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed.astype(np.float32)


def trim_silence_center(
    audio: np.ndarray,
    sr: int = 16000,
    top_db: int = 25,
    pad_ms: float = 50.0,
) -> np.ndarray:
    """
    Trim silence and add a small padding around the speech.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.
        top_db: Silence threshold in dB.
        pad_ms: Padding in milliseconds to add around speech.

    Returns:
        Trimmed and padded audio array.
    """
    import librosa
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    pad_samples = int(pad_ms * sr / 1000)
    padded = np.pad(trimmed, (pad_samples, pad_samples), mode="constant")
    return padded.astype(np.float32)


def clean_audio(
    audio: np.ndarray,
    sr: int = 16000,
    remove_dc: bool = True,
    normalize: bool = True,
    trim: bool = True,
    target_peak: float = 0.95,
) -> np.ndarray:
    """
    Full audio cleaning pipeline: DC removal → normalization → silence trimming.

    This is the recommended preprocessing step before feeding audio to models.
    Apply this consistently across training and inference.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.
        remove_dc: Remove DC offset.
        normalize: Apply peak normalization.
        trim: Trim leading/trailing silence.
        target_peak: Target peak amplitude for normalization.

    Returns:
        Cleaned audio array.

    Example:
        >>> audio, sr = load_audio("raw_recording.wav")
        >>> audio = clean_audio(audio, sr=sr)
        >>> # audio is now DC-free, peak-normalized, silence-trimmed
    """
    if remove_dc:
        audio = remove_dc_offset(audio)
    if normalize:
        audio = normalize_peak(audio, target_peak=target_peak)
    if trim:
        audio = trim_silence(audio, sr=sr)
    return audio


def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    p: float = 0.5,
) -> np.ndarray:
    """
    Apply random augmentation pipeline to audio with probability p.

    Convenience wrapper around AudioAugmentor for use in dataset __getitem__.

    Args:
        audio: 1-D float32 audio array, values in [-1.0, 1.0].
        sr: Sample rate in Hz.
        p: Probability of applying augmentation (0.0 = no augmentation).

    Returns:
        Augmented (or original) audio array, same length as input.

    Example:
        >>> # In dataset __getitem__:
        >>> if self.augment:
        >>>     audio = augment_audio(audio, sr=self.sr, p=0.5)
    """
    from model.data.augmentation import AudioAugmentor
    import random

    if random.random() > p:
        return audio

    augmentor = AudioAugmentor()
    return augmentor(audio, sr=sr)


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels.

    Args:
        signal: Clean signal array.
        noise: Noise array (same length as signal).

    Returns:
        SNR value in dB. Higher values indicate less noise.

    Example:
        >>> snr = compute_snr(clean_audio, noisy_audio - clean_audio)
        >>> print(f"SNR: {snr:.1f} dB")
    """
    signal = np.asarray(signal, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return float(snr_db)


# ---------------------------------------------------------------------------
# Audio Quality Checks
# ---------------------------------------------------------------------------

def check_audio_quality(
    audio: np.ndarray,
    sr: int = 16000,
    min_duration: float = 0.5,
    max_amplitude: float = 0.99,
    silence_threshold: float = 0.01,
    min_rms: float = 0.005,
) -> Dict[str, object]:
    """
    Check audio quality and return diagnostics.

    Detects common issues that can hurt model training:
    - Clipping (samples near ±1.0)
    - Near-silence (very low amplitude)
    - Very short clips
    - DC offset

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.
        min_duration: Minimum acceptable duration in seconds.
        max_amplitude: Threshold for clipping detection (0-1).
        silence_threshold: RMS threshold for silence detection.
        min_rms: Minimum RMS for non-silent audio.

    Returns:
        Dict with quality metrics:
            - "duration_sec": float
            - "rms": float
            - "peak_amplitude": float
            - "is_clipped": bool
            - "is_silent": bool
            - "is_too_short": bool
            - "dc_offset": float
            - "is_valid": bool (True if all checks pass)
            - "issues": list of str (descriptions of any problems found)

    Example:
        >>> quality = check_audio_quality(audio, sr=16000)
        >>> if not quality["is_valid"]:
        >>>     print(f"Audio issues: {quality['issues']}")
    """
    issues = []

    # Duration
    duration_sec = len(audio) / sr
    is_too_short = duration_sec < min_duration
    if is_too_short:
        issues.append(f"Too short: {duration_sec:.2f}s < {min_duration}s")

    # Amplitude statistics
    peak_amplitude = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio ** 2)))
    dc_offset = float(np.mean(audio))

    # Clipping
    is_clipped = peak_amplitude >= max_amplitude
    if is_clipped:
        issues.append(f"Clipping detected: peak={peak_amplitude:.3f}")

    # Silence
    is_silent = rms < silence_threshold
    if is_silent:
        issues.append(f"Near-silent: RMS={rms:.4f}")

    # DC offset
    if abs(dc_offset) > 0.1:
        issues.append(f"DC offset: {dc_offset:.4f}")

    is_valid = len(issues) == 0

    return {
        "duration_sec": round(duration_sec, 4),
        "rms": round(rms, 6),
        "peak_amplitude": round(peak_amplitude, 6),
        "is_clipped": is_clipped,
        "is_silent": is_silent,
        "is_too_short": is_too_short,
        "dc_offset": round(dc_offset, 6),
        "is_valid": is_valid,
        "issues": issues,
    }


def filter_audio_samples(
    audio_paths: List[str],
    sr: int = 16000,
    min_duration: float = 0.5,
) -> Tuple[List[str], List[Dict]]:
    """
    Filter a list of audio files, returning only valid ones.

    Args:
        audio_paths: List of paths to audio files.
        sr: Sample rate for loading.
        min_duration: Minimum duration in seconds.

    Returns:
        Tuple of (valid_paths, quality_reports) where quality_reports
        contains the check results for each file.
    """
    valid_paths = []
    reports = []

    for path in audio_paths:
        try:
            audio, _ = load_audio(path, sr=sr)
            quality = check_audio_quality(audio, sr=sr, min_duration=min_duration)
            reports.append({"path": path, **quality})

            if quality["is_valid"]:
                valid_paths.append(path)
        except Exception as e:
            reports.append({"path": path, "is_valid": False, "issues": [str(e)]})

    return valid_paths, reports


# ---------------------------------------------------------------------------
# Class Balancing
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    Useful for setting pos_weight in BCEWithLogitsLoss or class_weight in
    other loss functions. Higher weight for minority classes.

    Args:
        labels: 1-D array of binary labels (0 or 1) for a single class,
                or 2-D array (N, C) for multi-label.

    Returns:
        Dict mapping class index to weight. {0: weight_neg, 1: weight_pos}.
    """
    labels = np.asarray(labels).flatten()
    n_total = len(labels)
    n_pos = int(labels.sum())
    n_neg = n_total - n_pos

    if n_pos == 0 or n_neg == 0:
        return {0: 1.0, 1: 1.0}

    weight_neg = n_total / (2.0 * n_neg)
    weight_pos = n_total / (2.0 * n_pos)

    return {0: round(weight_neg, 4), 1: round(weight_pos, 4)}


def compute_pos_weight(labels: np.ndarray) -> float:
    """
    Compute pos_weight for BCEWithLogitsLoss from binary labels.

    pos_weight = n_neg / n_pos. This tells the loss to penalize
    false negatives more heavily when positives are rare.

    Args:
        labels: 1-D array of binary labels (0 or 1).

    Returns:
        pos_weight as float. Returns 1.0 if balanced.
    """
    labels = np.asarray(labels).flatten()
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    if n_pos == 0:
        return 1.0

    return round(n_neg / n_pos, 4)


def oversample_minority(
    indices: np.ndarray,
    labels: np.ndarray,
    target_ratio: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Oversample minority class indices to reach target_pos/neg ratio.

    Args:
        indices: Array of dataset indices to resample.
        labels: Binary labels aligned with indices (0 or 1).
        target_ratio: Desired ratio of positives to negatives. Default 1.0 (balanced).
        seed: Random seed for reproducibility.

    Returns:
        Resampled indices array with oversampled minority class.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    pos_idx = indices[labels == 1]
    neg_idx = indices[labels == 0]

    n_neg = len(neg_idx)
    target_pos = int(n_neg * target_ratio)

    if len(pos_idx) >= target_pos:
        return indices

    # Oversample positives with replacement
    oversampled = rng.choice(pos_idx, size=target_pos, replace=True)
    return np.concatenate([neg_idx, oversampled])


def create_balanced_sampler(labels: np.ndarray) -> "torch.utils.data.WeightedRandomSampler":
    """
    Create a PyTorch WeightedRandomSampler for balanced mini-batches.

    Each sample gets weight = 1/class_frequency. Minority class samples
    are sampled more frequently to balance each batch.

    Args:
        labels: 1-D array of binary labels for the dataset.

    Returns:
        WeightedRandomSampler instance.

    Example:
        >>> sampler = create_balanced_sampler(train_labels)
        >>> loader = DataLoader(dataset, batch_size=8, sampler=sampler)
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(labels).flatten()
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    weights = np.where(labels == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    weights = weights / weights.sum()

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(labels),
        replacement=True,
    )


if __name__ == "__main__":
    # Quick self-test: generate synthetic audio and run the full pipeline
    print("=== Swaraaha Preprocessing Pipeline — Self Test ===")

    test_audio = debug_generate_test_audio(duration=3.0, sr=16000)
    print(f"Test audio shape: {test_audio.shape}, dtype: {test_audio.dtype}")

    # Test cleaning pipeline
    cleaned = clean_audio(test_audio, sr=16000)
    print(f"Cleaned audio: peak={np.abs(cleaned).max():.3f}, rms={np.sqrt(np.mean(cleaned**2)):.4f}")

    # Test individual cleaning functions
    dc_removed = remove_dc_offset(test_audio)
    normalized = normalize_peak(test_audio, target_peak=0.9)
    trimmed = trim_silence(test_audio, sr=16000)
    print(f"DC removed: mean={np.mean(dc_removed):.6f}")
    print(f"Normalized: peak={np.abs(normalized).max():.3f}")
    print(f"Trimmed: {len(test_audio)} -> {len(trimmed)} samples")

    # Test class balancing
    fake_labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    weights = compute_class_weights(fake_labels)
    pw = compute_pos_weight(fake_labels)
    print(f"Class weights: {weights}, pos_weight: {pw}")

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
