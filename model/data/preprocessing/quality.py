"""Audio quality checks: SNR, clipping detection, filtering."""

from typing import Dict, List, Tuple

import numpy as np

from model.data.preprocessing.io import load_audio


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

    duration_sec = len(audio) / sr
    is_too_short = duration_sec < min_duration
    if is_too_short:
        issues.append(f"Too short: {duration_sec:.2f}s < {min_duration}s")

    peak_amplitude = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio ** 2)))
    dc_offset = float(np.mean(audio))

    is_clipped = peak_amplitude >= max_amplitude
    if is_clipped:
        issues.append(f"Clipping detected: peak={peak_amplitude:.3f}")

    is_silent = rms < silence_threshold
    if is_silent:
        issues.append(f"Near-silent: RMS={rms:.4f}")

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
