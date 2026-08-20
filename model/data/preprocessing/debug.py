"""Debug utilities: synthetic audio generation and debug artifact saving."""

import os

import numpy as np

from model.data.preprocessing.spectrogram import (
    generate_mel_spectrogram,
    plot_waveform,
    save_spectrogram,
    save_spectrogram_image,
)


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

    plot_waveform(audio, sr, waveform_path, title=f"Waveform — {prefix}")

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
