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

from model.data.preprocessing.balancing import (
    compute_class_weights,
    compute_pos_weight,
    create_balanced_sampler,
    oversample_minority,
)
from model.data.preprocessing.cleaning import (
    augment_audio,
    clean_audio,
    normalize_peak,
    normalize_rms,
    remove_dc_offset,
    trim_silence,
    trim_silence_center,
)
from model.data.preprocessing.debug import (
    debug_generate_test_audio,
    debug_save_spectrogram,
)
from model.data.preprocessing.frame_labels import (
    create_frame_labels,
    create_frame_labels_from_samples,
)
from model.data.preprocessing.io import (
    convert_to_wav,
    load_audio,
    load_audio_from_array,
    load_audio_input,
)
from model.data.preprocessing.quality import (
    check_audio_quality,
    compute_snr,
    filter_audio_samples,
)
from model.data.preprocessing.spectrogram import (
    audio_to_spectrogram,
    file_to_spectrogram,
    generate_mel_spectrogram,
    normalize_spectrogram,
    plot_waveform,
    save_spectrogram,
    save_spectrogram_image,
    spectrogram_to_image_array,
)
from model.data.preprocessing.utils import (
    pad_audio_and_labels,
    pad_to_length,
)

__all__ = [
    # io
    "convert_to_wav",
    "load_audio",
    "load_audio_from_array",
    "load_audio_input",
    # cleaning
    "remove_dc_offset",
    "normalize_peak",
    "normalize_rms",
    "trim_silence",
    "trim_silence_center",
    "clean_audio",
    "augment_audio",
    # spectrogram
    "generate_mel_spectrogram",
    "normalize_spectrogram",
    "spectrogram_to_image_array",
    "save_spectrogram",
    "save_spectrogram_image",
    "plot_waveform",
    "audio_to_spectrogram",
    "file_to_spectrogram",
    # quality
    "compute_snr",
    "check_audio_quality",
    "filter_audio_samples",
    # balancing
    "compute_class_weights",
    "compute_pos_weight",
    "oversample_minority",
    "create_balanced_sampler",
    # frame labels
    "create_frame_labels",
    "create_frame_labels_from_samples",
    # utils
    "pad_to_length",
    "pad_audio_and_labels",
    # debug
    "debug_save_spectrogram",
    "debug_generate_test_audio",
]
