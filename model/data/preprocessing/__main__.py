"""Quick self-test: generate synthetic audio and run the full pipeline."""

import numpy as np

from model.data.preprocessing import (
    audio_to_spectrogram,
    clean_audio,
    compute_class_weights,
    compute_pos_weight,
    create_frame_labels,
    debug_generate_test_audio,
    debug_save_spectrogram,
    normalize_peak,
    pad_audio_and_labels,
    remove_dc_offset,
    trim_silence,
)

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
intervals = [(1.0, 2.0)]
num_frames = spec.shape[2]
labels = create_frame_labels(intervals, num_frames=num_frames, sr=16000, hop_length=512)
print(f"Labels shape: {labels.shape}, sum (dysfluent frames): {labels.sum()}")
print(f"Labels: {labels}")

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
