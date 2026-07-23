# Swaraaha - Localization Models
# CNN-based spectrogram localization for dysfluency detection

from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer

DYSFLUENCY_CLASSES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]

__all__ = ["CNNSpectrogramLocalizer", "DYSFLUENCY_CLASSES"]
