# Swaraaha - Localization Models
# CNN-based spectrogram localization + Wav2Vec2-based localization

from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer
from model.localization.wav2vec2_localizer import Wav2Vec2Localizer

DYSFLUENCY_CLASSES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]

__all__ = ["CNNSpectrogramLocalizer", "Wav2Vec2Localizer", "DYSFLUENCY_CLASSES"]
