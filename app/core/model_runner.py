"""Model inference wrapper — classification, localization and transcription.

Uses model.registry (MultiTaskClassifier / Localizer) for real inference.
Classification errors propagate so the UI surfaces them instead of fabricating
results.
"""

import logging

import numpy as np

from app.core.transcription import AudioTranscriber

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self, models_dir: str = ""):
        self.models_dir = models_dir
        self._classifier = None
        self._localizer = None
        self.transcriber = AudioTranscriber()

    def transcribe(self, audio: np.ndarray, localizations=None, language: str = "english") -> dict:
        """Run transcription pipeline on audio."""
        return self.transcriber.transcribe(audio, localizations=localizations, language=language)

    def _get_classifier(self):
        if self._classifier is None:
            from model.registry import MultiTaskClassifier
            self._classifier = MultiTaskClassifier()
        return self._classifier

    def _get_localizer(self):
        if self._localizer is None:
            from model.registry import Localizer
            self._localizer = Localizer("cnn")
        return self._localizer

    def _classify(self, audio: np.ndarray) -> dict:
        """Multi-task classification results as {name: (stutter_present, confidence)}."""
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        raw = self._get_classifier().analyze(audio_np)
        return {
            name: (bool(result["label"]), float(result["confidence"]))
            for name, result in raw.items()
            if name != "summary"
        }

    def _localize(self, audio: np.ndarray) -> list:
        """Dysfluency regions as [(start_sec, end_sec, confidence), ...]."""
        try:
            from model.data.preprocessing import generate_mel_spectrogram
            audio_np = np.asarray(audio, dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            spec = generate_mel_spectrogram(audio_np, sr=16000)
            return list(self._get_localizer().predict(spec, threshold=0.3))
        except Exception as e:
            logger.warning("Localization unavailable: %s", e)
            return []

    def _combine(self, audio: np.ndarray, localizations: list) -> dict:
        """Fuse localization tuples with classifier saliency into combined regions."""
        try:
            from model.combiner import combine_regions
            from model.config.defaults import DYSFLUENCY_CLASSES

            audio_np = np.asarray(audio, dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            classifier = self._get_classifier()
            saliency = classifier.saliency(audio_np).squeeze(0)
            if hasattr(saliency, "cpu"):
                saliency = saliency.cpu()
            saliency = np.asarray(saliency, dtype=float)
            regions = [
                {"start": s, "end": e, "confidence": c}
                for s, e, c in localizations
            ]
            return combine_regions(
                regions,
                saliency,
                class_names=list(DYSFLUENCY_CLASSES),
                thresholds=getattr(classifier, "_thresholds", {}) or None,
                audio_duration=len(audio_np) / 16000,
            )
        except Exception as exc:
            logger.warning("Combined fusion unavailable: %s", exc)
            return {"error": str(exc)}

    def analyze(self, audio: np.ndarray, language: str = "english") -> dict:
        """Run classification + localization + transcription on audio. Returns structured results."""
        classifications = self._classify(audio)
        localizations = self._localize(audio)
        transcription = self.transcribe(audio, localizations=localizations, language=language)
        combined = self._combine(audio, localizations)

        return {
            "classifications": classifications,
            "localizations": localizations,
            "transcription": transcription,
            "combined": combined,
        }
