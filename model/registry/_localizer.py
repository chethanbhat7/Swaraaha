"""LocalizerRunner — dysfluency region detection with word/syllable alignment."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

from model.config.defaults import AUDIO_DURATION_SECONDS, SAMPLE_RATE

from ._utils import (
    _align_words_syllables,
    _audio_is_empty,
    _chunk_audio,
    _LOCALIZER_LOADERS,
    _LOCALIZER_PREDICTORS,
)


def _get_reg():
    import model.registry
    return model.registry


class LocalizerRunner:
    def __init__(self, model_type: Optional[str] = None):
        self.model_type = model_type
        self._models: Dict[str, Any] = {}

    def _load(self) -> None:
        _reg = _get_reg()
        registry = _reg._load_registry()
        localization = registry.get("localization", {})

        if not localization:
            raise FileNotFoundError(
                "No localization models in registry. "
                "Add entries to model/registry.json localization section."
            )

        types_to_load = (
            [self.model_type] if self.model_type else list(localization.keys())
        )

        for lt in types_to_load:
            if lt not in localization:
                available = list(localization.keys())
                raise ValueError(
                    f"Localizer '{lt}' not in registry. Available: {available}"
                )
            path = _reg._resolve_path(localization[lt])
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    f"Registry entry: localization.{lt}"
                )

            loader = _LOCALIZER_LOADERS.get(lt)
            if loader is None:
                raise ValueError(f"Unknown localizer type: {lt}")
            model = loader(path)
            model.max_length_seconds = AUDIO_DURATION_SECONDS
            self._models[lt] = model

    def predict(
        self, audio_tensor, threshold: float = 0.3
    ) -> Union[List[Tuple], Dict[str, List[Tuple]]]:
        if not self._models:
            self._load()

        if self.model_type is not None:
            return self._models[self.model_type].predict(audio_tensor, threshold=threshold)
        else:
            results = {}
            for name, model in self._models.items():
                results[name] = model.predict(audio_tensor, threshold=threshold)
            return results

    def analyze(
        self,
        audio,
        text: Optional[str] = None,
        language: str = "en",
        threshold: float = 0.3,
        max_length_seconds: Optional[float] = None,
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Analyze raw audio → dysfluency regions (+ words/syllables if text).

        Args:
            audio: File path, raw bytes, or 1-D numpy array.
            text: Optional transcript for word/syllable-level alignment.
            language: ISO language code for syllabification (en, kn, hi).
            threshold: Detection threshold for regions.
            max_length_seconds: Max audio length to process (defaults to model's
                fingerprint value).

        Returns:
            Single-type: {regions, [words], [syllables]}.
            All-types: {type: {...}} keyed by localizer type.
        """
        if _audio_is_empty(audio):
            types = [self.model_type] if self.model_type else list(self._models.keys())
            empty: Dict[str, Any] = {"regions": []}
            results = {lt: dict(empty) for lt in types}
            if self.model_type is not None:
                return results.get(self.model_type, empty)
            return results

        if not self._models:
            self._load()

        if max_length_seconds is None:
            first_model = next(iter(self._models.values()), None)
            max_length_seconds = getattr(first_model, "max_length_seconds", AUDIO_DURATION_SECONDS) if first_model is not None else AUDIO_DURATION_SECONDS

        from model.data.preprocessing import load_audio_input

        audio_array = load_audio_input(audio, sr=SAMPLE_RATE)
        audio_sec = len(audio_array) / SAMPLE_RATE

        types = [self.model_type] if self.model_type else list(self._models.keys())
        results: Dict[str, Any] = {}

        for lt in types:
            model = self._models[lt]

            if audio_sec <= max_length_seconds + 0.1:
                regions = self._localize_chunk(model, lt, audio_array, threshold, max_length_seconds)
            else:
                regions = []
                for chunk, start_sample in _chunk_audio(audio_array, max_length_seconds, sr=SAMPLE_RATE):
                    chunk_regions = self._localize_chunk(model, lt, chunk, threshold, max_length_seconds)
                    offset_sec = start_sample / SAMPLE_RATE
                    for s, e, c in chunk_regions:
                        regions.append((round(s + offset_sec, 3), round(e + offset_sec, 3), c))
                regions = self._dedupe_regions(regions)

            entry: Dict[str, Any] = {
                "regions": [
                    {"start": round(s, 3), "end": round(e, 3), "confidence": round(c, 4)}
                    for s, e, c in regions
                ]
            }

            if text:
                words, syllables = _align_words_syllables(
                    audio_array, text, language, sr=SAMPLE_RATE
                )
                entry["words"] = words
                entry["syllables"] = syllables

            results[lt] = entry

        if self.model_type is not None:
            return results[self.model_type]
        return results

    @staticmethod
    def _localize_chunk(model, lt, audio_array, threshold, max_length_seconds):
        """Run localizer on a single chunk, return list of (start, end, conf)."""
        predictor = _LOCALIZER_PREDICTORS.get(lt)
        if predictor is None:
            raise ValueError(f"Unknown localizer type: {lt}")
        return predictor(model, audio_array, threshold, max_length_seconds)

    @staticmethod
    def _dedupe_regions(regions, iou_threshold: float = 0.5):
        """Remove overlapping regions, keeping the one with higher confidence."""
        if not regions:
            return regions
        regions = sorted(regions, key=lambda r: r[2], reverse=True)
        kept = []
        for start, end, conf in regions:
            overlap = any(
                start < ke and end > ks for ks, ke, _ in kept
            )
            if not overlap:
                kept.append((start, end, conf))
        return sorted(kept, key=lambda r: r[0])

    @property
    def is_loaded(self) -> bool:
        return bool(self._models)
