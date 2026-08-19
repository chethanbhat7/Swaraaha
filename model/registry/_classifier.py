"""Classifier class — individual and all-class dysfluency classification."""

import os
from typing import Any, Dict, Optional, Tuple, Union

from model.config.defaults import AUDIO_DURATION_SECONDS, DYSFLUENCY_CLASSES, SAMPLE_RATE

from ._utils import (
    _audio_is_empty,
    _chunk_audio,
    _classifier_output,
    _empty_classifier_output,
    _preprocess_audio,
    _registry_classification_names,
)


def _get_reg():
    import model.registry
    return model.registry


class Classifier:
    def __init__(self, class_name: Optional[str] = None):
        self.class_name = class_name
        self._model = None
        self._models: Dict[str, Any] = {}

    def _load(self) -> None:
        _reg = _get_reg()
        registry = _reg._load_registry()
        classification = registry.get("classification", {})

        if self.class_name is not None:
            if self.class_name not in classification:
                available = list(classification.keys())
                raise ValueError(
                    f"Classifier '{self.class_name}' not in registry. "
                    f"Available: {available}"
                )
            path = _reg._resolve_path(classification[self.class_name])
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    f"Registry entry: classification.{self.class_name}"
                )

            self._model = _reg._load_classifier(self.class_name, path)
        else:
            missing = []
            for name in ["prolongation", "block", "soundrep", "wordrep", "interjection"]:
                if name not in classification:
                    missing.append(name)
                elif not os.path.exists(_reg._resolve_path(classification[name])):
                    missing.append(f"{name} (file not found)")
            if missing:
                raise FileNotFoundError(
                    f"Missing classification models in registry: {missing}"
                )

            from model.config.defaults import DYSFLUENCY_CLASSES

            self._models = {}
            for name, entry in classification.items():
                if name not in DYSFLUENCY_CLASSES:
                    print(
                        f"WARNING: ignoring unknown classification entry '{name}' "
                        f"in registry (expected one of {sorted(DYSFLUENCY_CLASSES)})"
                    )
                    continue
                path = _reg._resolve_path(entry)
                self._models[name] = _reg._load_classifier(name, path)

    def predict(
        self, audio_tensor, threshold: Optional[float] = None
    ) -> Union[Dict[str, Tuple[int, float]], Tuple[int, float]]:
        if self._model is None and not self._models:
            self._load()

        if self.class_name is not None:
            thr = self._resolve_thresholds(threshold)[self.class_name]
            return self._model.predict(audio_tensor, threshold=thr)
        else:
            return self.predict_all(audio_tensor, threshold)

    def predict_all(
        self, audio_tensor, threshold: Optional[float] = None
    ) -> Dict[str, Tuple[int, float]]:
        if self._model is None:
            self._load()

        if self.class_name is not None:
            raise ValueError(
                "predict_all() is only available when loading all classifiers "
                "(call Classifier() without class_name)"
            )

        thresholds = self._resolve_thresholds(threshold)
        results = {}
        for name, clf in self._models.items():
            results[name] = clf.predict(audio_tensor, threshold=thresholds[name])
        return results

    @staticmethod
    def _default_thresholds() -> Dict[str, float]:
        return {
            "prolongation": 0.5,
            "block": 0.5,
            "soundrep": 0.5,
            "wordrep": 0.5,
            "interjection": 0.5,
        }

    def _resolve_thresholds(self, threshold: Optional[float]) -> Dict[str, float]:
        registry = _get_reg()._load_registry()
        configured = registry.get("thresholds", {})
        defaults = self._default_thresholds()
        thresholds = {
            name: configured.get(name, defaults.get(name, 0.5))
            for name in defaults
        }
        if threshold is not None:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"threshold must be in [0, 1], got {threshold}")
            thresholds = {name: threshold for name in thresholds}
        return thresholds

    def _run_single(self, audio, threshold: Optional[float], include_logits: bool) -> Dict[str, Any]:
        if _audio_is_empty(audio):
            return _empty_classifier_output(include_logits)
        if self._model is None:
            self._load()
        tensor = _preprocess_audio(audio)
        thresholds = self._resolve_thresholds(threshold)
        thr = thresholds[self.class_name]
        return _classifier_output(self._model, tensor, thr, include_logits)

    def _run_all(self, audio, threshold: Optional[float], include_logits: bool) -> Dict[str, Any]:
        if _audio_is_empty(audio):
            if self._models:
                names = list(self._models)
            else:
                _reg = _get_reg()
                registry = _reg._load_registry()
                names = [n for n in registry.get("classification", {}) if n in DYSFLUENCY_CLASSES]
            results: Dict[str, Any] = {
                name: _empty_classifier_output(include_logits) for name in names
            }
            results["summary"] = {
                "detected": [],
                "primary": names[0] if names else None,
            }
            return results
        if not self._models:
            self._load()

        from model.data.preprocessing import load_audio_input

        audio_array = load_audio_input(audio, sr=SAMPLE_RATE)
        max_len = getattr(list(self._models.values())[0], "max_length_seconds", AUDIO_DURATION_SECONDS)
        audio_sec = len(audio_array) / SAMPLE_RATE

        if audio_sec <= max_len + 0.1:
            tensor = _preprocess_audio(audio)
            thresholds = self._resolve_thresholds(threshold)
            results: Dict[str, Any] = {}
            for name, clf in self._models.items():
                results[name] = _classifier_output(clf, tensor, thresholds[name], include_logits)
        else:
            thresholds = self._resolve_thresholds(threshold)
            merged: Dict[str, Dict[str, float]] = {}
            for chunk, _ in _chunk_audio(audio_array, max_len, sr=SAMPLE_RATE):
                tensor = _preprocess_audio(chunk)
                for name, clf in self._models.items():
                    chunk_out = _classifier_output(clf, tensor, thresholds[name], include_logits)
                    if name not in merged:
                        merged[name] = chunk_out
                    elif chunk_out["prob_present"] > merged[name]["prob_present"]:
                        merged[name] = chunk_out
            results = merged

        detected = [name for name, out in results.items() if out["label"] == 1]
        primary = max(results.items(), key=lambda kv: kv[1]["prob_present"])[0]
        results["summary"] = {"detected": detected, "primary": primary}
        return results

    def analyze(
        self, audio, threshold: Optional[float] = None
    ) -> Union[Dict[str, Any], Dict[str, float]]:
        """Simple analysis: label + confidence + probabilities per class."""
        if self.class_name is not None:
            return self._run_single(audio, threshold, include_logits=False)
        return self._run_all(audio, threshold, include_logits=False)

    def analyze_raw(
        self, audio, threshold: Optional[float] = None
    ) -> Union[Dict[str, Any], Dict[str, float]]:
        """Advanced analysis: everything in analyze() plus raw logits."""
        if self.class_name is not None:
            return self._run_single(audio, threshold, include_logits=True)
        return self._run_all(audio, threshold, include_logits=True)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or bool(self._models)
