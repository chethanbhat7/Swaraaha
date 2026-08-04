"""
Model Registry — loads trained models and exposes a clean predict API.

Usage:
    from model.registry import Classifier, Localizer, ModelRegistry

    # All classifiers (HybridClassifier)
    clf = Classifier()
    result = clf.predict(audio_tensor)           # {class_name: (label, confidence)}

    # Single classifier
    clf = Classifier("prolongation")
    result = clf.predict(audio_tensor)           # (label, confidence)

    # Localizer
    loc = Localizer("cnn")
    regions = loc.predict(audio_tensor)          # [(start, end, confidence), ...]

    # Everything at once
    m = ModelRegistry()
    all_results = m.run_all(audio_tensor)        # classify + localize
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.json")


def _load_registry() -> dict:
    if not os.path.exists(_REGISTRY_PATH):
        raise FileNotFoundError(
            f"Registry file not found: {_REGISTRY_PATH}\n"
            "Create model/registry.json with available model paths."
        )
    with open(_REGISTRY_PATH) as f:
        return json.load(f)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    project_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(project_root, path)


class Classifier:
    def __init__(self, class_name: Optional[str] = None):
        self.class_name = class_name
        self._model = None

    def _load(self) -> None:
        registry = _load_registry()
        classification = registry.get("classification", {})

        if self.class_name is not None:
            if self.class_name not in classification:
                available = list(classification.keys())
                raise ValueError(
                    f"Classifier '{self.class_name}' not in registry. "
                    f"Available: {available}"
                )
            path = _resolve_path(classification[self.class_name])
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    f"Registry entry: classification.{self.class_name}"
                )

            from model.classification.prolongation import ProlongationClassifier
            from model.classification.block import BlockClassifier
            from model.classification.soundrep import SoundRepClassifier
            from model.classification.wordrep import WordRepClassifier
            from model.classification.interjection import InterjectionClassifier

            cls_map = {
                "prolongation": ProlongationClassifier,
                "block": BlockClassifier,
                "soundrep": SoundRepClassifier,
                "wordrep": WordRepClassifier,
                "interjection": InterjectionClassifier,
            }
            self._model = cls_map[self.class_name].from_pretrained(path)
        else:
            missing = []
            for name in ["prolongation", "block", "soundrep", "wordrep", "interjection"]:
                if name not in classification:
                    missing.append(name)
                elif not os.path.exists(_resolve_path(classification[name])):
                    missing.append(f"{name} (file not found)")
            if missing:
                raise FileNotFoundError(
                    f"Missing classification models in registry: {missing}"
                )

            from model.classification import DYSFLUENCY_CLASSES
            from model.classification.hybrid import HybridClassifier, CombinerMLP
            from model.classification.prolongation import ProlongationClassifier
            from model.classification.block import BlockClassifier
            from model.classification.soundrep import SoundRepClassifier
            from model.classification.wordrep import WordRepClassifier
            from model.classification.interjection import InterjectionClassifier

            cls_map = {
                "prolongation": ProlongationClassifier,
                "block": BlockClassifier,
                "soundrep": SoundRepClassifier,
                "wordrep": WordRepClassifier,
                "interjection": InterjectionClassifier,
            }

            base_classifiers = []
            for name in DYSFLUENCY_CLASSES:
                path = _resolve_path(classification[name])
                base_classifiers.append(cls_map[name].from_pretrained(path))

            self._model = HybridClassifier.__new__(HybridClassifier)
            self._model.model_name = "facebook/wav2vec2-base"
            self._model.base_classifiers = base_classifiers
            self._model.combiner = CombinerMLP()

    def predict(
        self, audio_tensor, threshold: float = 0.5
    ) -> Union[Dict[str, Tuple[int, float]], Tuple[int, float]]:
        if self._model is None:
            self._load()

        if self.class_name is not None:
            return self._model.predict(audio_tensor)
        else:
            return self._model.predict(audio_tensor, threshold=threshold)

    def predict_all(
        self, audio_tensor
    ) -> Dict[str, Tuple[int, float]]:
        if self._model is None:
            self._load()

        if self.class_name is not None:
            raise ValueError(
                "predict_all() is only available when loading all classifiers "
                "(call Classifier() without class_name)"
            )

        results = {}
        for clf in self._model.base_classifiers:
            results[clf.class_name] = clf.predict(audio_tensor)
        return results

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


class Localizer:
    def __init__(self, model_type: Optional[str] = None):
        self.model_type = model_type
        self._models: Dict[str, Any] = {}

    def _load(self) -> None:
        registry = _load_registry()
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
            path = _resolve_path(localization[lt])
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    f"Registry entry: localization.{lt}"
                )

            if lt == "cnn":
                from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer
                self._models["cnn"] = CNNSpectrogramLocalizer.from_pretrained(path)
            elif lt == "wav2vec2":
                from model.localization.wav2vec2_localizer import Wav2Vec2Localizer
                self._models["wav2vec2"] = Wav2Vec2Localizer.from_pretrained(path)
            else:
                raise ValueError(f"Unknown localizer type: {lt}")

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

    @property
    def is_loaded(self) -> bool:
        return bool(self._models)


class ModelRegistry:
    def __init__(self):
        self.classifier = Classifier()
        self.localizer = Localizer()

    def run_all(
        self, audio_tensor, classify_threshold: float = 0.5, localize_threshold: float = 0.3
    ) -> Dict[str, Any]:
        results = {}

        try:
            results["classification"] = self.classifier.predict(
                audio_tensor, threshold=classify_threshold
            )
        except FileNotFoundError as e:
            results["classification"] = {"error": str(e)}

        try:
            results["localization"] = self.localizer.predict(
                audio_tensor, threshold=localize_threshold
            )
        except FileNotFoundError as e:
            results["localization"] = {"error": str(e)}

        return results

    @property
    def is_loaded(self) -> bool:
        return self.classifier.is_loaded and self.localizer.is_loaded
