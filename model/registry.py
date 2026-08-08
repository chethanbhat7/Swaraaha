"""
Model Registry — loads trained models and exposes a clean predict API.

Usage:
    from model.registry import Classifier, Localizer, ModelRegistry

    # All classifiers
    clf = Classifier()
    result = clf.analyze("recording.wav")        # path
    result = clf.analyze(audio_bytes)            # bytes
    result = clf.analyze(np_array, sr=16000)     # numpy array
    # result: {prolongation: {...}, ..., summary: {detected: [...], primary: ...}}

    # Single classifier
    clf = Classifier("prolongation")
    result = clf.analyze(audio)                  # {label, confidence, prob_present, prob_not_present}

    # Advanced: adds raw logits
    result = clf.analyze_raw(audio)

    # Per-call threshold override
    result = clf.analyze(audio, threshold=0.6)

    # Localizer (unchanged)
    loc = Localizer("cnn")
    regions = loc.predict(audio_tensor)

    # Everything at once
    m = ModelRegistry()
    all_results = m.run_all(audio_tensor)
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from model.fingerprint import model_name_from_path

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


def _load_classifier(class_name: str, path: str):
    """Load a classifier from a checkpoint, handling training and classifier formats."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if "model_name" in checkpoint:
        from model.classification import get_classifier_class
        return get_classifier_class(class_name).from_pretrained(path)

    from transformers import Wav2Vec2ForSequenceClassification

    from model.classification import DYSFLUENCY_CLASSES, BaseWav2VecClassifier

    model_name = model_name_from_path(path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    instance = BaseWav2VecClassifier.__new__(BaseWav2VecClassifier)
    instance.model_name = model_name
    instance._model = model
    instance.class_name = class_name
    instance.class_idx = DYSFLUENCY_CLASSES.index(class_name)
    return instance


def _preprocess_audio(
    audio, sr: int = 16000, max_length_seconds: float = 10.0
) -> "torch.Tensor":
    """Normalize audio input (path / bytes / ndarray) into a model-ready tensor.

    Returns a float32 tensor of shape [1, max_length_seconds * sr].
    """
    import torch

    from model.data.preprocessing import (
        clean_audio,
        load_audio_input,
        pad_to_length,
    )

    audio_array = load_audio_input(audio, sr=sr)
    audio_array = clean_audio(audio_array, sr=sr)
    audio_array = pad_to_length(
        audio_array, int(max_length_seconds * sr), axis=0, pad_value=0.0
    )
    tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    return tensor


def _classifier_output(clf, audio_tensor, threshold: float, include_logits: bool) -> Dict[str, float]:
    """Run one classifier and build the output dict (label/confidence/probs[/logits])."""
    import torch

    clf._model.eval()
    with torch.no_grad():
        logits = clf.forward(audio_tensor)
        probs = torch.softmax(logits, dim=-1)
        prob_present = probs[0, 1].item()
        prob_not_present = probs[0, 0].item()
        label = 1 if prob_present >= threshold else 0
        confidence = prob_present if label == 1 else prob_not_present

    result: Dict[str, float] = {
        "label": label,
        "confidence": confidence,
        "prob_present": prob_present,
        "prob_not_present": prob_not_present,
    }
    if include_logits:
        result["logits"] = {
            "not_present": logits[0, 0].item(),
            "present": logits[0, 1].item(),
        }
    return result


class Classifier:
    def __init__(self, class_name: Optional[str] = None):
        self.class_name = class_name
        self._model = None
        self._models: Dict[str, Any] = {}

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

            self._model = _load_classifier(self.class_name, path)
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

            self._models = {}
            for name, entry in classification.items():
                path = _resolve_path(entry)
                self._models[name] = _load_classifier(name, path)

    def predict(
        self, audio_tensor, threshold: float = 0.5
    ) -> Union[Dict[str, Tuple[int, float]], Tuple[int, float]]:
        if self._model is None and not self._models:
            self._load()

        if self.class_name is not None:
            return self._model.predict(audio_tensor)
        else:
            return self.predict_all(audio_tensor)

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
        for name, clf in self._models.items():
            results[name] = clf.predict(audio_tensor)
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
        registry = _load_registry()
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
        if self._model is None:
            self._load()
        tensor = _preprocess_audio(audio)
        thresholds = self._resolve_thresholds(threshold)
        thr = thresholds[self.class_name]
        return _classifier_output(self._model, tensor, thr, include_logits)

    def _run_all(self, audio, threshold: Optional[float], include_logits: bool) -> Dict[str, Any]:
        if not self._models:
            self._load()
        tensor = _preprocess_audio(audio)
        thresholds = self._resolve_thresholds(threshold)

        results: Dict[str, Any] = {}
        for name, clf in self._models.items():
            results[name] = _classifier_output(clf, tensor, thresholds[name], include_logits)

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
