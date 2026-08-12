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

    # Localizer (regions always; words/syllables when text is provided)
    loc = Localizer()                            # type comes from registry.json
    result = loc.analyze("recording.wav", text="the cat sat", language="en")
    # result: {regions: [...], words: [...], syllables: [...]}

    # Transcription (Whisper, word-level timestamps + stutter flagging)
    tr = Transcriber()
    result = tr.transcribe("recording.wav")

    # Everything at once — raw audio in, all results out
    m = ModelRegistry()
    all_results = m.run_all("recording.wav", text="the cat sat")
    # all_results: {classification: {...}, localization: {...}, transcription: {...}}
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from model.fingerprint import model_name_from_path
from model.transcription import Transcriber

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


def _audio_is_empty(audio) -> bool:
    """True when the input carries no samples (None or an empty ndarray).

    Mirrors Transcriber.transcribe's empty-audio guard so Classifier/Localizer
    analyze return a well-formed empty result instead of crashing on
    np.abs(audio).max() over a zero-size array.
    """
    import numpy as np

    return audio is None or (isinstance(audio, np.ndarray) and audio.size == 0)


def _empty_classifier_output(include_logits: bool) -> Dict[str, float]:
    """Well-formed 'not present' result for empty audio (no model run)."""
    result: Dict[str, float] = {
        "label": 0,
        "confidence": 0.0,
        "prob_present": 0.0,
        "prob_not_present": 1.0,
    }
    if include_logits:
        result["logits"] = {"not_present": 0.0, "present": 0.0}
    return result


def _registry_classification_names() -> List[str]:
    """Canonical classification model names from the registry (M1 filtering)."""
    from model.config.defaults import DYSFLUENCY_CLASSES

    registry = _load_registry()
    return [
        name for name in registry.get("classification", {}) if name in DYSFLUENCY_CLASSES
    ]


def _load_multitask_classifier(path: str):
    """Load a shared-backbone multitask classifier from its save format."""
    from model.classification.multitask import MultiTaskWav2VecClassifier

    return MultiTaskWav2VecClassifier.from_pretrained(path)


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


def _align_words_syllables(
    audio_array, text: str, language_code: str = "en", sr: int = 16000
):
    """Align transcript text to audio → word timestamps, then syllabify.

    Uses CTC forced alignment when available, falling back to the simple
    energy-based aligner. Returns ([word dicts], [syllable dicts]).
    """
    from model.localization.language_adapter import LanguageAdapterRegistry

    try:
        from model.localization.ctc_alignment import CTCTimeAligner

        aligner = CTCTimeAligner()
    except Exception:
        from model.localization.ctc_alignment import SimpleForcedAligner

        aligner = SimpleForcedAligner()

    try:
        word_timestamps = aligner.align(audio_array, text, sr=sr)
        word_list = [
            {
                "word": wt.word,
                "start": wt.start_sec,
                "end": wt.end_sec,
                "confidence": round(wt.confidence, 4),
            }
            for wt in word_timestamps
        ]

        registry = LanguageAdapterRegistry()
        adapter = registry.get(language_code)
        syllable_timestamps = adapter.adapt(word_timestamps, text)
        syllable_list = [
            {
                "syllable": s.syllable,
                "start": s.start_sec,
                "end": s.end_sec,
                "word": s.word,
                "index": s.syllable_index,
                "total": s.total_syllables,
            }
            for s in syllable_timestamps
        ]
        return word_list, syllable_list
    except Exception as e:
        print(f"Alignment/syllabification failed: {e}")
        return [], []


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

            from model.config.defaults import DYSFLUENCY_CLASSES

            self._models = {}
            for name, entry in classification.items():
                if name not in DYSFLUENCY_CLASSES:
                    print(
                        f"WARNING: ignoring unknown classification entry '{name}' "
                        f"in registry (expected one of {sorted(DYSFLUENCY_CLASSES)})"
                    )
                    continue
                path = _resolve_path(entry)
                self._models[name] = _load_classifier(name, path)

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
            names = list(self._models) if self._models else _registry_classification_names()
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


class MultiTaskClassifier:
    """Registry wrapper for the shared-backbone multitask classifier.

    analyze() runs ONE forward pass and returns per-class
    {label, confidence, prob_present, prob_not_present} plus a summary.
    """

    def __init__(self):
        self._model = None

    def _load(self) -> None:
        registry = _load_registry()
        entry = registry.get("classification_multitask")
        if not entry:
            raise FileNotFoundError(
                "No 'classification_multitask' entry in registry. "
                "Add model/registry.json classification_multitask.path."
            )
        path = _resolve_path(entry["path"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self._model = _load_multitask_classifier(path)

    @staticmethod
    def _empty_result(names: List[str]) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            name: _empty_classifier_output(False) for name in names
        }
        results["summary"] = {
            "detected": [],
            "primary": names[0] if names else None,
        }
        return results

    def analyze(self, audio, threshold: float = 0.5) -> Dict[str, Any]:
        """Run one forward pass and classify every class.

        Returns:
            {class_name: {label, confidence, prob_present, prob_not_present},
             summary: {detected, primary}}
        """
        import torch

        if _audio_is_empty(audio):
            names = (
                list(self._model.class_names)
                if self._model is not None
                else _registry_classification_names()
            )
            return self._empty_result(names)

        if self._model is None:
            self._load()

        tensor = _preprocess_audio(audio)
        self._model.model.eval()
        with torch.no_grad():
            logits = self._model.forward(tensor)

        results: Dict[str, Any] = {}
        for name, lg in logits.items():
            probs = torch.softmax(lg, dim=-1)
            prob_present = probs[0, 1].item()
            prob_not_present = probs[0, 0].item()
            label = 1 if prob_present >= threshold else 0
            confidence = prob_present if label == 1 else prob_not_present
            results[name] = {
                "label": label,
                "confidence": confidence,
                "prob_present": prob_present,
                "prob_not_present": prob_not_present,
            }

        detected = [name for name, r in results.items() if r["label"] == 1]
        primary = max(results.items(), key=lambda kv: kv[1]["prob_present"])[0]
        results["summary"] = {"detected": detected, "primary": primary}
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

    def analyze(
        self,
        audio,
        text: Optional[str] = None,
        language: str = "en",
        threshold: float = 0.3,
        max_length_seconds: float = 10.0,
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Analyze raw audio → dysfluency regions (+ words/syllables if text).

        Args:
            audio: File path, raw bytes, or 1-D numpy array.
            text: Optional transcript for word/syllable-level alignment.
            language: ISO language code for syllabification (en, kn, hi).
            threshold: Detection threshold for regions.
            max_length_seconds: Max audio length to process.

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

        from model.data.preprocessing import generate_mel_spectrogram, load_audio_input

        audio_array = load_audio_input(audio, sr=16000)

        types = [self.model_type] if self.model_type else list(self._models.keys())
        results: Dict[str, Any] = {}

        for lt in types:
            model = self._models[lt]
            if lt == "cnn":
                spec = generate_mel_spectrogram(audio_array, sr=16000)
                regions = model.predict(spec, sr=16000, threshold=threshold)
            elif lt == "wav2vec2":
                regions = model.predict(
                    audio_array,
                    sr=16000,
                    threshold=threshold,
                    max_length_seconds=max_length_seconds,
                )
            else:
                raise ValueError(f"Unknown localizer type: {lt}")

            entry: Dict[str, Any] = {
                "regions": [
                    {"start": round(s, 3), "end": round(e, 3), "confidence": round(c, 4)}
                    for s, e, c in regions
                ]
            }

            if text:
                words, syllables = _align_words_syllables(
                    audio_array, text, language, sr=16000
                )
                entry["words"] = words
                entry["syllables"] = syllables

            results[lt] = entry

        if self.model_type is not None:
            return results[self.model_type]
        return results

    @property
    def is_loaded(self) -> bool:
        return bool(self._models)


class ModelRegistry:
    def __init__(self):
        self.classifier = Classifier()
        self.localizer = Localizer()
        self.transcriber = Transcriber()
        self.multitask_classifier = MultiTaskClassifier()

    def run_all(
        self,
        audio,
        classify_threshold: float = 0.5,
        localize_threshold: float = 0.3,
        language: str = "english",
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run classification, localization, and transcription on raw audio.

        Args:
            audio: File path, raw bytes, or 1-D numpy array.
            classify_threshold: Label threshold for classifiers.
            localize_threshold: Detection threshold for the localizer.
            language: Whisper language name (english/kannada/hindi).
            text: Optional transcript for word/syllable-level localization.

        Returns:
            {"classification": ..., "localization": ..., "transcription": ...}
            Sub-results become {"error": ...} if a model is unavailable.
        """
        from model.transcription import WHISPER_LANG_CODES

        results = {}

        try:
            results["classification"] = self.classifier.analyze(
                audio, threshold=classify_threshold
            )
        except Exception as e:
            results["classification"] = {"error": str(e)}

        try:
            iso = WHISPER_LANG_CODES.get(language.lower(), "en")
            results["localization"] = self.localizer.analyze(
                audio,
                text=text,
                language=iso,
                threshold=localize_threshold,
            )
        except Exception as e:
            results["localization"] = {"error": str(e)}

        try:
            results["transcription"] = self.transcriber.transcribe(
                audio, language=language
            )
        except Exception as e:
            results["transcription"] = {"error": str(e)}

        return results

    @property
    def is_loaded(self) -> bool:
        return self.classifier.is_loaded and self.localizer.is_loaded
