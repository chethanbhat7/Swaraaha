"""Private helper functions for the model registry subsystem."""

import json
import os
from typing import Dict, List

from model.config.defaults import AUDIO_DURATION_SECONDS, DYSFLUENCY_CLASSES, SAMPLE_RATE
from model.fingerprint import parse_fingerprint_from_path

_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "registry.json")


def _load_cnn_localizer(path: str):
    from model.localization.cnn_spectrogram import CNNSpectrogramLocalizer
    return CNNSpectrogramLocalizer.from_pretrained(path)


def _load_wav2vec2_localizer(path: str):
    from model.localization.wav2vec2_localizer import Wav2Vec2Localizer
    return Wav2Vec2Localizer.from_pretrained(path)


_LOCALIZER_LOADERS = {
    "cnn": _load_cnn_localizer,
    "wav2vec2": _load_wav2vec2_localizer,
}


def _predict_cnn(model, audio_array, threshold, max_length_seconds):
    from model.data.preprocessing import generate_mel_spectrogram
    spec = generate_mel_spectrogram(audio_array, sr=SAMPLE_RATE)
    return model.predict(spec, sr=SAMPLE_RATE, threshold=threshold)


def _predict_wav2vec2(model, audio_array, threshold, max_length_seconds):
    return model.predict(
        audio_array,
        sr=SAMPLE_RATE,
        threshold=threshold,
        max_length_seconds=max_length_seconds,
    )


_LOCALIZER_PREDICTORS = {
    "cnn": _predict_cnn,
    "wav2vec2": _predict_wav2vec2,
}


def _max_length_from_path(path: str) -> float:
    """Extract max_length_seconds from a fingerprint-encoded checkpoint path.

    Falls back to ``AUDIO_DURATION_SECONDS`` when the path has no fingerprint
    or parsing fails.
    """
    try:
        params = parse_fingerprint_from_path(path)
        return float(params.get("max_length_seconds", AUDIO_DURATION_SECONDS))
    except (ValueError, KeyError):
        return AUDIO_DURATION_SECONDS


def _chunk_audio(audio_array, chunk_sec: float, sr: int = SAMPLE_RATE, overlap_sec: float = 0.5):
    """Yield (chunk, start_sample) pairs for sliding-window processing.

    Each chunk is ``chunk_sec`` seconds long with ``overlap_sec`` overlap.
    The last chunk may be shorter.
    """
    import numpy as np

    n_samples = len(audio_array)
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    stride = max(chunk_samples - overlap_samples, 1)

    start = 0
    while start < n_samples:
        end = min(start + chunk_samples, n_samples)
        yield audio_array[start:end], start
        if end >= n_samples:
            break
        start += stride


def _load_registry():
    import model.registry as _reg
    if not os.path.exists(_reg._REGISTRY_PATH):
        raise FileNotFoundError(
            f"Registry file not found: {_reg._REGISTRY_PATH}\n"
            "Create model/registry.json with available model paths."
        )
    with open(_reg._REGISTRY_PATH) as f:
        return json.load(f)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(project_root, path)


def _load_classifier(class_name: str, path: str):
    """Load a classifier from a checkpoint, handling training and classifier formats."""
    from model.evaluation.loader import load_classifier

    instance = load_classifier(class_name, path)
    instance.max_length_seconds = _max_length_from_path(path)
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
    registry = _load_registry()
    return [
        name for name in registry.get("classification", {}) if name in DYSFLUENCY_CLASSES
    ]


def _load_multitask_classifier(path: str):
    """Load a shared-backbone multitask classifier from either save format.

    Handles the model's own save format (``model_name`` key) and the training
    format produced by ``save_checkpoint`` (``model_state_dict``, no
    ``model_name``); see ``model.evaluation.loader.load_multitask``.
    """
    from model.evaluation.loader import load_multitask

    model = load_multitask(path)
    model.max_length_seconds = _max_length_from_path(path)
    return model


def _preprocess_audio(
    audio, sr: int = SAMPLE_RATE, max_length_seconds: float = AUDIO_DURATION_SECONDS
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
    audio_array, text: str, language_code: str = "en", sr: int = SAMPLE_RATE
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


def _resolve_multitask_thresholds(entry, model_path):
    import model.registry as _reg
    if entry.get('thresholds'):
        return {name: float(t) for name, t in entry['thresholds'].items()}
    thresholds_path = entry.get('thresholds_path')
    if thresholds_path:
        thresholds_path = _reg._resolve_path(thresholds_path)
        if not os.path.exists(thresholds_path):
            raise FileNotFoundError(f'Thresholds file not found: {thresholds_path}')
    else:
        candidate = os.path.join(os.path.dirname(model_path),
                                 'multitask_thresholds.json')
        thresholds_path = candidate if os.path.exists(candidate) else None
    if not thresholds_path:
        return {}
    with open(thresholds_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {name: spec['f1_threshold']
            for name, spec in data.get('thresholds', {}).items()}


def _load_multitask_registry_entry(registry, entry_key):
    import model.registry as _reg
    entry = registry.get(entry_key)
    if not entry:
        raise FileNotFoundError(
            f"No '{entry_key}' entry in registry. "
            f"Add model/registry.json {entry_key}.path."
        )
    path = _reg._resolve_path(entry['path'])
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found: {path}')
    model = _reg._load_multitask_classifier(path)
    thresholds = _reg._resolve_multitask_thresholds(entry, path)
    return model, thresholds
