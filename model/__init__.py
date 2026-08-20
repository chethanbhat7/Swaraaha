"""
Swaraaha ML Models — public API.

Easy mode (recommended):
    from model import init, analyze

    init()                    # pre-load best models at boot (optional)
    results = analyze(audio)  # classify + localize + transcribe + combine

Raw mode (full control):
    from model.registry import MultiTaskRunner, LocalizerRunner, ClassifierRunner

    clf = MultiTaskRunner()
    clf.analyze(audio)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_classifier = None
_localizer = None
_transcriber = None

_init_done = False
_init_classifier: Optional[str] = None
_init_localizer: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init(
    classifier: str = "multitask",
    localizer: str = "wav2vec2",
) -> None:
    """Pre-load models into memory.

    Call once at server boot. Subsequent ``analyze()`` calls reuse the
    loaded models with zero lazy-load overhead.

    Args:
        classifier: Classifier to load. Options: ``"multitask"`` (best),
            ``"cnn_multitask"``, ``"individual"`` (5 separate classifiers).
            Pass ``None`` to skip classification.
        localizer: Localizer to load. Options: ``"wav2vec2"`` (best),
            ``"cnn"``.
            Pass ``None`` to skip localization.

    If not called, ``analyze()`` lazy-loads the defaults on first use.
    """
    global _init_done, _init_classifier, _init_localizer

    _init_classifier = classifier
    _init_localizer = localizer

    if classifier is not None:
        _load_classifier(classifier)
    if localizer is not None:
        _load_localizer(localizer)

    _load_transcriber()
    _init_done = True


def analyze(
    audio: Any,
    *,
    language: str = "english",
    text: Optional[str] = None,
    classify_threshold: Optional[float] = None,
    localize_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Run classification + localization + transcription on audio.

    If ``init()`` was called, reuses pre-loaded models.
    Otherwise lazy-loads defaults (multitask classifier + wav2vec2 localizer).

    Args:
        audio: File path, raw bytes, or 1-D numpy array.
        language: Whisper language name (english / kannada / hindi).
        text: Optional transcript for word/syllable-level alignment.
        classify_threshold: Override per-class classification thresholds.
        localize_threshold: Detection threshold for the localizer.

    Returns::

        {
            "classification": {class_name: {label, confidence, ...}, summary: {...}},
            "localization":   {regions: [...], [words], [syllables]},
            "transcription":  {text, segments, ...},
            "combined":       {regions: [...], audio_duration, total_stutters},
        }

    Individual keys become ``{"error": "..."}`` if that pipeline fails.
    """
    _ensure_init()

    results: Dict[str, Any] = {}

    # --- classification ---
    if _classifier is not None:
        try:
            kwargs: Dict[str, Any] = {}
            if classify_threshold is not None:
                kwargs["threshold"] = classify_threshold
            results["classification"] = _classifier.analyze(audio, **kwargs)
        except Exception as exc:
            results["classification"] = {"error": str(exc)}
    else:
        results["classification"] = {"error": "no classifier loaded"}

    # --- localization ---
    if _localizer is not None:
        try:
            from model.config.defaults import SAMPLE_RATE
            from model.data.preprocessing import load_audio_input

            loc_kwargs: Dict[str, Any] = {
                "threshold": localize_threshold,
            }
            if text is not None:
                loc_kwargs["text"] = text
                loc_kwargs["language"] = _whisper_iso(language)
            results["localization"] = _localizer.analyze(audio, **loc_kwargs)

            # --- combined (fusion) ---
            try:
                loc = results["localization"]
                if isinstance(loc, dict) and "error" not in loc:
                    regions = loc.get("regions", [])
                    syllables = loc.get("syllables")
                else:
                    regions, syllables = [], None
                _fuse_combined(results, audio, regions, syllables)
            except Exception as exc:
                results["combined"] = {"error": str(exc)}
        except Exception as exc:
            results["localization"] = {"error": str(exc)}
            results["combined"] = {"error": str(exc)}
    else:
        results["localization"] = {"error": "no localizer loaded"}
        results["combined"] = {"error": "no localizer loaded"}

    # --- transcription ---
    if _transcriber is not None:
        try:
            results["transcription"] = _transcriber.transcribe(
                audio, language=language
            )
        except Exception as exc:
            results["transcription"] = {"error": str(exc)}
    else:
        results["transcription"] = {"error": "no transcriber loaded"}

    return results


def classify(
    audio: Any,
    *,
    classify_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run classification only.

    Args:
        audio: File path, raw bytes, or 1-D numpy array.
        classify_threshold: Override per-class classification thresholds.

    Returns::

        {class_name: {label, confidence, ...}, summary: {...}}
    """
    _ensure_init()
    if _classifier is None:
        return {"error": "no classifier loaded"}
    try:
        kwargs: Dict[str, Any] = {}
        if classify_threshold is not None:
            kwargs["threshold"] = classify_threshold
        return _classifier.analyze(audio, **kwargs)
    except Exception as exc:
        return {"error": str(exc)}


def localize(
    audio: Any,
    *,
    language: str = "english",
    text: Optional[str] = None,
    localize_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Run localization only (dysfluency region detection).

    Args:
        audio: File path, raw bytes, or 1-D numpy array.
        language: Whisper language name (english / kannada / hindi).
        text: Optional transcript for word/syllable-level alignment.
        localize_threshold: Detection threshold for the localizer.

    Returns::

        {regions: [...], [words], [syllables]}
    """
    _ensure_init()
    if _localizer is None:
        return {"error": "no localizer loaded"}
    try:
        loc_kwargs: Dict[str, Any] = {"threshold": localize_threshold}
        if text is not None:
            loc_kwargs["text"] = text
            loc_kwargs["language"] = _whisper_iso(language)
        return _localizer.analyze(audio, **loc_kwargs)
    except Exception as exc:
        return {"error": str(exc)}


def fuse(
    audio: Any,
    regions: list,
    *,
    syllables: Optional[list] = None,
) -> Dict[str, Any]:
    """Fuse localizer regions with classifier saliency into combined output.

    Args:
        audio: File path, raw bytes, or 1-D numpy array.
        regions: Localizer regions (list of dicts with start/end/confidence).
        syllables: Optional syllable-level data from localizer.

    Returns::

        {regions: [...], audio_duration, total_stutters}
    """
    _ensure_init()
    if _classifier is None:
        return {"error": "no classifier loaded"}
    results: Dict[str, Any] = {}
    _fuse_combined(results, audio, regions, syllables)
    return results.get("combined", {"error": "fusion failed"})


def status() -> Dict[str, bool]:
    """Check which models are loaded."""
    _ensure_init()
    return {
        "classifier": _classifier is not None and _classifier.is_loaded,
        "localizer": _localizer is not None and _localizer.is_loaded,
        "transcriber": _transcriber is not None,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_init() -> None:
    """Lazy-load defaults if init() was never called."""
    if _init_done:
        return
    init()  # uses defaults: multitask + wav2vec2


def _load_classifier(kind: str) -> None:
    global _classifier
    from model.registry._multitask import CNNMultiTaskRunner, MultiTaskRunner
    from model.registry._classifier import ClassifierRunner

    if kind == "multitask":
        _classifier = MultiTaskRunner()
    elif kind == "cnn_multitask":
        _classifier = CNNMultiTaskRunner()
    elif kind == "individual":
        _classifier = ClassifierRunner()
    else:
        raise ValueError(
            f"Unknown classifier type: {kind!r}. "
            f"Options: 'multitask', 'cnn_multitask', 'individual'"
        )


def _load_localizer(kind: str) -> None:
    global _localizer
    from model.registry._localizer import LocalizerRunner

    if kind in ("wav2vec2", "cnn"):
        _localizer = LocalizerRunner(kind)
    else:
        raise ValueError(
            f"Unknown localizer type: {kind!r}. Options: 'wav2vec2', 'cnn'"
        )


def _load_transcriber() -> None:
    global _transcriber
    from model.transcription import Transcriber
    _transcriber = Transcriber()


def _fuse_combined(
    results: Dict[str, Any],
    audio: Any,
    regions: list,
    syllables: Optional[list],
) -> None:
    """Fuse localizer regions with classifier saliency into combined output."""
    from model.config.defaults import DYSFLUENCY_CLASSES, SAMPLE_RATE

    if _audio_is_empty(audio):
        results["combined"] = {
            "regions": [], "audio_duration": 0.0, "total_stutters": 0,
        }
        return

    if not regions:
        results["combined"] = {
            "regions": [], "audio_duration": 0.0, "total_stutters": 0,
        }
        return

    from model.data.preprocessing import load_audio_input

    audio_array = load_audio_input(audio, sr=SAMPLE_RATE)
    audio_duration = len(audio_array) / SAMPLE_RATE

    if not hasattr(_classifier, "saliency"):
        results["combined"] = {"error": "classifier does not support saliency"}
        return

    sal = _classifier.saliency(audio_array)
    if hasattr(sal, "cpu"):
        sal = sal.cpu()
    import numpy as _np
    sal = _np.asarray(sal, dtype=float)

    from model.combiner import combine_regions
    results["combined"] = combine_regions(
        regions,
        sal.squeeze(0) if sal.ndim == 3 else sal,
        class_names=list(DYSFLUENCY_CLASSES),
        thresholds=getattr(_classifier, "_thresholds", {}) or None,
        syllables=syllables,
        audio_duration=audio_duration,
    )


def _audio_is_empty(audio: Any) -> bool:
    """Check if audio is None or empty."""
    if audio is None:
        return True
    try:
        import numpy as _np
        if isinstance(audio, _np.ndarray) and audio.size == 0:
            return True
    except ImportError:
        pass
    return False


def _whisper_iso(language: str) -> str:
    """Convert language name to ISO code for Whisper."""
    from model.transcription import WHISPER_LANG_CODES
    return WHISPER_LANG_CODES.get(language.lower(), "en")


__all__ = [
    "init",
    "analyze",
    "classify",
    "localize",
    "fuse",
    "status",
]
