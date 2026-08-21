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
            loc_result = _localizer.analyze(audio, **loc_kwargs)

            results["localization"] = loc_result

            # --- combined (fusion) ---
            try:
                loc = results["localization"]
                if isinstance(loc, dict) and "error" not in loc:
                    regions = loc.get("regions", [])
                    syllables = loc.get("syllables")
                else:
                    regions, syllables = [], None
                _fuse_combined(results, audio, regions, syllables)
                # Backfill localization if localizer was empty but fusion generated regions.
                if not regions:
                    combined = results.get("combined", {})
                    if isinstance(combined, dict) and "regions" in combined:
                        results["localization"] = {
                            "regions": combined["regions"],
                            "source": "saliency",
                        }
            except Exception as exc:
                results["combined"] = {"error": str(exc)}
        except Exception as exc:
            results["localization"] = {"error": str(exc)}
            results["combined"] = {"error": str(exc)}
    else:
        # No localizer — use saliency fallback via combined fusion.
        results["localization"] = {"regions": [], "source": "none"}
        try:
            _fuse_combined(results, audio, [], None)
            # Backfill localization with the regions generated by fusion.
            combined = results.get("combined", {})
            if isinstance(combined, dict) and "regions" in combined:
                results["localization"] = {
                    "regions": combined["regions"],
                    "source": "saliency",
                }
        except Exception as exc:
            results["combined"] = {"error": str(exc)}

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


def transcribe(
    audio: Any,
    *,
    language: str = "english",
    localizations: Optional[list] = None,
    passage_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Run transcription only (Whisper ASR with word-level timestamps).

    Falls back to a forced-alignment heuristic when Whisper produces no output.

    Args:
        audio: File path, raw bytes, or 1-D numpy array.
        language: Whisper language name (english / kannada / hindi).
        localizations: Optional list of (start, end, confidence) stutter regions
            to overlay onto word timestamps.
        passage_text: Optional reference passage for the fallback aligner.

    Returns::

        {text, words: [{word, start_sec, end_sec, confidence, stutter,
                         stutter_type}], duration_sec}
    """
    _ensure_init()
    if _transcriber is None:
        return {"error": "no transcriber loaded"}
    try:
        return _transcriber.transcribe(
            audio, language=language,
            localizations=localizations, passage_text=passage_text,
        )
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

    Falls back to classifier saliency when the localizer is unavailable.

    Args:
        audio: File path, raw bytes, or 1-D numpy array.
        language: Whisper language name (english / kannada / hindi).
        text: Optional transcript for word/syllable-level alignment.
        localize_threshold: Detection threshold for the localizer.

    Returns::

        {regions: [...], [words], [syllables], [source?]}
    """
    _ensure_init()

    if _localizer is not None:
        try:
            loc_kwargs: Dict[str, Any] = {"threshold": localize_threshold}
            if text is not None:
                loc_kwargs["text"] = text
                loc_kwargs["language"] = _whisper_iso(language)
            result = _localizer.analyze(audio, **loc_kwargs)
            # If the localizer found regions, return them directly.
            if result.get("regions"):
                return result
            # Otherwise fall through to saliency fallback.
        except Exception:
            pass

    # Fallback: saliency-based localization via classifier
    return _saliency_localize(audio)


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
    try:
        results: Dict[str, Any] = {}
        _fuse_combined(results, audio, regions, syllables)
        return results.get("combined", {"error": "fusion failed"})
    except Exception as exc:
        return {"error": str(exc)}


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


def _saliency_localize(audio: Any) -> Dict[str, Any]:
    """Saliency-based localization fallback using the classifier."""
    if _classifier is None or not hasattr(_classifier, "saliency"):
        return {"error": "no classifier available for saliency fallback"}
    try:
        from model.config.defaults import DYSFLUENCY_CLASSES, SAMPLE_RATE
        from model.data.preprocessing import load_audio_input
        from model.registry._pipelines import saliency_regions

        audio_array = load_audio_input(audio, sr=SAMPLE_RATE)
        duration_sec = len(audio_array) / SAMPLE_RATE
        sal = _classifier.saliency(audio_array)
        if hasattr(sal, "cpu"):
            sal = sal.cpu()
        import numpy as _np
        sal = _np.asarray(sal, dtype=float)
        sal = sal.squeeze(0) if sal.ndim == 3 else sal
        regions = saliency_regions(sal, list(DYSFLUENCY_CLASSES), duration_sec)
        return {"regions": regions, "duration_sec": duration_sec, "source": "saliency"}
    except Exception as exc:
        return {"error": str(exc)}


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

    from model.data.preprocessing import load_audio_input
    from model.registry._pipelines import saliency_regions

    audio_array = load_audio_input(audio, sr=SAMPLE_RATE)
    audio_duration = len(audio_array) / SAMPLE_RATE

    if not hasattr(_classifier, "saliency"):
        results["combined"] = {"error": "classifier does not support saliency"}
        return

    try:
        sal = _classifier.saliency(audio_array)
    except Exception as exc:
        results["combined"] = {"error": str(exc)}
        return
    if hasattr(sal, "cpu"):
        sal = sal.cpu()
    import numpy as _np
    sal = _np.asarray(sal, dtype=float)
    sal_2d = sal.squeeze(0) if sal.ndim == 3 else sal

    if not regions:
        regions = saliency_regions(sal_2d, list(DYSFLUENCY_CLASSES), audio_duration)

    from model.combiner import combine_regions
    results["combined"] = combine_regions(
        regions,
        sal_2d,
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
    "transcribe",
    "localize",
    "fuse",
    "status",
]

# Shared audio format support — used by both web and desktop frontends.
# The model layer handles all formats via ffmpeg (convert_to_wav), so
# this list controls what the file pickers show, not what can be processed.
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma")
SUPPORTED_AUDIO_GLOBS = [f"*{ext}" for ext in SUPPORTED_AUDIO_EXTENSIONS]
SUPPORTED_AUDIO_DESCRIPTION = "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg *.wma)"
