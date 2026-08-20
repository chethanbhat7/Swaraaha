"""ModelRegistry class and high-level audio pipeline functions."""

import json
import os
from typing import Any, Dict, Optional

from model.config.defaults import DYSFLUENCY_CLASSES, FRAME_DURATION, SAMPLE_RATE
from model.transcription import Transcriber

from ._classifier import ClassifierRunner
from ._localizer import LocalizerRunner
from ._multitask import CNNMultiTaskRunner, MultiTaskRunner
from ._utils import _REGISTRY_PATH, _audio_is_empty


class ModelRegistry:
    def __init__(self):
        self.classifier = ClassifierRunner()
        self.localizer = LocalizerRunner()
        self.transcriber = Transcriber()
        self.multitask_classifier = MultiTaskRunner()
        self.cnn_multitask_classifier = CNNMultiTaskRunner()

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
            {"classification": ..., "localization": ..., "transcription": ...,
             "multitask": ..., "cnn_multitask": ..., "combined": ...}
            "combined" fuses localizer regions with multitask saliency:
            {regions: [...], audio_duration, total_stutters}. Sub-results
            become {"error": ...} if a model is unavailable.
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
            results['cnn_multitask'] = self.cnn_multitask_classifier.analyze(
                audio, threshold=classify_threshold,
            )
        except Exception as e: # noqa: BLE001
            results['cnn_multitask'] = {'error' : str(e)}

        try:
            results['multitask'] = self.multitask_classifier.analyze(
                audio, threshold=classify_threshold,
            )
        except Exception as e: # noqa: BLE001
            results['multitask'] = {'error' : str(e)}

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

        try:
            from model.config.defaults import DYSFLUENCY_CLASSES
            from model.data.preprocessing import load_audio_input
            from model.combiner import combine_regions

            if _audio_is_empty(audio):
                results["combined"] = {
                    "regions": [], "audio_duration": 0.0, "total_stutters": 0,
                }
            else:
                loc = results.get("localization")
                if isinstance(loc, dict) and "error" not in loc:
                    regions = loc.get("regions", [])
                    syllables = loc.get("syllables") if isinstance(loc, dict) else None
                else:
                    regions, syllables = [], None
                saliency = self.multitask_classifier.saliency(audio).squeeze(0)  # (T, C)
                audio_array = load_audio_input(audio, sr=SAMPLE_RATE)
                audio_duration = len(audio_array) / SAMPLE_RATE
                results["combined"] = combine_regions(
                    regions,
                    saliency.cpu().numpy(),
                    class_names=list(DYSFLUENCY_CLASSES),
                    thresholds=getattr(self.multitask_classifier, "_thresholds", {}) or None,
                    syllables=syllables,
                    audio_duration=audio_duration,
                )
        except Exception as e:  # noqa: BLE001
            results["combined"] = {"error": str(e)}

        return results

    @property
    def is_loaded(self) -> bool:
        return self.classifier.is_loaded and self.localizer.is_loaded


def load_synthesis_config() -> dict:
    """Load localization_synthesis config from registry.json."""
    try:
        with open(_REGISTRY_PATH) as f:
            return json.load(f).get("localization_synthesis", {})
    except Exception:
        return {}


def load_audio_16k(audio_bytes: bytes):
    """Load audio bytes → (16kHz mono float32 array, duration_sec).

    Handles format conversion, resampling, and mono downmix in one call.
    """
    import io as _io

    import librosa
    import numpy as _np
    import soundfile as _sf

    from model.data.preprocessing import convert_to_wav

    wav_bytes = convert_to_wav(audio_bytes)
    audio_data, sr = _sf.read(_io.BytesIO(wav_bytes))
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio_data = _np.asarray(audio_data, dtype=_np.float32)
    duration_sec = round(len(audio_data) / SAMPLE_RATE, 3)
    return audio_data, duration_sec


def saliency_regions(saliency, class_names, duration_sec: float) -> list:
    """Extract contiguous high-saliency spans per class into regions.

    Adaptive thresholding per class: ``mean + adapt_k * std``, floored
    and capped. Spans shorter than ``min_span_sec`` are dropped.
    Overlapping regions are deduplicated by confidence.
    """
    import numpy as _np

    cfg = load_synthesis_config()
    min_span_frames = max(1, int(cfg.get("min_span_sec", 0.16) / FRAME_DURATION))
    adapt_k = float(cfg.get("adapt_k", 2.0))
    floor = float(cfg.get("floor", 0.6))
    max_threshold = float(cfg.get("max_threshold", 0.95))

    saliency = _np.asarray(saliency, dtype=float)
    regions = []
    n_frames = saliency.shape[0] if saliency.ndim == 2 else 0
    if n_frames == 0:
        return regions

    for c, name in enumerate(class_names):
        if c >= saliency.shape[1]:
            continue
        col = saliency[:, c]
        threshold = min(max_threshold, max(floor, col.mean() + adapt_k * col.std()))
        on = col >= threshold

        start = None
        for t in range(n_frames + 1):
            if t < n_frames and on[t] and start is None:
                start = t
            elif (t == n_frames or not on[t]) and start is not None:
                end = t
                if end - start >= min_span_frames:
                    seg = col[start:end]
                    regions.append({
                        "start": round(min(start * FRAME_DURATION, duration_sec), 3),
                        "end": round(min(end * FRAME_DURATION, duration_sec), 3),
                        "confidence": round(float(seg.max()), 4),
                        "type": name,
                    })
                start = None

    regions.sort(key=lambda r: r["start"])

    # Drop lower-confidence regions that overlap an already-accepted one
    regions.sort(key=lambda r: r["confidence"], reverse=True)
    kept = []
    for region in regions:
        overlaps = any(
            region["start"] < k["end"] and region["end"] > k["start"] for k in kept
        )
        if not overlaps:
            kept.append(region)
    kept.sort(key=lambda r: r["start"])
    return kept


def classify_audio_bytes(audio_bytes: bytes) -> dict:
    """Full classification pipeline: raw bytes → per-class results.

    Returns ``{class_name: {label, confidence, prob_present, prob_not_present},
    summary: {detected, primary}}``.
    """
    import model.registry as _reg

    audio_data, _duration = load_audio_16k(audio_bytes)
    clf = _reg.MultiTaskClassifier()
    return clf.analyze(audio_data)


def localize_audio_bytes(audio_bytes: bytes) -> dict:
    """Full localization pipeline: raw bytes → dysfluency regions.

    Tries the dedicated localizer first; falls back to multitask saliency
    when localizer weights are unavailable.

    Returns ``{regions: [...], duration_sec, source?}``.
    """
    import numpy as _np

    import model.registry as _reg
    from model.data.preprocessing import generate_mel_spectrogram

    audio_data, duration_sec = load_audio_16k(audio_bytes)
    spec = generate_mel_spectrogram(audio_data, sr=SAMPLE_RATE)

    try:
        loc = _reg.LocalizerRunner("wav2vec2")
        regions = loc.predict(spec, sr=SAMPLE_RATE)
        return {
            "regions": [
                {"start": round(s, 3), "end": round(e, 3), "confidence": round(c, 4)}
                for s, e, c in regions
            ],
            "duration_sec": duration_sec,
        }
    except Exception:
        pass

    # Dedicated localizer unavailable → saliency fallback
    try:
        mt = _reg.MultiTaskRunner()
        sal = mt.saliency(audio_data).squeeze(0)
        if hasattr(sal, "cpu"):
            sal = sal.cpu()
        sal = _np.asarray(sal, dtype=float)
        regions = saliency_regions(sal, list(DYSFLUENCY_CLASSES), duration_sec)
        return {
            "regions": regions,
            "duration_sec": duration_sec,
            "source": "saliency",
        }
    except Exception as exc:
        return {"regions": [], "error": str(exc), "duration_sec": duration_sec}


def combine_with_saliency(audio_bytes: bytes, regions: list) -> dict:
    """Full fusion pipeline: raw bytes + localizer regions → combined results.

    Computes multitask saliency and fuses it with the given regions via
    ``combine_regions``.

    Returns ``{regions: [...], audio_duration, total_stutters}`` or
    ``{"error": str}``.
    """
    import numpy as _np

    import model.registry as _reg
    from model.combiner import combine_regions

    try:
        audio_data, duration_sec = load_audio_16k(audio_bytes)
        clf = _reg.MultiTaskRunner()
        sal = clf.saliency(audio_data).squeeze(0)
        if hasattr(sal, "cpu"):
            sal = sal.cpu()
        sal = _np.asarray(sal, dtype=float)
        return combine_regions(
            regions,
            sal,
            class_names=list(DYSFLUENCY_CLASSES),
            thresholds=getattr(clf, "_thresholds", {}) or None,
            audio_duration=duration_sec,
        )
    except Exception as exc:
        return {"error": str(exc)}
