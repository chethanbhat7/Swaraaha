"""ModelRegistry class and high-level audio pipeline functions."""

import json
from typing import Any, Dict, Optional

from model.config.defaults import DYSFLUENCY_CLASSES, FRAME_DURATION, SAMPLE_RATE

from ._classifier import ClassifierRunner
from ._localizer import LocalizerRunner
from ._multitask import CNNMultiTaskRunner, MultiTaskRunner
from ._utils import _REGISTRY_PATH, _audio_is_empty


class ModelRegistry:
    def __init__(self):
        self.classifier = ClassifierRunner()
        self.localizer = LocalizerRunner()
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

        Prefer ``model.analyze()`` for new code.
        """
        from model.transcription import Transcriber, WHISPER_LANG_CODES

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
            transcriber = Transcriber()
            results["transcription"] = transcriber.transcribe(
                audio, language=language
            )
        except Exception as e:
            results["transcription"] = {"error": str(e)}

        try:
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
                saliency = self.multitask_classifier.saliency(audio).squeeze(0)
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


# ---------------------------------------------------------------------------
# Utility functions (use model.classify / model.localize / model.fuse instead)
# ---------------------------------------------------------------------------

def load_synthesis_config() -> dict:
    """Load localization_synthesis config from registry.json."""
    try:
        with open(_REGISTRY_PATH) as f:
            return json.load(f).get("localization_synthesis", {})
    except Exception:
        return {}


def load_audio_16k(audio_bytes: bytes):
    """Load audio bytes -> (16kHz mono float32 array, duration_sec)."""
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
    """Extract contiguous high-saliency spans per class into regions."""
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
    """Classify audio from raw bytes. Prefer ``model.classify()`` for new code."""
    from model import classify
    return classify(audio_bytes)


def localize_audio_bytes(audio_bytes: bytes) -> dict:
    """Localize audio from raw bytes. Prefer ``model.localize()`` for new code."""
    from model import localize
    return localize(audio_bytes)


def combine_with_saliency(audio_bytes: bytes, regions: list) -> dict:
    """Fuse regions with saliency. Prefer ``model.fuse()`` for new code."""
    from model import fuse
    return fuse(audio_bytes, regions)
