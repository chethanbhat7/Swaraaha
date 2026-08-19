"""Combine localizer regions with classifier per-frame saliency.

Pure data-fusion unit shared by the API (ModelRegistry.run_all) and the
evaluation harness. No model internals are coupled in here.

Frame alignment contract: wav2vec2 subsamples 320 samples/frame @ 16 kHz,
so frame_duration = 320 / 16000 = 0.02 s.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from model.config.defaults import DYSFLUENCY_CLASSES

FRAME_DURATION = 320 / 16000


def _frames_for_region(region: Dict[str, Any], frame_duration: float, n_frames: int):
    start_f = int(region["start"] / frame_duration)
    end_f = int(region["end"] / frame_duration)
    start_f = max(0, min(start_f, n_frames))
    end_f = max(0, min(end_f, n_frames))
    return start_f, end_f


def _syllable_snap(region: Dict[str, Any], syllables: List[Dict[str, Any]]) -> tuple:
    """Snap region boundaries to enclosing syllable edges.

    Returns (start, end, attached_syllables). When no syllable overlaps the
    region, boundaries are unchanged and no syllables are attached.
    """
    start, end = region["start"], region["end"]
    overlapping = [
        s for s in syllables
        if s["start"] < end and s["end"] > start
    ]
    if not overlapping:
        return start, end, []
    overlapping = sorted(overlapping, key=lambda s: s["start"])
    new_start = overlapping[0]["start"]
    new_end = max(s["end"] for s in overlapping)
    return new_start, new_end, overlapping


def combine_regions(
    regions: List[Dict[str, Any]],
    saliency,
    class_names: Optional[List[str]] = None,
    frame_duration: float = FRAME_DURATION,
    thresholds: Optional[Dict[str, float]] = None,
    syllables: Optional[List[Dict[str, Any]]] = None,
    audio_duration: float = 0.0,
    default_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Fuse type-agnostic localizer regions with per-frame class saliency.

    Args:
        regions: List of {start, end, confidence} from the localizer.
        saliency: (T, num_classes) array (or convertible) of per-frame
            per-class prob_present in [0, 1], ordered by ``class_names``.
        class_names: Class order matching ``saliency``'s last axis.
            Default: DYSFLUENCY_CLASSES.
        frame_duration: Seconds per saliency frame (default 0.02).
        thresholds: Per-class label thresholds; missing classes fall back to
            ``default_threshold``.
        syllables: Optional list of {syllable, start, end} for snapping.
        audio_duration: Total audio length in seconds (bounds regions).
        default_threshold: Threshold for classes without an entry in
            ``thresholds``.

    Returns:
        {regions: [{start, end, confidence, classes, primary_type, severity,
                    syllables}], audio_duration, total_stutters}
    """
    if class_names is None:
        class_names = list(DYSFLUENCY_CLASSES)
    saliency = np.asarray(saliency, dtype=float)
    n_frames = saliency.shape[0] if saliency.ndim == 2 else 0
    thresholds = thresholds or {}
    fused = []
    total_stutters = 0

    for region in regions:
        entry = dict(region)
        start_f, end_f = _frames_for_region(region, frame_duration, n_frames)
        classes: Dict[str, Any] = {}
        primary_type = None
        if n_frames > 0 and end_f > start_f:
            window = saliency[start_f:end_f]
            best_prob_detected = -1.0
            best_prob_any = -1.0
            primary_type_detected = None
            primary_type_any = None

            for i, name in enumerate(class_names):
                if i >= window.shape[1]:
                    continue
                prob_present = float(window[:, i].mean())
                prob_present = max(0.0, min(1.0, prob_present))
                prob_not_present = 1.0 - prob_present
                thr = thresholds.get(name, default_threshold)
                label = 1 if prob_present >= thr else 0
                confidence = prob_present if label == 1 else prob_not_present
                classes[name] = {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "prob_present": round(prob_present, 4),
                    "prob_not_present": round(prob_not_present, 4),
                }
                if label == 1 and prob_present > best_prob_detected:
                    best_prob_detected = prob_present
                    primary_type_detected = name
                if prob_present > best_prob_any:
                    best_prob_any = prob_present
                    primary_type_any = name

            primary_type = primary_type_detected if primary_type_detected is not None else primary_type_any

        entry["classes"] = classes
        entry["primary_type"] = primary_type
        entry["severity"] = None

        if syllables:
            snapped_start, snapped_end, attached = _syllable_snap(entry, syllables)
            entry["start"] = round(snapped_start, 3)
            entry["end"] = round(snapped_end, 3)
            entry["syllables"] = [
                {"syllable": s["syllable"], "start": round(s["start"], 3),
                 "end": round(s["end"], 3)}
                for s in attached
            ]
        else:
            entry["start"] = round(min(entry["start"], audio_duration), 3)
            entry["end"] = round(min(entry["end"], audio_duration), 3)
            entry["syllables"] = []

        if entry["start"] >= entry["end"]:
            continue

        if any(c["label"] == 1 for c in classes.values()):
            total_stutters += 1

        fused.append(entry)

    fused.sort(key=lambda r: r["start"])
    return {
        "regions": fused,
        "audio_duration": round(audio_duration, 3),
        "total_stutters": total_stutters,
    }


def mismatch_rate(
    regions: List[Dict[str, Any]],
    saliency,
    class_names: Optional[List[str]] = None,
    frame_duration: float = FRAME_DURATION,
    threshold: float = 0.5,
) -> float:
    """Fraction of high-saliency spans with no overlapping localizer region.

    Used by the eval probe to decide whether saliency synthesis is worth
    building later (spec: 'no synthesis in v1').
    """
    if class_names is None:
        class_names = list(DYSFLUENCY_CLASSES)
    saliency = np.asarray(saliency, dtype=float)
    if saliency.ndim != 2 or saliency.shape[0] == 0:
        return 0.0
    high = (saliency >= threshold).any(axis=1)  # (T,)
    spans = []
    in_span = False
    start = 0
    for i, flag in enumerate(high):
        if flag and not in_span:
            in_span = True
            start = i
        elif not flag and in_span:
            in_span = False
            spans.append((start, i))
    if in_span:
        spans.append((start, len(high)))

    if not spans:
        return 0.0

    uncovered = 0
    for span_start, span_end in spans:
        t_start = span_start * frame_duration
        t_end = span_end * frame_duration
        matched = any(
            r["start"] < t_end and r["end"] > t_start for r in regions
        )
        if not matched:
            uncovered += 1
    return uncovered / len(spans)