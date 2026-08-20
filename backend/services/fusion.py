"""Fusion service: combine localizer regions with classifier saliency.

Uses model.registry raw-mode MultiTaskRunner for saliency computation,
then fuses with the provided regions via combine_regions.
"""

import numpy as np

from model.config.defaults import DYSFLUENCY_CLASSES, SAMPLE_RATE
from model.registry import combine_regions, load_audio_16k


def combine_audio_bytes(audio_bytes: bytes, regions: list) -> dict:
    """Fuse localizer regions with classifier saliency."""
    import model.registry as _reg

    try:
        audio_data, duration_sec = load_audio_16k(audio_bytes)
        clf = _reg.MultiTaskRunner()
        sal = clf.saliency(audio_data).squeeze(0)
        if hasattr(sal, "cpu"):
            sal = sal.cpu()
        sal = np.asarray(sal, dtype=float)
        return combine_regions(
            regions,
            sal,
            class_names=list(DYSFLUENCY_CLASSES),
            thresholds=getattr(clf, "_thresholds", {}) or None,
            audio_duration=duration_sec,
        )
    except Exception as exc:
        return {"error": str(exc)}
