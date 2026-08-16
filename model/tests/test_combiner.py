import numpy as np
import pytest

from model.combiner import combine_regions, mismatch_rate

CLASS_NAMES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]
FRAME_DURATION = 320 / 16000  # 0.02 s


def test_combine_region_gets_class_scores_and_labels():
    saliency = np.zeros((100, 5))
    saliency[:, 1] = 0.9  # block active in every frame
    regions = [{"start": 0.5, "end": 1.0, "confidence": 0.8}]
    out = combine_regions(regions, saliency, class_names=CLASS_NAMES,
                          frame_duration=FRAME_DURATION, audio_duration=2.0)
    r = out["regions"][0]
    assert r["primary_type"] == "block"
    assert r["classes"]["block"]["label"] == 1
    assert r["classes"]["block"]["prob_present"] == pytest.approx(0.9, abs=1e-6)
    assert r["classes"]["block"]["prob_not_present"] == pytest.approx(0.1, abs=1e-6)
    assert r["classes"]["prolongation"]["label"] == 0
    assert r["severity"] is None


def test_combine_honors_per_class_thresholds():
    saliency = np.zeros((100, 5))
    saliency[:, 0] = 0.6  # prolongation prob 0.6
    regions = [{"start": 0.0, "end": 0.5, "confidence": 0.7}]
    thresholds = {"prolongation": 0.7, "block": 0.5}
    out = combine_regions(regions, saliency, class_names=CLASS_NAMES,
                          frame_duration=FRAME_DURATION, audio_duration=1.0,
                          thresholds=thresholds)
    r = out["regions"][0]
    assert r["classes"]["prolongation"]["label"] == 0  # 0.6 < 0.7 threshold
    assert r["classes"]["block"]["label"] == 0         # 0.0 < 0.5 threshold


def test_combine_total_stutters_counts_positive_regions():
    saliency = np.zeros((100, 5))
    saliency[5:25, 1] = 0.9   # block active only in frames 5-24 = 0.1s-0.5s
    regions = [
        {"start": 0.1, "end": 0.5, "confidence": 0.8},   # covers block span
        {"start": 0.6, "end": 1.0, "confidence": 0.7},   # no active class
    ]
    out = combine_regions(regions, saliency, class_names=CLASS_NAMES,
                          frame_duration=FRAME_DURATION, audio_duration=2.0)
    assert out["total_stutters"] == 1  # only first region has a positive class


def test_combine_snaps_region_boundaries_to_syllables():
    syllables = [
        {"syllable": "a", "start": 0.2, "end": 0.4},
        {"syllable": "b", "start": 0.4, "end": 0.6},
        {"syllable": "c", "start": 0.6, "end": 0.8},
    ]
    saliency = np.zeros((100, 5))
    saliency[:, 1] = 0.9
    regions = [{"start": 0.3, "end": 0.7, "confidence": 0.8}]
    out = combine_regions(regions, saliency, class_names=CLASS_NAMES,
                          frame_duration=FRAME_DURATION, audio_duration=1.0,
                          syllables=syllables)
    r = out["regions"][0]
    assert r["start"] == 0.2
    assert r["end"] == 0.8
    assert [s["syllable"] for s in r["syllables"]] == ["a", "b", "c"]


def test_combine_region_out_of_bounds_gets_empty_classes():
    saliency = np.zeros((10, 5))
    regions = [{"start": 5.0, "end": 6.0, "confidence": 0.5}]  # beyond 10 frames * 0.02 = 0.2s
    out = combine_regions(regions, saliency, class_names=CLASS_NAMES,
                          frame_duration=FRAME_DURATION, audio_duration=0.2)
    r = out["regions"][0]
    assert r["classes"] == {}
    assert r["primary_type"] is None


def test_combine_empty_audio():
    out = combine_regions([], np.zeros((0, 5)), class_names=CLASS_NAMES,
                          frame_duration=FRAME_DURATION, audio_duration=0.0)
    assert out["regions"] == []
    assert out["total_stutters"] == 0


def test_combine_regions_sorted_by_start():
    saliency = np.zeros((100, 5))
    saliency[:, 1] = 0.9
    regions = [
        {"start": 1.0, "end": 1.2, "confidence": 0.6},
        {"start": 0.0, "end": 0.3, "confidence": 0.8},
    ]
    out = combine_regions(regions, saliency, class_names=CLASS_NAMES,
                          frame_duration=FRAME_DURATION, audio_duration=1.5)
    starts = [r["start"] for r in out["regions"]]
    assert starts == sorted(starts)


def test_mismatch_rate_counts_uncovered_saliency_spans():
    saliency = np.zeros((100, 5))
    saliency[20:30, 1] = 0.9   # high span frames 20-29
    saliency[80:90, 2] = 0.8   # high span frames 80-89
    regions = [{"start": 0.4, "end": 0.6, "confidence": 0.9}]  # frames 20-29
    rate = mismatch_rate(regions, saliency, class_names=CLASS_NAMES,
                         frame_duration=FRAME_DURATION, threshold=0.5)
    assert rate == pytest.approx(0.5, abs=1e-6)
