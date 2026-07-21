"""Post-processing for localization model: frames to timestamped regions."""


def frames_to_regions(
    probs: list[float],
    threshold: float = 0.5,
    sr: int = 16000,
    hop_length: int = 512,
    min_duration: float = 0.1,
) -> list[dict]:
    """
    Convert per-frame probabilities to timestamped dysfluency regions.

    Args:
        probs: Per-frame probability values.
        threshold: Probability threshold for positive detection.
        sr: Audio sample rate.
        hop_length: Hop length used in spectrogram.
        min_duration: Minimum region duration in seconds to keep.

    Returns:
        List of dicts with keys: start, end, confidence.
    """
    frame_duration = hop_length / sr
    regions = []
    in_region = False
    start = 0.0
    conf_sum = 0.0
    count = 0

    for i, p in enumerate(probs):
        if p >= threshold and not in_region:
            in_region = True
            start = i * frame_duration
            conf_sum = p
            count = 1
        elif p >= threshold and in_region:
            conf_sum += p
            count += 1
        elif p < threshold and in_region:
            end = i * frame_duration
            conf = conf_sum / count
            if (end - start) >= min_duration:
                regions.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "confidence": round(conf, 4),
                })
            in_region = False

    if in_region:
        end = len(probs) * frame_duration
        conf = conf_sum / count if count > 0 else 0.0
        if (end - start) >= min_duration:
            regions.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "confidence": round(conf, 4),
            })

    return regions
