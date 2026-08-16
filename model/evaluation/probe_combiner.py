"""CLI + helpers to measure the region<->saliency mismatch rate.

Informs the 'no synthesis in v1' decision in the combiner spec: if many
high-confidence class spans have no overlapping localizer region, saliency
synthesis becomes a candidate for a later, data-backed version.

Usage:
    python -m model.evaluation.probe_combiner --data_dir data --max_length_seconds 3
"""

import argparse

import numpy as np

from model.combiner import FRAME_DURATION, mismatch_rate
from model.registry import ModelRegistry


def probe_from_regions(regions, saliency, threshold: float = 0.5) -> float:
    """Mismatch rate for one clip's regions + saliency matrix."""
    return mismatch_rate(regions, saliency, frame_duration=FRAME_DURATION,
                         threshold=threshold)


def aggregate_mismatch(per_clip_rates):
    """Aggregate per-clip mismatch rates into a summary dict."""
    rates = [r for r in per_clip_rates if r is not None]
    if not rates:
        return {"mean_mismatch": 0.0, "clips": 0, "clips_with_mismatch": 0}
    return {
        "mean_mismatch": float(np.mean(rates)),
        "clips": len(rates),
        "clips_with_mismatch": int(sum(1 for r in rates if r > 0)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Probe region<->saliency mismatch rate.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--max_length_seconds", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of clips evaluated (for smoke runs).")
    args = parser.parse_args(argv)

    from model.data.dataset import ClassificationDataset

    dataset = ClassificationDataset(
        args.data_dir, max_length_seconds=args.max_length_seconds,
    )
    reg = ModelRegistry()

    rates = []
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    for i in range(n):
        audio, _ = dataset[i]
        audio = np.asarray(audio).reshape(-1)
        loc = reg.localizer.analyze(
            audio, threshold=0.3, max_length_seconds=args.max_length_seconds,
        )
        regions = loc.get("regions", [])
        saliency = reg.multitask_classifier.saliency(
            audio, max_length_seconds=args.max_length_seconds,
        ).squeeze(0).cpu().numpy()
        rates.append(probe_from_regions(regions, saliency, threshold=args.threshold))

    print(aggregate_mismatch(rates))


if __name__ == "__main__":
    main()
