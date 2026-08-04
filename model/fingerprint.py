"""Parameter fingerprint helpers — shared by training and model registry.

Training encodes every hyperparameter into the output filename (a "fingerprint").
The registry decodes it to load models correctly. This module is the single
source of truth for both directions.
"""

import os
import re
from typing import Dict

# Args that must match for checkpoint resume
RESUME_KEYS = [
    "class_name", "data_dir", "model_name", "lr", "batch_size",
    "max_length_seconds", "warmup_steps", "weight_decay",
    "freeze_backbone_epochs", "loss_type", "focal_gamma", "seed",
    "gradient_accumulation_steps", "epochs",
]

FINGERPRINT_FMT = "{class_name}_e{epochs}_b{batch_size}_lr{lr}_frz{freeze_backbone_epochs}_{loss_type}_g{focal_gamma}_ga{gradient_accumulation_steps}_wu{warmup_steps}_wd{weight_decay}_ml{max_length_seconds}_s{seed}_{data_short}_{model_short}"

MODEL_ALIASES = {
    "facebook/wav2vec2-base": "w2v2base",
    "facebook/wav2vec2-large": "w2v2large",
}

MODEL_SHORT_TO_NAME = {v: k for k, v in MODEL_ALIASES.items()}


def _fmt_fp(v) -> str:
    if isinstance(v, float):
        s = f"{v:.10g}"
        s = re.sub(r'e([+-])0(\d)', r'e\1\2', s)
        return s
    return str(v)


def fingerprint(args) -> str:
    values = {k: _fmt_fp(getattr(args, k)) for k in RESUME_KEYS}
    values["data_short"] = os.path.basename(args.data_dir.rstrip("/"))
    values["model_short"] = MODEL_ALIASES.get(args.model_name, args.model_name.replace("/", "_"))
    return FINGERPRINT_FMT.format(**values)


def parse_fingerprint(fp: str) -> dict:
    """Parse a fingerprint string back into a dict of params."""
    pattern = (
        r'^(?P<class_name>\w+)'
        r'_e(?P<epochs>\d+)'
        r'_b(?P<batch_size>\d+)'
        r'_lr(?P<lr>[\d.e\-]+)'
        r'_frz(?P<freeze_backbone_epochs>\d+)'
        r'_(?P<loss_type>\w+)'
        r'_g(?P<focal_gamma>[\d.e\-]+)'
        r'_ga(?P<gradient_accumulation_steps>\d+)'
        r'_wu(?P<warmup_steps>\d+)'
        r'_wd(?P<weight_decay>[\d.e\-]+)'
        r'_ml(?P<max_length_seconds>[\d.e\-]+)'
        r'_s(?P<seed>\d+)'
        r'_(?P<data_short>\w+)'
        r'_(?P<model_short>\w+)$'
    )
    m = re.match(pattern, fp)
    if not m:
        raise ValueError(f"Cannot parse fingerprint: {fp}")
    d = m.groupdict()
    for k in ("epochs", "batch_size", "freeze_backbone_epochs",
              "gradient_accumulation_steps", "warmup_steps", "seed"):
        d[k] = int(d[k])
    for k in ("lr", "focal_gamma", "weight_decay", "max_length_seconds"):
        d[k] = float(d[k])
    ms = d.pop("model_short")
    d["model_name"] = MODEL_SHORT_TO_NAME.get(ms, ms)
    return d


def model_name_from_path(path: str) -> str:
    """Extract the HF model name from a fingerprint-encoded file path."""
    base = os.path.basename(path)
    m = re.search(r"_(\w+v2\w+)_", base)
    if m:
        short = m.group(1)
        if short in MODEL_SHORT_TO_NAME:
            return MODEL_SHORT_TO_NAME[short]
    return "facebook/wav2vec2-base"


def parse_fingerprint_from_path(path: str) -> Dict:
    """Parse a fingerprint filename (with suffix like _best.pt) into a params dict."""
    base = os.path.basename(path)
    for suffix in ("_best.pt", "_final.pt", "_checkpoint.pt", "_log.csv", "_curves.png", ".pt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return parse_fingerprint(base)
