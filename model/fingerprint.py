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
    try:
        params = parse_fingerprint_from_path(path)
    except ValueError:
        return "facebook/wav2vec2-base"
    return params.get("model_name", "facebook/wav2vec2-base")


def parse_fingerprint_from_path(path: str) -> Dict:
    """Parse a fingerprint filename (with suffix like _best.pt) into a params dict."""
    base = os.path.basename(path)
    for suffix in ("_best.pt", "_final.pt", "_checkpoint.pt", "_log.csv", "_curves.png", ".pt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return parse_fingerprint(base)


CNN_LOCALIZER_RESUME_KEYS = [
    "data_dir", "epochs", "batch_size", "lr", "n_mels", "hop_length",
    "max_length_seconds", "dropout", "patience", "weight_decay", "seed",
    "val_ratio",
]

W2V2_LOCALIZER_RESUME_KEYS = [
    "data_dir", "epochs", "batch_size", "lr", "max_length_seconds", "dropout",
    "hidden_dim", "patience", "weight_decay", "freeze_backbone_epochs",
    "model_name", "seed", "val_ratio", "warmup_steps",
]

LOCALIZER_RESUME_KEYS = {
    "loc": CNN_LOCALIZER_RESUME_KEYS,
    "wav2vec": W2V2_LOCALIZER_RESUME_KEYS,
}

CNN_LOCALIZER_FMT = (
    "cnnloc_e{epochs}_b{batch_size}_lr{lr}_n{n_mels}_h{hop_length}"
    "_ml{max_length_seconds}_d{dropout}_pa{patience}_wd{weight_decay}"
    "_vr{val_ratio}_s{seed}_{data_short}"
)

W2V2_LOCALIZER_FMT = (
    "w2v2loc_e{epochs}_b{batch_size}_lr{lr}_frz{freeze_backbone_epochs}"
    "_wu{warmup_steps}_hd{hidden_dim}_d{dropout}_wd{weight_decay}"
    "_ml{max_length_seconds}_pa{patience}_vr{val_ratio}_s{seed}"
    "_{data_short}_{model_short}"
)

LOCALIZER_FINGERPRINT_FMTS = {
    "loc": CNN_LOCALIZER_FMT,
    "wav2vec": W2V2_LOCALIZER_FMT,
}


def localizer_fingerprint(args, pipeline: str) -> str:
    """Build a fingerprint string for a localizer pipeline."""
    if pipeline not in LOCALIZER_FINGERPRINT_FMTS:
        raise ValueError(f"Unknown localizer pipeline: {pipeline}")
    fmt = LOCALIZER_FINGERPRINT_FMTS[pipeline]
    keys = LOCALIZER_RESUME_KEYS[pipeline]
    values = {k: _fmt_fp(getattr(args, k)) for k in keys if hasattr(args, k)}
    values["data_short"] = os.path.basename(args.data_dir.rstrip("/"))
    if pipeline == "wav2vec":
        values["model_short"] = MODEL_ALIASES.get(
            args.model_name, args.model_name.replace("/", "_")
        )
    return fmt.format(**values)


def parse_localizer_fingerprint(fp: str) -> dict:
    """Parse a localizer fingerprint string back into a params dict."""
    if fp.startswith("cnnloc_"):
        pattern = (
            r'^cnnloc_e(?P<epochs>\d+)_b(?P<batch_size>\d+)_lr(?P<lr>[\d.e\-]+)'
            r'_n(?P<n_mels>\d+)_h(?P<hop_length>\d+)'
            r'_ml(?P<max_length_seconds>[\d.e\-]+)_d(?P<dropout>[\d.e\-]+)'
            r'_pa(?P<patience>\d+)_wd(?P<weight_decay>[\d.e\-]+)'
            r'_vr(?P<val_ratio>[\d.e\-]+)_s(?P<seed>\d+)_(?P<data_short>\w+)$'
        )
    elif fp.startswith("w2v2loc_"):
        pattern = (
            r'^w2v2loc_e(?P<epochs>\d+)_b(?P<batch_size>\d+)_lr(?P<lr>[\d.e\-]+)'
            r'_frz(?P<freeze_backbone_epochs>\d+)_wu(?P<warmup_steps>\d+)'
            r'_hd(?P<hidden_dim>\d+)_d(?P<dropout>[\d.e\-]+)'
            r'_wd(?P<weight_decay>[\d.e\-]+)_ml(?P<max_length_seconds>[\d.e\-]+)'
            r'_pa(?P<patience>\d+)_vr(?P<val_ratio>[\d.e\-]+)_s(?P<seed>\d+)'
            r'_(?P<data_short>\w+)_(?P<model_short>\w+)$'
        )
    else:
        raise ValueError(f"Cannot parse localizer fingerprint: {fp}")
    m = re.match(pattern, fp)
    if not m:
        raise ValueError(f"Cannot parse localizer fingerprint: {fp}")
    d = m.groupdict()
    int_keys = ("epochs", "batch_size", "patience", "seed")
    if "n_mels" in d:
        int_keys += ("n_mels", "hop_length")
    if "freeze_backbone_epochs" in d:
        int_keys += ("freeze_backbone_epochs",)
    if "warmup_steps" in d:
        int_keys += ("warmup_steps",)
    if "hidden_dim" in d:
        int_keys += ("hidden_dim",)
    for k in int_keys:
        d[k] = int(d[k])
    for k in ("lr", "max_length_seconds", "dropout", "weight_decay", "val_ratio"):
        d[k] = float(d[k])
    d.pop("data_short", None)
    if "model_short" in d:
        ms = d.pop("model_short")
        d["model_name"] = MODEL_SHORT_TO_NAME.get(ms, ms)
    return d
