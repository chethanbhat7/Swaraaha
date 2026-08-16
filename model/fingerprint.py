"""Parameter fingerprint helpers — shared by training and model registry.

Training encodes every hyperparameter into the output filename (a "fingerprint").
The registry decodes it to load models correctly. This module is the single
source of truth for both directions.
"""

import os
import re
from typing import Dict

from model.config.defaults import DYSFLUENCY_CLASSES

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


# Args that must match for multitask checkpoint resume
MULTITASK_RESUME_KEYS = [
    "data_dir", "model_name", "lr", "batch_size",
    "max_length_seconds", "warmup_steps", "weight_decay",
    "freeze_backbone_epochs", "focal_gamma", "seed",
    "gradient_accumulation_steps", "epochs",
]

MULTITASK_FINGERPRINT_FMT = (
    "multi_e{epochs}_b{batch_size}_lr{lr}_frz{freeze_backbone_epochs}"
    "_focal_g{focal_gamma}_ga{gradient_accumulation_steps}_wu{warmup_steps}"
    "_wd{weight_decay}_ml{max_length_seconds}_s{seed}_{data_short}_{model_short}"
)


def multitask_fingerprint(args) -> str:
    """Build a fingerprint string for the multitask classifier pipeline."""
    values = {k: _fmt_fp(getattr(args, k)) for k in MULTITASK_RESUME_KEYS}
    values["data_short"] = os.path.basename(args.data_dir.rstrip("/"))
    values["model_short"] = MODEL_ALIASES.get(
        args.model_name, args.model_name.replace("/", "_")
    )
    return MULTITASK_FINGERPRINT_FMT.format(**values)


def parse_multitask_fingerprint(fp: str) -> dict:
    """Parse a multitask fingerprint string back into a dict of params."""
    pattern = (
        r'^multi_e(?P<epochs>\d+)_b(?P<batch_size>\d+)_lr(?P<lr>[\d.e\-]+)'
        r'_frz(?P<freeze_backbone_epochs>\d+)_focal_g(?P<focal_gamma>[\d.e\-]+)'
        r'_ga(?P<gradient_accumulation_steps>\d+)_wu(?P<warmup_steps>\d+)'
        r'_wd(?P<weight_decay>[\d.e\-]+)_ml(?P<max_length_seconds>[\d.e\-]+)'
        r'_s(?P<seed>\d+)_(?P<data_short>\w+)_(?P<model_short>\w+)$'
    )
    m = re.match(pattern, fp)
    if not m:
        raise ValueError(f"Cannot parse multitask fingerprint: {fp}")
    d = m.groupdict()
    for k in ("epochs", "batch_size", "freeze_backbone_epochs",
              "gradient_accumulation_steps", "warmup_steps", "seed"):
        d[k] = int(d[k])
    for k in ("lr", "focal_gamma", "weight_decay", "max_length_seconds"):
        d[k] = float(d[k])
    ms = d.pop("model_short")
    d["model_name"] = MODEL_SHORT_TO_NAME.get(ms, ms)
    d.pop("data_short", None)
    return d


CNN_CLASSIFIER_RESUME_KEYS = [
    'data_dir', 'epochs', 'batch_size', 'lr', 'n_mels', 'hop_length', 'n_fft',
    'max_length_seconds', 'hidden_dim', 'dropout', 'patience', 'warmup_steps',
    'weight_decay', 'gradient_accumulation_steps', 'seed', 'aggregator',
    'num_lstm_layers', 'num_transformer_layers', 'class_names',
]

DEFAULT_N_FFT = 2048

CNN_CLASSIFIER_FINGERPRINT_FMT = (
    'cnnclf_agg{aggregator_short}_e{epochs}_b{batch_size}_lr{lr}_n{n_mels}'
    '_h{hop_length}{n_fft_segment}_ml{max_length_seconds}_hd{hidden_dim}_d{dropout}'
    '_pa{patience}_wu{warmup_steps}_wd{weight_decay}'
    '_ga{gradient_accumulation_steps}_s{seed}_{data_short}_{classes_short}'
)

_CNN_CLASSIFIER_FP_PATTERN = re.compile(
    r'^cnnclf_agg(?P<aggregator_short>[a-z]+\d*)_e(?P<epochs>\d+)_b(?P<batch_size>\d+)'
    r'_lr(?P<lr>[\d.eE+-]+)_n(?P<n_mels>\d+)_h(?P<hop_length>\d+)'
    r'(?:_f(?P<n_fft>\d+))?'
    r'_ml(?P<max_length_seconds>[\d.eE+-]+)_hd(?P<hidden_dim>\d+)_d(?P<dropout>[\d.eE+-]+)'
    r'_pa(?P<patience>\d+)_wu(?P<warmup_steps>\d+)_wd(?P<weight_decay>[\d.eE+-]+)'
    r'_ga(?P<gradient_accumulation_steps>\d+)_s(?P<seed>\d+)'
    r'_(?P<data_short>[^_]+)_(?P<classes_short>[a-z_]+)$'
)


def _aggregator_short(aggregator, num_lstm_layers=1, num_transformer_layers=1):
    if aggregator == 'pool':
        return 'pool'
    if aggregator == 'lstm':
        return f'lstm{num_lstm_layers}'
    if aggregator == 'transformer':
        return f'tf{num_transformer_layers}'
    return aggregator


def cnn_classifier_fingerprint(args):
    data_short = os.path.basename(args.data_dir.rstrip('/')) if args.data_dir else 'data'
    class_names = getattr(args, 'class_names', None) or list(DYSFLUENCY_CLASSES)
    classes_short = ('all' if set(class_names) == set(DYSFLUENCY_CLASSES)
                     else '_'.join(class_names))
    n_fft = getattr(args, 'n_fft', None)
    n_fft_segment = f'_f{n_fft}' if n_fft and n_fft != DEFAULT_N_FFT else ''
    return CNN_CLASSIFIER_FINGERPRINT_FMT.format(
        aggregator_short=_aggregator_short(
            args.aggregator,
            getattr(args, 'num_lstm_layers', 1),
            getattr(args, 'num_transformer_layers', 1),
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=_fmt_fp(args.lr),
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        n_fft_segment=n_fft_segment,
        max_length_seconds=_fmt_fp(args.max_length_seconds),
        hidden_dim=args.hidden_dim,
        dropout=_fmt_fp(args.dropout),
        patience=args.patience,
        warmup_steps=args.warmup_steps,
        weight_decay=_fmt_fp(args.weight_decay),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        data_short=data_short,
        classes_short=classes_short,
    )


def parse_cnn_classifier_fingerprint(fp):
    match = _CNN_CLASSIFIER_FP_PATTERN.match(fp)
    if not match:
        raise ValueError(f'Invalid CNN classifier fingerprint: {fp}')
    groups = dict(match.groupdict())
    aggregator_short = groups.pop('aggregator_short')
    data_short = groups.pop('data_short')
    classes_short = groups.pop('classes_short')
    params = {}
    n_fft = groups.pop('n_fft', None)
    if n_fft is None:
        n_fft = DEFAULT_N_FFT
    params['n_fft'] = int(n_fft)
    for key, value in groups.items():
        if key in {'epochs', 'batch_size', 'n_mels', 'hop_length', 'hidden_dim',
                   'patience', 'warmup_steps', 'gradient_accumulation_steps', 'seed'}:
            params[key] = int(value)
        else:
            params[key] = float(value)
    if aggregator_short.startswith('lstm'):
        params['aggregator'] = 'lstm'
        params['num_lstm_layers'] = int(aggregator_short[4:] or 1)
        params['num_transformer_layers'] = 1
    elif aggregator_short.startswith('tf'):
        params['aggregator'] = 'transformer'
        params['num_lstm_layers'] = 1
        params['num_transformer_layers'] = int(aggregator_short[2:] or 1)
    else:
        params['aggregator'] = 'pool'
        params['num_lstm_layers'] = 1
        params['num_transformer_layers'] = 1
    params['data_short'] = data_short
    params['class_names'] = (list(DYSFLUENCY_CLASSES) if classes_short == 'all'
                             else classes_short.split('_'))
    return params


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
