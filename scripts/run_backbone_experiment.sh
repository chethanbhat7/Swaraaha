#!/usr/bin/env bash
# Run the backbone & imbalance experiment arms sequentially.
# Each arm trains a multitask classifier; resume is automatic (same fingerprint).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
DATA_DIR="${DATA_DIR:-$ROOT/data/train}"
OUT_DIR="${OUT_DIR:-$ROOT/model/weights}"

# ---- Imbalance study (base backbone, focal gamma + bce_posweight) ----
"$VENV_PY" -m model.training.train_multitask_classifier \
  --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/wav2vec2-base \
  --loss_type focal --focal_gamma 2.0 --seed 42

"$VENV_PY" -m model.training.train_multitask_classifier \
  --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/wav2vec2-base \
  --loss_type focal --focal_gamma 3.0 --seed 42

"$VENV_PY" -m model.training.train_multitask_classifier \
  --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/wav2vec2-base \
  --loss_type bce_posweight --focal_gamma 2.0 --seed 42

# ---- Backbone study (focal gamma 2.0 recipe) ----
"$VENV_PY" -m model.training.train_multitask_classifier \
  --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/wav2vec2-large-960h \
  --batch_size 8 --gradient_accumulation_steps 2 --seed 42

"$VENV_PY" -m model.training.train_multitask_classifier \
  --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name microsoft/wavlm-base-plus --seed 42

"$VENV_PY" -m model.training.train_multitask_classifier \
  --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/hubert-base-ls960 --seed 42