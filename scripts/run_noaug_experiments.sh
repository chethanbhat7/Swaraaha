#!/usr/bin/env bash
# Retrain all comparison-study + experiment arms WITHOUT augmentation (A/B ablation).
# Every arm appends _naug to its fingerprint, so aug-on checkpoints are untouched.
# Run sequentially; resume is automatic (same fingerprint). User runs this on GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
DATA_DIR="${DATA_DIR:-$ROOT/data/train}"
OUT_DIR="${OUT_DIR:-$ROOT/model/weights}"
MULTI=("$VENV_PY" -m model.training.train_multitask_classifier)
SINGLE=("$VENV_PY" -m model.training.train_classifier)
CNN=("$VENV_PY" -m model.training.train_cnn_classifier)
CNN_LOC=("$VENV_PY" -m model.training.train_localizer)
W2V2_LOC=("$VENV_PY" -m model.training.train_wav2vec2_localizer)

run_single() { # $1=class_name
  "${SINGLE[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
    --class_name "$1" --model_name facebook/wav2vec2-base \
    --epochs 20 --batch_size 8 --lr 3e-5 --max_length_seconds 3.0 \
    --warmup_steps 500 --weight_decay 0.01 --loss_type focal --focal_gamma 2.0 \
    --seed 42 --freeze_backbone_epochs 3 --gradient_accumulation_steps 1 \
    --no-augmentation
}

run_cnn() { # $1=aggregator $2=class_names
  "${CNN[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
    --aggregator "$1" --class_names "$2" \
    --epochs 20 --batch_size 16 --lr 3e-5 --n_mels 128 --hop_length 512 \
    --max_length_seconds 3.0 --hidden_dim 128 --dropout 0.4 --patience 5 \
    --warmup_steps 500 --weight_decay 0.01 --gradient_accumulation_steps 1 \
    --seed 42 --no-augmentation
}

run_cnn_lstm() { # $1=class_names, $2=num_lstm_layers
  "${CNN[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
    --aggregator lstm --num_lstm_layers "$2" --class_names "$1" \
    --epochs 20 --batch_size 16 --lr 3e-5 --n_mels 128 --hop_length 512 \
    --max_length_seconds 3.0 --hidden_dim 128 --dropout 0.4 --patience 5 \
    --warmup_steps 500 --weight_decay 0.01 --gradient_accumulation_steps 1 \
    --seed 42 --no-augmentation
}

run_cnn_tf() { # $1=class_names, $2=num_transformer_layers
  "${CNN[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
    --aggregator transformer --num_transformer_layers "$2" --class_names "$1" \
    --epochs 20 --batch_size 16 --lr 3e-5 --n_mels 128 --hop_length 512 \
    --max_length_seconds 3.0 --hidden_dim 128 --dropout 0.4 --patience 5 \
    --warmup_steps 500 --weight_decay 0.01 --gradient_accumulation_steps 1 \
    --seed 42 --no-augmentation
}

echo "=== arm01: single-class w2v2 (no-aug) ==="
for c in prolongation block soundrep wordrep interjection; do run_single "$c"; done

echo "=== arm02: multitask w2v2 frz3 (no-aug) ==="
"${MULTI[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/wav2vec2-base --freeze_backbone_epochs 3 --seed 42 \
  --no-augmentation

echo "=== arm03: multitask w2v2 frz20 (no-aug) ==="
"${MULTI[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/wav2vec2-base --freeze_backbone_epochs 20 --seed 42 \
  --no-augmentation

echo "=== arm04/05/06/07: CNN arms (no-aug) ==="
run_cnn pool all
for c in prolongation block soundrep wordrep interjection; do run_cnn pool "$c"; done
run_cnn_lstm all 1
run_cnn_tf all 1

echo "=== armL1: CNN localizer ml3 (no-aug) ==="
"${CNN_LOC[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --epochs 30 --batch_size 8 --lr 0.001 --n_mels 128 --hop_length 512 \
  --max_length_seconds 3.0 --dropout 0.4 --patience 7 --weight_decay 0.0001 \
  --val_ratio 0.2 --seed 42 --no-augmentation

echo "=== armL2: w2v2 localizer ml3 (no-aug) ==="
"${W2V2_LOC[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/wav2vec2-base --epochs 20 --batch_size 4 --lr 3e-5 \
  --freeze_backbone_epochs 5 --warmup_steps 500 --hidden_dim 256 --dropout 0.3 \
  --weight_decay 0.01 --max_length_seconds 3.0 --patience 5 --val_ratio 0.2 \
  --seed 42 --no-augmentation

echo "=== I-2 / I-3: imbalance study on wav2vec2-base (no-aug) ==="
"${MULTI[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/wav2vec2-base --loss_type focal --focal_gamma 3.0 \
  --seed 42 --no-augmentation
"${MULTI[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/wav2vec2-base --loss_type bce_posweight --focal_gamma 2.0 \
  --seed 42 --no-augmentation  # --focal_gamma ignored by bce_posweight; kept to match aug-on arm fingerprint

echo "=== B-3 / B-4: backbone study (no-aug) ==="
"${MULTI[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name microsoft/wavlm-base-plus --seed 42 --no-augmentation
"${MULTI[@]}" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR" \
  --model_name facebook/hubert-base-ls960 --seed 42 --no-augmentation

echo "=== Done: no-aug grid complete. ==="
echo "NOTE: B-2 (w2v2-large-960h) intentionally SKIPPED — its aug-on run went NaN
from epoch 1 (use_amp autocast, no GradScaler). Debug separately before retraining."
