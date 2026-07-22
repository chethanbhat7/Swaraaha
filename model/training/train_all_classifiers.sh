#!/bin/bash
# Train all 5 Wav2Vec 2.0 binary classifiers sequentially.
#
# Usage:
#   bash model/training/train_all_classifiers.sh [DATA_DIR] [EPOCHS] [BATCH_SIZE]
#
# Examples:
#   bash model/training/train_all_classifiers.sh
#   bash model/training/train_all_classifiers.sh data 20 8
#   bash model/training/train_all_classifiers.sh data 30 4

set -e

DATA_DIR="${1:-data}"
EPOCHS="${2:-20}"
BATCH_SIZE="${3:-8}"
OUTPUT_DIR="model/weights"

echo "=============================================="
echo "  Training All Classifiers"
echo "  Data: $DATA_DIR | Epochs: $EPOCHS | Batch: $BATCH_SIZE"
echo "=============================================="

for cls in prolongation block soundrep wordrep interjection; do
    echo ""
    echo "----------------------------------------------"
    echo "  Training: $cls"
    echo "----------------------------------------------"
    python -m model.training.train_classifier \
        --class_name "$cls" \
        --data_dir "$DATA_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR"
done

echo ""
echo "=============================================="
echo "  All classifiers trained!"
echo "  Weights saved to: $OUTPUT_DIR/"
echo "=============================================="
