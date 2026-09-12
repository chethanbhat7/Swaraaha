#!/usr/bin/env bash
# Full no-aug evaluation for the augmentation A/B ablation.
# Mirrors the aug-on protocol exactly:
#   Phase 1: val threshold sweeps (data/train -> val_split 20% seed 42)
#            -> multitask_thresholds.json  (picks per-class optimal thresholds)
#   Phase 2: test split   (data/test --full, tuned thresholds applied)
#   Phase 3: boli subset  (data/test --sources boli --full, tuned thresholds applied)
# Results land in model/evaluation/reports/arms_noaug/<arm>/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
PY="$VENV_PY -m model.evaluation.evaluate"
W="$ROOT/model/weights"
OUT="$ROOT/model/evaluation/reports/arms_noaug"
TRAIN="$ROOT/data/train"
TEST="$ROOT/data/test"
ALL="prolongation,block,soundrep,wordrep,interjection"

mkdir -p "$OUT"
for arm in arm01_5x_w2v2 arm02_mt_w2v2_frz3 arm03_mt_w2v2_frz20 arm04_cnn_pool \
          arm05_cnn_single arm06_cnn_lstm arm07_cnn_tf armI2_max_f3 armI3_bcepos \
          armB3_wavlm armB4_hubert armL1_cnn_loc armL2_w2v2_loc; do
  mkdir -p "$OUT/$arm/val" "$OUT/$arm/test" "$OUT/$arm/boli"
  for c in prolongation block soundrep wordrep interjection; do
    mkdir -p "$OUT/$arm/val/$c" "$OUT/$arm/test/$c" "$OUT/$arm/boli/$c"
  done
done

# =============== Phase 1: val sweeps ===============

# ---- arm01: 5x single-class w2v2 ----
echo "=== arm01 val sweeps (no-aug) ==="
for c in prolongation block soundrep wordrep interjection; do
  $PY --model_type classifier --class_name "$c" \
    --model_path "$W/${c}_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base_naug_best.pt" \
    --data_dir "$TRAIN" --sweep_thresholds \
    --output_dir "$OUT/arm01_5x_w2v2/val/$c"
done

# ---- multitask / CNN arms ----
for spec in \
  "arm02_mt_w2v2_frz3:multi_e20_b16_lr3e-5_frz3_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base" \
  "arm03_mt_w2v2_frz20:multi_e20_b16_lr3e-5_frz20_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base" \
  "arm04_cnn_pool:cnnclf_aggpool_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_all" \
  "arm06_cnn_lstm:cnnclf_agglstm1_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_all" \
  "arm07_cnn_tf:cnnclf_aggtf1_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_all" \
  "armI2_max_f3:multi_e20_b16_lr3e-5_frz3_ltfocal_g3_ga1_wu500_wd0.01_ml3_s42_train_w2v2base" \
  "armI3_bcepos:multi_e20_b16_lr3e-5_frz3_ltbce_posweight_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base" \
  "armB3_wavlm:multi_e20_b16_lr3e-5_frz3_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_wavlmbase" \
  "armB4_hubert:multi_e20_b16_lr3e-5_frz3_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_hubertbase" \
  ; do
  arm="${spec%%:*}"; fp="${spec#*:}"
  echo "=== $arm val sweep (no-aug) ==="
  $PY --model_type multitask --model_path "$W/${fp}_naug_best.pt" \
    --data_dir "$TRAIN" --sweep_thresholds --output_dir "$OUT/$arm" | tee "$OUT/$arm/sweep.log"
done

# ---- arm05: CNN 5x single-class ----
echo "=== arm05 val sweeps (no-aug) ==="
for c in prolongation block soundrep wordrep interjection; do
  $PY --model_type multitask \
    --model_path "$W/cnnclf_aggpool_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_${c}_naug_best.pt" \
    --data_dir "$TRAIN" --sweep_thresholds \
    --output_dir "$OUT/arm05_cnn_single/val/$c" | tee "$OUT/arm05_cnn_single/val/$c/sweep.log"
done

# =============== Assemble combined thresholds (arm01, arm05) ===============
echo "=== Assemble combined multitask_thresholds.json (arm01/arm05) ==="
$VENV_PY - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
# arm01: from per-class val classifier reports (best_f1 threshold)
d = {}
for c in ["prolongation","block","soundrep","wordrep","interjection"]:
    r = json.load(open(os.path.join(out, "arm01_5x_w2v2", "val", c, f"{c}_report.json")))
    d[c] = {"f1_threshold": r["threshold_sweep"]["best_f1"]["threshold"]}
with open(os.path.join(out, "arm01_5x_w2v2", "multitask_thresholds.json"), "w") as f:
    json.dump({"model_path": "arm01_5x_w2v2", "source": "val_split (20%, seed 42)",
               "thresholds": d}, f, indent=2)
print("  wrote arm01 combined thresholds:", {k: round(v["f1_threshold"],2) for k,v in d.items()})
# arm05: from per-class val multitask thresholds
thr = {}
for c in ["prolongation","block","soundrep","wordrep","interjection"]:
    r = json.load(open(os.path.join(out, "arm05_cnn_single", "val", c, "multitask_thresholds.json")))
    thr[c] = r["thresholds"][c]
with open(os.path.join(out, "arm05_cnn_single", "multitask_thresholds.json"), "w") as f:
    json.dump({"model_path": "arm05_cnn_single", "source": "val_split (20%, seed 42)",
               "thresholds": thr}, f, indent=2)
print("  wrote arm05 combined thresholds:", {k: round(v["f1_threshold"],2) for k,v in thr.items()})
PY

# =============== Phase 2+3: test & boli with tuned thresholds ===============

# ---- multitask / CNN arms ----
for spec in \
  "arm02_mt_w2v2_frz3:multi_e20_b16_lr3e-5_frz3_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base" \
  "arm03_mt_w2v2_frz20:multi_e20_b16_lr3e-5_frz20_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base" \
  "arm04_cnn_pool:cnnclf_aggpool_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_all" \
  "arm06_cnn_lstm:cnnclf_agglstm1_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_all" \
  "arm07_cnn_tf:cnnclf_aggtf1_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_all" \
  "armI2_max_f3:multi_e20_b16_lr3e-5_frz3_ltfocal_g3_ga1_wu500_wd0.01_ml3_s42_train_w2v2base" \
  "armI3_bcepos:multi_e20_b16_lr3e-5_frz3_ltbce_posweight_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base" \
  "armB3_wavlm:multi_e20_b16_lr3e-5_frz3_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_wavlmbase" \
  "armB4_hubert:multi_e20_b16_lr3e-5_frz3_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_hubertbase" \
  ; do
  arm="${spec%%:*}"; fp="${spec#*:}"
  cp="$OUT/$arm/multitask_thresholds.json"
  echo "=== $arm test split (no-aug, tuned thr) ==="
  $PY --model_type multitask --model_path "$W/${fp}_naug_best.pt" \
    --data_dir "$TEST" --full --thresholds_path "$cp" \
    --output_dir "$OUT/$arm/test" | tee "$OUT/$arm/test/sweep.log"
  echo "=== $arm boli subset (no-aug, tuned thr) ==="
  $PY --model_type multitask --model_path "$W/${fp}_naug_best.pt" \
    --data_dir "$TEST" --sources boli --full --thresholds_path "$cp" \
    --output_dir "$OUT/$arm/boli" | tee "$OUT/$arm/boli/sweep.log"
done

# ---- arm01: 5x single-class w2v2 ----
for c in prolongation block soundrep wordrep interjection; do
  echo "=== arm01 $c test split (no-aug, tuned thr) ==="
  $PY --model_type classifier --class_name "$c" \
    --model_path "$W/${c}_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base_naug_best.pt" \
    --data_dir "$TEST" --full \
    --thresholds_path "$OUT/arm01_5x_w2v2/multitask_thresholds.json" \
    --output_dir "$OUT/arm01_5x_w2v2/test"
  echo "=== arm01 $c boli subset (no-aug, tuned thr) ==="
  $PY --model_type classifier --class_name "$c" \
    --model_path "$W/${c}_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base_naug_best.pt" \
    --data_dir "$TEST" --sources boli --full \
    --thresholds_path "$OUT/arm01_5x_w2v2/multitask_thresholds.json" \
    --output_dir "$OUT/arm01_5x_w2v2/boli"
done

# ---- arm05: CNN 5x single-class ----
for c in prolongation block soundrep wordrep interjection; do
  echo "=== arm05 $c test split (no-aug, tuned thr) ==="
  $PY --model_type multitask \
    --model_path "$W/cnnclf_aggpool_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_${c}_naug_best.pt" \
    --data_dir "$TEST" --full \
    --thresholds_path "$OUT/arm05_cnn_single/multitask_thresholds.json" \
    --output_dir "$OUT/arm05_cnn_single/test/$c"
  echo "=== arm05 $c boli subset (no-aug, tuned thr) ==="
  $PY --model_type multitask \
    --model_path "$W/cnnclf_aggpool_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_${c}_naug_best.pt" \
    --data_dir "$TEST" --sources boli --full \
    --thresholds_path "$OUT/arm05_cnn_single/multitask_thresholds.json" \
    --output_dir "$OUT/arm05_cnn_single/boli/$c"
done

# ---- Localizers (mirror aug-on: test split full + boli subset, thr 0.5) ----
for spec in "armL1_cnn_loc:cnn:data/test:cnnloc_e30_b8_lr0.001_n128_h512_ml3_d0.4_pa7_wd0.0001_vr0.2_s42_train" \
            "armL2_w2v2_loc:wav2vec2:data/test:w2v2loc_e20_b4_lr3e-5_frz5_wu500_hd256_d0.3_wd0.01_ml3_pa5_vr0.2_s42_train_w2v2base" ; do
  arm="${spec%%:*}"; rest="${spec#*:}"; lt="${rest%%:*}"; rest="${rest#*:}"; ddir="${rest%%:*}"; fp="${rest#*:}"
  echo "=== $arm test split (no-aug) ==="
  $PY --model_type localizer --localizer_type "$lt" \
    --model_path "$W/${fp}_naug_best.pt" \
    --data_dir "$ROOT/$ddir" --full \
    --output_dir "$OUT/$arm/test" | tee "$OUT/$arm/test/sweep.log"
  echo "=== $arm boli subset (no-aug) ==="
  $PY --model_type localizer --localizer_type "$lt" \
    --model_path "$W/${fp}_naug_best.pt" \
    --data_dir "$TEST" --sources boli --full \
    --output_dir "$OUT/$arm/boli" | tee "$OUT/$arm/boli/sweep.log"
done

echo "=== Done: no-aug eval complete. ==="