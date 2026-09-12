#!/usr/bin/env python3
"""A/B comparison: augmentation ON vs OFF, val macro F1 @ optimal thresholds."""
import json
import os

REPORTS = "model/evaluation/reports"

DYS = ["prolongation", "block", "soundrep", "wordrep", "interjection"]

MULTITASK_ARMS = [
    ("arm02_mt_w2v2_frz3", "multi_e20_b16_lr3e-5_frz3_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base"),
    ("arm03_mt_w2v2_frz20", "multi_e20_b16_lr3e-5_frz20_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base"),
    ("arm04_cnn_pool", "cnnclf_aggpool_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_all"),
    ("arm06_cnn_lstm", "cnnclf_agglstm1_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_all"),
    ("arm07_cnn_tf", "cnnclf_aggtf1_e20_b16_lr3e-5_n128_h512_ml3_hd128_d0.4_pa5_wu500_wd0.01_ga1_s42_train_all"),
    ("armI2_max_f3", "multi_e20_b16_lr3e-5_frz3_ltfocal_g3_ga1_wu500_wd0.01_ml3_s42_train_w2v2base"),
    ("armI3_bcepos", "multi_e20_b16_lr3e-5_frz3_ltbce_posweight_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base"),
    ("armB3_wavlm", "multi_e20_b16_lr3e-5_frz3_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_wavlmbase"),
    ("armB4_hubert", "multi_e20_b16_lr3e-5_frz3_ltfocal_g2_ga1_wu500_wd0.01_ml3_s42_train_hubertbase"),
]

MONO_ARMS = [  # arm01 (w2v2, classifier) and arm05 (cnn, multitask-single)
    ("arm01_5x_w2v2", "classifier"),
    ("arm05_cnn_single", "multitask_single"),
]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def val_macro(arm, non_aug):
    p = os.path.join(REPORTS, "arms_noaug" if non_aug else "arms", arm, "multitask_thresholds.json")
    d = load(p)
    return d.get("macro_f1_at_optimal") if d else None


def mono_macro(arm, non_aug):
    base = os.path.join(REPORTS, "arms_noaug" if non_aug else "arms", arm)
    f1s = []
    for c in DYS:
        if arm == "arm01_5x_w2v2":
            p = os.path.join(base, "val", c, f"{c}_report.json")
            r = load(p)
            if r and "threshold_sweep" in r:
                f1s.append(r["threshold_sweep"]["best_f1"]["f1"])
        else:
            # arm05 stores per-class val threshold files under val/<c> (noaug) or test/<c> (aug-on)
            sub = "val" if non_aug else "test"
            p = os.path.join(base, sub, c, "multitask_thresholds.json")
            r = load(p)
            if r and "thresholds" in r and c in r["thresholds"]:
                f1s.append(r["thresholds"][c]["f1_at_optimal"])
    return sum(f1s) / len(f1s) if len(f1s) == len(DYS) else None


def test_macro(arm, non_aug, tuned=False):
    if arm in ("arm01_5x_w2v2", "arm05_cnn_single"):
        base = os.path.join(REPORTS, "arms_noaug" if non_aug else "arms", arm, "test")
        f1s = []
        for c in DYS:
            if arm == "arm01_5x_w2v2":
                r = load(os.path.join(base, f"{c}_report.json"))
                if not r:
                    continue
                entry = r.get("per_class", {}).get(c)
                if entry is None:
                    entry = r  # classifier reports store metrics at top level
                v = entry.get("threshold_tuned", {}).get("f1") if tuned else entry.get("binary", {}).get("f1")
                if v is not None:
                    f1s.append(v)
            else:
                r = load(os.path.join(base, c, "multitask_report.json"))
                if not r:
                    continue
                per = r.get("per_class", {}).get(c, {})
                v = per.get("threshold_tuned", {}).get("f1") if tuned else per.get("binary", {}).get("f1")
                if v is not None:
                    f1s.append(v)
        return sum(f1s) / len(f1s) if f1s else None
    p = os.path.join(REPORTS, "arms_noaug" if non_aug else "arms", arm, "test", "multitask_report.json")
    r = load(p)
    if not r:
        return None
    return r.get("macro_f1_tuned") if tuned else r.get("macro_f1")


def localizer_f1(arm, split, non_aug):
    d = os.path.join(REPORTS, "arms_noaug" if non_aug else "arms", arm, split)
    if not os.path.isdir(d):
        return None
    for fname in os.listdir(d):
        if fname.endswith("_localizer_report.json"):
            return load(os.path.join(d, fname))["frame_level"]["f1"]
    return None


def fmt(v):
    return f"{v:.4f}" if v is not None else "   n/a"


rows = []
print(f"{'arm':>18s} | {'val macro@opt':>30s} | {'test macro':>34s} | {'winner'}")
print("-" * 100)
for arm, _ in MULTITASK_ARMS:
    av, nv = val_macro(arm, False), val_macro(arm, True)
    at, nt = test_macro(arm, False, tuned=True), test_macro(arm, True, tuned=True)
    win = "no-aug" if (nv or 0) > (av or 0) else ("aug" if (av or 0) > (nv or 0) else "tie")
    delta = (nv - av) if av is not None and nv is not None else None
    rows.append(win)
    print(f"{arm:>18s} | aug={fmt(av)}  noaug={fmt(nv)} | aug={fmt(at)}  noaug={fmt(nt)} | {win:>6s} ({delta:+.4f})" if delta is not None else f"{arm:>18s} | aug={fmt(av)}  noaug={fmt(nv)} | aug={fmt(at)}  noaug={fmt(nt)} | {win:>6s}")

print()
print("Single-class arms:")
for arm, _ in MONO_ARMS:
    av, nv = mono_macro(arm, False), mono_macro(arm, True)
    at, nt = test_macro(arm, False, tuned=True), test_macro(arm, True, tuned=True)
    win = "no-aug" if (nv or 0) > (av or 0) else ("aug" if (av or 0) > (nv or 0) else "tie")
    delta = (nv - av) if av is not None and nv is not None else None
    rows.append(win)
    d = f" ({delta:+.4f})" if delta is not None else ""
    print(f"{arm:>18s} | aug={fmt(av)}  noaug={fmt(nv)} | aug={fmt(at)}  noaug={fmt(nt)} | {win:>6s}{d}")

print()
print("Localizers (frame F1 @ 0.5):")
for arm, lt in [("armL1_cnn_loc", "cnn"), ("armL2_w2v2_loc", "wav2vec2")]:
    for split in ["test", "boli"]:
        av, nv = localizer_f1(arm, split, False), localizer_f1(arm, split, True)
        win = "no-aug" if (nv or 0) > (av or 0) else ("aug" if (av or 0) > (nv or 0) else "tie")
        delta = (nv - av) if av is not None and nv is not None else None
        rows.append(win)
        d = f" ({delta:+.4f})" if delta is not None else ""
        print(f"{arm:>18s} {split:>4s} | aug={fmt(av)}  noaug={fmt(nv)} |            aug={fmt(at)}  noaug={fmt(nt)} | {win:>6s}{d}" if False else f"{arm:>18s} {split:>4s} | aug={fmt(av)}  noaug={fmt(nv)} | {win:>6s}{d}")

print()
wins = rows.count("no-aug")
print(f"Summary: no-aug wins {wins}/{len(rows)} comparisons, aug wins {rows.count('aug')}, ties {rows.count('tie')}.")