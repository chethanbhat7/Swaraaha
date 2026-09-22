# Swaraaha Project Report — Review Findings

**Date:** 2026-09-20
**Scope:** `docs/report/` only (papers out of scope). Read-only review; no fixes applied.
**Ground-truth rule (user-mandated):** The authoritative results are the **NO-AUGMENTATION ("no-aug")** runs matching the shipped `model/registry.json` (`_naug_best.pt` checkpoints). Sources of truth:
- `model/evaluation/reports/comparative_study_report.json` (per-arm test/boli F1, all `augmentation: false`)
- `model/evaluation/reports/arms_noaug/**` (per-arm per-class reports, tracked in git)
- `model/registry.json` (shipped defaults/thresholds)

**STALE, do NOT use as truth:** top-level `model/evaluation/reports/multitask_report.json`, `multitask_thresholds.json`, per-class `*_report.json`, `evaluation_summary.json`, and the `arms/` directory (all reference the old augmented `..._best.pt` checkpoints). **The report currently reproduces these stale sources almost everywhere.**

---

## 1. Severity summary

| Severity | Count | Key items |
|---|---|---|
| **Critical** | 11 | All Ch6/Ch7 results are stale (augmented) run numbers; Ch1 Scope + Organization sections are swallowed by a stray comment block; Ch2 has no rendered Chapter Summary (inside a comment); localizer narrative is factually wrong for the shipped model |
| **Major** | 13 | "Primary classifier = multitask" vs shipped `single`; Ch4 "not combined/fused" contradicts fusion code; primer matrix figures inconsistent; Ch2/Ch3 duplicate chapters; multitask-vs-5x terminology; stale "Table 4.x" references; augmentation params mismatch; "temporal attention pooling" false claim; etc. |
| **Minor** | 18 | Typos, naming, boundary rounding, unreferenced assets, metadata nits, bib duplicate suspicions |

---

## 2. CRITICAL findings

### C1. Table 4.1 — per-class results are the AUGMENTED multitask run, not the shipped no-aug model
`docs/report/chapters/chapter6.typ:47-61`
Report values (P/R/F1/AUROC/AUPRC/support) come from `multitask_report.json` (`multi_..._focal_..._train_w2v2base_best.pt`, augmented). The shipped registry model is `multi_..._ltfocal_..._naug_best.pt`. No-aug ground truth (from `arms_noaug/arm02_mt_w2v2_frz3/test/multitask_report.json`):

| Class | Report (aug) | GT (no-aug) |
|---|---|---|
| Prolongation | 0.560/0.418/0.479/0.844/0.467/400 | 0.566/0.440/0.495/0.845/0.491/400 |
| Block | 0.639/0.150/0.243/0.735/0.395/553 | 0.584/0.264/0.364/0.768/0.444/553 |
| Sound Repetition | 0.564/0.490/0.524/0.860/0.552/494 | 0.705/0.502/0.586/0.878/0.624/494 |
| Word Repetition | 0.581/0.383/0.461/0.830/0.477/366 | 0.621/0.399/0.486/0.855/0.550/366 |
| Interjection | 0.836/0.666/0.741/0.932/0.836/834 | 0.810/0.746/0.776/0.935/0.848/834 |
| Macro F1 | 0.490 | **0.542** |
| Total samples | 3,715 (OK) | 3,715 |

**Fix:** replace all values with the no-aug column.

### C2. "Primary classifier" is multitask, but the shipped classifier is `single` (5 × separate Wav2Vec2)
`docs/report/chapters/chapter6.typ:45` (and ch5/ch7). `model/registry.json:3` → `defaults.classifier: "single"` (five independent wav2vec2 binary classifiers, `_naug`). The report describes the multitask shared-backbone as "The primary classifier". Either re-anchor the results chapter to the 5x single pipeline (which is also the model whose confusion matrices ship), or make clear the multitask is an experimental variant, not the default.

### C3. Table 4.2 — threshold optimization is from the AUGMENTED sweep
`docs/report/chapters/chapter6.typ:72-86`
Values (default→tuned, optimal threshold) come from `multitask_thresholds.json` (aug). No-aug GT (registry multitask thresholds; defaults = no-aug F1): prolongation 0.495→0.457 @0.55; block 0.364→0.429 @0.35; soundrep 0.586→0.621 @0.4; wordrep 0.486→0.499 @0.35; interjection 0.776→0.773 @0.45; **macro 0.542→0.556 (+2.6%)**. Note the narrative changes: tuning *hurts* prolongation & interjection and helps far less than the claimed +8.8%.

### C4. Table 4.3 — Comparative Architecture Study is AUGMENTED `arms/` data
`docs/report/chapters/chapter6.typ:93-108`
Every F1 cell except `MT frz20 @0.5 ≈0.141` is from the augmented `arms/` directory. Param counts (94.6M/97.3M/0.34M/0.27M/0.46M/0.54M) are correct. No-aug GT from `comparative_study_report.json`:

| Arm | Params | Test@0.5 report→GT | Tuned report→GT | Boli report→GT |
|---|---|---|---|---|
| 5x Wav2Vec2 | 94.6M ✓ | 0.511→0.573 | 0.518→0.570 | 0.154→0.292 |
| MT W2V2 (frz3) | 97.3M ✓ | 0.490→0.542 | 0.522→0.556 | 0.160→0.238 |
| MT W2V2 (frz20) | 97.3M ✓ | 0.142→0.141 ✓ | 0.339→0.378 | 0.295→0.214 |
| CNN Pool | 0.34M ✓ | 0.188→0.149 | 0.253→0.256 | 0.359→0.332 |
| CNN Single | 0.27M ✓ | 0.221→0.212 | 0.253→0.262 | 0.441→0.476 |
| CNN-LSTM | 0.46M ✓ | 0.248→0.242 | 0.259→0.249 | 0.521→0.508 |
| CNN-Transformer | 0.54M ✓ | 0.255→0.269 | 0.264→0.271 | 0.477→0.345 |

### C5. Table 4.4 — Wav2Vec2 localizer numbers are from a stale, non-shipped checkpoint
`docs/report/chapters/chapter6.typ:141-144`
Reported P0.676/R0.065/F1 0.119/Sp0.973/det 0.210/IoU 0.751/FA 8.95/events 2,852 match the **top-level `wav2vec2_localizer_report.json`** whose `model_path` is `w2v2loc_..._train_w2v2base_best.pt` (old non-naug). No-aug GT (`arms_noaug/armL2_w2v2_loc/test/wav2vec2_localizer_report.json`, uses registry `..._naug_best.pt`, 3,715 samples, thr 0.5):

| Metric | Stale (report) | GT (no-aug) |
|---|---|---|
| Frame Precision | 0.676 | 0.8045 |
| Frame Recall | 0.065 | 0.738 |
| Frame F1 | 0.119 | 0.7698 |
| Frame Specificity | 0.973 | 0.8469 |
| Detection Accuracy | 0.210 | 0.5859 |
| Mean IoU | 0.751 | 0.7793 |
| False Alarms/min | 8.95 | 4.28 (1,263) |
| Predicted Events | 2,852 | 1,846 |

### C6. Localizer narrative is factually wrong for the shipped model
`docs/report/chapters/chapter6.typ:149,170` — "precision high (0.676), recall low (0.065)… conservative, few false alarms, many missed events". For the shipped `_naug` localizer recall is **0.738** and precision 0.805 — the "misses most events" story is false. **Rewrite from the no-aug numbers** and re-render `localizer_metrics.png` (chapter6.typ:151).

### C7. CNN localizer paragraph is stale and references a non-existent checkpoint
`docs/report/chapters/chapter6.typ:153-154` — "evaluated on only 2 test samples… predicted a single dysfluency event with 100% recall but zero precision". This matches the top-level `cnn_localizer_report.json` whose `model_path` is `model/weights/localizer_best.pt` — **that file does not exist on disk**. (Its "100% recall" is also vacuous: `num_true_events: 0`.) No-aug GT (`arms_noaug/armL1_cnn_loc/test/cnn_localizer_report.json`, registry `cnnloc_..._naug_best.pt`): evaluated on **3,715** samples; frame P **0.527**, R **0.8731**, F1 **0.6572**, Sp 0.3303; det 0.1548; IoU 0.6785; 4,483 predicted events; 4,329 false alarms (23.49/min); threshold 0.5. **Report the real 3,715-sample numbers; drop the "2 samples / data availability" framing.**

### C8. Chapter 7 repeats every stale number
`docs/report/chapters/chapter7.typ:26,28,39,41` — "0.490 rising to 0.533"; interjection F1=0.741 AUROC=0.932; block 0.243→0.401; CNN-LSTM 0.521 Boli; multitask F1 "0.522 → 0.160 on Boli". No-aug GT: 0.542→0.556; interj 0.776/0.935; block 0.364→0.429; CNN-LSTM Boli 0.508; multitask 0.556→0.238. Also affects `chapter7.typ:16` (describes shipped classifier as multitask → contradicts registry `single`).

### C9. Chapter 1 Scope and Organization sections are swallowed by a stray comment block — they render blank
`docs/report/chapters/chapter1.typ:65-92` — the second `/* ... */` opens at :65 and only closes at :92, swallowing **SCOPE OF THE PROJECT (:73-84) and ORGANIZATION OF THE REPORT (:86-92)**. **Fix: close the comment right after the "Visual Annotation System" bullet (:71).** (The old draft text inside [:65-71] can be deleted.)

### C10. Chapter 2 has no rendered Chapter Summary (it is inside a comment block)
`docs/report/chapters/chapter2.typ:210-285` — a large commented-out block (FEASIBILITY STUDY, USE CASE, ACTIVITY, SEQUENCE, DATA FLOW, CHAPTER SUMMARY) is near-verbatim chapter 3 content and contains ch2's only CHAPTER SUMMARY. Fix: delete the comment block (or the whole ch2 if merged with ch3, see M3).

### C11. Chapter 6 "threshold of 0.5" describes the eval, not the shipped inference threshold
`docs/report/chapters/chapter6.typ:133` — "evaluated … at threshold 0.5" is correct for the eval, but `model/registry.json` stores **0.3** as the shipped wav2vec2 localizer inference threshold. A user running the registry default will not reproduce Table 4.4. Fix: state "evaluated at 0.5; shipped registry inference threshold is 0.3."

---

## 3. MAJOR findings

### M1. Ch4/Ch5 "not combined or fused" contradicts the code's fusion
`docs/report/chapters/chapter4.typ:13` ("outputs presented together without being combined or fused") vs `chapter5.typ:69-72` combiner section and `chapter7.typ:19`; `analyze()/fuse()/combine_regions` do fuse (localizer regions + classifier saliency, adaptive fallback). Pick one story (fusion is what ships).

### M2. Multitask-vs-5x terminology inconsistency across chapters
`chapter5.typ:46-49` describes the classification pipeline as a single Wav2Vec2 backbone + MultiTaskClassifier with five heads, while `chapter2.typ:62`, `chapter3.typ:19,55`, `chapter4.typ:37,68` describe five parallel/independent binary classifiers (chapter 5's own registry list at :85-90 acknowledges both). Align on one description; the shipped default is the 5x single.

### M3. Chapters 2 and 3 are duplicate requirement chapters
Ch2 "REQUIREMENT SPECIFICATION AND ANALYSIS" (`chapter2.typ:5`) vs Ch3 "ANALYSIS AND REQUIREMENT SPECIFICATION" (`chapter3.typ:4`) — overlapping FUNCTIONAL / NON-FUNCTIONAL / HARDWARE / SOFTWARE sections. Merge or drop one; at minimum rename so titles aren't near-identical.

### M4. Stale "Table 4.x" references — tables render as 6.x
`lib.typ:37-51` (`add_table`/`add_image`) number figures as `chapter.figure` using the current chapter counter, so all chapter 6 tables render as **Table 6.1–6.4**. In-text references at `chapter6.typ:45,70,91,133` say "Table 4.1/4.2/4.3/4.4" → stale (leftover from an earlier chapter layout). Fix to 6.x (or use `@label` cross-references).

### M5. "Optimal thresholds" in Table 4.2 match no shipped threshold set
`chapter6.typ:82` reports optimal thresholds 0.45/0.35/0.45/0.45/0.45. `registry.json` `single` = 0.55/0.5/0.45/0.4/0.5; `multitask` = 0.55/0.35/0.4/0.35/0.45; `cnn_multitask` = 0.5/0.55/0.6/0.4/0.1. None match the table.

### M6. Confusion matrices are internally inconsistent (mixed AUG/NO-AUG)
`docs/report/chapters/chapter6.typ:116-128` — none of the five `*.png` assets is byte-identical to any source (all re-rendered, 1012×882 vs sources 1028×885). Content decode: **block & interjection ≈ NO-AUG `arm01_5x`; prolongation ≈ AUG `arm01_5x`; soundrep → weakly NO-AUG; wordrep → ambiguous**. Fix: re-render all five from one family; recommend NO-AUG `arms_noaug/arm01_5x_w2v2/test/*confusion_matrix.png`.

### M7. Result figures are the AUGMENTED run — root cause is `presentation.py` hardcoding aug paths
`chapter6.typ:65,67,114,151` — `classifier_test_f1.png`, `classifier_test_auc.png`, `localizer_metrics.png`, `classifier_boli_f1.png` all encode the augmented run. **Root cause:** `model/evaluation/presentation.py` hardcodes AUG report paths at the top (`ARM02_TEST = ".../arms/arm02_mt_w2v2_frz3/test/multitrack_report.json"`, `ARM02_BOLI`, `ARML2_TEST`, `ARML2_BOLI`). Regenerate from no-aug by repointing those four constants to the `arms_noaug/` equivalents, then run `python -m model.evaluation.presentation`. See §8 for exact commands.

### M8. Augmentation parameters in Ch5 don't match `AudioAugmentor`
`docs/report/chapters/chapter5.typ:44` claims SNR 10–20 dB noise, ±2 semitone pitch shift, ±20% time shift. `AudioAugmentor` defaults: noise `σ=0.005`, pitch **±1 semitone**, time shift **±0.1 s**. Align the prose with the implemented defaults.

### M9. Localizer falsely credited with "temporal attention pooling"
`docs/report/chapters/chapter5.typ:58` — the Wav2Vec2 localizer is a plain per-frame MLP (fingerprint `hd256 d0.3`), not an attention-pooling model. Correct the description.

### M10. "F1 up to 0.522" / "CNN-LSTM 0.521 best cross-corpus" wording
`chapter6.typ:110` — with no-aug data the max in-distribution F1 is **0.573** (5x single W2V2) and CNN-LSTM Boli is **0.508** (still best cross-corpus; conclusion survives, but numbers/ranking change — cross-corpus ranking among the CNNs also shifts).

### M11. Chapter 1 Organization list is self-contradictory
`docs/report/chapters/chapter1.typ:87` (inside the C9 block) — says "organized into two chapters" then lists Chapter 1–4. Rewrite to enumerate the real 7 chapters.

### M12. 80:10:10 is the dataset split, but training actually used an 80/20 reselection
`chapter6.typ:21`, `chapter3.typ:153`, `chapter4.typ:96` — `model/data/prepare.py` does 80/10/10 (`test_only_sources=("boli",)`), but training re-split `data/train` 80/20 (`stratified_split(val_ratio=0.2)`, `TRAIN_VAL_SPLIT=0.8`, threshold sweep on the 20% val). Add one sentence clarifying the two-level split.

### M13. Ch5 severity boundary at 15% mismatches code
`chapter5.typ:75`, `chapter6.typ:70` — report says Moderate (5–15%) / Severe (>15%); code (`backend/services/severity.py`, `shared/reporting/report_builder.py`) uses `>=15` severe, `>=5` moderate, `>=2` mild. An index of **exactly 15** is "severe" in code. Use "2–5 / 5–15 / ≥15".

---

## 4. MINOR findings

### Ch1 (Introduction)
- `chapter1.typ:15` — lists stutter types as "repetitions, prolongations, blocks, and fillers" (only four; "fillers" should be the fifth class "interjection"). Standardize to the five-class taxonomy.
- `chapter1.typ:18` — stray mid-sentence `// and the detected stutter timestamps …` line-comment silently drops the trailing sentence; remove leftover or restore text.
- `chapter1.typ:41-42` — heading contains `// PROPOSED SYSTEM` (Typst line-comment merges the label; renders only "PROBLEM STATEMENT"); leftover merge artifact → clean to `== PROBLEM STATEMENT`.
- `chapter1.typ:45-58`, `chapter1.typ:65-71` — commented-out draft blocks (PROBLEM STATEMENT, OBJECTIVES); delete (commented-out content does not render but is confusing in source).
- `chapter1.typ:74` — typo "etects" (inside C9 block; remove with it).

### Ch2/Ch3 (requirements)
- `chapter2.typ:155` "SOFTWARE REQUIREMENT" (singular) vs `chapter3.typ:76` "SOFTWARE REQUIREMENTS" — unify.
- `chapter3.typ:12` — "repetition of sound, repetition of word" vs the standard "sound repetition, word repetition" used elsewhere.
- `chapter3.typ:1-4` — archive comment header says "Chapter 3: Analysis and Requirement Specification" (confirms the ch2/ch3 duplication, M3).

### Ch3/Ch4 (registry, desktop)
- `chapter3.typ:82` and `chapter4.typ:135` — "registry module (`model/registry.py`)" is stale: the registry is now the **`model/registry/` package** (no `model/registry.py` file); `ModelRegistry`/`run_all` were removed (PR #111).
- `chapter4.typ:136` — desktop app is said to include "AudioTranscriber"; no such module exists (`app/` has `ModelRunner`, `AudioHandler`; transcription goes through the shared `model` package's `Transcriber`). Rename to the real text.

### Ch6 (results)
- `chapter6.typ:58` — macro "Support" shown as 3,715 (total multi-label samples) while per-class supports sum to 2,647. Expected for multi-label, but label it "total samples" (or per-class "positive samples") to avoid confusion.
- `chapter6.typ:83` — "+8.8%" computed from rounded 0.490/0.533; the precise change is +8.9%. (Also moot once switched to no-aug, C3.)

### Ch6 (implementation claims)
- `chapter6.typ:161-162` — "Gradient clipping removed" and "GradScaler removed; autocast only" are true for the **classifier** loops (`train_classifier.py`/`train_multitask_classifier.py`; `GRADIENT_CLIP_MAX_NORM` in `defaults.py:31` is unused) but the **localizers still clip** (`train_localizer.py:295`, `train_wav2vec2_localizer.py:365`). Also `model/training/README.md:124` still documents `torch.amp.GradScaler` — stale, contradicts both code and report. Qualify the claim and fix README:124.

### Figures/housekeeping
- `docs/report/assets/` — 14 of 16 assets are tracked and referenced. No missing assets. (Asset mtimes are newer than the commit that added them, so if files were re-generated locally ensure the change is committed.)
- `architecture-horizontal.png` — tracked but never referenced (only `architecture-verticle.png` is used, chapter4.typ:59). Use or remove. (Filename "verticle" is a consistent misspelling.)
- `resultpage.html` — tracked but never imported; only `resultpage.png` referenced (chapter5.typ:101). Keep or remove.
- `chapter6.typ:127` — confusion-matrix caption notation "(TP, FP, TN, FN)" doesn't match rendered order (TN/FP top-left, FN/TP bottom-right per `metrics.py:506-552`); reword.
- No equation-numbering issues; TOC is auto-generated and consistent; ch5 figures number 5.1/5.2 correctly.

### Front matter / metadata
- `docs/report/meta.typ:58` — `project_coordinator` is defined, never referenced, and equals the guide. Use it or remove it (identical guide/coordinator reads as placeholder).
- `docs/report/frontmatter/toc.typ:30` — heading "TABLE OF CONTENT" → "TABLE OF CONTENTS".
- `docs/report/frontmatter/abbreviations.typ:30` — "Sep-28k" mixed case vs "SEP-28K" elsewhere; expansion says "Prediction" while the paper/podcast framing uses "Event Detection".

### Literature survey / bibliography (verify before submission)
- `docs/report/bibliography.typ:41` — bib #12 (R. Ahmed & J. Park, "Stutter-Solver: …") duplicates the title of bib #9 (X. Zhou et al., "Stutter-Solver: …").
- `docs/report/bibliography.typ:29` and `:43` — both list "Large Language Models for Dysfluency Detection in Stuttered Speech" under different authors (Wagner; Rahimi & Torres). Suspicious duplicate.
- `docs/report/contents/literature_survey.typ:381` — table row 8 lists "Y. Zhang et al." for YOLO-Stutter, but prose (:96) and bib #8 say X. Zhou et al.
- `literature_survey.typ` table typos: :374 "thorugh…seelction", :375 "acoutic", :379 "Requres", :380 "suttering", :381 "stuter", :383 "Geenrates", :385 "datsets"; rows 9 & 13 share copied Pros/Cons cells.
- Reference count is internally consistent (prose 30 ↔ table 30 ↔ bib #1–30).

### Dataset prose — verified consistent (no change needed)
`chapter6.typ:15-21` — Boli (GitHub `projectboli/Project_Boli_Dataset`, "Hindi, Kannada, Telugu… read and spontaneous speech"), SEP-28K (~28,000 podcast clips, Kaggle `ikrbasak/sep-28k`), UCLASS (Kaggle `vudominhgiang/uclass-…`), `combined_labels.csv`, 80:10:10, 16 kHz mono, DC removal / peak norm / silence trim / 48,000-sample (3 s) padding — all match `model/data/README.md`, `config.py`, `model/config/defaults.py:8-10`, `prepare.py`. *(Only caveat: the exact language list for Boli is not enumerated in repo docs — verify externally before submission.)*

### Training stories — verified consistent (no change needed)
All TRAINING CHALLENGES claims match `model/training/DEBUGGING_LOG.md`: BCEWithLogits single-logit collapse (§3h), CrossEntropy two-logit, Focal γ=2 (§3j), gradient clipping suppression (§3i), GradScaler NaN → autocast-only (§3a/§3f), waveform-vs-spectrogram augmentation separation (§9), freeze-3 beats freeze-20 (§7/§23). No invented evidence found.

### Cross-chapter consistency note
Ch6 ↔ Ch7 numbers currently agree with each other — **but both are the stale augmented set** (fixed together via C1–C8, not independently).

---

## 5. Confirmed-accurate implementation claims (for reference — do not "fix")

- `chapter5.typ:33` — public API `classify()/localize()/transcribe()/analyze()/fuse()` all exist in `model/__init__.py`. ✓
- `chapter4.typ:68-69` — five per-class two-logit softmax classifiers (matches default `single`). ✓
- `chapter4.typ:67`, `chapter3.typ:146` — mel 128 / hop 512 / FFT 2048, 32 ms CNN frames, 20 ms w2v2 frames. ✓
- `chapter5.typ:40` — preprocessing: 16 kHz, DC offset, peak norm 0.95, silence trim `top_db=25`, pad to 48k samples. ✓
- `chapter5.typ:47` — 20 ms embeddings; `chapter5.typ:49` — multitask head detail (Linear→Tanh→Linear → 2 logits). ✓
- `chapter5.typ:60-62` — CNN localizer (BN/ReLU convs, 32 ms). ✓
- `chapter5.typ:65-67` — Whisper-tiny en/kn/hi with word timestamps. ✓
- `chapter5.typ:70-72` — combiner + adaptive saliency fallback. ✓
- `chapter5.typ:78,81-90` — registry structure, lazy loading, fingerprint-encoded filenames. ✓
- `chapter5.typ:93-105` — frontend: sidebar routes, MediaRecorder, formats WAV/MP3/FLAC/M4A/OGG/WMA, IndexedDB `utils/db.ts` + localStorage, endpoint list. ✓
- `chapter6.typ:43` — freeze 3 epochs, 0.1× LR on unfreeze, Focal γ2, AdamW 3e-5, warmup 500, patience 5. ✓
- `chapter6.typ:70` — threshold sweep grid 0.1–0.9 step 0.05. ✓
- `chapter6.typ:159,163,164` — debugging lessons. ✓
- `chapter6.typ:45,133` — 3,715 test clips. ✓
- `report.typ:39-46,97-106` — all 7 frontmatter files + 7 chapters + bibliography included. ✓
- `meta.typ` — title, B.E., 7th sem, VTU Belagavi, VCET Puttur, AI&ML dept, 4 authors/USNs, guide, HOD, principal, AY 2026-27, Nov-2026 submission; internally consistent across titlepage/certificate/declaration/acknowledgement. ✓

---

## 6. Files required but NOT in the GitHub repo

> Repo root = `/home/kshku/Git/Swaraaha`; local/`origin`/`upstream` `main` all at `fcc10bf`.

### 6.1 Required to reproduce/run, gitignored (too large for a normal GitHub repo — decide how to handle)

| Path | Size | Why required | Status |
|---|---|---|---|
| `model/weights/` | **84 GB** (254 files) | All checkpoints referenced by `model/registry.json` and every eval report (`*_naug_best.pt` + `..._best.pt`). Without them nothing runs / no result reproducible. | gitignored (`.gitignore:26`), only `.gitkeep` tracked |
| `data/` | **32 GB** | Dataset (`combined_labels.csv`, `splits.json`, `sources.csv`, `train/`, `test/`, `val/`, `cache/`, `labels/`). Source of the 3,715 test clips and all Boli/SEP-28K/UCLASS content. | gitignored, nothing tracked |

*(If weights/data must live in GitHub: consider Git LFS, or a separate release asset / archive; otherwise the report can still be read but results can't be re-verified.)*

### 6.2 Untracked but small / referenced by tracked code — should probably be committed

| Path | Size | Why |
|---|---|---|
| `model/evaluation/reports/arms/armB3_wavlm/` | 120 K | WavLM multitask ablation arm; referenced by tracked `model/evaluation/compare_ablation.py:18` |
| `model/evaluation/reports/arms/armB4_hubert/` | 40 K | HuBERT ablation arm; referenced by `compare_ablation.py:19` |
| `model/evaluation/reports/arms/armI2_max_f3/` | 132 K | Focal-γ3 ablation arm; referenced by `compare_ablation.py:16` |
| `model/evaluation/reports/arms/armI3_bcepos/` | 44 K | BCE-with-pos-weight ablation arm; referenced by `compare_ablation.py:17` |

All other `arms/arm0X*` and the whole `arms_noaug/` tree are tracked — only these four are untracked.

### 6.3 Untracked artifacts — your call (reference material / deliverables, not needed for the build)

| Path | What it is | Notes |
|---|---|---|
| `docs/MajorProjectReport.pdf` | Compiled project report | Could be added as a release artifact |
| `2501.15877v3.pdf` | Reference paper PDF | Optional |
| `DADS_final_edit.pdf` | Reference paper PDF | Optional |
| `papers/` (root, ~20 PDFs + `Archive.zip` + `__MACOSX/`) | Downloaded literature for bibliography | Optional; `__MACOSX/` junk should never be committed |
| `docs/architecture*.dot/.png`, `docs/pipeline*.dot/.png` | Source diagrams for the report's architecture figures | Optional (report uses the already-tracked `docs/report/assets/*.png`) |
| `Modelfile` | Ollama modelfile (`FROM qwen3-coder:30b`) | Unrelated to the project — exclude |
| `session-ses_008d.md`, `session-ses_f42f.md` | Agent session logs | **Do not commit** |

### 6.4 Presence caveat discovered during review
- `model/evaluation/reports/cnn_localizer_report.json` (tracked, stale source) references `model/weights/localizer_best.pt`, which **does not exist** anywhere on disk — further evidence that the CNN-localizer story in the report must be rebuilt from `arms_noaug/armL1_cnn_loc` (registry `cnnloc_..._naug_best.pt` weights **do** exist on disk, just untracked).

---

## 7. Recommended fix order
1. **Blocking structural (do first):** C9 (Ch1 comment block), C10 (Ch2 comment block), M3 (Ch2/Ch3 duplication decision).
2. **Results re-baseline to no-aug (C1–C8, M5, M7, M10):** rewrite Tables 4.1–4.4 + all in-text/ch7 numbers from `comparative_study_report.json` + `arms_noaug/`; regenerate figures and confusion matrices from the no-aug family.
3. **Consistency/accuracy edits:** M1, M2, M4 (Table 6.x refs), M6, M8, M9, M11–M13, plus all Minor items.
4. **Optional housekeeping:** the four ablation arms `armB3_wavlm`/`armB4_hubert`/`armI2_max_f3`/`armI3_bcepos` in `model/evaluation/reports/arms/` are now committed (branch `fix/eval-arm-reports`, commit `f124d48`); decide on weights/data/PDFs (6.1, 6.3).
5. Recompile `report.pdf` via Typst and verify rendering before submission.

---

## 8. Commands for the fixing agent

Report root: `docs/report/` (repo root `/home/kshku/Git/Swaraaha`). Typst binary is installed (`/usr/bin/typst`).

### 8.1 Regenerate the report PDF
```bash
typst compile docs/report/report.typ docs/report/report.pdf
```
Check `docs/report/frontmatter/toc.typ`, `meta.typ`, `lib.typ` are all imported by `report.typ` (they are). Verify the C9/C10 sections actually render after the comment-block fixes.

### 8.2 Regenerate the four result figures from NO-AUG data (M7)
Root cause: `model/evaluation/presentation.py` hardcodes AUG paths at the top of the file:
```python
ARM02_TEST = "model/evaluation/reports/arms/arm02_mt_w2v2_frz3/test/multitrack_report.json"
ARM02_BOLI = "model/evaluation/reports/arms/arm02_mt_w2v2_frz3/boli/multitrack_report.json"
ARML2_TEST = "model/evaluation/reports/arms/armL2_w2v2_loc/test/wav2vec2_localizer_report.json"
ARML2_BOLI = "model/evaluation/reports/arms/armL2_w2v2_loc/boli/wav2vec2_localizer_report.json"
```
Repoint those four constants to the `arms_noaug/` equivalents (i.e. `.../arms_noaug/arm02_mt_w2v2_frz3/...` and `.../arms_noaug/armL2_w2v2_loc/...`), then run:
```bash
python -m model.evaluation.presentation --output_dir model/evaluation/reports/presentation
```
Copy the four PNGs (`classifier_test_f1.png`, `classifier_test_auc.png`, `classifier_boli_f1.png`, `localizer_metrics.png`) into `docs/report/assets/` (overwrite the stale ones referenced at `chapter6.typ:65,67,114,151`).

### 8.3 Regenerate the five confusion matrices from NO-AUG (M6)
Source-of-truth matrices are already tracked in `model/evaluation/reports/arms_noaug/arm01_5x_w2v2/test/*_confusion_matrix.png` (one per class). Copy those five PNGs over `docs/report/assets/*_confusion_matrix.png`. If re-rendering from scratch is preferred, regenerate via the documented eval tool:
```bash
python -m model.evaluation.evaluate --model_type classifier --class_name <class> \
    --model_path model/weights/<class>_<fingerprint>_naug_best.pt --data_dir data
```
(class ∈ `prolongation, block, soundrep, wordrep, interjection`; fingerprint per `model/registry.json`.) Requires local `data/` + `model/weights/` — both gitignored, see §6.1.

### 8.4 Verify swapped numbers against ground truth
No-aug GT is machine-readable. Spot-check macro F1 from the CLI:
```bash
python3 -c "
import json
r = json.load(open('model/evaluation/reports/arms_noaug/arm02_mt_w2v2_frz3/test/multitrack_report.json'))
print('test macro_f1:', r['macro_f1'], '| tuned:', r['macro_f1_tuned'])
"
```
Cross-check any arm/cell you rewrote against `model/evaluation/reports/comparative_study_report.json` (`augmentation: false` blocks) or `model/evaluation/compare_ablation.py` output (`python3 model/evaluation/compare_ablation.py`). After the arms commit (§6.2) the script no longer prints `n/a` for the four new arms' val columns.

### 8.5 Dataset/severity notes
- Boli language list (Hindi/Kannada/Telugu/…) is verified only from `projectboli/Project_Boli_Dataset` externally; do not over-specify in prose.
- Severity boundaries from code (`shared/reporting/report_builder.py`): `>=2` mild, `>=5` moderate, `>=15` severe (note `>=` not `>`).