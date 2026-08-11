# model/ Audit Ledger

Systematic audit of `model/` — findings, severity, and fix status.
Each fixed item is addressed by a failing test first (TDD) and lands as its own small commit.

## HIGH — silent wrong results

| # | Location | Bug | Status |
|---|----------|-----|--------|
| H1 | data/dataset.py:213-231, localization/wav2vec2_dataset.py:127-135 | `clean_audio` trims leading/trailing silence but frame labels keep the original timebase → every label is shifted late by the trim offset. Also train/serve skew: `Localizer.analyze` (registry.py:424) feeds the untrimmed spectrogram at inference. | OPEN |
| H2 | evaluation/full_evaluate.py:67-85 | `_eval_args()` never sets `full`; `evaluate.py` reads `args.full` → AttributeError → every classifier/localizer reports `"status": "error"`. full_evaluate is fully broken. | DONE |
| H3 | fingerprint.py:80 | `model_name_from_path` regex `_(\w+v2\w+)_` can't isolate the model short (`\w` matches `_`) → always returns `facebook/wav2vec2-base`; a `w2v2large` checkpoint loads the base architecture → shape RuntimeError. | OPEN |
| H4 | localization/ctc_alignment.py:34-54 | `Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")` — base is a pre-training model with a randomly initialized CTC head → alignment timestamps are garbage, silently. | OPEN |
| H5 | classification/__init__.py:138 | `from_pretrained` uses `strict=False`: compiled checkpoints (`_orig_mod.` prefixes) load with ALL keys dropped → silently random weights; also masks head-shape mismatches. | OPEN |

## MEDIUM

| # | Location | Bug | Status |
|---|----------|-----|--------|
| M1 | registry.py:234-237,307 | Extra classification registry entries (6th class) crash `Classifier()` / `run_all` (ValueError or thresholds KeyError). | OPEN |
| M2 | registry.py:239-265, classification/__init__.py:102 | `Classifier.predict`/`predict_all` ignore the `threshold` param (hardcoded 0.5); `analyze` honors registry thresholds → inconsistent API. | OPEN |
| M3 | evaluation/loader.py:90 | CNN eval hardcodes `n_mels=128`, ignoring the checkpoint's trained n_mels → silent quality degradation. | OPEN |
| M4 | evaluation/metrics.py:348-359,432-433 | All-negative perfect predictions → frame F1=0.0 while event acc/IoU=1.0 (contradictory report). | OPEN |
| M5 | evaluation/metrics.py:80-114 | Macro-F1 wrong on degenerate splits (all-positive/all-negative): perfect model scores 0.5. | OPEN |
| M6 | registry.py:492-515 | `run_all` catches only `FileNotFoundError` for classification/localization; other errors crash instead of `{"error": ...}`. | OPEN |
| M7 | evaluation/evaluate.py:68-69 | `--sweep_thresholds` is a dead flag for the localizer path. | OPEN |
| M8 | evaluation/metrics.py:260-305 | `find_optimal_threshold(metric="youden")` returns `(0.5, 0.0)` for worse-than-chance (never updates best); sweep range `(0.05,0.96,0.05)` disagrees with the printed sweep `(0.1,0.91,0.05)`. | OPEN |
| M9 | data/dataset.py:356-388 | Classification pickle cache key = (clip_id, label_signature) only; `sr`/`max_samples`/audio content changes reuse stale cache. | OPEN |
| M10 | data/merge.py:391-393 | Per-clip interval CSVs are never regenerated on re-merge without `--force`; combined_labels.csv always overwritten → classification/localization labels can disagree. | OPEN |
| M11 | data/download.py:45-46,158-159 | Partial downloads count as success (dir-exists guard); exit codes never propagate through setup.py. | OPEN |
| M12 | data/augmentation.py:36-46 | `_resample` semantics inverted (factor>1 decimates = speed-up) + always forces output to input length → tail truncation; stretch/pitch don't do what params claim. | OPEN |
| M13 | data/merge.py:236,197-198 | SEP-28K positives use `>0` annotator votes (single-annotator) instead of majority (>=2) → inflated positives; `int(EpId)` aborts whole merge on malformed rows. | OPEN |
| M14 | data/preprocessing.py:531-534 | `create_frame_labels` floors end_frame → final partial frame of an interval dropped (systematic off-by-one). | OPEN |
| M15 | data/augmentation.py:69 | `time_shift` hardcodes 16000 instead of `sample_rate`. | OPEN |
| M16 | data/status.py:34-38 | ZeroDivisionError on empty CSV; KeyError on missing columns. | OPEN |
| M17 | data/prepare.py:66-80 | All clips filtered → silently writes empty splits and exits 0. | OPEN |
| M18 | localization/wav2vec2_localizer.py:155-190 | `predict` scores zero-padding to 10s and emits regions up to 9.98s for short clips (false tail regions). | OPEN |
| M19 | registry.py:294,416 + preprocessing.py:95 | `Classifier.analyze`/`Localizer.analyze` crash on empty audio (`np.abs(audio).max()` on empty); Transcriber guards it. | OPEN |
| M20 | transcription.py:116,120 | `duration_sec = len/16000` hardcodes rate; path input with sample_rate=44100 gives wrong duration AND hands 44.1k audio to Whisper. | OPEN |
| M21 | transcription.py:76-83 | Forced-decoder/no-timestamps setup failure swallowed (`except: pass`) → wrong-language transcripts silently. | OPEN |
| M22 | localization/wav2vec2_dataset.py:82-107 | Missing header-only WAV guard (CNN dataset has it) → NaN waveforms → NaN training. | OPEN |
| M23 | localization/wav2vec2_localizer.py:67-97 | `temporal_attention` computed then discarded — dead params, no effect on logits. | OPEN |

## LOW

| # | Location | Bug | Status |
|---|----------|-----|--------|
| L1 | fingerprint.py:60-61 | `parse_fingerprint` `\w+` rejects `-`/`.` in data/model names → ValueError → reports lose hyperparameters. | OPEN |
| L2 | evaluation/loader.py:167 | `model_info_from_path` strips `_best/_final/_checkpoint/_log.csv/.pt` but not `_curves.png`. | OPEN |
| L3 | localization/ctc_alignment.py:133-138,167-173 | No CTC collapse for adjacent duplicate tokens; token→word mapping never checks the decoded text (character-count heuristic). | OPEN |
| L4 | localization/wav2vec2_dataset.py:161-162 | Frame mapping uses `int(sec/0.02)` float division vs multiply convention elsewhere → 1-frame shift for boundary values. | OPEN |
| L5 | transcription.py:182 | Whisper word "confidence" fabricated (0.9) — pipeline emits no per-word confidence. | OPEN |
| L6 | transcription.py:131-134 | Text-without-timestamps returns non-empty text + `words: []`, localization overlay silently skipped. | OPEN |
| L7 | transcription.py:113-118 | `transcribe(b"")` raises instead of returning the empty result. | OPEN |
| L8 | transcription.py:175-185 | Whisper word timestamps never clamped to duration_sec. | OPEN |
| L9 | localization/cnn_spectrogram.py:211 | Region end can overshoot audio duration (313 frames × 512/16000 = 10.016s). | OPEN |
| L10 | localization/cnn_spectrogram.py:181-185 | Batched 3-D `(B,n_mels,T)` input silently mis-interpreted (extra axis added). | OPEN |
| L11 | localization/language_adapter.py:268 | `_adapters` is shared class state mutated by register() — leaks across instances. | OPEN |
| L12 | registry.py:310 | `_run_all` "primary" can name a class that wasn't detected (max prob even when all below threshold). | OPEN |
| L13 | classification/__init__.py:124-139 | `from_pretrained` ignores stored `class_name` — loading a block ckpt as prolongation silently succeeds. | OPEN |

## Dead code (verified no callers)

- preprocessing.py: `audio_to_spectrogram`, `file_to_spectrogram`, `save_spectrogram`, `save_spectrogram_image`, `plot_waveform`, `debug_save_spectrogram`, `debug_generate_test_audio`, `create_frame_labels_from_samples`, `pad_audio_and_labels`, `normalize_rms`, `trim_silence_center`, `augment_audio`, `compute_snr`, `check_audio_quality`, `filter_audio_samples`, `compute_class_weights`, `oversample_minority`, `create_balanced_sampler`, `compute_pos_weight`, `normalize_spectrogram`
- training/utils.py: `update_registry_localizers`, `find_latest_localizer`
- training/train_localizer.py: `create_frame_loss_weights`
- training/train_classifier.py: `compute_pos_weight` (duplicate of preprocessing version)
- localization/ctc_alignment.py: `align_with_syllables` (never called)
- evaluation/summary.py:240-241: `sys.stdout.encoding` check does nothing
