# Swaraaha Model Plan

Design and experiment plan for the ML core. This is the working contract between
design intent and the week of experiments. Every experiment has a numeric gate;
every branch has a fallback chain. No dead ends.

---

## 1. Requirements (confirmed with product owner)

The **model API is the deliverable**. Apps (web/desktop) are demonstration
consumers — the output *shape* matters, not the apps.

- **API contract**: audio file of **any length** in → classification labels +
  confidence + localization results (so UIs can draw graphs and highlight
  things).
- **Two access levels**:
  - *Easy*: `analyze()` → labels + confidence.
  - *Power-user*: `analyze_raw()` → + logits etc.
- **Stable, opaque API**: the API completely abstracts model internals.
- **Localization is a hard MVP requirement, aiming for syllable-level.**
- **Multilingual**: prefer one model that handles multiple languages; the system
  must support adding languages over time.
- **Data**: only publicly available datasets (no collection).
- **Evaluation**: detection metrics (per-type AUROC/F1/PR) + localization
  metrics (frame + event) + **accuracy** (secondary metric in reports).

## 2. Current-state audit (what exists / needs update / new)

| Requirement | Status |
|---|---|
| Model API abstraction (registry) | **Exists** — `Classifier()`/`Localizer()`/`Transcriber()`/`ModelRegistry.run_all()` |
| Easy + power-user access | **Exists** — `analyze()` vs `analyze_raw()` (logits) |
| Input path/bytes/numpy | **Exists** — `load_audio_input` |
| Classification output (per-type label + conf + probs) | **Exists** — 5 binary outputs + `summary` |
| **Any-length audio** | **Needs update** — currently pads/truncates to `max_length_seconds=10.0`; no chunking |
| Localization output (regions for UI) | **Exists but type-agnostic** — `{start, end, confidence}`, no dysfluency type |
| Per-type localization | **New** — no mechanism links classification types to region positions |
| Syllable-level localization | **Partial** — CTC alignment produces word+syllable timestamps, but not annotated fluent/dysfluent |
| Auto transcription → annotation | **Needs update** — `text` currently optional; must be automatic |
| Multilingual | **Partial** — Whisper + `WHISPER_LANG_CODES` + `LanguageAdapterRegistry`; backbones + all data are English-only |
| Accuracy metric | **Exists in code**, not surfaced in reports — surface it |
| Detection + localization eval | **Exists** |

## 3. Target architecture

```
Input audio (any length)
    │
    ├─► Multi-label classifier (1 model, shared backbone + 5 heads)
    │     └─► per-type {label, confidence, logits} + summary
    │
    ├─► Generic localizer (1 model: dysfluent vs fluent frames)
    │     └─► candidate regions [{start, end, confidence}]
    │
    ├─► Transcriber (Whisper) + CTC alignment + syllable snap
    │     └─► words + syllables with timestamps
    │
    └─► Linker: regions ↔ type (from classifier) ↔ syllables
          └─► per-type annotated regions at syllable granularity
```

All behind the **unchanged registry API**. Consumers see no change.

**Diagram caveat:** the Transcriber → alignment → syllable branch above is
currently trustworthy **only** for scripted/reading input. Freeform speech
relies on this branch too (see the transcript-gap section below) and is
unproven until that gap closes.

### Decisions locked in

- **Classification: shared-backbone multi-label** (not 5 binaries). Per-class
  specialists only for classes that fail after measuring.
- **Backbone: wav2vec2-family API kept; `--model_name` ablation** —
  `facebook/wav2vec2-base` vs `facebook/wav2vec2-xls-r-300m` (XLS-R) vs
  `facebook/hubert-base`. Winner becomes default. XLS-R is the multilingual
  candidate (drop-in: same architecture, one-line swap, fingerprint absorbs it).
- **Localization: generic localizer + output-level type linking + syllable
  snap** (no per-type localizers — event-level supervision is data-starved).
- **Semi-supervised**: train localizer on UCLASS → pseudo-label SEP-28K →
  retrain. Data multiplier; optional, not required.
- **Accuracy**: surfaced in reports as secondary metric.
- **Scope (decided)**: freeform spontaneous speech is a production requirement,
  not just reading-exercise input. Syllable naming therefore cannot assume a
  known reference transcript — the Whisper transcript dependency is a real gap
  (see the section below), and per the fidelity probe the transcript is
  unreliable on disfluent audio (wordrep flags 0/500; soundrep recall ≈ noise).

### Type linking (unvalidated hypothesis — how regions get their dysfluency type)

Regions from the generic localizer are type-agnostic `{start, end, confidence}`.
Nothing in the current pipeline assigns a dysfluency type to a region — this is
the biggest open design gap. Per-type localizers are **infeasible**: UCLASS
centers every dysfluent clip on a single event point where all of its types
co-occur (e.g. `M_0030…` has Block *and* SoundRep both at 1.25–1.75 s), so no
training data shows two types at different positions.

Intended mechanism — types are assigned at **output level** from the
multi-label classifier's own temporal signal:

1. Generic localizer → candidate regions (*where*).
2. Per-class saliency maps from the classifier's wav2vec2 backbone
   (attention-pooling weights or gradient-CAM over time frames) → *which* type
   is active where.
3. Type per region = `argmax` over classes of class saliency averaged over that
   region's frames.

**Risk:** per-class attention may not be spatially class-discriminative (the
classifier may attend to confirm *presence*, not *identity*). This is exactly
why the fallback chain (classifier-attention CAM → classification +
forced-alignment only) exists — see Branch 1.

**Status: HYPOTHESIS, not a result.** The "primary" label is a design note, not
a validated mechanism. Per review, it must be probed — saliency maps on clips
with real region+type ground truth (UCLASS precise event intervals) — before
being trusted in the API. The probe needs the classifier backbone's attention
to be spatially class-discriminative, so it realistically runs after C0 (the
shared-backbone classifier exists), not before.

**Probe ground truth — Boli (arXiv:2501.15877v3, PDF in repo root).** Boli is
the better probe target than UCLASS: it has **word-level annotations with
timestamps**, one annotated stutter event per clip ("every file contains a
single stuttered word"), and is multilingual (Hindi/Telugu/Bengali/Marathi/
Assamese) with **both read and spontaneous** speech — matching the freeform
scope decision. It is open access but **audio is not downloaded yet** (acquire
before the probe). Caveats: tiny per-class counts (SR=140, B=70, PR=41, WR=21,
IN=8), so any Boli validation must report per-class counts, not just F1. The
paper's SEP-28K→Boli cross-dataset transfer is non-degenerate (a classical
MFCC-averaged baseline still gets signal, Table V), which legitimizes Boli as a
transfer validation target for the linker.

### Freeform speech: the transcript gap (must close for full scope)

`align_with_syllables` (model/localization/ctc_alignment.py:193) requires
`text`; syllable naming is a linguistic labeling step (LanguageAdapter splits
word strings), not a temporal one. For scripted/reading input the reference
passage supplies the text and the path works. For **freeform spontaneous
speech there is no reference text**, so the transcript must come from Whisper —
and the fidelity probe showed that transcript is unreliable on disfluent audio
(sound-reps smoothed to 1/100, blocks to 0/100; wordrep never flagged because
dedup runs before flagging).

Two candidate paths (decide by a targeted probe, not upfront):

- **Path A — stutter-aware ASR:** fine-tune Whisper(-tiny) on stuttered speech
  (StutterTTS synthetic + real disfluent clips), re-run the fidelity probe on
  the fine-tuned model, and only then trust syllable names from alignment.
  Smallest delta: reuses `align_with_syllables` unchanged; restores a (better)
  transcript dependency.
- **Path B — text-free segmentation:** phone-level CTC from wav2vec2/HuBERT
  with **no text reference** → phone sequence + timestamps → syllabify by
  sonority/onset-max rules → syllable spans straight from audio. No transcript
  dependency at all (consistent with "transcript can't be trusted"). Risks:
  phone recognizer trained on fluent speech degrading on stutters; syllables
  are phonetic, not lexical (no word identity on freeform).

Recommendation: **Path A first** (smaller delta, reuses everything, StutterTTS
provides labeled training data); Path B only if A's probe still fails.

### Known blockers already found (root-caused)

- **Localizers train to F1=0.** Root causes:
  1. `AugmentedDataset` ran the **waveform** augmentor on the CNN localizer's
     spectrogram input → destroyed input (measured corr ≈ −0.11).
  2. For the wav2vec2 localizer, waveform augmentors shift/resample audio but
     not the frame labels → misaligned targets.
  3. wav2vec2 loss has **no `pos_weight`** (rare positive frames ≈ 5%) → model
     predicts low everywhere; predictions never cross 0.5.
  4. Early stopping on **frame F1@0.5** is a dead metric for rare positives
     (model improves loss/AUROC yet never crosses 0.5 → looks like F1=0).
- Fixes: `SpectrogramAugmentor` (SpecAugment masking — label-safe by
  construction), label-aligned `AudioAugmentor` for wav2vec2 (resample/roll the
  frame mask in lockstep), `pos_weight` on w2v2 loss, threshold-free early
  stopping, and a `--augmentation` on/off flag for a cheap ablation.

---

## 4. Experiment decision trees

### Principles

1. **Every experiment has a numeric gate** — pass/fail is per-class or macro
   F1/AUROC on held-out data.
2. **Cheapest experiments first** — localizer run ≈ 30–60 min (3k UCLASS
   clips); classifier run ≈ 2–4 h (29k clips). Fail cheap.
3. **One variable per step** — never change architecture + data + loss together.
4. **Fallback chain on every branch** — nothing has a dead end.

### Branch 1 — Localizer

```
L0  Fix augmentation bug (SpectrogramAugmentor + label-aligned w2v2)
    │  Gate: full test suite green (correctness gate, not a model metric)
    ▼
L1  CNN localizer retrain on UCLASS (--sources uclass)
    │  Gate: frame F1 > 0.10 on UCLASS val
    │
    ├── PASS → L2
    │
    └── FAIL → check labels first (visualize masks — is the centered interval
              landing on the stutter?), then, one at a time:
              L1a  widen UCLASS intervals (1.5s ± 0.25 may miss the event)
              L1b  sweep pos_weight (currently fixed 5.0)
              L1c  train WITHOUT augmentation (--augmentation off) — isolate
                   whether augmentation still hurts
              if still F1 ≈ 0 →
    ▼
L2  wav2vec2 localizer retrain on UCLASS (same data, different arch)
    │  Gate: frame F1 > 0.10
    │
    ├── PASS → L3 (also: does CNN improve with the same fixes? pick best arch)
    │
    └── FAIL → LOCALIZATION FALLBACK CHAIN (stop training localizers):
         CNN → wav2vec2 → classifier-attention (CAM on classifier hidden
         states; no localizer at all) → classification + forced-alignment only
         (assume region ≈ aligned syllable spans of detected-type words).
         This still satisfies the MVP API contract with weaker spatial
         precision.
    ▼
L3  Semi-supervised: pseudo-label SEP-28K clips with the L2 localizer, retrain
    on UCLASS + pseudo-labels
    │  Gate: frame F1 on UCLASS val > L2 (data multiplier must actually help)
    │
    ├── PASS → keep as final localizer
    └── FAIL → keep L2 (semi-supervision is optional, not required)
```

### Branch 2 — Classifier

```
C0  Build + train multi-label classifier (shared backbone, 5 heads,
    wav2vec2-base)
    │  Gate: macro F1 > 0.36 (current 5-binary baseline) AND every class > 0.2
    │
    ├── PASS → C2
    │
    └── FAIL → diagnose WHICH classes fail:
         │
         ├── rare classes only (soundrep/wordrep) →
         │     C1a  focal loss γ sweep / class-weighted loss
         │     C1b  oversampling / weighted sampler
         │     C1c  retrain WITHOUT augmentation (is augmentation hurting?)
         │       Gate: rare-class F1 > 0.2
         │       Note: WR imbalance-sensitivity is independently confirmed by
         │       the Boli paper (Table V: RF 0.94 balanced → 0.32 imbalanced,
         │       the steepest fall of any class on every model) — C1a/b is
         │       expected to be necessary, not hypothetical.
         │
         └── all classes fail → likely data/config, not class-specific:
               C1d  verify labels on a sample, try cross_entropy loss, longer
                    freeze schedule
               C1e  if still failing → per-class binary specialists (old
                    design), retrain only problem classes
    ▼
C2  Backbone ablation (SAME data, SAME script, only --model_name changes)
    wav2vec2-base  vs  XLS-R-300m  vs  HuBERT-base
    │  Gate: per-class AUROC/F1 table. XLS-R wins if multilingual is
    │        equal-or-better (it is the multilingual-aligned choice).
    │        Tie rule: XLS-R within ~2–3% macro F1 of best → take XLS-R.
    │
    └── Pick winner as registry default.
    ▼
C3  Per-class escalation — any class still below F1 0.2 after C2:
    train that class as a binary specialist
    │  Gate: that class's F1 improves vs multi-label's
    │
    └── FAIL → keep multi-label result (specialists didn't help — measured)
    ▼
C4  DADS CNN models as FINAL fallback only if everything above underperforms
    them (last resort).
```

---

## 5. Week budget with contingencies

| Run | Cost | Fires when |
|---|---|---|
| L0 augmentation fixes | — | always |
| L1 CNN retrain | ~1 h | always |
| L1a/b/c recovery | ~1 h each | L1 gate fails |
| L2 w2v2 retrain | ~1 h | L1 passes or L1 recovery exhausted |
| L3 semi-supervised | ~2 h | L2 passes |
| C0 multi-label train | ~3 h | always |
| C1 a–e recovery | ~2–3 h each | C0 gate fails |
| C2 two ablate runs | ~6 h | C0 passes |
| C3 specialist(s) | ~2 h/class | C2 leaves a failing class |

Worst case ≈ 4 days of GPU; ~1 week of wall-clock with human tweaking between
runs. Day-7 buffer for tweaks, docs, and registry updates.

### Week skeleton

| Day | Work | Gate |
|---|---|---|
| 1 | L0 augmentation fixes + tests; fix pre-existing test failure | full suite green |
| 2 | L1 retrain both localizers on UCLASS | **frame F1 > 0** |
| 3 | C0 build + train multi-label classifier | per-class F1 baseline, one model |
| 4 | C2 backbone ablation (XLS-R, HuBERT) | per-class AUROC/F1 table → pick |
| 5 | API updates: any-length chunking, type-linking, auto-transcription, accuracy in reports | API spec + tests |
| 6 | Full re-eval, reports, registry.json, README + debugging.md | evaluation summary + docs |
| 7 | Buffer: C3 specialist, L3 semi-supervised, tweaks | best achievable numbers |

---

## 6. Open items / assumptions (flag if wrong)

1. **Syllable-level = events snapped to syllables** (option A), not per-syllable
   classification (option B).
2. **Per-class escalation** and **backbone choice** are decided by the ablation
   numbers, not upfront.
3. Semi-supervised SEP-28K pseudo-labeling happens *after* a UCLASS localizer
   proves F1 > 0.
4. C2 tie rule: XLS-R within ~2–3% macro F1 of best English model → take XLS-R
   (multilingual goal wins ties).
5. Localization fallback chain (CAM → alignment-only) acceptable as last
   resort.
6. Freeform syllable-naming path (stutter-aware ASR vs text-free segmentation)
   is chosen by a targeted probe, not upfront; recommendation is Path A first.
