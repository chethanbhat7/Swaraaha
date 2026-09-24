# Swaraaha — Diagrams

Professional diagrams of the Swaraaha speech-dysfluency detection and
localization system, drawn with Graphviz (DOT) in the style of an
engineering report.

## Contents

| # | Source (DOT) | Rendered |
|---|---|---|
| 1 | `01-system-flowchart.dot` | System flowchart — end-to-end analysis of a speech recording, showing the five dysfluency classes in the output |
| 2 | `02-workflow-diagram.dot` | Detailed workflow — data-to-deployment pipeline as ten concrete steps in a serpentine layout, with a retrain/tune feedback loop |
| 3 | `03-use-case-diagram.dot` | Use case diagram — the single User actor and the essential capabilities of the system boundary |
| 4 | `04-sequence-diagram.dot` | Sequence diagram — Web-app runtime interaction between the frontend, backend, and ML model components |

These are deliberately reduced to the **essential blocks** so the labels
stay legible when the figures are scaled into a report page. Natural
sizes (Graphviz coordinates, inches):

| Diagram | Width | Height |
|---|---|---|
| 01 System flowchart | 9.6 | 12.7 |
| 02 Workflow | 9.8 | 6.1 |
| 03 Use case | 5.0 | 9.8 |
| 04 Sequence | 10.4 | 11.5 |

Diagram 01 is portrait. Diagram 02 uses a serpentine (snake) layout —
four steps down, a break to the right, four steps up, a second break to
the right, then the final two steps down — so it stays close to a
landscape report block. Diagram 04 is the widest; scale it to the full
text width (or give it a full page) for best legibility.

All diagrams are rendered as both SVG and PNG in `rendered/`:

```
docs/diagrams/
├── 01-system-flowchart.dot
├── 02-workflow-diagram.dot
├── 03-use-case-diagram.dot
├── 04-sequence-diagram.dot
├── rendered/
│   ├── 01-system-flowchart.svg / .png
│   ├── 02-workflow-diagram.svg  / .png
│   ├── 03-use-case-diagram.svg  / .png
│   └── 04-sequence-diagram.svg  / .png
└── README.md
```

## Rendering

Graphviz version used: `dot - graphviz version 16.1.0 (20260904.0139)`.

```sh
dot -Tsvg docs/diagrams/01-system-flowchart.dot -o docs/diagrams/rendered/01-system-flowchart.svg
dot -Tpng docs/diagrams/01-system-flowchart.dot -o docs/diagrams/rendered/01-system-flowchart.png
```

Repeat for `02-workflow-diagram`, `03-use-case-diagram`, and
`04-sequence-diagram`.

## Architecture Summary

Swaraaha detects and localizes five types of speech dysfluency —
prolongation, block, sound repetition, word repetition, and
interjection — in audio recordings, and produces a clinical-style
report. The system ships as both a React web application
(`frontend/` + FastAPI backend in `backend/`) and a PySide6 desktop
application (`app/`). Both consume the shared `model/` package through
a model registry (`model/registry.json`).

The model pipeline (as depicted in the diagrams) is:

1. **Classification** — a single Wav2Vec2-base backbone with five
   per-class heads (the multi-head classifier, `MultiTaskRunner`) runs
   one forward pass and emits per-class present/absent results,
   applying per-class thresholds from the registry.
2. **Localization** — a Wav2Vec2-base temporal localizer proposes
   dysfluent regions as start/end timestamps.
3. **Transcription** — a Whisper tiny transcriber (per language:
   English/Kannada/Hindi) produces the transcript with word-level
   timestamps and lightweight stutter flagging.
4. **Fusion** — localizer regions are fused with per-frame per-class
   head saliency (`model/combiner.py`); a saliency fallback can
   generate regions when the localizer returns none.
5. **Scoring & reporting** — a stutter index and severity level
   (Fluent/Mild/Moderate/Severe) are computed, and a Typst-built PDF
   clinical report is generated (`shared/reporting/`).

## Purpose of Each Diagram

- **01 — System flowchart:** the runtime analysis of a single
  recording, from speech input through preprocessing, the shared-backbone
  ML engine, and the fused analysis output, which branches into one
  block per dysfluency class (prolongation, block, sound rep, word rep,
  interjection) before the clinical report.
- **02 — Workflow diagram:** the offline development pipeline as ten
  concrete steps in a serpentine layout — datasets (SEP-28K/UCLASS/
  Project Boli), merge & label normalisation, cleaning, feature
  extraction, balancing & augmentation, dataset split, multi-head model
  training, checkpoints, evaluation & threshold tuning (→ registry.json),
  and deployment — with the retrain/tune feedback loop from evaluation
  back to training.
- **03 — Use case diagram:** the single runtime actor (User) and the
  essential capabilities of the system boundary.
- **04 — Sequence diagram:** the Web-app runtime interaction for a full
  analysis and PDF report download; internal backend steps (region
  fusion, severity, PDF generation) are shown as self-messages.

## Assumptions & Limitations

- Per request, the diagrams depict only the **multi-head, shared
  backbone** classifier configuration (Wav2Vec2-base, five heads); the
  per-class single classifiers and CNN variants are intentionally not
  shown even though the codebase also contains them.
- The original visual style of the reference report figures was not
  directly available; the figures use the specified academic palette
  (light blue node fills `#83C5E5`, black outlines, serif
  `Times New Roman`).
- No metric values are fabricated. Per-class thresholds quoted in the
  diagrams come from `model/registry.json` (source of truth for the
  active configuration). Published README evaluation numbers are not
  part of the diagrams.
- The figures are intentionally high-level: implementation details that
  are not essential to the system (per-class thresholds, the localizer's
  saliency fallback, dataset download mechanics, experiment arms) are
  omitted for legibility. The underlying behaviour is still described in
  the Architecture Summary above.
- Sequence-diagram labels that span multiple participants may cross the
  grey lifelines of intermediate participants — normal for UML
  sequence diagrams and accepted here for readability.