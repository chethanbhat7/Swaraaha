# Swaraaha — Diagrams

Diagrams of the Swaraaha speech-dysfluency detection and localization system, authored as Graphviz DOT sources with rendered SVG/PNG outputs.

## Contents

| # | Source (DOT) | Rendered | Description |
|---|---|---|---|
| 1 | `01-system-flowchart.dot` | `rendered/01-system-flowchart.svg`, `.png` | End-to-end runtime analysis flow; shows the five dysfluency classes in the output. |
| 2 | `02-workflow-diagram.dot` | `rendered/02-workflow-diagram.svg`, `.png` | Offline data-to-deployment workflow (serpentine layout, fixed coordinates). |
| 3 | `03-use-case-diagram.dot` | `rendered/03-use-case-diagram.svg`, `.png` | Runtime use cases and the primary user actor. |
| 4 | `04-sequence-diagram.dot` | `rendered/04-sequence-diagram.svg`, `.png` | Web-app runtime interaction (frontend, backend, model components). |
| — | `architecture.dot` | `rendered/architecture.svg`, `.png` | High-level model/pipeline architecture overview. |

```text
docs/diagrams/
├── 01-system-flowchart.dot
├── 02-workflow-diagram.dot
├── 03-use-case-diagram.dot
├── 04-sequence-diagram.dot
├── architecture.dot
├── rendered/
│   ├── 01-system-flowchart.svg / .png
│   ├── 02-workflow-diagram.svg / .png
│   ├── 03-use-case-diagram.svg / .png
│   ├── 04-sequence-diagram.svg / .png
│   └── architecture.svg / .png
└── README.md
```

## Rendering

Requires [Graphviz](https://graphviz.org/).

```sh
# 01 — System flowchart
dot -Tsvg docs/diagrams/01-system-flowchart.dot -o docs/diagrams/rendered/01-system-flowchart.svg
dot -Tpng docs/diagrams/01-system-flowchart.dot -o docs/diagrams/rendered/01-system-flowchart.png

# 02 — Workflow diagram (layout=neato with fixed coordinates; use neato -n2)
neato -n2 -Tsvg docs/diagrams/02-workflow-diagram.dot -o docs/diagrams/rendered/02-workflow-diagram.svg
neato -n2 -Tpng docs/diagrams/02-workflow-diagram.dot -o docs/diagrams/rendered/02-workflow-diagram.png

# 03 — Use case diagram
dot -Tsvg docs/diagrams/03-use-case-diagram.dot -o docs/diagrams/rendered/03-use-case-diagram.svg
dot -Tpng docs/diagrams/03-use-case-diagram.dot -o docs/diagrams/rendered/03-use-case-diagram.png

# 04 — Sequence diagram
dot -Tsvg docs/diagrams/04-sequence-diagram.dot -o docs/diagrams/rendered/04-sequence-diagram.svg
dot -Tpng docs/diagrams/04-sequence-diagram.dot -o docs/diagrams/rendered/04-sequence-diagram.png

# Architecture overview
dot -Tsvg docs/diagrams/architecture.dot -o docs/diagrams/rendered/architecture.svg
dot -Tpng docs/diagrams/architecture.dot -o docs/diagrams/rendered/architecture.png
```

## Architecture summary

Swaraaha detects and localizes five dysfluency types — `prolongation`, `block`, `soundrep` (sound repetition), `wordrep` (word repetition), and `interjection` — in speech recordings.

The active model/registry setup (see `model/registry.json`) includes:
- **Classification:** Wav2Vec2-based classifiers (default: single per-class classifiers; multitask and CNN-multitask variants are also registered), producing per-class results with configured thresholds.
- **Localization:** Wav2Vec2 and CNN localizers (default: `wav2vec2`), producing time regions.
- **Transcription/fusion (runtime):** Whisper-based transcription and region/saliency fusion via the shared `model/` API (`model.transcription`, `model.combiner`, `model.analyze`/`model.fuse`).

The web app (`frontend/` + `backend/`) and desktop app (`app/`) consume the shared `model/` package through the registry/API. Diagrams are simplified for readability; when diagrams and code disagree, treat the code and `model/registry.json` as the source of truth.
