# Documentation

Index of tracked documentation and deliverables in this repo. Authoring sources are kept here; compiled/exported outputs may also be present (no moves/renames done as part of cleanup — structure kept as-is).

## Report (major project)
- `report/` — project report (Typst). Key entry: `report/report.typ`. Includes `chapters/`, `contents/`, `frontmatter/`, `assets/`, `lib.typ`, `meta.typ`, `bibliography.typ`, and related files. Compiled/exported artifacts (e.g. PDFs/HTML/DOCX where present) are kept alongside sources per project workflow.

## Papers
- `paper/` — paper source (`paper.typ`, `references.bib`).
- `papers/localization/` — localization paper source (`paper.typ`, `references.bib`).
- `papers/comparison-study/` — comparative study (Typst: `paper.typ`, `references.bib`).
- `papers/comparison-study-latex/` — comparative study (LaTeX: `paper.tex`, `references.bib`, `figures/`, and build artifacts where tracked).

## Diagrams
- `diagrams/` — Graphviz/DOT sources and rendered outputs. See [`diagrams/README.md`](diagrams/README.md) for the figure list and rendering notes.

## Project context
- `project.md` — project/architecture/context notes (treat as reference; verify against current code/README when needed).

## Local-only (not tracked)
- `superpowers/` — local specs/plans (ignored by git).

For model/data/training/evaluation details, see the `README.md` files under `model/` and the repository root `README.md`.
