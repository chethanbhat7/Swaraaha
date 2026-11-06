# Documentation

Index of documentation and deliverables in this repository.

## Report (major project)
- `report/` — project report (Typst). Key entry: `report/report.typ`. Includes `chapters/`, `contents/`, `frontmatter/`, `assets/`, `lib.typ`, `meta.typ`, and `bibliography.typ`.
  - `report/report.html` / `report/report.pdf` may be present as compiled/exported outputs when generated.

## Papers
- `paper/` — paper source (`paper.typ`, `references.bib`).
- `papers/localization/` — localization paper source (`paper.typ`, `references.bib`).
- `papers/comparison-study/` — comparative study (Typst: `paper.typ`, `references.bib`).
- `papers/comparison-study-latex/` — comparative study (LaTeX: `paper.tex`, `references.bib`, `figures/`, and related build files as present).

## Diagrams
- `diagrams/` — Graphviz/DOT sources and rendered outputs. See [`diagrams/README.md`](diagrams/README.md) for figure details and rendering commands.

## Project context
- `project.md` — project and architecture context notes. Use alongside the root [`README.md`](../README.md) and code/registry for the most current behavior.

## Local-only (not tracked)
- `superpowers/` — local specs/plans (ignored by git).

For model, data, training, and evaluation details, see the `README.md` files under [`model/`](../model/) and the repository root [`README.md`](../README.md).
