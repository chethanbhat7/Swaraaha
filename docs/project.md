# PROJECT.md — Swaraaha

> Context file for AI coding assistants (Antigravity, Claude Code, Cursor, etc).
> Read this fully before making changes. This is a **fresh project, built
> from scratch** — not a fork or continuation of any prior repo. Treat
> anything not explicitly described here as **not yet built**, not as
> "check the code for it."
> Update this file whenever architecture, conventions, model design, or
> active work changes. This is the persistent memory across sessions —
> nothing here should be assumed stale just because a session ended.

## 1. What this project is

Swaraaha is a speech dysfluency (stuttering) detection tool. It analyzes a speech
recording and does two things:
1. **Classifies** which type(s) of dysfluency are present
   (`prolongation`, `block`, `soundrep`, `wordrep`, `interjection`).
2. **Localizes** — pinpoints *where in the audio* a dysfluency occurs, not
   just that it occurred somewhere.

It ships as both a **web app** and a **desktop app**.

## 2. Top-level structure

```
Swaraaha/
├── frontend/     # Web app UI
├── backend/      # Web app API / server
├── models/       # ML models, training code, weights — shared source of truth
└── app/          # PyQt5 desktop app
```

- `frontend/` + `backend/` together make up the web app.
- `app/` is a separate, independently-runnable **desktop app** (PyQt5).
  It is not a legacy leftover to be replaced — it's a maintained, parallel
  interface to the same models.
- `models/` is the single source of truth for model architecture, weights,
  and training/eval code. Both `app/` and `backend/` should load from here
  rather than each keeping their own copy of model code.

## 3. Tech stack

**Not yet fixed in code — these are working defaults based on prior stack
experience. Confirm before treating as final, but assume these unless told
otherwise:**
- **Frontend:** React
- **Backend:** FastAPI (Python) — natural fit since it can share process
  space / imports with the PyTorch model code in `models/`
- **Desktop app (`app/`):** PyQt5
- **ML:** PyTorch, plus `transformers` (for Wav2Vec 2.0), `librosa`
- **Containerization:** Docker (see §6)
- **Deployment:** Render (see §6)

## 4. Model architecture

There are **two independent model pipelines**, run separately, whose
outputs are shown together rather than merged into one score:

### 4a. Classification pipeline — "what kind of stutter"
- **Five binary classifiers**, one per dysfluency class: `prolongation`,
  `block`, `soundrep`, `wordrep`, `interjection`.
- Each classifier is based on **Wav2Vec 2.0**, fine-tuned per class.
- A **hybrid model** sits on top of these five and combines their outputs
  into a final classification result. Combination strategy (e.g. weighted
  ensemble, learned meta-classifier, stacking) is **not decided yet** —
  flag this as an open design decision if asked to implement it.

### 4b. Localization pipeline — "where in the audio it happened"
- A **CNN-based image model** that runs convolutional kernels over the
  audio's **spectrogram image** to detect irregularities in the waveform
  and pinpoint the specific time location of a stutter event.
- This is treated as image processing (spectrogram-as-image), not raw
  waveform/sequence modeling.
- **Active exploration:** also investigating whether comparable
  localization can be achieved **without image processing** — e.g. running
  models directly over audio transcription/sequence features instead of a
  spectrogram image. This is a parallel approach being evaluated alongside
  the CNN one, not a settled replacement. If asked to build "the
  localization model," check which approach (CNN-image vs.
  transcript/sequence-based) is the one intended before assuming.

### How the two pipelines relate
They are **independent**: the hybrid classifier gives "this audio contains
X type of dysfluency," the localization model separately gives "at this
point in the audio." Results are presented together to the user, not
fused into a single model or score. Do not build tight coupling between
them (e.g. one depending on the other's output as input) unless explicitly
asked to.

## 5. Infrastructure

- **Docker:** used to containerize the app for scaling — most likely
  separate containers/services for `backend` (API) and `models` (inference),
  possibly `frontend` as a static build served separately. Exact
  docker-compose / multi-service layout not decided yet.
- **Deployment:** **Render** is the target platform for the web app
  (frontend + backend). The desktop app (`app/`) is not deployed anywhere —
  it stays a local install.

## 6. Conventions

- **Commit messages:** `[TYPE] message`, types: `[FIX]`, `[ADD]`, `[DOCS]`,
  `[MNT]` (refactor/maintenance), `[TEST]`.
- Keep `models/` framework-agnostic where possible so both `backend/`
  (FastAPI) and `app/` (PyQt5) can import from it without pulling in
  web-only or desktop-only dependencies.

## 7. Known constraints / things to be careful about

- **Two separate "app" concepts exist**: `app/` (desktop) is a real,
  distinct piece from the web app (`frontend/` + `backend/`). Never assume
  "the app" means the web app by default — check context.
- The hybrid classifier's combination method is undecided — don't invent
  and hardcode a specific ensemble strategy as if it were settled; flag it
  as a design choice when it comes up.
- Localization approach (image/CNN vs. transcript/sequence-based) is under
  active comparison, not finalized — same caution as above.
- No dataset, trained weights, or scaffolding exist yet — this is a ground-
  up build. Treat every directory in §2 as starting empty.

## 8. Current state / open items

_(Keep this section updated across sessions — note what's in progress, what
was last worked on, and any known bugs or TODOs here.)_

- [ ] Repo scaffolding for `frontend/`, `backend/`, `models/`, `app/` —
      not yet started.
- [ ] Wav2Vec 2.0 per-class binary classifiers — not yet built.
- [ ] Hybrid combiner model — design + implementation open.
- [ ] CNN spectrogram-image localization model — not yet built.
- [ ] Explore non-image (transcript/sequence-based) localization as an
      alternative to the CNN approach.
- [ ] Web app (frontend/backend) — not yet scaffolded.
- [ ] Docker setup for scaling — not yet defined.
- [ ] Render deployment — not yet configured.
