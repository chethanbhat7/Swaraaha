# `model/` — Swaraaha ML Core

The `model/` package contains Swaraaha’s ML code: classification, localization, transcription, data pipeline, training, evaluation, and the registry-driven runtime API used by the backend, desktop app, and scripts.

## Public API (recommended)

Use the top-level `model` API and registry-backed runners instead of importing model classes or loading checkpoints manually.

```python
import model

# Load defaults from model/registry.json (or pass explicit classifier/localizer)
model.init()

# Full analysis (classification + localization + transcription + fusion)
result = model.analyze("recording.wav", text="optional transcript", language="en")

# Individual steps
clf = model.classify("recording.wav", threshold=0.55)
loc = model.localize("recording.wav", text="optional transcript", language="en")
tr  = model.transcribe("recording.wav")
combined = model.fuse(loc, "recording.wav", text="optional transcript")
```

Lower-level registry runners (when you need finer control):

```python
from model.registry import ClassifierRunner, LocalizerRunner

clf = ClassifierRunner()                 # uses registry defaults/config
loc = LocalizerRunner()                 # e.g. wav2vec2/cnn per registry
```

Notes:
- Audio inputs are handled by the pipelines/runners (commonly file paths; some APIs also accept bytes/arrays).
- When a component is unavailable (e.g. missing localizer checkpoint), the registry/model API returns structured errors rather than requiring ad-hoc fallbacks.

## Registry configuration

- `model/registry.json` is the source of truth for active checkpoints, defaults, thresholds, and localization synthesis settings.
- To change the active model, update `registry.json` (no code changes required).
- Training scripts produce checkpoints; **registry selection is manual**.

## Package layout

- `classification/` — classifier models and inference helpers (single, multitask, CNN multitask variants).
- `localization/` — CNN spectrogram and Wav2Vec2 localizers.
- `registry/` — registry loading and runners.
- `transcription.py` — Whisper-based transcription API.
- `combiner.py` — fusion of localizer regions with classifier saliency.
- `data/` — dataset download/merge/prepare, preprocessing, augmentation. See [`data/README.md`](data/README.md).
- `training/` — training pipelines and utilities. See [`training/README.md`](training/README.md).
- `evaluation/` — evaluation CLIs and metrics. See [`evaluation/README.md`](evaluation/README.md).
- `config/` — defaults and configuration.
- `tests/` — model package tests.

## Common commands

```bash
python -m model.data.setup
python -m model.training.train
python -m model.evaluation.evaluate --help
```

For the project overview and app setup, see the repository root [`README.md`](../README.md).
