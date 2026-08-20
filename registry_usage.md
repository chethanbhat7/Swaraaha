# Model Registry Usage in Swaraaha Application

## Overview

The Swaraaha application uses models registered in `model/registry.py` and `model/registry.json` for all audio classification, localization, and transcription tasks. The system implements a clean API through which both the desktop application and webapp access these models.

## Registry Structure

The registry is defined in two files:
1. `model/registry.py` - Core logic for loading, managing, and interfacing with models
2. `model/registry.json` - Configuration listing paths to various model weights

The registry supports multiple model types:
- Classification models (prolongation, block, soundrep, wordrep, interjection)
- Multitask classifiers (shared backbone approach)
- Localization models (CNN and wav2vec2-based)
- Transcription support via Whisper models

## Combiner (localize + classify fusion)

`ModelRegistry.run_all` fuses localizer regions with the multitask classifier's
per-frame per-class saliency into a `combined` key:

```python
m = ModelRegistry()
result = m.run_all("recording.wav", text="the cat sat")
combined = result["combined"]
# {regions: [{start, end, confidence, classes, primary_type, severity,
#             syllables[]}], audio_duration, total_stutters}
```

Per-region `classes` mirrors the classification output format
(`{label, confidence, prob_present, prob_not_present}`), `primary_type` is the
class with the max `prob_present`, `severity` is a `null` placeholder in v1, and
`start`/`end` snap to enclosing syllables when text is provided. Fusion logic
lives in `model/combiner.py` (`combine_regions`, `mismatch_rate`); saliency
comes from `MultiTaskClassifier.saliency` (per-frame CAM-style probs).

To measure the region↔saliency mismatch rate over an eval split (informs the
"no saliency synthesis in v1" decision):

```bash
python -m model.evaluation.probe_combiner --data_dir data/test --max_length_seconds 3
```

## Desktop Application Usage

In the desktop application (`app/` directory), model usage follows this pattern:

### Key Components

1. **ModelRunner** (`app/core/model_runner.py`)
   - Main interface for executing model operations
   - Imports and instantiates `MultiTaskClassifier` and `Localizer` from `model.registry`
   - Provides high-level analysis functions that combine classification, localization, and transcription
   - Uses lazy loading for models to improve startup performance

2. **Main Application Flow** (`app/ui/main_window.py`)
   - When user clicks "Analyze" button, it starts an `AnalysisWorker` thread
   - The worker calls `ModelRunner.analyze()` which orchestrates:
     - Multi-task classification using `MultiTaskClassifier`
     - Localization using `Localizer("cnn")`
     - Transcription using `AudioTranscriber`

### Usage Patterns

```python
# In app/core/model_runner.py
from model.registry import MultiTaskClassifier, Localizer

class ModelRunner:
    def _get_classifier(self):
        if self._classifier is None:
            self._classifier = MultiTaskClassifier()
        return self._classifier

    def _get_localizer(self):
        if self._localizer is None:
            self._localizer = Localizer("cnn")  # CNN localizer specifically
        return self._localizer

    def _classify(self, audio: np.ndarray) -> dict:
        # Calls self._get_classifier().analyze(audio_np)
        pass
    
    def _localize(self, audio: np.ndarray) -> list:
        # Calls self._get_localizer().predict(spec, threshold=0.3)
        pass

    def analyze(self, audio: np.ndarray, language: str = "english") -> dict:
        # Orchestrates all three operations
        classifications = self._classify(audio)
        localizations = self._localize(audio)
        transcription = self.transcribe(audio, localizations=localizations, language=language)
        return {
            "classifications": classifications,
            "localizations": localizations,
            "transcription": transcription,
        }
```

### How Registry API Is Used

1. **MultiTaskClassifier** is used for classification of all dysfluency types
2. **Localizer** with "cnn" type is used for detecting dysfluency regions  
3. **Transcription** uses Whisper pipelines (not directly from registry but integrated)

The registry paths in `model/registry.json` are resolved using `_resolve_path()` function which handles both absolute and relative paths correctly.

## Key Integration Points

1. **Lazy Loading**: Models are loaded only when needed through the `_load()` methods in `Classifier`, `Localizer`, and other registry classes
2. **Threshold Management**: Both default and configurable thresholds are used through registry configuration
3. **Error Handling**: All model operations are wrapped in try-catch blocks to prevent crashes and provide error information

## Reference Usage

The models are imported and used in the following locations:

- `app/core/model_runner.py`: Contains the core logic that uses `MultiTaskClassifier` and `Localizer`
- `app/core/transcription.py`: Uses Whisper models (external to registry)
- `app/ui/main_window.py`: Coordinates the analysis flow using the `ModelRunner`

This architecture allows for clean separation between UI code and model logic while providing a consistent API for accessing all registered models.
