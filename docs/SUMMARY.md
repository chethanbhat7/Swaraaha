# Swaraaha — Speech Dysfluency Detection System

## Complete Project Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement and Motivation](#2-problem-statement-and-motivation)
3. [System Architecture](#3-system-architecture)
4. [Core Machine Learning Pipeline](#4-core-machine-learning-pipeline)
5. [Classification Models](#5-classification-models)
6. [Localization Models](#6-localization-models)
7. [Fusion and Combiner](#7-fusion-and-combiner)
8. [Transcription Pipeline](#8-transcription-pipeline)
9. [Training Pipeline](#9-training-pipeline)
10. [Data Pipeline](#10-data-pipeline)
11. [Evaluation Framework](#11-evaluation-framework)
12. [Backend API](#12-backend-api)
13. [Web Frontend](#13-web-frontend)
14. [Desktop Application](#14-desktop-application)
15. [Shared Utilities](#15-shared-utilities)
16. [Deployment](#16-deployment)
17. [Dependencies and Setup](#17-dependencies-and-setup)
18. [File Structure Reference](#18-file-structure-reference)

---

## 1. Project Overview

**Swaraaha** is an end-to-end speech dysfluency (stuttering) detection system that analyzes speech recordings to:

1. **Classify** what types of dysfluency are present in the audio
2. **Localize** precisely where in the audio timeline each dysfluency event occurs
3. **Transcribe** the speech with word-level timestamps
4. **Fuse** classification and localization results into a unified output with severity scoring
5. **Generate** clinical PDF reports for speech-language pathologists

The system detects **five clinically recognized dysfluency types**:

| Dysfluency Type | Clinical Definition |
|---|---|
| **Prolongation** | Unnecessarily prolonged sounds (e.g., "ssssssnake") |
| **Block** | Complete stoppage of airflow/sound (silent pause before a word) |
| **Sound Repetition** | Repeating sounds/syllables (e.g., "b-b-ball", "ba-ba-ball") |
| **Word Repetition** | Repeating whole words (e.g., "I-I-I went") |
| **Interjection** | Filler sounds/words (e.g., "um", "uh", "like") |

### Technology Stack

| Component | Technology |
|---|---|
| **Frontend** | React 19 + TypeScript + Vite + Tailwind CSS v4 |
| **Backend** | FastAPI (Python 3.11) + Uvicorn |
| **Desktop App** | PySide6 (Qt 6) + sounddevice |
| **ML Framework** | PyTorch + Hugging Face Transformers |
| **Pre-trained Model** | facebook/wav2vec2-base (95M parameters) |
| **ASR** | OpenAI Whisper (tiny variant) via HuggingFace |
| **PDF Generation** | Typst |
| **Deployment** | Docker + Render |

---

## 2. Problem Statement and Motivation

### Why This Project Exists

Speech dysfluency assessment is a critical component of speech-language pathology. Currently, the process is:

1. **Manual**: A clinician listens to recordings, manually noting timestamps and types of dysfluencies
2. **Time-consuming**: A 5-minute recording can take 30-60 minutes to annotate
3. **Subjective**: Inter-rater reliability varies between clinicians
4. **Inaccessible**: Expert clinicians are scarce, especially in developing regions

### What Swaraaha Addresses

Swaraaha automates the detection and localization of stuttering events using deep learning, providing:

- **Objective measurement**: Consistent classification across all recordings
- **Temporal localization**: Precise timestamps for each dysfluency event (not just "is it present?")
- **Severity quantification**: A stutter index (percentage of speech time containing dysfluency)
- **Multi-language support**: English, Kannada, and Hindi
- **Clinical reporting**: Auto-generated PDF reports for documentation

### Research Significance

This project contributes to the field of speech processing by:

- Applying Wav2Vec 2.0 (a self-supervised speech representation model) to dysfluency detection
- Implementing a multi-task learning framework with shared backbone and per-class heads
- Developing a fusion mechanism that combines localization regions with per-class saliency maps
- Creating a complete end-to-end system from audio input to clinical report output

---

## 3. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   Web Frontend   │  │  Desktop App     │  │  REST API    │  │
│  │  (React/Vite)    │  │  (PySide6/Qt)    │  │  (FastAPI)   │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘  │
│           │                     │                    │          │
│           └─────────────────────┼────────────────────┘          │
│                                 │                               │
│                    ┌────────────▼────────────┐                  │
│                    │    model/ (ML Engine)    │                  │
│                    │  ┌──────────────────┐   │                  │
│                    │  │  Model Registry   │   │                  │
│                    │  │  (registry.py)    │   │                  │
│                    │  └────────┬─────────┘   │                  │
│                    │           │              │                  │
│                    │  ┌────────▼─────────┐   │                  │
│                    │  │  Classifier      │   │                  │
│                    │  │  (Wav2Vec2)      │   │                  │
│                    │  └──────────────────┘   │                  │
│                    │  ┌──────────────────┐   │                  │
│                    │  │  Localizer       │   │                  │
│                    │  │  (Wav2Vec2/CNN)  │   │                  │
│                    │  └──────────────────┘   │                  │
│                    │  ┌──────────────────┐   │                  │
│                    │  │  Transcriber     │   │                  │
│                    │  │  (Whisper)       │   │                  │
│                    │  └──────────────────┘   │                  │
│                    │  ┌──────────────────┐   │                  │
│                    │  │  Combiner        │   │                  │
│                    │  │  (Saliency Fusion)│  │                  │
│                    │  └──────────────────┘   │                  │
│                    └─────────────────────────┘                  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    shared/ (Report Builder)              │   │
│  │              Typst-based clinical PDF generation         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow (Web Pipeline)

```
1. User uploads/records audio on UploadPage
        │
        ▼
2. Frontend sends POST /api/analyze (FormData: audio + language)
        │
        ▼
3. Backend orchestrates 5 sequential operations:
        │
        ├──► Classification: MultiTaskClassifier.analyze()
        │       → Wav2Vec2 backbone → 5 binary heads → per-class {label, confidence}
        │
        ├──► Localization: Wav2Vec2Localizer.predict()
        │       → Wav2Vec2 backbone → temporal classifier → per-frame probabilities → regions
        │
        ├──► Transcription: Whisper ASR
        │       → word-level timestamps → hallucination collapse → stutter flagging
        │
        ├──► Severity: compute_severity()
        │       → stutter_index = (dysfluency_coverage / total_duration) × 100
        │
        └──► Fusion: combine_with_saliency()
                → multitask saliency × localizer regions → enriched regions with class scores
        │
        ▼
4. Response JSON: {classification, localization, transcription, severity, combined}
        │
        ▼
5. Frontend renders: severity badge, classification bars, timeline, waveform overlays
```

### Data Flow (Desktop Pipeline)

```
1. User records/loads audio in MainWindow
        │
        ▼
2. Background TranscriptionWorker runs Whisper ASR
        │
        ▼
3. User clicks "Analyze" → AnalysisWorker thread
        │
        ├──► ModelRunner.analyze(audio, language)
        │       ├── classify_audio_bytes() → MultiTaskClassifier
        │       ├── localize_audio_bytes() → Localizer("wav2vec2")
        │       ├── AudioTranscriber.transcribe() → Whisper + stutter alignment
        │       └── combine_with_saliency() → Combiner
        │
        ▼
4. Results displayed on AnalysisPage
        │
        ▼
5. User exports PDF → build_report_data() → generate_report_pdf() → Typst compilation
```

---

## 4. Core Machine Learning Pipeline

### Why Wav2Vec 2.0?

Wav2Vec 2.0 is a self-supervised speech representation model pre-trained on 960 hours of unlabeled speech (LibriSpeech). The choice of this backbone is motivated by:

1. **Transfer Learning**: Pre-trained on massive speech data, captures universal speech patterns
2. **Raw Waveform Processing**: No hand-crafted features (MFCC, etc.) needed; learns optimal representations
3. **Contextual Embeddings**: Transformer layers capture long-range temporal dependencies
4. **Proven Architecture**: State-of-the-art on many speech tasks (ASR, speaker verification, emotion recognition)

### Why Not Fine-Tune the Full Model?

Fine-tuning the entire Wav2Vec2 model for dysfluency detection risks:
- **Catastrophic forgetting**: Losing general speech representations
- **Overfitting**: Dysfluency datasets are small (hundreds to low thousands of samples)
- **Computational cost**: Full fine-tuning is expensive

**Solution**: Freeze the first 3-5 transformer layers (feature extractor) and fine-tune only the top layers + classification head. This preserves general speech features while adapting to dysfluency-specific patterns.

### Model Architecture Overview

```
Raw Audio (16kHz, mono)
        │
        ▼
┌───────────────────────────────────────┐
│        Wav2Vec2 Feature Encoder       │
│  (Convolutional layers, 7 layers)    │
│  Input: raw waveform                 │
│  Output: frame-level representations │
│  Frame rate: 50 Hz (20ms/frame)      │
│  Frame size: 320 samples             │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│      Wav2Vec2 Contextual Encoder     │
│  (Transformer layers, 12 layers)     │
│  Hidden dim: 768                     │
│  Attention heads: 12                 │
│  Output: (B, T, 768)                │
│  T = audio_length / 320              │
└───────────────────┬───────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐    ┌──────────────────────┐
│  Classifier  │    │     Localizer        │
│  (per-frame  │    │  (temporal attention │
│   mean-pool  │    │   + classifier)      │
│   + heads)   │    │                      │
└──────────────┘    └──────────────────────┘
```

---

## 5. Classification Models

### 5.1 Individual Binary Classifiers

**File**: `model/classification/__init__.py` (BaseWav2VecClassifier)
**File**: `model/classification/{prolongation,block,soundrep,wordrep,interjection}.py`

#### Architecture

```
Input: Raw audio waveform (B, 160000) — 10 seconds at 16kHz
        │
        ▼
Wav2Vec2ForSequenceClassification (HuggingFace)
  ├── Wav2Vec2Model (backbone)
  │     ├── FeatureExtractor: 7 conv layers (Conv1d + LayerNorm + Dropout)
  │     │     Input: raw waveform
  │     │     Output: (B, T, 768) where T ≈ 499 for 10s audio
  │     │     Subsampling factor: 320 samples/frame
  │     │
  │     └── Encoder: 12 Transformer layers
  │           Hidden dim: 768
  │           Attention heads: 12
  │           Feed-forward dim: 3072
  │           Dropout: 0.1
  │           Output: (B, T, 768) contextual embeddings
  │
  └── Classification Head
        ├── Linear(768, 2)  — binary: [not_present, present]
        Output: (B, 2) logits
```

#### Why Individual Classifiers?

Each dysfluency type has distinct acoustic characteristics:
- **Prolongation**: Extended energy in specific frequency bands
- **Block**: Sudden silence or energy drop
- **Sound repetition**: Rapid periodic patterns
- **Word repetition**: Longer repeated segments
- **Interjection**: Filler sounds with distinct spectral signature

Training separate classifiers allows each to specialize in its dysfluency type's unique patterns.

#### How They Work

```python
# model/classification/__init__.py
class BaseWav2VecClassifier:
    def predict(self, audio_tensor, threshold=0.5):
        logits = self.forward(audio_tensor)           # (B, 2)
        probs = torch.softmax(logits, dim=-1)         # (B, 2)
        prob_present = probs[0, 1].item()             # P(dysfluency)
        label = 1 if prob_present >= threshold else 0
        confidence = prob_present if label == 1 else 1.0 - prob_present
        return label, confidence
```

#### Five Concrete Subclasses

| Class | File | class_name | class_idx |
|---|---|---|---|
| ProlongationClassifier | `prolongation.py` | "prolongation" | 0 |
| BlockClassifier | `block.py` | "block" | 1 |
| SoundRepClassifier | `soundrep.py` | "soundrep" | 2 |
| WordRepClassifier | `wordrep.py` | "wordrep" | 3 |
| InterjectionClassifier | `interjection.py` | "interjection" | 4 |

Each subclass is a thin wrapper that sets `class_name` and `class_idx`. All logic resides in `BaseWav2VecClassifier`.

### 5.2 Multi-Task Classifier (Shared Backbone)

**File**: `model/classification/multitask.py` (MultiTaskWav2VecClassifier)

#### Architecture

```
Input: Raw audio waveform (B, 160000)
        │
        ▼
Wav2Vec2Model (backbone, facebook/wav2vec2-base)
  ├── FeatureExtractor (7 conv layers)
  └── Encoder (12 Transformer layers)
        Output: last_hidden_state (B, T, 768)
        │
        ▼
Mean Pooling: hidden.mean(dim=1) → (B, 768)
        │
        ├──► Head_prolongation: Linear(768, 768) → Tanh → Linear(768, 2)
        ├──► Head_block:        Linear(768, 768) → Tanh → Linear(768, 2)
        ├──► Head_soundrep:     Linear(768, 768) → Tanh → Linear(768, 2)
        ├──► Head_wordrep:      Linear(768, 768) → Tanh → Linear(768, 2)
        └──► Head_interjection: Linear(768, 768) → Tanh → Linear(768, 2)

Output: {class_name: (B, 2) logits}
```

#### Why Multi-Task?

1. **Parameter efficiency**: One backbone (95M params) instead of five (475M params)
2. **Shared representations**: Common speech features learned jointly help rare classes
3. **Regularization**: Multi-task learning acts as implicit regularization
4. **Faster inference**: One forward pass instead of five

#### Key Methods

```python
class MultiTaskWav2VecClassifier:
    def predict(self, audio_tensor, threshold=0.5):
        """Classify with all heads in one forward pass."""
        logits = self.forward(audio_tensor)  # {class_name: (B, 2)}
        results = {}
        for name, lg in logits.items():
            probs = torch.softmax(lg, dim=-1)
            prob_present = probs[0, 1].item()
            label = 1 if prob_present >= threshold else 0
            confidence = prob_present if label == 1 else 1.0 - prob_present
            results[name] = (label, confidence)
        return results

    def saliency(self, input_values):
        """Per-frame per-class probability saliency map.

        CAM-style: runs each frame's hidden state through the trained heads
        and softmax over each class's 2 logits.

        Returns: (B, T, num_classes) tensor with values in [0, 1]
        """
        hidden = self.model.wav2vec2(input_values).last_hidden_state  # (B, T, d)
        frames = [
            torch.softmax(self.model.heads[name](hidden), dim=-1)[..., 1]
            for name in self.class_names
        ]
        return torch.stack(frames, dim=-1)  # (B, T, 5)
```

The `saliency()` method is critical — it produces a per-frame, per-class probability map that is later fused with localization regions in the combiner.

### 5.3 CNN Multi-Task Classifier

**File**: `model/classification/cnn_multitask.py` (CNNMultitaskClassifier)

#### Architecture

```
Input: Mel-spectrogram (B, 1, 128, T) — 128 mel bins
        │
        ▼
CNN Encoder (4 blocks):
  Block 1: Conv2d(1, 32, 3×3) → BN → ReLU → MaxPool2d(2,1) → Dropout2d
           Output: (B, 32, 64, T)
  Block 2: Conv2d(32, 64, 3×3) → BN → ReLU → MaxPool2d(2,1) → Dropout2d
           Output: (B, 64, 32, T)
  Block 3: Conv2d(64, 128, 3×3) → BN → ReLU → MaxPool2d(2,1) → Dropout2d
           Output: (B, 128, 16, T)
  Block 4: Conv2d(128, 128, 3×3) → BN → ReLU → Dropout2d
           Output: (B, 128, 16, T)
        │
        ▼
Sequence Aggregator (one of three options):
  ├── Pool:     mean(dim=(2,3)) → Linear(128, hidden_dim)
  ├── LSTM:     mean(dim=2) → LSTM → hidden state
  └── Transformer: mean(dim=2) → Linear → TransformerEncoder → mean(dim=1)
        │
        ▼
Per-class heads (5 heads, same as multitask):
  Head_prolongation: Linear(hidden_dim, hidden_dim) → Tanh → Linear(hidden_dim, 2)
  Head_block: ...
  Head_soundrep: ...
  Head_wordrep: ...
  Head_interjection: ...
```

#### Why CNN Multi-Task?

- **Ablation study**: Compare Wav2Vec2 vs CNN architectures
- **Mel-spectrogram input**: Leverages time-frequency representations directly
- **Pluggable aggregators**: Test pool/LSTM/Transformer for temporal aggregation
- **Lighter weight**: No pre-trained backbone; trains from scratch on spectrograms

#### MaxPool2d(2,1) Design Choice

The `MaxPool2d(kernel_size=(2, 1))` halves the frequency dimension while preserving the time dimension. This is intentional:
- Frequency reduction: 128 → 64 → 32 → 16 (progressive compression)
- Time preservation: Every time frame is retained for precise localization

---

## 6. Localization Models

### 6.1 Wav2Vec2 Localizer

**File**: `model/localization/wav2vec2_localizer.py`

#### Architecture

```
Input: Raw waveform (B, max_samples) — up to 10 seconds at 16kHz
        │
        ▼
Wav2Vec2Model (backbone)
  Output: last_hidden_state (B, T_frames, 768)
  T_frames = max_samples / 320
  Frame resolution: 20ms per frame
        │
        ▼
Temporal Classifier Head:
  Linear(768, 256) → ReLU → Dropout(0.3)
  Linear(256, 128) → ReLU → Dropout(0.3)
  Linear(128, 1)
        │
        ▼
Output: (B, 1, T_frames) — per-frame dysfluency logits
        │
        ▼
Sigmoid → per-frame probabilities [0, 1]
        │
        ▼
Threshold + contiguous region extraction:
  → List of (start_sec, end_sec, confidence)
```

#### Why This Architecture?

1. **Frame-level prediction**: Unlike classifiers (clip-level), the localizer predicts dysfluency at every 20ms frame
2. **Temporal attention**: The deep classifier head (3 layers) learns to attend to temporal patterns
3. **Dropout regularization**: Prevents overfitting on small localization datasets
4. **Frame resolution**: 20ms frames provide fine-grained temporal localization

#### Region Extraction Algorithm

```python
def predict(self, audio, sr=16000, threshold=0.5, max_length_seconds=10.0):
    probs = self.predict_proba(tensor)  # (1, 1, T_frames)
    probs_np = probs.squeeze().cpu().numpy()

    frame_duration = 320 / sr  # 0.02 seconds per frame

    regions = []
    in_region = False
    for i, p in enumerate(probs_np):
        if p >= threshold and not in_region:
            in_region = True
            start_frame = i
            max_conf = p
        elif p >= threshold and in_region:
            max_conf = max(max_conf, p)
        elif p < threshold and in_region:
            in_region = False
            regions.append((start_frame * frame_duration,
                           i * frame_duration,
                           float(max_conf)))

    return regions  # [(start_sec, end_sec, confidence), ...]
```

### 6.2 CNN Spectrogram Localizer

**File**: `model/localization/cnn_spectrogram.py`

#### Architecture

```
Input: Mel-spectrogram (B, 1, n_mels, T) — 128 mel bins
        │
        ▼
CNN Backbone:
  Block 1: Conv2d(1, 32, 3×3, padding=1) → BN → ReLU → MaxPool2d(2,1) → Dropout2d(0.4)
  Block 2: Conv2d(32, 64, 3×3, padding=1) → BN → ReLU → MaxPool2d(2,1) → Dropout2d(0.4)
  Block 3: Conv2d(64, 128, 3×3, padding=1) → BN → ReLU → MaxPool2d(2,1) → Dropout2d(0.4)
  Block 4: Conv2d(128, 128, 3×3, padding=1) → BN → ReLU → AdaptiveAvgPool2d((1, None)) → Dropout2d(0.4)
        │
        ▼
Per-frame Classifier:
  Conv2d(128, 64, 1×1) → ReLU → Dropout(0.4) → Conv2d(64, 1, 1×1)
        │
        ▼
Output: (B, 1, T) — per-frame logits
```

#### AdaptiveAvgPool2d((1, None)) Design Choice

The `AdaptiveAvgPool2d((1, None))` layer collapses the frequency dimension to 1 while preserving the time dimension:
- Input: (B, 128, ~16, T) — 128 frequency features across ~16 reduced mel bins
- Output: (B, 128, 1, T) — frequency-collapsed, time-preserved

This is critical because:
1. Frequency information is encoded in the 128 channels
2. Time resolution must be preserved for precise localization
3. The subsequent 1×1 conv reduces channels to a single prediction per frame

### 6.3 CTC Forced Alignment

**File**: `model/localization/ctc_alignment.py`

#### Purpose

Aligns transcript text to audio using CTC (Connectionist Temporal Classification) forced alignment. This produces word-level timestamps when a reference transcript is available.

#### How It Works

1. Run Wav2Vec2 CTC model on audio to get per-frame character probabilities
2. Use dynamic programming to find the optimal alignment between the transcript and audio frames
3. Map character-level alignment back to word boundaries
4. Return word-level timestamps: `[(word, start_sec, end_sec, confidence), ...]`

#### Fallback: SimpleForcedAligner

When CTC alignment is unavailable (e.g., model not loaded), a simple energy-based aligner distributes words evenly across the audio duration.

### 6.4 Language Adapters

**File**: `model/localization/language_adapter.py`

#### Purpose

Converts word-level timestamps into syllable-level timestamps for different languages. Syllable-level alignment is important because:
- Many dysfluencies occur at the syllable level (sound repetitions)
- Clinical assessment often works at the syllable level
- Different languages have different syllabification rules

#### Supported Languages

| Language | Adapter | Syllabification Rules |
|---|---|---|
| English | EnglishAdapter | Vowel-consonant pattern detection |
| Kannada | KannadaAdapter | Akshara-based syllabification |
| Hindi | HindiAdapter | Devanagari syllable structure |

---

## 7. Fusion and Combiner

**File**: `model/combiner.py`

### Why Fusion?

The localization model tells us **where** dysfluency might occur (type-agnostic regions). The multi-task classifier tells us **what type** of dysfluency is present at each frame (per-class saliency). Neither alone is sufficient:

- **Localizer alone**: "There's dysfluency at 1.2s-2.1s" (but what type?)
- **Classifier alone**: "Prolongation is present" (but where exactly?)
- **Fused**: "Prolongation detected at 1.2s-2.1s with 87% confidence"

### Fusion Algorithm

```python
def combine_regions(regions, saliency, class_names, thresholds, ...):
    """
    For each localizer region:
      1. Map region boundaries to saliency frame indices
      2. Extract the saliency window for that region
      3. Average the per-class probabilities across the window
      4. Apply per-class thresholds to determine labels
      5. Select the primary type (highest confidence detected class)
      6. Optionally snap boundaries to syllable edges
    """
    for region in regions:
        start_f = int(region["start"] / frame_duration)
        end_f = int(region["end"] / frame_duration)
        window = saliency[start_f:end_f]  # (window_frames, num_classes)

        classes = {}
        for i, name in enumerate(class_names):
            prob_present = float(window[:, i].mean())  # average across frames
            label = 1 if prob_present >= threshold else 0
            classes[name] = {
                "label": label,
                "confidence": prob_present if label else 1 - prob_present,
                "prob_present": prob_present,
            }

        # Primary type = detected class with highest probability
        primary_type = max(classes.items(), key=lambda x: x[1]["prob_present"])

        entry["classes"] = classes
        entry["primary_type"] = primary_type
        entry["severity"] = None  # computed separately
```

### Saliency Fallback (When Localizer Unavailable)

When the dedicated localizer weights are unavailable, the system falls back to **saliency synthesis**:

```python
def saliency_regions(saliency, class_names, duration_sec):
    """Extract regions directly from saliency using adaptive thresholding."""
    for c, name in enumerate(class_names):
        col = saliency[:, c]
        # Adaptive threshold: mean + k * std, floored and capped
        threshold = min(max_threshold, max(floor, col.mean() + adapt_k * col.std()))

        # Find contiguous runs above threshold
        # Drop spans shorter than min_span_sec
        # Deduplicate overlapping regions by confidence
```

### Mismatch Rate Probe

The `mismatch_rate()` function measures what fraction of high-saliency spans have no overlapping localizer region. This metric guides the decision of whether saliency synthesis is needed.

---

## 8. Transcription Pipeline

**File**: `model/transcription.py`

### Architecture

```
Input: Audio bytes
        │
        ▼
Format Conversion: convert_to_wav() → 16kHz mono WAV
        │
        ▼
Language-specific Whisper Model:
  ├── English:     openai/whisper-tiny
  ├── Kannada:     vasista22/whisper-kannada-tiny
  └── Hindi:       collabora/whisper-tiny-hindi
        │
        ▼
Whisper ASR Pipeline (HuggingFace):
  return_timestamps=True → word-level timestamps
        │
        ▼
Post-processing:
  ├── _collapse_runs(): Deduplicate Whisper hallucination loops
  │   (Whisper often repeats phrases 3-5 times)
  │   Keeps up to max_run=3 repetitions (preserves genuine stutters)
  │
  └── _collapse_phrase_runs(): Detect multi-word phrase repetitions
      (e.g., "I was nervous I was nervous I was nervous")
        │
        ▼
Output: {text, language, chunks: [{text, start, end, language}]}
```

### Why Hallucination Collapse?

Whisper models are known to produce hallucinated repeated phrases, especially on short audio or audio with pauses. The collapse algorithms:

1. **Chunk-level**: Deduplicates consecutive identical text chunks
2. **Phrase-level**: Detects multi-word phrase repetitions using sliding window comparison
3. **Preserves genuine stutters**: Keeps up to 3 repetitions (configurable `max_run`)

### Desktop Transcription (app/core/transcription.py)

The desktop app extends transcription with:

- **Word-level timestamps**: `return_timestamps="word"` for granular timing
- **Stutter flagging**: Detects hyphenated syllable repeats ("s-s", "ba-ba") and stretched characters ("sss")
- **Repetition marking**: Consecutive identical words flagged as `wordrep`, repeated fragments as `soundrep`
- **Forced alignment fallback**: When Whisper fails, uses `SimpleForcedAligner` from `model.registry`

---

## 9. Training Pipeline

**Files**: `model/training/`

### 9.1 Training Scripts

| Script | Purpose | Model |
|---|---|---|
| `train_classifier.py` | Train 5 independent binary classifiers | Wav2Vec2ForSequenceClassification |
| `train_multitask_classifier.py` | Train shared-backbone multitask classifier | MultiTaskWav2VecClassifier |
| `train_cnn_classifier.py` | Train CNN multitask with aggregator ablation | CNNMultitaskClassifier |
| `train_localizer.py` | Train CNN spectrogram localizer | CNNSpectrogramLocalizer |
| `train_wav2vec2_localizer.py` | Train Wav2Vec2 localizer | Wav2Vec2Localizer |

### 9.2 Training Configuration

```python
# model/config/defaults.py
SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 3
MAX_AUDIO_LENGTH = 48000  # 3 seconds × 16000 Hz

LEARNING_RATE = 3e-5
BATCH_SIZE = 8
NUM_EPOCHS = 20
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP_MAX_NORM = 1.0
```

### 9.3 Loss Functions

#### FocalLoss (Classification)

```python
class FocalLoss(nn.Module):
    """Handles class imbalance by down-weighting easy negatives.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=2.0 reduces the contribution of well-classified examples,
    focusing training on hard-to-classify samples.
    """
    def __init__(self, gamma=2.0, reduction="mean"):
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = softmax(logits).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()
```

**Why Focal Loss?**
- Dysfluency datasets are imbalanced (fewer positive samples than negative)
- Standard cross-entropy is dominated by easy negative examples
- Focal loss down-weights easy examples, focusing on hard cases

#### BCEWithLogitsLoss (Localization)

```python
# With pos_weight to handle rare positive frames
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

**Why pos_weight?**
- Frame labels are heavily imbalanced (dysfluent frames are rare)
- Without weighting, the model learns to predict "not dysfluent" for all frames
- pos_weight = n_neg / n_pos penalizes missed positive frames

### 9.4 Learning Rate Schedule

```
Linear warmup: 0 → 3e-5 over 500 steps
Linear decay: 3e-5 → 0 over remaining steps
```

**Why warmup?**
- Prevents large initial gradients from destabilizing the pre-trained Wav2Vec2 weights
- Allows the classification heads to adapt gradually

### 9.5 Freezing Strategy

```python
# Freeze first 3 transformer layers (default)
for layer in model.wav2vec2.encoder.layers[:3]:
    for param in layer.parameters():
        param.requires_grad = False

# Unfreeze after 5 epochs (configurable)
if epoch >= freeze_backbone_epochs:
    unfreeze_backbone()
```

**Why freeze layers?**
- Lower layers capture general speech features (phonemes, formants)
- Higher layers capture task-specific features (dysfluency patterns)
- Freezing lower layers prevents catastrophic forgetting
- Progressive unfine-tuning allows adaptation without destabilization

### 9.6 Checkpoint Management

```python
# model/training/utils.py
def save_checkpoint(model, optimizer, epoch, metrics, path, scheduler, extra):
    state = {
        "epoch": epoch,
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(state, path)

def resume_checkpoint_path(args, fp):
    """Resume checkpoint keyed by fingerprint string."""
    return os.path.join(args.output_dir, f"{fp}_checkpoint.pt")
```

### 9.7 Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="max"):
        self.patience = patience
        self.best_score = None
        self.counter = 0

    def step(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
```

---

## 10. Data Pipeline

**Files**: `model/data/`

### 10.1 Audio Preprocessing

```python
# model/data/preprocessing.py

def clean_audio(audio, sr=16000, remove_dc=True, normalize=True, trim=True):
    """Full cleaning pipeline: DC removal → peak normalization → silence trimming."""
    if remove_dc:
        audio = audio - np.mean(audio)       # Center around zero
    if normalize:
        audio = audio * (0.95 / np.abs(audio).max())  # Peak normalize
    if trim:
        audio, _ = librosa.effects.trim(audio, top_db=25)  # Remove silence
    return audio
```

#### Why These Steps?

1. **DC removal**: Microphone recordings often have a DC bias; centering the signal prevents feature extraction bias
2. **Peak normalization**: Ensures consistent amplitude across recordings (different microphones, distances)
3. **Silence trimming**: Removes leading/trailing silence that adds no dysfluency information

### 10.2 Spectrogram Generation

```python
def generate_mel_spectrogram(audio, sr=16000, n_mels=128, hop_length=512, n_fft=2048):
    """Generate log-mel spectrogram from raw audio."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec.astype(np.float32)
```

#### Parameters Justification

| Parameter | Value | Justification |
|---|---|---|
| `n_mels` | 128 | Captures fine frequency detail; standard for speech |
| `hop_length` | 512 | At 16kHz: 32ms per frame; good temporal resolution |
| `n_fft` | 2048 | 128ms window; captures formant structure |
| `fmin` | 0.0 | Includes all frequencies up to Nyquist |
| `fmax` | sr/2 = 8000 | Covers speech fundamental frequency range |

### 10.3 Frame-Level Labels (Localization)

```python
def create_frame_labels(dysfluency_intervals, num_frames, sr=16000, hop_length=512):
    """Convert (start_sec, end_sec) intervals to binary frame mask."""
    labels = np.zeros(num_frames, dtype=np.uint8)
    for start_sec, end_sec in dysfluency_intervals:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        start_frame = start_sample // hop_length
        end_frame = (end_sample + hop_length - 1) // hop_length  # Ceil
        labels[start_frame:end_frame] = 1
    return labels
```

**Frame alignment**: Frame `i` covers audio samples `[i * hop_length, (i+1) * hop_length)`.

### 10.4 Data Augmentation

**File**: `model/data/augmentation.py`

#### Audio Augmentations

| Augmentation | Probability | Description | Why |
|---|---|---|---|
| Additive noise | 0.4 | Random white/pink noise | Robustness to recording conditions |
| Time stretch | 0.3 | Speed change (0.8x-1.2x) | Speed invariance |
| Pitch shift | 0.3 | ±2 semitones | Speaker invariance |
| Time masking | 0.2 | Random time segment dropout | Encourages local feature learning |
| Frequency masking | 0.2 | Random frequency band dropout | Encourages multi-band feature learning |

#### Spectrogram Augmentations (SpecAugment)

Applied directly to spectrograms during training:
- Time warping: Non-linear time distortion
- Frequency masking: Zero out random frequency bands
- Time masking: Zero out random time segments

### 10.5 Class Balancing

```python
def create_balanced_sampler(labels):
    """WeightedRandomSampler for balanced mini-batches."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    weights = np.where(labels == 1, 1.0/n_pos, 1.0/n_neg)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
```

---

## 11. Evaluation Framework

**Files**: `model/evaluation/`

### 11.1 Classification Metrics

**File**: `model/evaluation/metrics.py`

```python
def compute_binary_metrics(y_true, y_scores, threshold=0.5):
    """Comprehensive binary classification metrics."""
    return {
        "auroc": _compute_auroc(y_true, y_scores),      # Threshold-independent
        "auprc": _compute_auprc(y_true, y_scores),      # Better for imbalanced data
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": 2 * precision * recall / (precision + recall),
        "specificity": tn / (tn + fp),                   # Critical for screening
        "accuracy": (tp + tn) / total,
    }
```

#### Why AUROC and AUPRC?

- **AUROC**: Threshold-independent; measures ranking quality
- **AUPRC**: More informative than AUROC for imbalanced data (rare positive class)
- **Specificity**: Critical for clinical screening (low false alarm rate)

#### Optimal Threshold Search

```python
def find_optimal_threshold(y_true, y_scores, metric="f1"):
    """Sweep thresholds 0.1-0.9 (step 0.05) to find optimal operating point."""
    for thresh in np.arange(0.1, 0.91, 0.05):
        # Compute metric at each threshold
        # Return (best_threshold, best_metric_value)
```

### 11.2 Localization Metrics

```python
def compute_localization_metrics(y_true_frames, y_pred_frames, threshold=0.5, iou_threshold=0.5):
    """Frame-level and event-level localization metrics."""
    return {
        "frame_level": {"precision", "recall", "f1", "specificity"},
        "event_level": {
            "detection_accuracy": matched_true / total_true,  # % of true events found
            "mean_iou": average_iou_of_matched_pairs,         # Temporal overlap quality
            "false_alarm_rate_per_min": false_alarms / duration_min,
        }
    }
```

#### Event Matching (IoU)

Two events are "matched" if their IoU (Intersection over Union) ≥ 0.5:

```
IoU = |intersection| / |union|
```

This ensures that only well-aligned predictions are counted as correct.

### 11.3 Severity Scoring

```python
def compute_severity(regions, duration_sec):
    """Compute stutter index from localized regions."""
    total_dysfluency_coverage = sum(r["end"] - r["start"] for r in regions)
    stutter_index = (total_dysfluency_coverage / duration_sec) * 100

    if stutter_index >= 15:   severity = "severe"
    elif stutter_index >= 5:  severity = "moderate"
    elif stutter_index >= 2:  severity = "mild"
    else:                     severity = "fluent"

    return {"index_pct": stutter_index, "severity": severity}
```

### 11.4 Evaluation Reports

**File**: `model/evaluation/summary.py`, `comparative_study.py`, `presentation.py`

- **Summary**: Aggregates per-class metrics into a Markdown report
- **Comparative Study**: Multi-arm comparison of different model configurations
- **Presentation**: Charts and visualizations for model performance

---

## 12. Backend API

**Files**: `backend/`

### 12.1 Endpoints

| Method | Path | Purpose | Input | Output |
|---|---|---|---|---|
| `GET` | `/health` | Health check | — | `{"status": "ok"}` |
| `POST` | `/api/classify` | Classify dysfluency types | FormData: file, language | Classification + transcription |
| `POST` | `/api/localize` | Localize dysfluency regions | FormData: file, language | Regions list |
| `POST` | `/api/analyze` | Full analysis pipeline | FormData: file, language | All results combined |
| `POST` | `/api/report` | Generate PDF report | JSON: patient info + results | PDF blob |

### 12.2 Service Layer

```
backend/services/
├── classifier.py      → Wraps MultiTaskClassifier.analyze()
├── localizer.py       → Wraps Localizer("wav2vec2").predict() with saliency fallback
├── transcriber.py     → Language-specific Whisper + hallucination collapse
├── fusion.py          → Wraps combine_with_saliency()
├── severity.py        → Pure computation: stutter index calculation
├── report_generator.py → Typst-based PDF generation
└── audio_utils.py     → Re-exports convert_to_wav from model.registry
```

### 12.3 Report Generation

```python
# backend/services/report_generator.py
def build_typ_source(data):
    """Build Typst source document from analysis data."""
    # Sections: Title, Patient Details, Audio Details,
    # Stuttering Classes (table), Diagnostic Severity

def generate_report_pdf(data):
    """Compile Typst source to PDF."""
    source = build_typ_source(data)
    with tempfile.NamedTemporaryFile(suffix=".typ", mode="w") as f:
        f.write(source)
        return typst.compile(f.name)
```

---

## 13. Web Frontend

**Files**: `frontend/`

### 13.1 Technology Choices

| Technology | Version | Why |
|---|---|---|
| React | 19.1 | Component-based UI, hooks for state management |
| TypeScript | 5.8 | Type safety, better IDE support |
| Vite | 6.3 | Fast dev server, HMR, optimized builds |
| Tailwind CSS | 4 | Utility-first CSS, no custom CSS files |
| React Router | 7 | Client-side routing without page reloads |
| Lucide React | Latest | Consistent icon library |

### 13.2 Component Architecture

```
App.tsx (BrowserRouter + MainAppShell)
├── Sidebar (navigation links with active highlighting)
├── Routes:
│   ├── / → UploadPage
│   │   ├── AudioPlayer (preview with seek, mute)
│   │   ├── Upload mode (drag-drop, file picker)
│   │   ├── Record mode (MediaRecorder API, live waveform)
│   │   └── Language selector (English, Kannada, Hindi)
│   │
│   ├── /results → ResultsPage
│   │   ├── Loading screen (progress bar, step messages)
│   │   ├── Severity card (Fluent/Mild/Moderate/Severe)
│   │   ├── Classification card (5 categories with confidence bars)
│   │   ├── Localization card (regions list with timestamps)
│   │   ├── Timeline visualization (color-coded segments)
│   │   ├── WaveformView (interactive canvas with overlays)
│   │   └── PDF report export (patient info dialog)
│   │
│   ├── /documents → DocumentsPage (standardized reading passages + custom PDFs)
│   ├── /history → HistoryPage (localStorage + IndexedDB)
│   ├── /settings → SettingsPage (placeholder)
│   └── /about → AboutPage (project info)
│
└── Modals: Help/FAQ, How It Works, Recording Tips
```

### 13.3 State Management

| Storage | Key | Purpose |
|---|---|---|
| `localStorage` | `theme` | Light/dark mode preference |
| `localStorage` | `swaraaha_history` | Assessment history records |
| `sessionStorage` | `results` | Cached analysis results (survives page refresh) |
| `sessionStorage` | `filename`, `filesize`, `duration` | Current audio metadata |
| `sessionStorage` | `transcription_language` | Selected language |
| `IndexedDB` | `SwaraahaDB.audioFiles` | Raw audio File objects for history replay |

### 13.4 API Communication

```typescript
// frontend/src/api/client.ts
const API_BASE = "/api";  // Proxied to localhost:5173 → localhost:8000

export async function analyzeAudio(file: File, language: string): Promise<AnalyzeResults> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("language", language);

    const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
    });
    return response.json();
}
```

### 13.5 Reading Passages

Three standardized passages are provided for consistent assessment:

1. **Assessment Passages**: Grandfather Passage + Rainbow Passage (standard stuttering assessment texts)
2. **Phonetically Balanced**: 10 sentences covering all English phonemes
3. **Diagnostic Exercises**: Targeted exercises for prolongation, block, and conversational flow detection

---

## 14. Desktop Application

**Files**: `app/`

### 14.1 Architecture

```
app/main.py                    → QApplication bootstrap, font loading, stylesheet
app/core/                      → Pure logic (no Qt imports)
├── audio_handler.py           → Recording, playback, WAV I/O (sounddevice)
├── model_runner.py            → ML pipeline orchestrator (classification + localization + transcription)
├── transcription.py           → Whisper ASR with stutter alignment
├── report_data.py             → Results → report data contract
├── pdf_handler.py             → PDF text extraction (pdfplumber)
└── recent_files.py            → QSettings-backed recent file persistence
app/ui/                        → PySide6 widgets
├── main_window.py             → QMainWindow, page navigation, drag-drop
├── home_page.py               → Landing page with passage/files nav
├── analysis_page.py           → Results display + PDF export
├── audio_controls.py          → Record/Stop/Load/Play/Analyze buttons
├── waveform_view.py           → QGraphicsView waveform renderer
├── results_panel.py           → Classification table + localization timeline
├── transcription_panel.py     → Full transcription display
├── compact_transcript.py      → Compact home-page transcript
├── transcription_worker.py    → Background QThread for ASR
├── file_panel.py              → Directory tree + recent files
├── pdf_viewer.py              → PDF page renderer with zoom
├── language_dialog.py         → Language selection modal
├── table_utils.py             → Table sizing/population helpers
├── report_export.py           → Patient name dialog for export
├── wait_dialog.py             → Indeterminate spinner modal
├── theme.py                   → Design tokens, font loading, theme toggle
└── styles.py                  → QSS stylesheet generator
```

### 14.2 Theme System

```python
# app/ui/theme.py
LIGHT_COLORS = {
    "primary": "#7C3AED",       # Purple
    "surface": "#FFFFFF",
    "record": "#EF4444",        # Red
    "dysfluency_prolongation": "#F59E0B",
    "dysfluency_block": "#EF4444",
    "dysfluency_soundrep": "#3B82F6",
    "dysfluency_wordrep": "#10B981",
    "dysfluency_interjection": "#8B5CF6",
}

DARK_COLORS = {
    "primary": "#A78BFA",
    "surface": "#1E293B",
    # ...
}
```

### 14.3 Background Workers

```python
# app/ui/main_window.py
class AnalysisWorker(QThread):
    finished = Signal(dict)

    def run(self):
        runner = ModelRunner()
        self.finished.emit(runner.analyze(self.audio, self.language))
```

Long-running operations (analysis, transcription) run in background threads to keep the UI responsive.

---

## 15. Shared Utilities

**Files**: `shared/reporting/`

### Report Builder

**File**: `shared/reporting/report_builder.py`

```python
def build_report_source(data):
    """Build complete Typst source document."""
    # Sections:
    # 1. Title: "Swaraaha Stutter Analysis Report"
    # 2. Patient Details: name, phone
    # 3. Audio Details: filename, size, duration
    # 4. Stuttering Classes: table with Dysfluency Category / Clinical Label / Confidence
    # 5. Localized Events: table with timestamps and types
    # 6. Summary: total events, severity
    # 7. Transcript: full text with timestamps
    # 8. Medical Disclaimer

def generate_report_pdf(data):
    """Compile Typst source → PDF bytes."""
    source = build_report_source(data)
    with tempfile.NamedTemporaryFile(suffix=".typ", mode="w") as f:
        f.write(source)
        return typst.compile(f.name)
```

### Why Typst?

- **Modern**: Rust-based typesetting system, faster than LaTeX
- **Simple syntax**: Easier to generate programmatically
- **Good PDF quality**: Professional-looking clinical reports
- **Python bindings**: `typst` package for direct compilation

---

## 16. Deployment

### Docker (Backend)

```yaml
# docker-compose.yml
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./model/weights:/app/model/weights
```

### Render (Cloud)

```yaml
# render.yaml
services:
  - type: web
    name: swaraaha-backend
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

### Development

```bash
# Backend
cd backend && uvicorn backend.main:app --reload --port 8000

# Frontend (with proxy to backend)
cd frontend && npm run dev  # Vite dev server on :5173, proxies /api to :8000

# Desktop
cd app && python -m app.main
```

---

## 17. Dependencies and Setup

### Backend Dependencies

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.6
typst>=0.6
```

### ML Dependencies (model/)

```
torch>=2.0
transformers>=4.30.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
```

### Frontend Dependencies

```json
{
  "react": "^19.1.0",
  "react-dom": "^19.1.0",
  "react-router-dom": "^7.0.0",
  "lucide-react": "^0.500.0",
  "typescript": "^5.8.0",
  "vite": "^6.3.0",
  "tailwindcss": "^4.0.0",
  "@tailwindcss/vite": "^4.0.0"
}
```

### Desktop Dependencies

```
PySide6>=6.5
sounddevice>=0.4
soundfile>=0.12
librosa>=0.10
pdfplumber>=0.10
pypdfium2>=4.0
```

---

## 18. File Structure Reference

```
Swaraaha/
├── model/                          # Core ML engine
│   ├── __init__.py                 # Exports Classifier, Localizer, Transcriber, ModelRegistry
│   ├── registry.py                 # Central inference API (1033 lines)
│   ├── registry.json               # Weight paths + per-class thresholds
│   ├── combiner.py                 # Region + saliency fusion (194 lines)
│   ├── transcription.py            # Whisper transcription + hallucination collapse
│   ├── fingerprint.py              # Hyperparameter fingerprinting
│   ├── config/
│   │   └── defaults.py             # Constants: SAMPLE_RATE, DYSFLUENCY_CLASSES, training defaults
│   ├── classification/
│   │   ├── __init__.py             # BaseWav2VecClassifier (156 lines)
│   │   ├── prolongation.py         # ProlongationClassifier
│   │   ├── block.py                # BlockClassifier
│   │   ├── soundrep.py             # SoundRepClassifier
│   │   ├── wordrep.py              # WordRepClassifier
│   │   ├── interjection.py         # InterjectionClassifier
│   │   ├── multitask.py            # MultiTaskWav2VecClassifier (167 lines)
│   │   └── cnn_multitask.py        # CNNMultitaskClassifier (208 lines)
│   ├── localization/
│   │   ├── __init__.py             # Lazy loaders
│   │   ├── wav2vec2_localizer.py   # Wav2Vec2Localizer (297 lines)
│   │   ├── cnn_spectrogram.py      # CNNSpectrogramLocalizer (311 lines)
│   │   ├── ctc_alignment.py        # CTCTimeAligner, SimpleForcedAligner
│   │   ├── language_adapter.py     # English/Kannada/Hindi adapters
│   │   └── wav2vec2_dataset.py     # Wav2Vec2LocalizationDataset
│   ├── data/
│   │   ├── config.py               # Dataset paths, splits, class labels
│   │   ├── dataset.py              # ClassificationDataset, LocalizationDataset
│   │   ├── augmentation.py         # AudioAugmentor, SpectrogramAugmentor
│   │   ├── preprocessing.py        # load_audio, mel-spectrogram, normalize (1144 lines)
│   │   ├── download.py             # Kaggle dataset download
│   │   ├── prepare.py              # Data preparation pipeline
│   │   └── merge.py                # Multi-dataset merging
│   ├── training/
│   │   ├── train.py                # Orchestrator with auto-resource-detection
│   │   ├── train_classifier.py     # Binary classifier training
│   │   ├── train_multitask_classifier.py  # Shared-backbone multitask training
│   │   ├── train_cnn_classifier.py # CNN multitask with ablation
│   │   ├── train_localizer.py      # CNN spectrogram localizer training
│   │   ├── train_wav2vec2_localizer.py    # Wav2Vec2 localizer training
│   │   └── utils.py                # Checkpointing, EarlyStopping, FocalLoss (461 lines)
│   ├── evaluation/
│   │   ├── evaluate.py             # Unified evaluation
│   │   ├── metrics.py              # All metrics from scratch (635 lines)
│   │   ├── summary.py              # Summary aggregation + Markdown
│   │   ├── comparative_study.py    # Multi-arm comparative study
│   │   └── presentation.py         # Charts + visualizations
│   ├── weights/                    # Trained model checkpoints
│   └── tests/                      # 18 test files
│
├── backend/                        # REST API server
│   ├── main.py                     # FastAPI entry point
│   ├── routes/
│   │   ├── classification.py       # POST /api/classify
│   │   ├── localization.py         # POST /api/localize, POST /api/analyze
│   │   └── report.py               # POST /api/report
│   ├── services/
│   │   ├── classifier.py           # MultiTaskClassifier wrapper
│   │   ├── localizer.py            # Localizer wrapper with fallback
│   │   ├── transcriber.py          # Language-specific Whisper
│   │   ├── fusion.py               # Saliency fusion wrapper
│   │   ├── severity.py             # Stutter index computation
│   │   ├── report_generator.py     # Typst PDF generation
│   │   └── audio_utils.py          # Audio format conversion
│   └── tests/                      # 8 test files
│
├── frontend/                       # Web UI (React)
│   ├── src/
│   │   ├── main.tsx                # Entry point
│   │   ├── App.tsx                 # App shell, routing, modals (552 lines)
│   │   ├── pages/
│   │   │   ├── UploadPage.tsx      # Audio upload/record (756 lines)
│   │   │   └── ResultsPage.tsx     # Results dashboard (1100 lines)
│   │   ├── api/
│   │   │   └── client.ts           # API client + TypeScript types (139 lines)
│   │   ├── utils/
│   │   │   └── db.ts               # IndexedDB wrapper
│   │   ├── components/
│   │   │   └── PdfViewer.tsx       # PDF document viewer (331 lines)
│   │   └── data/
│   │       └── readingDocuments.json  # Reading passage metadata
│   └── public/documents/           # Standardized reading passage PDFs
│
├── app/                            # Desktop application (PySide6)
│   ├── main.py                     # Entry point
│   ├── core/
│   │   ├── audio_handler.py        # Recording, playback, WAV I/O
│   │   ├── model_runner.py         # ML pipeline orchestrator
│   │   ├── transcription.py        # Whisper + stutter alignment
│   │   ├── report_data.py          # Results → report data
│   │   ├── pdf_handler.py          # PDF text extraction
│   │   └── recent_files.py         # QSettings persistence
│   ├── ui/
│   │   ├── main_window.py          # QMainWindow
│   │   ├── home_page.py            # Landing page
│   │   ├── analysis_page.py        # Results display
│   │   ├── waveform_view.py        # QGraphicsView waveform
│   │   ├── results_panel.py        # Classification table
│   │   ├── transcription_panel.py  # Full transcription display
│   │   ├── theme.py                # Design tokens
│   │   ├── styles.py               # QSS stylesheet generator
│   │   └── ... (15+ widget files)
│   └── assets/                     # Fonts, passages
│
├── shared/                         # Shared utilities
│   └── reporting/
│       └── report_builder.py       # Typst-based PDF report generator
│
├── docs/                           # Documentation
│   ├── SUMMARY.md                  # This file
│   ├── project.md                  # Project overview
│   ├── project-understanding.md    # Detailed understanding
│   ├── accuracy-estimates.md       # Model accuracy estimates
│   ├── *.typ                       # Typst document templates
│   └── chapters/                   # Chapter files for report
│
├── data/                           # Audio datasets
├── scripts/                        # Utility scripts
├── docker-compose.yml              # Container orchestration
└── render.yaml                     # Cloud deployment config
```

---

## Appendix A: Model Weights Registry

```json
{
  "classification": {
    "prolongation": "model/weights/prolongation_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base_best.pt",
    "block": "model/weights/block_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base_best.pt",
    "soundrep": "model/weights/soundrep_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base_best.pt",
    "wordrep": "model/weights/wordrep_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base_best.pt",
    "interjection": "model/weights/interjection_e20_b8_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base_best.pt"
  },
  "classification_multitask": {
    "path": "model/weights/multi_e20_b16_lr3e-5_frz3_focal_g2_ga1_wu500_wd0.01_ml3_s42_train_w2v2base_best.pt",
    "model_name": "facebook/wav2vec2-base",
    "thresholds": {
      "prolongation": 0.45,
      "block": 0.4,
      "soundrep": 0.4,
      "wordrep": 0.4,
      "interjection": 0.4
    }
  },
  "localization": {
    "cnn": "model/weights/cnnloc_e30_b8_lr0.001_n128_h512_ml3_d0.4_pa7_wd0.0001_vr0.2_s42_train_best.pt",
    "wav2vec2": "model/weights/w2v2loc_e20_b4_lr3e-5_frz5_wu500_hd256_d0.3_wd0.01_ml3_pa5_vr0.2_s42_train_w2v2base_best.pt"
  }
}
```

### Weight Filename Convention

The checkpoint filenames encode training hyperparameters:
- `e20`: 20 epochs
- `b8`/`b16`: batch size 8/16
- `lr3e-5`: learning rate 3×10⁻⁵
- `frz3`/`frz5`: froze backbone for 3/5 epochs
- `focal_g2`: FocalLoss with gamma=2
- `ga1`: gradient accumulation steps = 1
- `wu500`: warmup steps = 500
- `wd0.01`: weight decay = 0.01
- `ml3`: max length = 3 seconds
- `s42`: random seed = 42
- `w2v2base`: Wav2Vec2 base model

---

## Appendix B: Glossary

| Term | Definition |
|---|---|
| **Dysfluency** | Any break, irregularity, or non-lexical vocable that occurs within the flow of otherwise fluent speech |
| **Prolongation** | A sound is held for a longer period than normal (e.g., "mmmmmy") |
| **Block** | Airflow stops completely; the speaker is unable to produce sound |
| **Sound Repetition** | Repeating a sound or syllable (e.g., "b-b-ball") |
| **Word Repetition** | Repeating a whole word (e.g., "I-I-I") |
| **Interjection** | Filler words/sounds (e.g., "um", "uh", "like") |
| **Wav2Vec 2.0** | Self-supervised speech representation model by Facebook AI |
| **Whisper** | Automatic speech recognition model by OpenAI |
| **Mel-spectrogram** | Time-frequency representation of audio using mel-scaled filter banks |
| **Saliency** | Per-frame probability of dysfluency presence (from multi-task classifier) |
| **IoU** | Intersection over Union; metric for temporal overlap between predicted and true regions |
| **AUROC** | Area Under Receiver Operating Characteristic curve |
| **AUPRC** | Area Under Precision-Recall Curve |
| **CTC** | Connectionist Temporal Classification; loss for unaligned sequence labeling |
| **Focal Loss** | Loss function that down-weights easy examples to focus on hard cases |
| **Typst** | Modern typesetting system used for PDF report generation |
