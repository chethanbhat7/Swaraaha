# TODO — Swaraaha Project

> **Scope:** Desktop Application + Model Building only.
> **Excluded:** Docker, Render, Backend (FastAPI), Frontend (React).
> **Team:** Shreekrishna, Chethan, Srinivas, Skanda.

---

## Team Assignments & Work Streams

| Member | Work Stream | Focus |
|---|---|---|
| **Shreekrishna** | Classification Pipeline | Wav2Vec 2.0 per-class classifiers + hybrid combiner |
| **Chethan** | Localization Pipeline | CNN spectrogram model + transcript/sequence exploration |
| **Srinivas** | Desktop Application | PyQt5 app, UI, audio, model integration |
| **Skanda** | Data & Training Infrastructure | Dataset, loaders, training loops, evaluation |

---

## Dependency Graph

### Per-Person Task Chains

**Shreekrishna (Classification)**
```
1.1 models/ scaffold ──► 2.1 Prolongation ─┐
                  ├──► 2.2 Block         ─┤
                  ├──► 2.3 Sound Rep     ─┤──► 3.1 Hybrid Combiner
                  ├──► 2.4 Word Rep      ─┤
                  └──► 2.5 Interjection  ─┘
```

**Chethan (Localization)**
```
1.2 Spectrogram pipeline ──► 2.6 CNN Localization ──┐
                          └──► 2.7 Audio Preproc ─────┤──► 3.5 (training, with Skanda)
                                                     └──► (transcript exploration)
```

**Srinivas (Desktop App)**
```
1.3 app/ scaffold + UI ──► 2.8 Audio Record/Play ──┐
                         └──► 2.9 File Browser    ──┤──► 3.2 Wire Audio-UI ──► 3.3 Model Runner ──► 3.7 Model Integration ──► 3.8 Results UI ──► 3.9 E2E Testing
```

**Skanda (Data & Training)**
```
1.4 Dataset + Loaders ──► 2.10 Classifier Training ──┐
                            └──► 2.11 Localization Training ──┤──► 3.4 Train Classifiers ──┐
                                    └──► 2.12 Eval Framework   ──┤──► 3.5 Train Localization ──┤──► 3.6 Full Eval ──► 3.9
```

### Cross-Person Dependencies

| This task | Waits for | Why |
|---|---|---|
| **3.1** Hybrid Combiner | 2.1–2.5 (all classifiers) | Combiner needs all 5 classifier architectures to be defined |
| **3.4** Train Classifiers | 2.10 + 1.4 | Training pipeline + dataset must both be ready |
| **3.5** Train Localization | 2.11 + 2.7 | Localization training pipeline + preprocessing must both be ready |
| **3.6** Full Evaluation | 3.1, 3.4, 3.5, 2.12 | All models trained + eval framework ready |
| **3.7** Model Integration | 3.3 + 3.6 | App inference wrapper ready + models evaluated |
| **3.9** E2E Testing | 3.8 | Full UI with results display complete |

### Critical Path

```
1.1 → 2.1–2.5 → 3.1 → 3.6 → 3.7 → 3.8 → 3.9
                                    ↑
1.4 → 2.10 → 3.4 ─────────────────┘
```

The longest chain is **Shreekrishna's classifiers → hybrid combiner → evaluation → integration**. This is the bottleneck — everything else can run ahead of it.

---

## Phase 1 — Foundation (All parallel, no blockers)

### Task 1.1: Scaffold `models/` Directory
| | |
|---|---|
| **Assignee** | Shreekrishna |
| **Depends on** | Nothing |
| **Blocks** | Task 2.1, 2.2, 2.3, 2.4, 2.5, 3.1 |
| **Effort** | Small (1–2 hours) |

**Description:**
Create the full directory structure and Python package scaffolding under `models/`.

**Guidance:**
```
models/
├── __init__.py
├── classification/
│   ├── __init__.py
│   ├── prolongation.py      # Wav2Vec2 binary classifier for prolongation
│   ├── block.py             # Wav2Vec2 binary classifier for block
│   ├── soundrep.py          # Wav2Vec2 binary classifier for sound repetition
│   ├── wordrep.py           # Wav2Vec2 binary classifier for word repetition
│   ├── interjection.py      # Wav2Vec2 binary classifier for interjection
│   └── hybrid.py            # Hybrid combiner that sits on top of the 5
├── localization/
│   ├── __init__.py
│   ├── cnn_spectrogram.py   # CNN model over spectrogram images
│   └── sequence_based.py    # Transcript/sequence-based localization (exploration)
├── data/
│   ├── __init__.py
│   ├── dataset.py           # PyTorch Dataset class
│   └── preprocessing.py     # Audio preprocessing (resampling, normalization, spectrogram generation)
├── training/
│   ├── __init__.py
│   ├── train_classifier.py  # Training loop for classifiers
│   ├── train_localizer.py   # Training loop for localization models
│   └── utils.py             # Checkpointing, logging, LR schedulers
├── evaluation/
│   ├── __init__.py
│   └── metrics.py           # Precision, recall, F1, localization accuracy
├── config/
│   └── defaults.py          # Hyperparameters, paths, constants
└── weights/                 # Saved model weights (gitignored)
    └── .gitkeep
```

- Each classifier file should define a class that wraps `Wav2Vec2ForSequenceClassification` from HuggingFace `transformers`.
- Use `pytorch` and `transformers` — confirm these are the chosen libraries.
- Add a `requirements.txt` at `models/requirements.txt` listing: `torch`, `transformers`, `librosa`, `soundfile`, `numpy`.
- Do NOT implement model logic yet — just the skeleton files with placeholder classes and docstrings.
- Create `config/defaults.py` with all hyperparameters as named constants (learning rate, batch size, epochs, audio sample rate = 16000, num_classes = 5, etc.).

---

### Task 1.2: Spectrogram Generation Pipeline
| | |
|---|---|
| **Assignee** | Chethan |
| **Depends on** | Nothing |
| **Blocks** | Task 2.6, 2.7 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Build the pipeline that converts raw audio waveforms into spectrogram images suitable for CNN input.

**Guidance:**
- Work inside `models/data/preprocessing.py`.
- Use `librosa` to load audio files and generate mel-spectrograms.
- The output should be a 2D numpy array (or torch tensor) shaped like an image: `[channels, height, width]` where channels=1 (grayscale spectrogram).
- Key functions to implement:
  - `load_audio(path, sr=16000)` — loads and resamples audio to 16kHz.
  - `generate_mel_spectrogram(audio, sr=16000, n_mels=128, hop_length=512)` — returns a mel-spectrogram array.
  - `normalize_spectrogram(spec)` — min-max or z-score normalization.
  - `save_spectrogram(spec, path)` — optionally save as image for debugging.
- Also create a utility to visualize spectrograms with `matplotlib` for debugging (save to a `debug/` folder, gitignored).
- Test with a sample `.wav` file if available; if not, generate a synthetic sine wave to verify the pipeline works.
- Document the expected input/output shapes in docstrings.

---

### Task 1.3: Scaffold `app/` Directory + Design UI Layout
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Nothing |
| **Blocks** | Task 2.8, 2.9, 3.2, 3.3 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Set up the PyQt5 desktop application structure and design the main UI layout (without model integration).

**Guidance:**
```
app/
├── main.py                  # Entry point — QApplication setup
├── requirements.txt         # PyQt5, pyaudio (or sounddevice), numpy
├── ui/
│   ├── __init__.py
│   ├── main_window.py       # QMainWindow with menu bar, central widget, status bar
│   ├── audio_panel.py       # Widget: record/stop/play buttons, waveform display
│   ├── file_panel.py        # Widget: file browser, drag-and-drop for .wav files
│   ├── results_panel.py     # Widget: classification results + localization timeline
│   └── styles.py            # QSS stylesheets for consistent theming
├── core/
│   ├── __init__.py
│   ├── audio_handler.py     # Audio recording and playback (stub for now)
│   └── model_runner.py      # Model inference wrapper (stub for now)
└── assets/
    └── icons/               # App icons (optional)
```

**UI Design Requirements:**
- **Main Window**: Split into 3 sections — left sidebar (file browser), center (audio waveform + controls), right (results display).
- **Audio Panel**: Record button, Stop button, Play button, a `QGraphicsView` or `matplotlib` canvas for waveform visualization.
- **File Panel**: `QTreeView` showing local files, filter for `.wav` files, double-click to load.
- **Results Panel**: Top section shows classification results as a table (5 rows, one per dysfluency type, with confidence scores). Bottom section shows a timeline/heatmap for localization.
- Use `QSplitter` so panels are resizable.
- Set a minimum window size of 1200x800.
- Make it look clean — use consistent fonts, spacing, and a light color scheme.

---

### Task 1.4: Dataset Research & Data Loaders
| | |
|---|---|
| **Assignee** | Skanda |
| **Depends on** | Nothing |
| **Blocks** | Task 2.10, 2.11, 2.12, 3.4 |
| **Effort** | Large (5–6 hours) |

**Description:**
Research available stuttering datasets, decide which to use, and build PyTorch data loaders.

**Guidance:**
- **Dataset Research** — investigate these options:
  - [UCLASS](https://github.com/shanemhutchinson/UCLASS) — labeled stuttering events.
  - [FluencyBank](https://www.fluencybank.org/) — part of the TalkBank project.
  - [SEP-28k](https://github.com/ShirshoSaha-NC/SEP-28k) — 28k clips with dysfluency labels.
  - Any other publicly available stuttering/speech dysfluency dataset.
- **Decision criteria**: Must have labels for at least some of the 5 dysfluency types (prolongation, block, soundrep, wordrep, interjection). Audio format should be `.wav` or easily convertible.
- Once a dataset is chosen:
  - Create `models/data/dataset.py` with a `StutterDataset(torch.utils.data.Dataset)` class.
  - It should accept a root directory path, scan for `.wav` files + corresponding label files.
  - `__getitem__` should return: `(audio_tensor, label_vector)` where label_vector is a multi-hot tensor of shape `[5]` (one per dysfluency class).
  - Support optional augmentation: time stretching, pitch shifting, adding noise (via `librosa.effects`).
- Create `models/data/preprocessing.py` (if not already done by Chethan for spectrograms — coordinate) with:
  - `load_audio(path, sr=16000)` — shared utility.
  - `pad_or_truncate(audio, target_length)` — ensures fixed-length input for batched training.
- Write `models/config/defaults.py` with dataset paths, sample rate, target audio length, augmentation flags.
- Document the dataset format expected (directory layout, label file format) in a `models/data/README.md`.

---

## Phase 2 — Core Model Development (All parallel, each depends on their Phase 1 task)

### Task 2.1: Wav2Vec 2.0 Classifier — Prolongation
| | |
|---|---|
| **Assignee** | Shreekrishna |
| **Depends on** | Task 1.1 |
| **Blocks** | Task 3.1 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Build a binary Wav2Vec 2.0 classifier that detects prolongation dysfluency.

**Guidance:**
- File: `models/classification/prolongation.py`
- Use `Wav2Vec2ForSequenceClassification` from `transformers` with `num_labels=2` (prolongation present / not present).
- Base model: `facebook/wav2vec2-base` (95M params, good balance of performance and speed).
- Class structure:
  ```python
  class ProlongationClassifier:
      def __init__(self, model_name="facebook/wav2vec2-base"):
          # Load pretrained Wav2Vec2 + classification head
      
      def forward(self, input_values, attention_mask=None):
          # Returns logits of shape [batch, 2]
      
      def predict(self, audio_tensor):
          # Returns (label: int, confidence: float)
      
      @staticmethod
      def from_pretrained(path):
          # Load saved checkpoint
      
      def save(self, path):
          # Save checkpoint
  ```
- The model expects raw waveform (not spectrograms) as input — Wav2Vec 2.0 handles its own feature extraction.
- Input shape: `[batch_size, sequence_length]` where sequence_length corresponds to audio at 16kHz.
- Add a `forward` method that handles both training (returns loss) and inference (returns predictions).
- Reference: See HuggingFace docs for `Wav2Vec2ForSequenceClassification`.

---

### Task 2.2: Wav2Vec 2.0 Classifier — Block
| | |
|---|---|
| **Assignee** | Shreekrishna |
| **Depends on** | Task 1.1 |
| **Blocks** | Task 3.1 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Build a binary Wav2Vec 2.0 classifier that detects block dysfluency.

**Guidance:**
- File: `models/classification/block.py`
- Same structure as Task 2.1 but for the `block` dysfluency class.
- Can reuse the base class from Task 2.1 — consider creating a `BaseWav2VecClassifier` in `models/classification/__init__.py` that all 5 classifiers inherit from, to avoid code duplication.
- The only difference between classifiers is the class name and potentially class-specific hyperparameters.
- After implementing, verify that all 5 classifiers (once built) share the same interface.

---

### Task 2.3: Wav2Vec 2.0 Classifier — Sound Repetition
| | |
|---|---|
| **Assignee** | Shreekrishna |
| **Depends on** | Task 1.1 |
| **Blocks** | Task 3.1 |
| **Effort** | Small (2–3 hours) |

**Description:**
Build a binary Wav2Vec 2.0 classifier for sound repetition (`soundrep`).

**Guidance:**
- File: `models/classification/soundrep.py`
- Same pattern as Tasks 2.1 and 2.2.
- If a `BaseWav2VecClassifier` was created, this is nearly trivial — just instantiate with the right class label.
- Sound repetition = repeating a sound/syllable (e.g., "b-b-ball"). The model learns to distinguish this from fluent speech.

---

### Task 2.4: Wav2Vec 2.0 Classifier — Word Repetition
| | |
|---|---|
| **Assignee** | Shreekrishna |
| **Depends on** | Task 1.1 |
| **Blocks** | Task 3.1 |
| **Effort** | Small (2–3 hours) |

**Description:**
Build a binary Wav2Vec 2.0 classifier for word repetition (`wordrep`).

**Guidance:**
- File: `models/classification/wordrep.py`
- Same pattern. Word repetition = repeating whole words (e.g., "I-I-I want").
- Ensure the class naming is consistent with the other classifiers.

---

### Task 2.5: Wav2Vec 2.0 Classifier — Interjection
| | |
|---|---|
| **Assignee** | Shreekrishna |
| **Depends on** | Task 1.1 |
| **Blocks** | Task 3.1 |
| **Effort** | Small (2–3 hours) |

**Description:**
Build a binary Wav2Vec 2.0 classifier for interjection dysfluency.

**Guidance:**
- File: `models/classification/interjection.py`
- Same pattern. Interjection = filler words/sounds (e.g., "um", "uh", "like" used as filler).
- Once all 5 classifiers are done, verify they all have the same API: `__init__`, `forward`, `predict`, `from_pretrained`, `save`.

---

### Task 2.6: CNN Spectrogram Localization Model
| | |
|---|---|
| **Assignee** | Chethan |
| **Depends on** | Task 1.2 |
| **Blocks** | Task 3.5 |
| **Effort** | Large (5–6 hours) |

**Description:**
Build a CNN model that takes spectrogram images and outputs dysfluency location as a time-series probability map.

**Guidance:**
- File: `models/localization/cnn_spectrogram.py`
- **Input**: Mel-spectrogram of shape `[batch, 1, n_mels, time_steps]` (grayscale image).
- **Output**: Per-frame probability map of shape `[batch, 1, time_steps]` — probability that each time frame contains a dysfluency event. This is a sequence labeling / segmentation task.
- Architecture suggestion:
  - Use a small CNN backbone (3–4 conv layers with batch norm and ReLU, max pooling).
  - After convolutional layers, use a fully connected head or transposed convolutions to upsample back to the original time resolution.
  - Final activation: sigmoid (per-frame probability between 0 and 1).
- Class structure:
  ```python
  class CNNSpectrogramLocalizer:
      def __init__(self, n_mels=128, num_classes=1):
          # Define CNN layers
      
      def forward(self, spectrograms):
          # Input: [B, 1, n_mels, T]
          # Output: [B, 1, T] — per-frame probabilities
      
      def predict(self, spectrogram, threshold=0.5):
          # Returns list of (start_time, end_time, confidence) tuples
      
      def save(self, path):
          ...
      
      @staticmethod
      def from_pretrained(path):
          ...
  ```
- The model should output raw probabilities; thresholding is done at inference time.
- For training, use `BCELoss` (binary cross-entropy) since each frame is an independent binary prediction.
- Coordinate with Skanda (Task 2.10) on how the training data labels are formatted — the localization labels need to indicate which time frames are dysfluent.
- Consider using dropout (0.3–0.5) between conv layers to prevent overfitting, since stuttering datasets are typically small.

---

### Task 2.7: Audio Preprocessing for Localization
| | |
|---|---|
| **Assignee** | Chethan |
| **Depends on** | Task 1.2 |
| **Blocks** | Task 2.6 (partial), Task 3.5 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Extend the preprocessing pipeline to support localization-specific needs: aligned spectrogram + label generation.

**Guidance:**
- Work in `models/data/preprocessing.py` and `models/data/dataset.py`.
- For localization, each training sample needs:
  - Input: mel-spectrogram of the audio clip.
  - Label: a binary mask over time frames indicating which frames are dysfluent.
- The label mask must be the same length as the spectrogram's time dimension.
- Key function: `create_frame_labels(dysfluency_intervals, spectrogram_length, sr, hop_length)` — takes a list of `(start_sec, end_sec)` intervals and produces a binary numpy array aligned to the spectrogram frames.
- Ensure that the spectrogram generation (from Task 1.2) and frame label generation use the same `hop_length` and `sr` so they are aligned.
- Create a `LocalizationDataset(torch.utils.data.Dataset)` class that returns `(spectrogram_tensor, frame_label_tensor)`.
- Handle variable-length audio by padding spectrograms to a max length within a batch (use `torch.nn.utils.rnn.pad_sequence` or manual padding).
- Test with synthetic data: create a 5-second sine wave, mark frames 50–100 as "dysfluent", verify the label mask is correct.

---

### Task 2.8: Audio Recording & Playback in Desktop App
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Task 1.3 |
| **Blocks** | Task 3.2 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Implement audio recording from microphone and playback in the PyQt5 app.

**Guidance:**
- File: `app/core/audio_handler.py`
- Use `sounddevice` (preferred) or `pyaudio` for recording. Add to `app/requirements.txt`.
- Class structure:
  ```python
  class AudioHandler:
      def __init__(self, sample_rate=16000, channels=1):
          ...
      
      def start_recording(self):
          # Start capturing audio from default mic into a buffer
      
      def stop_recording(self) -> np.ndarray:
          # Stop recording, return the captured audio as numpy array
      
      def play_audio(self, audio: np.ndarray):
          # Play audio through speakers
      
      def save_audio(self, audio: np.ndarray, path: str):
          # Save to .wav file using soundfile
      
      def load_audio(self, path: str) -> np.ndarray:
          # Load .wav file, return numpy array at 16kHz
  ```
- Recording should happen in a background thread to avoid freezing the UI.
- Use `QThread` or `threading.Thread` for non-blocking recording.
- Connect recording start/stop to the Audio Panel buttons (from Task 1.3).
- After recording, automatically update the waveform display in the Audio Panel.
- Test: record 3 seconds of audio, save to file, play back, verify it matches.

---

### Task 2.9: File Browser & Import in Desktop App
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Task 1.3 |
| **Blocks** | Task 3.2 |
| **Effort** | Small (2–3 hours) |

**Description:**
Implement file browsing and .wav file import in the desktop app.

**Guidance:**
- File: `app/core/audio_handler.py` (extend) and `app/ui/file_panel.py` (connect).
- Use `QFileSystemModel` for the `QTreeView` in the file panel.
- Filter to show only `.wav` files (or audio files in general).
- Double-click on a file should:
  1. Load the audio using `AudioHandler.load_audio()`.
  2. Update the waveform display in the Audio Panel.
  3. Store the loaded audio path for later use by the model.
- Add a "Recent Files" section using `QSettings` to persist recently opened files across app restarts.
- Add drag-and-drop support: user can drag a `.wav` file from their OS file manager onto the app window, and it loads automatically. Implement via `setAcceptDrops(True)` on the main window and override `dragEnterEvent` / `dropEvent`.

---

### Task 2.10: Training Pipeline for Classifiers ✅ DONE
| | |
|---|---|
| **Assignee** | Skanda |
| **Depends on** | Task 1.4 |
| **Blocks** | Task 3.4 |
| **Effort** | Large (5–6 hours) |
| **Status** | ✅ Completed |

**Description:**
Build the training loop and infrastructure for training the Wav2Vec 2.0 binary classifiers.

**Implementation:**
- File: `model/training/train_classifier.py`
- CLI arguments: `--class_name`, `--data_dir`, `--epochs`, `--batch_size`, `--lr`, `--output_dir`, `--warmup_steps`, `--patience`, `--model_name`, `--seed`
- Loads `ClassificationDataset` from `model/data/dataset.py`
- 80/20 stratified train/val split (first positive class per sample used for stratification)
- AdamW optimizer with linear warmup schedule
- Class imbalance handled via `pos_weight` in `BCEWithLogitsLoss`
- CSV logging via `CSVLogger` utility
- Early stopping on validation F1 (configurable patience, default 5)
- Saves best + final checkpoints, training curve PNGs
- Shell script: `model/training/train_all_classifiers.sh`

```bash
bash model/training/train_all_classifiers.sh data 20 8
# Or individually:
python -m model.training.train_classifier --class_name prolongation --data_dir data --epochs 20
```

---

### Task 2.11: Training Pipeline for Localization Models ✅ DONE
| | |
|---|---|
| **Assignee** | Skanda |
| **Depends on** | Task 1.4, Task 2.7 |
| **Blocks** | Task 3.5 |
| **Effort** | Medium (4–5 hours) |
| **Status** | ✅ Completed |

**Description:**
Build the training loop for the CNN spectrogram localization model.

**Implementation:**
- File: `model/training/train_localizer.py`
- CLI arguments: `--data_dir`, `--epochs`, `--batch_size`, `--lr`, `--output_dir`, `--n_mels`, `--hop_length`, `--cnn_type` (wrapper/module), `--dropout`, `--patience`, `--seed`
- Loads `LocalizationDataset` from `model/data/dataset.py`
- Frame-level `BCEWithLogitsLoss` with `pos_weight=5.0` for class imbalance
- AdamW optimizer + cosine annealing LR scheduler
- Evaluates frame-level P/R/F1 and event-level IoU each epoch
- Early stopping on validation frame F1
- Saves best + final checkpoints, training curve PNGs

```bash
python -m model.training.train_localizer --data_dir data --epochs 30 --batch_size 8
```

---

### Task 2.12: Evaluation Framework ✅ DONE
| | |
|---|---|
| **Assignee** | Skanda |
| **Depends on** | Task 1.4 |
| **Blocks** | Task 3.6 |
| **Effort** | Medium (3–4 hours) |
| **Status** | ✅ Completed |

**Description:**
Build a unified evaluation script that can assess any trained model and produce a report.

**Implementation:**
- Files: `model/evaluation/metrics.py` and `model/evaluation/evaluate.py`
- `metrics.py`:
  - `compute_classification_metrics(y_true, y_pred)` — per-class P/R/F1, macro averages, accuracy
  - `compute_localization_metrics(y_true_frames, y_pred_frames)` — frame-level P/R/F1 + event-level detection accuracy & mean IoU
  - `confusion_matrix()` + `save_confusion_matrix_plot()` — generates and saves PNG
  - `save_report()` — JSON export
  - `print_classification_report()` / `print_localization_report()` — human-readable stdout
- `evaluate.py`:
  - CLI: `--model_type classifier|localizer`, `--class_name`, `--model_path`, `--data_dir`, `--output_dir`, `--threshold`, `--save_misclassified`
  - Loads trained model, runs inference, computes all metrics
  - Saves JSON report + confusion matrix PNG (classifiers)
  - Optionally saves misclassified samples for manual review

```bash
python -m model.evaluation.evaluate --model_type classifier --class_name prolongation --model_path model/weights/prolongation_best.pt --data_dir data
python -m model.evaluation.evaluate --model_type localizer --model_path model/weights/localizer_best.pt --data_dir data
```

---

## Phase 3 — Hybrid Model, Alternative Localization, Integration (Parallel where possible)

### Task 3.1: Hybrid Combiner Model
| | |
|---|---|
| **Assignee** | Shreekrishna |
| **Depends on** | Tasks 2.1–2.5 (all 5 classifiers must be architecturally complete) |
| **Blocks** | Task 3.6 |
| **Effort** | Large (5–6 hours) |

**Description:**
Build the hybrid model that combines outputs from the 5 individual binary classifiers into a final classification result.

**Guidance:**
- File: `models/classification/hybrid.py`
- **Design decision needed**: The combination strategy is not yet decided. Consider these options:
  1. **Weighted voting**: Each classifier votes, weighted by confidence. Simple but may not be optimal.
  2. **Learned meta-classifier**: Train a small neural network (e.g., 2-layer MLP) on the 5 classifier logits to produce the final prediction. More powerful but needs joint training data.
  3. **Stacking**: Train the meta-classifier using cross-validation predictions from the base classifiers.
- **Recommendation**: Start with option 2 (learned meta-classifier) since it's the most flexible. The meta-classifier takes 5 logits as input and outputs 5 probabilities (one per class).
- Class structure:
  ```python
  class HybridClassifier:
      def __init__(self, base_classifiers: list, combiner):
          # base_classifiers: list of 5 trained Wav2Vec2 classifiers
          # combiner: the meta-classifier (MLP or similar)
      
      def forward(self, input_values):
          # 1. Run input through each base classifier → get 5 logit vectors
          # 2. Concatenate logits → input to combiner
          # 3. Return final predictions
      
      def predict(self, audio_tensor):
          # Returns dict: {class_name: (label, confidence)}
      
      def save(self, path):
          ...
      
      @staticmethod
      def from_pretrained(path):
          ...
  ```
- The combiner MLP architecture:
  - Input: 5 logits (one per base classifier)
  - Hidden layer: 32 units, ReLU, dropout 0.3
  - Output: 5 units (one per class), sigmoid
  - This is intentionally small since it only processes 5 numbers.
- For training the combiner: freeze the base classifiers, feed their logits into the MLP, train with `BCELoss`.
- Document the design decision and rationale in the file's docstring.

---

### Task 3.2: Connect Audio Recording/Playback to UI
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Tasks 2.8, 2.9 |
| **Blocks** | Task 3.3 |
| **Effort** | Small (2–3 hours) |

**Description:**
Wire up the audio handler to the UI buttons and displays.

**Guidance:**
- File: `app/ui/audio_panel.py` (modify) and `app/main.py` (modify).
- Connect buttons:
  - "Record" button → `audio_handler.start_recording()`
  - "Stop" button → `audio_handler.stop_recording()`
  - "Play" button → `audio_handler.play_audio(current_audio)`
  - "Load File" button → open `QFileDialog` → load audio
- After recording or loading, update the waveform display:
  - Plot the audio waveform using `matplotlib` embedded in a `QGraphicsView` (use `FigureCanvasQTAgg`).
  - Show the time axis in seconds.
  - Highlight the current playback position with a moving vertical line during playback.
- Add a status bar message: "Ready", "Recording...", "Playing...", "Loaded: filename.wav".
- Add keyboard shortcuts: `Space` = play/stop, `R` = record, `Ctrl+O` = open file.
- Test the full flow: record → waveform appears → play back → verify.

---

### Task 3.3: Model Runner (Inference Wrapper) in Desktop App
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Task 3.2 |
| **Blocks** | Task 3.7 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Build the inference wrapper that loads trained models and runs predictions from the desktop app.

**Guidance:**
- File: `app/core/model_runner.py`
- This wrapper imports from `models/` (the shared model code).
- Class structure:
  ```python
  class ModelRunner:
      def __init__(self, models_dir: str):
          # Load all trained models from models_dir
          # - 5 individual classifiers
          # - hybrid classifier
          # - localization model
      
      def classify(self, audio: np.ndarray) -> dict:
          # Run hybrid classifier on audio
          # Returns: {"prolongation": (bool, float), "block": (bool, float), ...}
      
      def localize(self, audio: np.ndarray) -> list:
          # Run localization model on audio
          # Returns: [(start_sec, end_sec, confidence), ...]
      
      def analyze(self, audio: np.ndarray) -> dict:
          # Run both pipelines
          # Returns: {"classifications": ..., "localizations": ...}
  ```
- Model loading should be lazy (load on first use, not at app startup) to avoid slow startup.
- Handle the case where model weights don't exist yet — show a user-friendly message: "Model weights not found. Please train models first."
- Run inference in a background thread (`QThread`) with a progress bar ("Analyzing audio...").
- After analysis completes, emit a Qt signal with the results to update the Results Panel.
- Ensure the model's expected input format (sample rate, length) matches what `AudioHandler` provides.

---

### Task 3.4: Train All Classifiers (Execution)
| | |
|---|---|
| **Assignee** | Skanda |
| **Depends on** | Tasks 2.10, 1.4 |
| **Blocks** | Task 3.6 |
| **Effort** | Small (1–2 hours active, long compute time) |

**Description:**
Execute training for all 5 Wav2Vec 2.0 binary classifiers using the training pipeline.

**Guidance:**
- Use the script from Task 2.10.
- Run each classifier training on a machine with sufficient resources.
- Monitor training — watch for overfitting (train loss decreasing but val loss increasing).
- Save all trained weights to `models/weights/`.
- Document training results: epochs trained, final F1 scores, any issues encountered.
- If dataset is small, consider using `facebook/wav2vec2-large` for better representations (but slower training).
- Store training logs in `models/weights/logs/` for reference.

---

### Task 3.5: Train Localization Model (Execution)
| | |
|---|---|
| **Assignee** | Skanda |
| **Depends on** | Tasks 2.11, 2.7 |
| **Blocks** | Task 3.6 |
| **Effort** | Small (1–2 hours active, long compute time) |

**Description:**
Execute training for the CNN spectrogram localization model.

**Guidance:**
- Use the script from Task 2.11.
- Run training and monitor frame-level F1 and IoU metrics.
- Save trained weights to `models/weights/`.
- Generate training curve plots for documentation.
- If results are poor, iterate: adjust learning rate, add data augmentation, try a larger CNN, or adjust the spectrogram parameters (n_mels, hop_length).

---

### Task 3.6: Run Full Evaluation on All Trained Models
| | |
|---|---|
| **Assignee** | Skanda |
| **Depends on** | Tasks 3.1, 3.4, 3.5, 2.12 |
| **Blocks** | Task 3.7 |
| **Effort** | Medium (2–3 hours) |

**Description:**
Run the evaluation framework on all trained models and produce a comprehensive report.

**Guidance:**
- Use `models/evaluation/evaluate.py` from Task 2.12.
- Evaluate each of the 5 classifiers individually AND the hybrid combiner.
- Evaluate the localization model.
- Save results to `models/evaluation/reports/` as JSON + human-readable summary.
- Key metrics to report:
  - Per-class F1 for each dysfluency type (classification).
  - Macro-averaged F1 across all 5 classes.
  - Frame-level precision/recall/F1 for localization.
  - Event-level detection accuracy and mean IoU for localization.
- Flag any classes with F1 < 0.7 — these need attention.
- This report is the final word on model performance before integration.

---

### Task 3.7: Integrate Models into Desktop App
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Tasks 3.3, 3.6 (models should be trained and evaluated) |
| **Blocks** | Task 3.8 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Connect the trained models to the desktop app so users can analyze audio files.

**Guidance:**
- Modify `app/core/model_runner.py` to point to the actual trained weights.
- Add an "Analyze" button to the Audio Panel that triggers `model_runner.analyze(current_audio)`.
- Wire the results from `ModelRunner` to the Results Panel (Task 3.8).
- Add error handling:
  - If no audio is loaded → show "Please load or record audio first."
  - If model weights are missing → show "Models not trained yet."
  - If inference fails → show error message with details.
- Add a "Loading models..." indicator on first analysis (models can take a few seconds to load).
- Test the full flow: load audio → click Analyze → results appear.

---

### Task 3.8: Results Visualization in Desktop App
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Task 3.7 |
| **Blocks** | Task 3.9 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Build the results display in the desktop app showing classification scores and localization timeline.

**Guidance:**
- File: `app/ui/results_panel.py` (build out from Task 1.3 stub).
- **Classification Results** (top half):
  - Display a table with 5 rows: Prolongation, Block, Sound Repetition, Word Repetition, Interjection.
  - Columns: Class Name | Detected (Yes/No) | Confidence (%).
  - Color-code: green for detected, gray for not detected.
  - Use `QTableWidget` or custom `QFrame` widgets.
- **Localization Timeline** (bottom half):
  - Show the audio waveform (reuse from Audio Panel).
  - Overlay colored regions where dysfluency is detected (red/orange highlights on the waveform).
  - Show a legend mapping colors to dysfluency types.
  - If multiple types overlap at the same time, stack or use semi-transparent overlays.
  - Clicking on a highlighted region should jump playback to that point.
- Use `matplotlib` embedded via `FigureCanvasQTAgg` for the waveform + overlay plot.
- Add export option: "Save Results as Image" (save the visualization as PNG).
- Add export option: "Save Results as JSON" (structured output for downstream use).

---

### Task 3.9: End-to-End Testing & Polish
| | |
|---|---|
| **Assignee** | Srinivas (lead), all team members |
| **Depends on** | Task 3.8 |
| **Blocks** | None (final task) |
| **Effort** | Medium (3–4 hours) |

**Description:**
Test the complete desktop application end-to-end and fix any issues.

**Guidance:**
- Test cases to run:
  1. Record audio → Analyze → verify results appear.
  2. Load a `.wav` file → Analyze → verify results.
  3. Load a file with known stuttering → verify correct classification.
  4. Load a fluent audio file → verify no false positives.
  5. Check localization timeline aligns with actual dysfluency in audio.
  6. Test edge cases: very short audio (<1s), very long audio (>60s), silent audio.
  7. Test app stability: rapid record/stop/play cycles, loading multiple files in succession.
- Polish:
  - Ensure window resizing works correctly (panels resize proportionally).
  - Verify all keyboard shortcuts work.
  - Add tooltips to buttons.
  - Ensure the app doesn't crash on invalid inputs.
  - Add a "About" dialog with project info.
- Each team member should review their own module for edge cases.
- Document any known limitations or bugs in `app/KNOWN_ISSUES.md`.

---

## Summary Table

| Task | Assignee | Phase | Depends On | Effort | Status |
|---|---|---|---|---|---|
| 1.1 Scaffold `models/` | Shreekrishna | 1 | — | Small | ✅ |
| 1.2 Spectrogram Pipeline | Chethan | 1 | — | Medium | ✅ |
| 1.3 Scaffold `app/` | Srinivas | 1 | — | Medium | ✅ |
| 1.4 Dataset & Loaders | Skanda | 1 | — | Large | ✅ |
| 2.1 Classifier: Prolongation | Shreekrishna | 2 | 1.1 | Medium | ✅ |
| 2.2 Classifier: Block | Shreekrishna | 2 | 1.1 | Medium | ✅ |
| 2.3 Classifier: Sound Rep | Shreekrishna | 2 | 1.1 | Small | ✅ |
| 2.4 Classifier: Word Rep | Shreekrishna | 2 | 1.1 | Small | ✅ |
| 2.5 Classifier: Interjection | Shreekrishna | 2 | 1.1 | Small | ✅ |
| 2.6 CNN Localization Model | Chethan | 2 | 1.2 | Large | ✅ |
| 2.7 Audio Preprocessing (Local.) | Chethan | 2 | 1.2 | Medium | ✅ |
| 2.8 Audio Recording/Playback | Srinivas | 2 | 1.3 | Medium | ✅ |
| 2.9 File Browser & Import | Srinivas | 2 | 1.3 | Small | ⚠️ Partial |
| 2.10 Classifier Training | Skanda | 2 | 1.4 | Large | ✅ |
| 2.11 Localization Training | Skanda | 2 | 1.4, 2.7 | Medium | ✅ |
| 2.12 Evaluation Framework | Skanda | 2 | 1.4 | Medium | ✅ |
| 3.1 Hybrid Combiner | Shreekrishna | 3 | 2.1–2.5 | Large | ✅ |
| 3.2 Audio-UI Wiring | Srinivas | 3 | 2.8, 2.9 | Small | ✅ |
| 3.3 Model Runner | Srinivas | 3 | 3.2 | Medium | ❌ Stub only |
| 3.4 Train Classifiers | Shreekrishna | 3 | 2.10, 1.4 | Small | ❌ |
| 3.5 Train Localization | Shreekrishna | 3 | 2.11, 2.7 | Small | ❌ |
| 3.6 Full Evaluation | Skanda | 3 | 3.1, 3.4, 3.5, 2.12 | Medium | ❌ |
| 3.7 Model Integration | Srinivas | 3 | 3.3, 3.6 | Medium | ❌ |
| 3.8 Results Visualization | Srinivas | 3 | 3.7 | Medium | ❌ |
| 3.9 E2E Testing | All (Srinivas lead) | 3 | 3.8 | Medium | ❌ |

**Legend:** ✅ Done | ⚠️ Partial | ❌ Pending

> **Detailed Phase 3 tasks, assignments, and optimization tasks:** see `docs/todo-phase3.md`

---

## Parallelism Summary

**Maximum parallelism at each phase:**
- **Phase 1**: 4 tasks in parallel (1 per person)
- **Phase 2**: 8 tasks in parallel (Shreekrishna: 5 classifiers, Chethan: 2 tasks, Srinivas: 2 tasks, Skanda: 3 tasks)
- **Phase 3**: Shreekrishna (training + optimization), Chethan (data augmentation + cleanup), Srinivas (UI integration), Skanda (evaluation + analysis)

**Critical path:** `3.4/3.5 → 3.6 → 3.7 → 3.8 → 3.9`
