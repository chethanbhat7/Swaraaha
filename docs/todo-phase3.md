# TODO — Phase 3: Training, Integration & Optimization

> **Scope:** Model training execution, UI integration, optimization, and E2E testing.
> **Prerequisites:** Phase 1 ✅ complete, Phase 2 ✅ complete (all pipelines & frameworks built).
> **GPU Owner:** Shreekrishna — all training execution tasks run on his machine.

---

## Team Assignments

| Member | Role | Focus |
|---|---|---|
| **Shreekrishna** | Model & GPU Owner | Training execution, model optimization, hyperparameter tuning |
| **Chethan** | Data & Preprocessing | Data augmentation, preprocessing improvements, code cleanup |
| **Srinivas** | Desktop Application | Model runner, results UI, file browser completion, E2E testing |
| **Skanda** | Training & Evaluation | Training support, full evaluation, analysis & reporting |

---

## Dependency Graph

```
2.9 (partial) ─┐
                ├──► 3.3 ──► 3.7 ──► 3.8 ──► 3.9
4.1 ──┐              ↑
      ├──► 4.2 ──┐   │
4.3 ──┘          │   │
                 ├──► 4.4 ──► 4.5 ──► 4.6
4.7 ──► 4.8 ────┘
```

### Critical Path
```
4.1 → 4.2 → 4.4 → 4.5 → 4.6 → 3.7 → 3.8 → 3.9
```

---

## Phase 3 — Training Execution

### Task 3.4: Train All Classifiers (Execution)
| | |
|---|---|
| **Assignee** | Shreekrishna (GPU) |
| **Depends on** | Task 2.10 ✅, Task 1.4 ✅ |
| **Blocks** | Task 4.6 |
| **Effort** | Small (1–2 hours active, long compute time) |

**Description:**
Execute training for all 5 Wav2Vec 2.0 binary classifiers using the training pipeline from Task 2.10.

**Guidance:**
- Run on GPU machine: `bash model/training/train_all_classifiers.sh data 20 8`
- Or individually per classifier for better control:
  ```bash
  python -m model.training.train_classifier --class_name prolongation --data_dir data --epochs 20 --batch_size 8 --output_dir model/weights
  ```
- Monitor training for overfitting (train loss decreasing but val loss increasing)
- Save all trained weights to `model/weights/`
- Document: epochs trained, final F1 scores, any issues
- If dataset is small, consider `facebook/wav2vec2-large` for better representations
- Store training logs in `model/weights/` for reference

---

### Task 3.5: Train Localization Model (Execution)
| | |
|---|---|
| **Assignee** | Shreekrishna (GPU) |
| **Depends on** | Task 2.11 ✅, Task 2.7 ✅ |
| **Blocks** | Task 4.6 |
| **Effort** | Small (1–2 hours active, long compute time) |

**Description:**
Execute training for the CNN spectrogram localization model.

**Guidance:**
- Run: `python -m model.training.train_localizer --data_dir data --epochs 30 --batch_size 8 --output_dir model/weights`
- Monitor frame-level F1 and IoU metrics
- Save trained weights to `model/weights/`
- Generate training curve plots for documentation
- If results are poor, iterate: adjust LR, add augmentation, try larger CNN, adjust spectrogram params

---

## Phase 3 — Data & Preprocessing (Parallel with training)

### Task 4.1: Data Augmentation Pipeline
| | |
|---|---|
| **Assignee** | Chethan |
| **Depends on** | Task 1.4 ✅, Task 2.7 ✅ |
| **Blocks** | Task 4.2 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Implement audio data augmentation to improve model generalization with small stuttering datasets.

**Guidance:**
- File: `model/data/augmentation.py`
- Implement augmentations using `librosa.effects`:
  - **Time stretching**: randomly speed up/slow down (rate 0.8–1.2)
  - **Pitch shifting**: randomly shift pitch (±2 semitones)
  - **Additive noise**: white noise, pink noise at varying SNR (20–40 dB)
  - **Time masking**: randomly mask short time segments (spec augment)
  - **Frequency masking**: randomly mask mel frequency bands
- Create an `AugmentedDataset` wrapper that applies random augmentations on-the-fly
- Each augmentation should have configurable probability
- Add augmentation flags to `model/config/defaults.py`
- Test: verify augmented audio sounds reasonable, spectrograms look correct
- Document augmentation strategies and their expected impact

---

### Task 4.2: Preprocessing Improvements
| | |
|---|---|
| **Assignee** | Chethan |
| **Depends on** | Task 4.1 |
| **Blocks** | Task 4.4 |
| **Effort** | Small (2–3 hours) |

**Description:**
Improve the preprocessing pipeline to support training with augmentation and better data quality.

**Guidance:**
- File: `model/data/preprocessing.py` (extend)
- Add `augment_audio(audio, sr, p=0.5)` — applies random augmentation pipeline
- Add `compute_snr(signal, noise)` — signal-to-noise ratio utility
- Add audio quality checks: detect clipping, silence, very short clips
- Integrate augmentation into `ClassificationDataset` and `LocalizationDataset` as optional flag
- Ensure augmented data is compatible with both classifier (raw waveform) and localizer (spectrogram) inputs

---

### Task 4.3: Clean Up Duplicate Localization Code
| | |
|---|---|
| **Assignee** | Chethan |
| **Depends on** | Nothing |
| **Blocks** | Nothing |
| **Effort** | Small (1 hour) |

**Description:**
Remove duplicate localization files that overlap with the primary implementations.

**Guidance:**
- Files to review:
  - `model/localization/cnn.py` — duplicate of `cnn_spectrogram.py` (different API: nn.Module vs wrapper)
  - `model/localization/spectrogram.py` — duplicate of `model/data/preprocessing.py`
  - `model/localization/inference.py` — overlaps with `cnn_spectrogram.py`'s `predict()` method
- Decision: keep the wrapper-based `cnn_spectrogram.py` (consistent with classification pattern), deprecate or remove `cnn.py`
- If `cnn.py` (nn.Module) is preferred for training efficiency, consolidate into `cnn_spectrogram.py`
- Update imports in any files that reference the deprecated modules
- Ensure `model/localization/__init__.py` exports only the canonical classes

---

## Phase 3 — Training Support & Analysis (Parallel with training)

### Task 4.4: Training Monitoring & Analysis
| | |
|---|---|
| **Assignee** | Skanda |
| **Depends on** | Task 4.2 |
| **Blocks** | Task 4.5 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Monitor training runs, analyze results, and provide feedback for hyperparameter tuning.

**Guidance:**
- Monitor Shreekrishna's training runs (Tasks 3.4, 3.5)
- Track metrics per epoch: loss, F1, IoU, learning rate
- Identify overfitting/underfitting patterns
- Generate comparison plots: training vs validation curves
- Document observations: which classes are hardest, what augmentation helps
- Prepare analysis report for the team
- Flag any classes with F1 < 0.7 — these need attention

---

### Task 4.5: Hyperparameter Tuning Recommendations
| | |
|---|---|
| **Assignee** | Skanda |
| **Depends on** | Task 4.4 |
| **Blocks** | Task 4.6 |
| **Effort** | Medium (2–3 hours) |

**Description:**
Based on training analysis, recommend hyperparameter adjustments for optimum performance.

**Guidance:**
- Analyze training logs from Tasks 3.4 and 3.5
- Recommend adjustments for:
  - Learning rate (try 1e-5, 3e-5, 1e-4)
  - Batch size (try 4, 8, 16)
  - Warmup steps
  - Dropout rates
  - Augmentation probabilities
- For the CNN localizer: experiment with n_mels (64, 128, 256), hop_length (256, 512)
- Document recommended configs in `model/config/defaults.py`
- Prioritize: which changes give the biggest F1 improvement per compute hour

---

### Task 4.6: Full Evaluation
| | |
|---|---|
| **Assignee** | Skanda |
| **Depends on** | Tasks 3.4, 3.5, 4.5 |
| **Blocks** | Task 3.7 |
| **Effort** | Medium (2–3 hours) |

**Description:**
Run the evaluation framework on all trained models and produce a comprehensive report.

**Guidance:**
- Use `model/evaluation/evaluate.py` from Task 2.12
- Evaluate each of the 5 classifiers individually AND the hybrid combiner
- Evaluate the localization model
- Save results to `model/evaluation/reports/` as JSON + human-readable summary
- Key metrics:
  - Per-class F1 for each dysfluency type (classification)
  - Macro-averaged F1 across all 5 classes
  - Frame-level precision/recall/F1 for localization
  - Event-level detection accuracy and mean IoU
- Flag any classes with F1 < 0.7
- This report is the final word on model performance before integration

---

## Phase 3 — UI Integration (Parallel with training)

### Task 3.9: Complete File Browser & Import
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Task 2.9 (partial) |
| **Blocks** | Task 3.3 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Complete the file browser functionality that was left partial in Task 2.9.

**Guidance:**
- File: `app/ui/file_panel.py` (create or extend home_page.py)
- Implement `QTreeView` with `QFileSystemModel` for browsing local files
- Filter to show only `.wav` (and optionally `.mp3`, `.flac`) files
- Double-click loads the audio file into the app
- Add "Recent Files" section using `QSettings` to persist across restarts
- Add drag-and-drop: `setAcceptDrops(True)` on main window, override `dragEnterEvent` / `dropEvent`
- Add a "Browse" button that opens `QFileDialog` as fallback
- Coordinate with Srinivas's existing `main_window.py` file loading flow

---

### Task 3.3: Real Model Runner Implementation
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Task 3.9, Task 4.6 |
| **Blocks** | Task 3.7 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Replace the stub `ModelRunner` with real model loading and inference.

**Guidance:**
- File: `app/core/model_runner.py`
- Load trained models from `model/weights/`:
  - 5 individual classifiers
  - Hybrid classifier (`HybridClassifier`)
  - Localization model (`CNNSpectrogramLocalizer`)
- Lazy loading: load on first use, not at app startup
- Handle missing weights gracefully: "Model weights not found. Please train models first."
- Run inference in background thread (`QThread`) with progress bar
- Emit Qt signal with results when done
- Ensure input format matches what `AudioHandler` provides (16kHz, float32)
- Use `model.data.preprocessing.load_audio_from_array()` for format normalization

---

### Task 3.7: Model Integration
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Task 3.3, Task 4.6 |
| **Blocks** | Task 3.8 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Wire the real ModelRunner into the desktop app flow.

**Guidance:**
- Modify `app/ui/main_window.py`:
  - "Analyze" button triggers `model_runner.analyze(current_audio)`
  - Show "Loading models..." indicator on first analysis
  - Wire results to `ResultsPanel.set_results()`
- Error handling:
  - No audio loaded → "Please load or record audio first."
  - Model weights missing → "Models not trained yet."
  - Inference fails → error message with details
- Test full flow: load audio → click Analyze → results appear

---

### Task 3.8: Enhanced Results Visualization
| | |
|---|---|
| **Assignee** | Srinivas |
| **Depends on** | Task 3.7 |
| **Blocks** | Task 3.9 |
| **Effort** | Medium (3–4 hours) |

**Description:**
Enhance the results display with full classification and localization visualization.

**Guidance:**
- File: `app/ui/results_panel.py` (enhance existing)
- **Classification Results**: table with 5 rows, color-coded by class, confidence bars
- **Localization Timeline**: waveform with colored overlays per dysfluency type
  - Semi-transparent overlays for overlapping regions
  - Click to jump playback to that point
  - Legend mapping colors to dysfluency types
- Add export: "Save Results as Image" (PNG) and "Save Results as JSON"
- Use `matplotlib` via `FigureCanvasQTAgg` for waveform + overlay plot

---

## Phase 3 — Model Optimization (After initial training)

### Task 4.7: Hyperparameter Optimization
| | |
|---|---|
| **Assignee** | Shreekrishna (GPU) |
| **Depends on** | Task 4.5 (recommendations from Skanda) |
| **Blocks** | Task 4.8 |
| **Effort** | Medium (3–4 hours + GPU time) |

**Description:**
Run hyperparameter sweeps to find optimal training configurations.

**Guidance:**
- Based on Skanda's recommendations from Task 4.5
- Train classifiers with different configs:
  - Learning rate: [1e-5, 3e-5, 1e-4]
  - Batch size: [4, 8, 16]
  - Augmentation on/off
- Train localizer with different configs:
  - n_mels: [64, 128, 256]
  - hop_length: [256, 512]
  - CNN dropout: [0.3, 0.4, 0.5]
- Use `model/weights/hparam_search/` for experiment tracking
- Compare results: which config gives best F1 per compute hour
- Document final optimal configs in `model/config/defaults.py`

---

### Task 4.8: Model Size & Speed Optimization
| | |
|---|---|
| **Assignee** | Shreekrishna (GPU) |
| **Depends on** | Task 4.7 |
| **Blocks** | Task 3.7 (integration needs final models) |
| **Effort** | Medium (2–3 hours) |

**Description:**
Optimize trained models for inference speed and size, suitable for desktop app deployment.

**Guidance:**
- Consider model distillation: train a smaller student model from the Wav2Vec2 teacher
- Evaluate `facebook/wav2vec2-base` vs `facebook/wav2vec2-large` — is the accuracy gain worth 3x more params?
- Quantize models to FP16 or INT8 for faster CPU inference
- Benchmark inference time per audio clip on CPU (target: < 2 seconds for 10s audio)
- Export final models in a deployment-ready format
- Document model sizes and inference speeds
- If using the hybrid combiner, evaluate whether the MLP overhead is worth the accuracy gain

---

## Phase 3 — Final Testing

### Task 3.9 (revised): End-to-End Testing & Polish
| | |
|---|---|
| **Assignee** | Srinivas (lead), all team members |
| **Depends on** | Task 3.8, Task 4.8 |
| **Blocks** | None (final task) |
| **Effort** | Medium (3–4 hours) |

**Description:**
Test the complete desktop application end-to-end and polish for release.

**Guidance:**
- Test cases:
  1. Record audio → Analyze → verify results appear
  2. Load a `.wav` file → Analyze → verify results
  3. Load file with known stuttering → verify correct classification
  4. Load fluent audio → verify no false positives
  5. Check localization timeline alignment with actual dysfluency
  6. Edge cases: very short audio (<1s), very long audio (>60s), silent audio
  7. Stability: rapid record/stop/play cycles, loading multiple files
- Polish:
  - Window resizing, keyboard shortcuts, tooltips
  - "About" dialog with project info
  - Handle invalid inputs gracefully
- Each team member reviews their own module for edge cases
- Document known limitations in `app/KNOWN_ISSUES.md`

---

## Summary Table

| Task | Assignee | Phase | Depends On | Effort | Status |
|---|---|---|---|---|---|
| 3.4 Train Classifiers | Shreekrishna (GPU) | 3 | 2.10, 1.4 | Small | ❌ |
| 3.5 Train Localization | Shreekrishna (GPU) | 3 | 2.11, 2.7 | Small | ❌ |
| 3.9 Complete File Browser | Srinivas | 3 | 2.9 | Medium | ❌ |
| 3.3 Real Model Runner | Srinivas | 3 | 3.9, 4.6 | Medium | ❌ |
| 3.7 Model Integration | Srinivas | 3 | 3.3, 4.6 | Medium | ❌ |
| 3.8 Enhanced Results Viz | Srinivas | 3 | 3.7 | Medium | ❌ |
| 3.9 E2E Testing & Polish | All (Srinivas lead) | 3 | 3.8, 4.8 | Medium | ❌ |
| 4.1 Data Augmentation | Chethan | 3 | 1.4, 2.7 | Medium | ❌ |
| 4.2 Preprocessing Improve | Chethan | 3 | 4.1 | Small | ❌ |
| 4.3 Clean Duplicate Code | Chethan | 3 | — | Small | ❌ |
| 4.4 Training Monitoring | Skanda | 3 | 4.2 | Medium | ❌ |
| 4.5 Hyperparam Tuning Recs | Skanda | 3 | 4.4 | Medium | ❌ |
| 4.6 Full Evaluation | Skanda | 3 | 3.4, 3.5, 4.5 | Medium | ❌ |
| 4.7 Hyperparam Optimization | Shreekrishna (GPU) | 3 | 4.5 | Medium | ❌ |
| 4.8 Model Size/Speed Opt | Shreekrishna (GPU) | 3 | 4.7 | Medium | ❌ |

---

## Workload Distribution

| Member | Tasks | Total Effort |
|---|---|---|
| **Shreekrishna** (GPU) | 3.4, 3.5, 4.7, 4.8 | Small + Medium + Medium + Medium |
| **Chethan** | 4.1, 4.2, 4.3 | Medium + Small + Small |
| **Srinivas** | 3.9, 3.3, 3.7, 3.8, 3.9-test | Medium × 5 |
| **Skanda** | 4.4, 4.5, 4.6 | Medium × 3 |

**Parallelism:**
- **Wave 1** (parallel): Chethan (4.1, 4.3), Shreekrishna (3.4, 3.5), Srinivas (3.9), Skanda (4.4 after Chethan's 4.2)
- **Wave 2**: Chethan (4.2), Skanda (4.5), Srinivas (3.3)
- **Wave 3**: Shreekrishna (4.7, 4.8), Skanda (4.6), Srinivas (3.7, 3.8)
- **Wave 4**: All (3.9 E2E testing)

**Critical path:** `3.4/3.5 → 4.6 → 3.3 → 3.7 → 3.8 → 3.9`

---

## Notes for Optimum Model Performance

1. **Data augmentation is critical** — stuttering datasets are small. Time stretching, pitch shifting, and noise injection can improve generalization significantly.
2. **Class imbalance** — most samples are fluent. Use `pos_weight` in loss and monitor per-class F1, not just macro F1.
3. **Learning rate schedule** — warmup is essential for Wav2Vec2 fine-tuning. Start with 3e-5 and warmup over 500 steps.
4. **Early stopping** — prevents overfitting on small datasets. Monitor val F1, not just val loss.
5. **Ensemble consideration** — if individual classifiers perform well but hybrid doesn't improve, the combiner MLP may need more training data or a different architecture.
6. **Localization is harder** — frame-level annotation is noisy. Focus on event-level detection accuracy as the primary metric.
7. **Augment before training** — make sure augmentation is applied in the Dataset `__getitem__`, not pre-computed, so each epoch sees different augmentations.
