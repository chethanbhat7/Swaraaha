#import "../lib.typ": *
#import "../meta.typ": *

// --- Chapter 5: Implementation Details ---
#chapter_heading[IMPLEMENTATION DETAILS]

== INTRODUCTION
This chapter describes how the #project_title was built: the development environment, project structure, algorithms, libraries, and how the components fit together.
Each component was developed and tested on its own before integration into the full pipeline.

== DEVELOPMENT ENVIRONMENT
The system was developed and tested in the following environment.

- Operating System: Ubuntu 22.04 LTS (Linux) for development and training; Windows 10/11 and macOS for deployment.
- Programming Language: Python 3.11 for backend and machine learning components; TypeScript for the web frontend.
- Deep Learning Framework: PyTorch 2.0+ with CUDA for GPU-accelerated training and inference.
- Code Editor: Visual Studio Code with Python and TypeScript extensions.
- Version Control: Git, with a modular monorepo layout.
- Package Management: pip (Python), npm (Node.js).
- Containerization: Docker and Docker Compose for reproducible deployment.

== PROJECT STRUCTURE
The codebase is a monorepo with clearly separated directories.

- `model/`: the core machine learning package. Contains classification, localization, transcription, training, evaluation, and data pipelines. This is the primary deliverable.
- `backend/`: a FastAPI REST API server that wraps the model package and exposes HTTP endpoints for the web frontend.
- `frontend/`: a React 19 single-page application for the web interface.
- `app/`: a PySide6 desktop application with its own GUI.
- `shared/`: utilities shared across components, including the Typst-based PDF report builder.
- `deploy/`: production Docker configuration and deployment blueprints.
- `docs/`: project documentation and this report.

The `model/` package exposes a public API through its `__init__.py` with functions like `classify()`, `localize()`, `transcribe()`, `analyze()`, and `fuse()`. Both the web backend and the desktop application call this API directly.

== KEY ALGORITHMS AND IMPLEMENTATION

=== Audio Preprocessing Pipeline
The preprocessing pipeline applies a fixed sequence of transformations to every input audio clip.

Audio is loaded and resampled to 16 kHz using librosa. DC offset is removed by subtracting the signal mean. Peak normalization scales the waveform to a target amplitude of 0.95. Silence trimming uses energy-based detection with a top dB threshold of 25 to strip non-speech segments from the boundaries.

The processed audio is padded or truncated to a fixed length of 48,000 samples (3 seconds at 16 kHz). All downstream models receive inputs of this uniform size.

During training, data augmentation applies both waveform-level and spectrogram-level transforms. Waveform augmentations include random noise injection (SNR 10 to 20 dB), time stretching (rate 0.9 to 1.1), pitch shifting (plus or minus 2 semitones), temporal shifting (plus or minus 20%), and amplitude scaling (0.8 to 1.2). Spectrogram augmentations include time masking and frequency masking with configurable parameters.

=== Classification Pipeline
The classification pipeline uses a Wav2Vec 2.0 pretrained backbone (`facebook/wav2vec2-base`) as a shared feature extractor. Input audio passes through the Wav2Vec2 feature extractor and transformer encoder, producing contextual embeddings at roughly 20 ms frame resolution.

The MultiTaskClassifier applies mean-pooling over the time dimension to get a fixed-length representation, then passes it through five independent binary classification heads. Each head is a Linear layer followed by Tanh activation and a second Linear layer producing two logits (present vs. not present). Softmax converts these to probabilities.

During training, the backbone stays frozen for the first three epochs so the classification heads can stabilize. After that, it is unfrozen with a learning rate scaled to 0.1 of the head learning rate. Training uses Focal Loss with gamma equal to 2.0 to address class imbalance, and the AdamW optimizer with a learning rate of $3 times 10^(-5)$ and weight decay of 0.01. A linear warm-up of 500 steps is applied, and early stopping with patience of 5 epochs prevents overfitting. The best checkpoint is the one with the highest macro F1-score on the validation set.

At inference time, per-class probabilities are compared against class-specific thresholds stored in the model registry. These thresholds come from threshold sweep analysis on the validation set.

=== Localization Pipeline
The localization pipeline finds the temporal regions where dysfluencies occur.

The Wav2Vec2-based localizer processes audio through the Wav2Vec2 backbone and applies temporal attention pooling to produce per-frame dysfluency probabilities at roughly 20 ms resolution. The architecture has a hidden layer of 256 units with dropout of 0.3, followed by a classification head that outputs frame-level probabilities.

The CNN-based spectrogram localizer computes mel-spectrograms (128 mel bands, hop length 512) and processes them through convolutional layers with batch normalization and ReLU activations. The output is upsampled to per-frame probability maps at roughly 32 ms resolution.

Both localizers output contiguous regions of predicted dysfluency as (start, end, confidence) tuples. Post-processing merges overlapping regions and applies minimum duration constraints.

=== Transcription Pipeline
The transcription pipeline uses Whisper for generating timestamped transcripts. Language-specific Whisper-tiny variants handle English, Kannada, and Hindi.

The transcriber produces word-level timestamps alongside the transcript text. Dysfluency regions from the localization pipeline are overlaid onto these timestamps to identify which words or syllables are affected.

=== Combiner and Fusion
The combiner merges localizer regions with classifier per-frame saliency. For each localized region, it computes per-class saliency scores by averaging the classifier's frame-level activations within that region. The region is then labeled with its primary dysfluency type (the class with the highest saliency score) along with confidence values for all five classes.

When the localizer is unavailable, the combiner synthesizes regions from the classifier's saliency map using adaptive thresholding.

=== Severity Scoring
The severity module computes a stutter index as the ratio of total dysfluent speech duration to overall speech duration. The index maps to severity levels: Fluent (less than 2%), Mild (2 to 5%), Moderate (5 to 15%), and Severe (greater than 15%).

=== Report Generation
The report generation module produces clinical-style PDF reports using Typst. The template includes patient details, classification results with confidence scores, a table of localized dysfluency events with timestamps, the timestamped transcript, and the severity assessment. The Typst source is compiled to PDF through the Python `typst` library.

== MODEL REGISTRY
A centralized model registry manages all trained checkpoints through a JSON configuration file (`registry.json`). It maps task names (classification, localization) to checkpoint file paths and per-class thresholds.

Models load lazily on first use and stay cached in memory. Swapping active checkpoints means editing the JSON file only; no code changes are required. Checkpoint filenames encode all training hyperparameters through a fingerprint naming convention, which helps with version tracking and reproducibility.

The registry supports these model types:
- Individual binary classifiers (one per dysfluency class)
- Multitask classifier (shared backbone with five heads)
- CNN multitask classifier (spectrogram-based)
- CNN spectrogram localizer
- Wav2Vec2 frame-level localizer

== WEB APPLICATION IMPLEMENTATION
The web frontend is built with React 19, TypeScript, and Tailwind CSS 4. The layout uses a sidebar with routes for the input page, results page, history, and reading passages.

The input page supports audio file upload (WAV, MP3, FLAC, M4A, OGG, WMA) and direct microphone recording via the MediaRecorder API. Files are validated on the client side before being sent to the backend as multipart form data.

#add_image(align(center, image("/assets/Home Page.png", width: 100%)), caption: [Web application home page with audio upload and microphone recording])

The results page shows classification results with confidence scores, an interactive waveform with color-coded dysfluency region overlays, a timestamped transcript, and a button to generate and download a PDF report.

#add_image(align(center, image("/assets/resultpage.png", width: 100%)), caption: [Web application results page showing classification results, waveform overlays, and the timestamped transcript])

Assessment history is stored in the browser. Audio files go into IndexedDB for offline access; analysis metadata goes into localStorage.

The backend uses FastAPI served via Uvicorn. Endpoints include `/api/classify`, `/api/localize`, `/api/analyze`, and `/api/report`. CORS is configured to allow requests from the frontend development server.

== DESKTOP APPLICATION IMPLEMENTATION
The desktop application uses PySide6 (Qt for Python) for a native cross-platform interface.

It includes modules for audio recording and playback (sounddevice), file handling (soundfile), model inference (calling the shared model package), and PDF report generation (Typst with pypdfium2 for viewing). The desktop application calls the model package directly with no separate server, so it works fully offline.

== CHAPTER SUMMARY
This chapter covered the development environment, project structure, algorithms, and how the components fit together. The monorepo layout lets each part be developed and tested on its own. The model registry keeps model management consistent across the web and desktop applications.

The next chapter presents the testing methodology and evaluation results.

#pagebreak()
