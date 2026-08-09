#import "../lib.typ": *

// --- Chapter 4: System Design ---
#chapter_heading[SYSTEM DESIGN]

== Introduction
This chapter presents the overall system design for an end-to-end framework developed for automated stutter detection and localization.
The design outlines how different components of the system interact to process speech data and generate meaningful analytical results.

A key feature of the system is the use of two independent model pipelines.
The first pipeline focuses on classification, identifying the type of dysfluency present in the speech.
The second pipeline is responsible for localization, determining the exact position in the audio where the dysfluency occurs.
These pipelines operate independently to preserve clarity and interpretability, and their outputs are presented together without being combined or fused.

The system follows a multi-layered architecture.
User interaction is supported through both a web-based interface developed using React and a desktop application built with PySide6.
These interfaces communicate with a FastAPI-based backend, which handles the core processing tasks.
The backend relies on a shared model package that contains all machine learning components, while a centralized model registry ensures consistent loading and management of trained models.

The design is guided by several important objectives.
Modularity allows different components to be developed and maintained independently.
Maintainability ensures that the system can be updated and extended with minimal effort.
Interpretability is prioritized so that outputs remain clear and useful, especially in analytical or clinical contexts.
Scalability enables the addition of new dysfluency types and language support.
Additionally, the desktop application is designed to function offline, ensuring accessibility even in environments without internet connectivity.

== Overall System Architecture
The overall system architecture is designed to support efficient and structured processing of speech data, from input acquisition to final result generation.
The process begins when the user either records audio directly or uploads a pre-recorded file through the web interface or desktop application.
The web application uses MediaRecorder APIs, while the desktop application utilizes sounddevice for capturing audio input.

Once the audio is received, it is sent to the backend, where it undergoes normalization.
This involves converting the input into a standardized 16 kHz mono WAV format using FFmpeg, ensuring consistency across all processing stages.

After preprocessing, the system processes the audio through three parallel pipelines.
The first pipeline focuses on classification.
In this stage, Wav2Vec 2.0 is used to extract meaningful speech representations, which are then passed through five independent binary classifiers.
The outputs of these classifiers are aggregated into a multi-label result that reports the presence probability of each dysfluency type and summarizes the detected classes and the primary dysfluency.

The second pipeline handles transcription.
The Whisper Automatic Speech Recognition (ASR) model is used to generate a timestamped transcript of the audio.
This component supports multiple languages, including English, Kannada, and Hindi.

The third pipeline is responsible for localization.
Dysfluency regions are identified at the frame level using either a CNN-based spectrogram approach or Wav2Vec2 temporal attention mechanisms.
This enables the system to detect the precise segments of speech where dysfluencies occur.

To connect these outputs meaningfully, a Connectionist Temporal Classification (CTC) based time alignment process is used.
This step maps the detected dysfluency regions to specific words or syllables in the transcript.
Language-specific adapters are incorporated to ensure accurate alignment across English, Kannada, and Hindi.

The final results are presented through multiple visual and textual outputs, including waveform overlays, spectrogram visualizations, timestamped transcripts, confidence scores, and a detailed clinical-style report.

Both the web and desktop applications rely on a centralized model registry for loading trained models.
This registry, implemented using a registry module and a configuration file, ensures that model checkpoints can be updated or replaced without requiring changes to the application code, thereby improving flexibility and maintainability.

== System Architecture Blcok Diagram

#image("/assets/architecture.jpeg")

== Module Description
The system is organized into multiple functional modules, each responsible for a specific stage in the speech processing pipeline.
This modular design improves clarity, maintainability, and ease of extension.
- *Audio Acquisition Module:*
  This module handles input collection. It allows users to either record speech using a microphone or upload pre-recorded audio files in formats such as WAV, MP3, FLAC, or M4A.
- *Audio Conversion Module:*
  The acquired audio is converted into a standardized format using FFmpeg. Specifically, the audio is transformed into 16 kHz mono WAV format. A fallback mechanism is provided in case FFmpeg is not available.
- *Preprocessing Module:*
  This module prepares the audio for further analysis. It includes resampling, removal of DC offset, peak normalization (set to 0.95), and trimming of silent segments. These steps ensure consistent input quality across the system.
- *Feature Extraction Module:*
  In this stage, meaningful representations of the audio are generated. Wav2Vec 2.0 is used to produce contextual embeddings, while mel-spectrograms (with 128 mel bands, hop length of 512, and FFT size of 2048) are computed for spectral analysis.
- *Classification Module:*
  The classification module consists of five parallel Wav2Vec2-based binary classifiers, each responsible for detecting a specific type of dysfluency.
  Each classifier uses a two-logit output with softmax activation and yields the presence probability of its dysfluency type.
  The per-classifier outputs are aggregated into a multi-label result that reports each class probability and summarizes the detected classes and the primary dysfluency.
- *Localization Module:*
  This module identifies the temporal regions of dysfluencies within the audio. It uses two approaches: a CNN-based spectrogram localizer with multiple convolutional layers operating at approximately 32 ms frame resolution, and a Wav2Vec2-based localizer that works at around 20 ms resolution.
- *Speech-to-Text Module:*
  The system uses Whisper-based pipelines to convert speech into text. It supports multiple languages, including English, Kannada, and Hindi, and generates word-level timestamps for accurate alignment.
- *Timestamp Alignment Module:*
  This module aligns detected dysfluency regions with corresponding words or syllables. It uses a CTC-based alignment approach, with a fallback mechanism for forced alignment. Language-specific adapters are used to improve accuracy across supported languages.
- *Model Registry Module:*
  A centralized model registry manages all trained models, including classifiers and localization models. It is designed to be configuration-driven and supports lazy loading, ensuring efficient resource utilization and easy model updates.
- *Visualization Module:*
  The system provides multiple visualization outputs, including waveform displays with highlighted dysfluency regions, spectrograms, transcripts, confidence scores, and a timeline view. These visualizations improve interpretability of results.
- *Report Generation Module:*
  This module generates a structured report containing patient details, classification results, and localized dysfluency events. It also maintains a history of analyses using local storage mechanisms such as LocalStorage or IndexedDB.

== Data Flow Design
The data flow design describes how audio data is processed through different stages of the system, from input acquisition to final output generation.

The process begins with the input audio, which is either recorded or uploaded by the user.
This audio is first converted into a standardized 16 kHz mono WAV format using FFmpeg.
To ensure uniform input length, the audio is then padded or truncated to 160,000 samples, corresponding to a duration of 10 seconds.

After normalization, the audio is processed through three parallel pipelines: classification, transcription, and localization.
The classification pipeline generates dysfluency probabilities, the transcription pipeline produces a timestamped transcript, and the localization pipeline identifies frame-level dysfluency regions.

These outputs are then combined using a CTC-based alignment process, which maps detected dysfluencies to specific words or syllables.
The final processed data is used to generate per-word annotations, which are visualized through waveform and spectrogram displays and included in the final analysis report.

In addition to inference, the system also defines a structured data flow for training.
The training process utilizes three primary datasets: Project Boli (sourced via Git repositories), SEP-28K (approximately 28,000 audio clips), and UCLASS (both obtained via Kaggle).
Each dataset undergoes normalization through dataset-specific preprocessing functions.

The processed datasets are merged into a unified format consisting of a combined_labels.csv file containing multi-label binary annotations, along with individual interval CSV files for each audio clip.
The complete dataset is then divided into training, validation, and testing subsets using an 80:10:10 split.
These subsets are organized using symbolic links for efficient access, and preprocessed audio files are cached to improve training performance.

== Algorithm
The inference process of the system follows a structured sequence of steps, ensuring accurate and efficient detection and localization of dysfluencies.
- *Step 1: Input Acquisition:*
  The system acquires speech input either through real-time recording or by uploading an audio file.
- *Step 2: Audio Conversion:*
  The input audio is converted into a 16 kHz mono WAV format using FFmpeg to maintain consistency.
- *Step 3: Preprocessing:*
  The audio is cleaned by removing DC offset, applying peak normalization, and trimming silent segments.
- *Step 4: Length Normalization:*
  The processed audio is adjusted to a fixed length of 160,000 samples by padding or truncating as required.
- *Step 5: Parallel Processing:*
  The system processes the audio simultaneously through three parallel branches:
  + *Classification:* Wav2Vec 2.0 embeddings are generated and passed through five binary classifiers.
    The outputs are aggregated into a multi-label result with a probability score for each dysfluency type.
  + *Transcription:* The Whisper model generates a timestamped transcript of the speech.
  + *Localization:* A spectrogram-based CNN or a Wav2Vec2-based model identifies frame-level dysfluency regions within the audio.
- *Step 6: Alignment:*
  The detected dysfluency regions are aligned with corresponding words or syllables using a CTC-based alignment method.
- *Step 7: Visualization:*
  The system displays the results through waveform overlays, spectrograms, transcripts, and confidence scores, along with clearly marked dysfluency regions.
- *Step 8: Report Generation:*
  A detailed clinical-style report is generated and stored for future reference.

In addition to inference, the training procedure for each classifier follows a structured approach. Initially, the backbone model is frozen for the first three epochs to stabilize learning. It is then unfrozen with a reduced learning rate (scaled by a factor of 0.1). Training is performed using Focal Loss with a gamma value of 2 to handle class imbalance, and optimization is carried out using the AdamW optimizer with a learning rate of $3 times 10^(-5)$. A warm-up phase of 500 steps is applied, followed by early stopping to prevent overfitting. The best-performing model is selected based on the highest F1-score and saved as the final checkpoint.

== Design Considerations
The system is designed with a focus on accuracy, scalability, interpretability, usability, maintainability, and performance.
Key design goals and their corresponding implementation strategies are outlined below: \
- *Accuracy:*
  Achieved using Wav2Vec 2.0 embeddings combined with per-class fine-tuned binary classifiers.
  A focal loss function is used to handle hard examples and improve classification robustness.
- *Scalability:*
  The use of independent binary classifiers allows new dysfluency classes to be added without requiring architectural redesign, ensuring easy extensibility.
- *Interpretability:*
  Timestamp alignment enables word- and syllable-level outputs.
  Visual overlays on waveform and spectrogram provide intuitive understanding of detected dysfluencies.
- *Usability:*
  The system provides both a React-based web interface and a PySide6 desktop application, featuring dark mode and guided workflows for ease of use.
- *Maintainability:*
  A modular monorepo structure is adopted, along with a centralized model registry and fingerprint-based checkpoint naming for version control and reproducibility.
- *Performance:*
  Models are lazy-loaded and cached to reduce latency.
  Training leverages GPU acceleration with mixed precision and torch.compile for efficiency.
- *Class Imbalance Handling:*
  Focal loss is used alongside positive class weighting (pos_weight) in the localizer.
  Performance is monitored using per-class evaluation metrics.

== Component Interaction
The system components interact through clearly defined interfaces across different layers: \
- *Frontend #sym.arrow.l.r Backend:*
  Communication occurs via REST APIs exposed by FastAPI, including endpoints such as `/api/classify`, `/api/localize`, `/api/analyze`, and `/health`.
- *Backend Services:*
  Core services include audio processing utilities, classification, localization, and transcription modules.
  These services interact with a shared model registry (`model/registry.py`) to dynamically load models.
- *Desktop Components:*
  The desktop application includes modules such as ModelRunner, AudioHandler, and AudioTranscriber, currently structured for inference and real-time transcription.
- *Data Pipeline:*
  The data layer follows a structured workflow: `download → merge → prepare → train → evaluate`, ensuring reproducibility and consistency across experiments.

== Chapter Summary
This chapter presented the complete system design of the proposed framework, covering its architecture, data flow, modules, and processing algorithms.
The design clearly separates key functional stages, including audio acquisition, preprocessing, classification and localization pipelines, transcription, alignment, visualization, and report generation.

The modular and registry-driven architecture ensures flexibility, maintainability, and ease of integration of new models or features.
By keeping classification and localization as independent pipelines, the system improves interpretability while maintaining high accuracy.
Additionally, the design supports scalability for future extensions such as new dysfluency classes and multilingual capabilities.

Overall, the system design provides a robust foundation for efficient and interpretable stutter detection and analysis.
The next chapter focuses on the implementation details of the system.

#pagebreak()
