#import "../lib.typ": *
#import "../meta.typ": *

// --- Chapter 2: Requirement Specification and Analysis ---
#chapter_heading[REQUIREMENT SPECIFICATION AND ANALYSIS]

== INTRODUCTION

The requirements analysis and specification stage of the #project_title defines the main functions of the system, the expected performance, the dependencies on technical issues, and the needs of the users (developers, clinicians, and end-users).
This phase brings all interested parties together (the developing agency, clinicians, and end users) to agree on what the system will do.
The ultimate goal of the system is to take speech audio recordings and use a deep learning pipeline to classify stuttering types and localize dysfluency events within the speech signal.
Writing functional and non-functional requirements up front reduces the chance of design errors and helps ensure the finished system meets the clinical and technical needs.

== FUNCTIONAL REQUIREMENTS

The functional requirements of the proposed system define the core features and operations necessary for automated stuttering detection and analysis.

=== Dataset

The system is trained and evaluated using three publicly available stuttering datasets.
Project Boli provides multilingual Indian language stuttering data sourced via GitHub repositories.
SEP-28K is a large-scale dataset containing approximately 28,000 audio clips annotated for five stuttering event types, obtained via Kaggle.
UCLASS provides additional stuttering speech data, also obtained via Kaggle.

Each dataset is normalized into a unified format consisting of a #raw("combined_labels.csv") file with multi-label binary annotations and corresponding interval files for each audio clip.
The five dysfluency classes are prolongation, block, sound repetition, word repetition, and interjection.
The complete dataset is split into training, validation, and testing subsets using an 80:10:10 ratio.
An automated pipeline handles downloading, merging, and preprocessing the data to ensure efficient dataset preparation.

=== Preprocessing

Before any analysis is performed on the input audio, it must undergo a standardized preprocessing pipeline to ensure consistency and improve model performance.

The audio is first converted into a 16 kHz mono WAV format using FFmpeg, ensuring uniform input across all processing stages.
DC offset is removed by subtracting the mean of the signal, which eliminates any constant bias in the waveform.
Peak normalization scales the audio to a target peak amplitude of 0.95, ensuring consistent volume levels across different recordings.
Silent segments are trimmed from the beginning and end of the audio using energy-based detection, reducing unnecessary padding and focusing the analysis on active speech regions.

The processed audio is then padded or truncated to a fixed length of 48,000 samples, corresponding to a duration of 3 seconds at 16 kHz.
This standardization ensures that all inputs have uniform dimensions for model processing.

Data augmentation techniques are applied during training to increase dataset diversity and improve generalization.
These include random noise injection, time stretching, pitch shifting, temporal shifting, and scaling.
Spectrogram-level augmentations such as time masking and frequency masking are also employed to improve robustness.

=== User Requirements

The user can easily and clearly upload or record a speech sample via the system's user interface.
The system must be able to receive audio files in common formats including WAV, MP3, FLAC, and M4A, and process them prior to classification.

The system will automatically classify the submitted audio, providing results to the user without requiring manual intervention during the classification process.
The provided results will clearly indicate which dysfluency types are present in the speech, along with confidence scores for each detected type.

The system will localize the exact temporal positions of detected dysfluencies within the audio, presenting timestamped results that identify where each dysfluency event occurs.
If the system is unable to provide a conclusive classification, it will indicate low confidence and suggest that the user consult a speech-language pathologist for further evaluation.

=== System Requirements

The system shall preprocess input audio by converting it to 16 kHz mono format, removing DC offset, applying peak normalization (up to 0.95), and trimming silence segments.
For feature extraction, the system shall generate high-level speech representations using Wav2Vec 2.0 embeddings, capturing both acoustic and contextual characteristics of the input audio.

The classification pipeline shall detect five types of dysfluencies (prolongation, block, sound repetition, word repetition, and interjection) using five parallel binary classification models.
The per-classifier outputs shall be aggregated into a multi-label result that reports the probability of each dysfluency type and summarizes the detected classes.

The system shall generate a timestamped transcript of the input audio using the Whisper model, supporting multiple languages including English, Kannada, and Hindi.
Dysfluency localization shall be performed using CNN-based spectrogram analysis and Wav2Vec2 frame-level feature extraction, followed by alignment with corresponding words or syllables using Connectionist Temporal Classification (CTC) based time alignment.

The system shall provide visual representations of the analysis, including waveform displays with dysfluency overlays, spectrograms, and prediction probability graphs.
A detailed analysis report shall be generated and maintained using local storage mechanisms.

== NON-FUNCTIONAL REQUIREMENTS

The non-functional requirements define the quality attributes and operational constraints of the system.
These requirements cover how well the system performs, how easy it is to use, and how it can be maintained and extended over time.

=== Reliability

Reliability is critical in any system used to assist clinical assessment, as incorrect results can lead to misdiagnosis and inappropriate therapy decisions.
The system must produce consistent and reliable results regardless of variations in input audio quality, speaker characteristics, or recording environments.

The deep learning models are trained on diverse datasets that include multiple speakers, accents, and recording conditions, enabling the system to generalize across different populations.
The use of Wav2Vec 2.0 pretrained embeddings provides robust speech representations that have been learned from large-scale speech corpora, improving generalization to unseen speakers and environments.

A confidence threshold mechanism prevents the system from producing unreliable outputs for ambiguous inputs.
Predictions falling below the defined threshold are flagged as uncertain rather than being presented as definitive results.
Error handling routines capture exceptions that occur due to invalid file formats, corrupted audio, or unexpected input types, preventing complete system failure during operation.

=== Performance

System performance is essential for making the system practical and usable in real-world clinical environments.
The system is designed to deliver analysis results within a few seconds when processing a single audio clip.

This rapid response time is achieved through optimized processing techniques such as lazy loading and caching of models, efficient audio conversion using FFmpeg, and standardizing input to a fixed duration of 3 seconds at 16 kHz.
Utilizing pretrained Wav2Vec 2.0 model weights via transfer learning eliminates the considerable resources that would be required to train the model from scratch.

If GPU hardware is available, it is used for accelerated computation during both training and inference.
The modular design of the classification and localization pipelines allows parallel processing, improving overall throughput.

=== Usability

The system is designed to be intuitive and accessible for both clinicians and non-technical users.
The user interface provides a clean graphical interface with a minimal number of steps, simply uploading or recording audio and getting results.

Results are presented in plain language that is easy to understand, avoiding technical jargon where possible.
Confidence scores accompany all predictions so that users can assess the reliability of the results.
For cases where results indicate low confidence, the system suggests consulting a speech-language pathologist.

The system supports both web-based and desktop interfaces, ensuring accessibility across different platforms without requiring specialized hardware or software installation.

=== Scalability

The system is designed to be scalable from the beginning, with the ability to extend both the size and usage of the system without requiring a complete redesign.

The classification pipeline uses independent binary classifiers, allowing new dysfluency classes to be added by training and incorporating additional models with minimal modifications to the existing codebase.
Language-specific adapters enable the system to support additional languages without major architectural changes.

The modular architecture supports horizontal scaling when hosted on cloud platforms, accommodating larger volumes of concurrent users.
The separation of preprocessing, classification, localization, and reporting components allows individual modules to be upgraded or replaced independently.

=== Maintainability

The codebase follows a modular monorepo architecture, with preprocessing, classification, localization, transcription, and reporting developed as independent but loosely coupled components.
This separation of concerns allows developers to modify one section of the system with minimal risk of creating unintended consequences in other sections.

Model files are stored separately and can be replaced with newer versions once retrained on updated data.
A centralized model registry with configuration-driven checkpoint management ensures consistent model loading across the web and desktop applications.
Unique fingerprinting of model checkpoints enables version tracking and reproducibility.

Extensive documentation and consistent code quality standards, enforced through linting tools, reduce the ramp-up time for new contributors and support ongoing maintenance.

== USER INTERFACE REQUIREMENT

The system provides both a web-based interface and a desktop application to support different user preferences and environments.

=== Input Page

The input page allows users to interact with the system by providing audio input.
Users can upload pre-recorded audio files in WAV, MP3, FLAC, or M4A formats through a clearly marked upload button.
Alternatively, users can record speech directly using a microphone via the MediaRecorder API in the web application or the sounddevice library in the desktop application.

The system validates that the uploaded file is in a supported format before proceeding with analysis.
A preview of the uploaded audio may be displayed to confirm the correct file has been selected.
A clearly visible button initiates the analysis pipeline once the user is ready.

=== Result Page

The results page displays the output of the analysis pipeline.
The classification results indicate which dysfluency types are present in the speech, along with confidence scores for each type.
A waveform visualization with highlighted dysfluency regions provides a visual representation of where dysfluencies occur in the audio.

Spectrogram visualizations offer additional insight into the spectral characteristics of the speech signal.
A timestamped transcript of the speech is displayed, with detected dysfluencies marked at their corresponding positions.
The system also provides an option to generate and download a detailed clinical-style report in PDF format.

== SOFTWARE REQUIREMENT

The software requirements of the proposed system include the tools, frameworks, and technologies used for developing, deploying, and maintaining the application across different platforms.

=== PyTorch

PyTorch is the primary deep learning framework used for developing and training all machine learning models in the system.
It provides the computational backend for Wav2Vec 2.0 feature extraction, binary classification, multitask classification, CNN-based localization, and Wav2Vec2-based localization.
PyTorch's dynamic computation graph and automatic differentiation capabilities enable efficient model training and inference.
The framework supports GPU acceleration through CUDA and mixed-precision training for improved performance.

=== Transformers

The Hugging Face Transformers library provides pretrained speech models and integration tools.
Wav2Vec 2.0 models (facebook/wav2vec2-base) are used for extracting contextual speech embeddings that serve as input features for classification and localization.
The Whisper model is used for automatic speech recognition, generating timestamped transcripts of input audio.
Language-specific Whisper variants support English, Kannada, and Hindi transcription.

=== librosa

librosa is a Python library for audio analysis and processing.
It is used throughout the preprocessing pipeline for loading audio files, resampling to the target sample rate, computing mel-spectrograms (128 mel bands, hop length of 512, FFT size of 2048), and performing silence trimming.
librosa provides the foundational audio manipulation capabilities that support both training and inference workflows.

=== Matplotlib

Matplotlib is used for generating visual representations of audio data and model performance.
Mel-spectrograms are plotted using Matplotlib for visualization and debugging purposes.
Training curves showing accuracy and loss metrics are plotted during model development to monitor learning progress.
Confusion matrices and other evaluation visualizations are generated using Matplotlib to assess model performance across different dysfluency classes.

=== NumPy

NumPy is used extensively throughout the system for numerical computations on array data.
Audio signals are represented as NumPy arrays for preprocessing operations such as DC offset removal, peak normalization, and padding.
Model outputs and probability scores are processed using NumPy for aggregation and thresholding.
Data preparation and augmentation pipelines rely on NumPy for efficient array manipulation and transformation.

=== scikit-learn

scikit-learn provides evaluation metrics and tools for assessing model performance.
Classification metrics including precision, recall, F1-score, AUROC, and AUPRC are computed using scikit-learn to evaluate the reliability and diagnostic performance of the system.
These metrics are used during training to select the best-performing model checkpoints and during evaluation to compare different model architectures and configurations.

== HARDWARE REQUIREMENTS

The hardware requirements define the minimum and recommended system specifications necessary for efficient execution of the proposed system.
These requirements ensure smooth performance during both development and deployment phases.

- Processor: a system with at least an Intel Core i5 or AMD Ryzen 5 (or higher) to handle audio processing and model inference efficiently.
- Memory (RAM): 8 GB RAM for basic functionality, with 16 GB recommended for smoother performance when handling multiple recordings or resource-intensive tasks.
- Storage: at least 10 GB of available storage to accommodate datasets, trained model weights, and application files. Additional storage may be needed depending on usage and data accumulation.
- Graphics Processing Unit (GPU): an NVIDIA GPU is recommended for training deep learning models, as it enables faster computation through features such as torch.compile, mixed precision, and TensorFloat-32 (TF32). A GPU is not mandatory for inference; the system can run on CPU for deployment.
- Audio Input Device: a functional microphone is required for recording speech input within the application.

/*
// --- Commented out sections (from original chapter 3) ---

== FEASIBILITY STUDY
The feasibility study evaluates the practicality of the proposed system from technical, economic, operational, and future expansion perspectives.

- *Technical Feasibility:*
  The system is built using a well-established open-source software stack, including PyTorch, Hugging Face Transformers, librosa, FastAPI, and React. These technologies are widely adopted, thoroughly tested, and supported by strong developer communities, making implementation both reliable and manageable.
- *Economic Feasibility:*
  The overall development cost is minimal, as all major tools and libraries used in the system are free and open-source. The primary expense is limited to computational resources required for model training, which can be managed using platforms such as Kaggle or Google Colab.
- *Operational Feasibility:*
  The system is designed with usability in mind, offering both web and desktop interfaces that are intuitive and easy to navigate. This reduces the learning curve for users, including clinicians and non-technical individuals, and allows for smooth day-to-day operation without extensive training.
- *Schedule and Data Feasibility:*
  The project is supported by the availability of publicly accessible datasets such as Project Boli, SEP-28K, and UCLASS. These datasets can be obtained through platforms like GitHub and Kaggle. An automated pipeline is used to download, merge, and preprocess the data, ensuring efficient dataset preparation.
- *Future Feasibility:*
  The system is designed with extensibility in mind. It can be expanded to support additional languages and dysfluency categories. Future enhancements may also include cloud-based synchronization, as well as features for tracking therapy progress over time, further increasing its practical value.

== USE CASE DIAGRAM AND DESCRIPTIONS
The use case diagram represents the interaction between the user and the system.
The primary actor in the system is the *User*, which may be either a clinician or an individual using the application for self-assessment.

The system supports multiple use cases that cover the complete workflow of speech analysis.
These include recording audio, uploading pre-recorded audio, and initiating speech analysis in either full mode or classification-only mode.
Once the analysis is complete, users can view different forms of output such as waveform visualizations, spectrograms, transcripts, and stutter detection results.

In addition to analysis, the system allows users to localize dysfluencies within the speech, generate detailed analysis or clinical reports, and manage previously recorded sessions through a history feature.
Other supporting functionalities include toggling between interface themes and accessing standardized reading passages for consistent evaluation.

The primary flow of interaction follows a simple sequence: the user records or uploads audio, initiates analysis, reviews visualizations and results, and finally generates or saves the report.

== ACTIVITY DIAGRAM
The activity diagram illustrates the step-by-step workflow of the system.
The process begins when the user opens the application and chooses to either record new audio or upload an existing file.
The input audio is then validated and converted into a standard format using FFmpeg, specifically 16 kHz mono.

Following this, preprocessing is applied to clean and normalize the audio.
The processed audio is then passed through multiple stages: classification, transcription, localization, and alignment.
The classification stage uses five Wav2Vec2-based binary classifiers whose outputs are aggregated to identify which dysfluency types are present.
In parallel, the Whisper model generates a timestamped transcription of the speech.

Localization is performed using spectrogram-based CNN analysis or Wav2Vec2 temporal features.
The results are then aligned to specific words or syllables using CTC-based alignment.
Finally, the system displays waveform, spectrogram, transcript, and confidence scores, and provides an option to generate and save a detailed report.

== SEQUENCE DIAGRAM
The sequence diagram describes the interaction between different system components during execution.
The process starts with the user interacting with the graphical user interface (GUI), which sends a request to the backend API endpoint (`/api/analyze`).

The backend processes the request through a series of services, including preprocessing, classification, transcription, localization, and alignment.
Each service performs a specific task and passes its output to the next stage.
Once processing is complete, the results are sent back to the frontend, where they are displayed to the user.
The system also stores the results for report generation and history management.

== DATA FLOW DESCRIPTION
The data flow within the system begins with the input audio, which is first converted into a standardized 16 kHz mono WAV format using FFmpeg.
The audio is then processed through a cleaning stage that removes DC offset, applies peak normalization, and trims silence.

The cleaned audio is routed through three parallel processing paths.
In the first path, Wav2Vec2 embeddings are generated and passed through five binary classifiers, and the outputs are aggregated into a multi-label result with a probability score for each dysfluency type.
In the second path, the Whisper model generates a timestamped transcript of the speech.
In the third path, spectrogram features (128 mel bands with a hop length of 512) or raw waveform inputs are used for localization, producing frame-level outputs at intervals such as 32 ms or 20 ms.

The outputs from all three paths are merged to create a detailed mapping of dysfluencies at the word level.
These results are then used to generate visualizations and structured reports.

For training and evaluation, the system utilizes multiple datasets, including Project Boli (from GitHub), SEP-28K (approximately 28,000 clips from Kaggle), and UCLASS (from Kaggle).
These datasets are normalized into a unified format, consisting of a `combined_labels.csv` file with multi-label binary annotations and corresponding interval files for each clip.
The dataset is split into training, validation, and testing sets in an 80:10:10 ratio.

== CHAPTER SUMMARY
This chapter presented a detailed analysis of the system requirements, covering both functional and non-functional aspects.
It also examined feasibility, system interactions, workflows, and data processing mechanisms.
The requirements highlight the need for an accurate, interpretable, and multilingual-ready stuttering detection system that can operate efficiently in both online and offline environments.

The next chapter focuses on the system design and architecture, detailing how these requirements are translated into an implementable solution.
*/

#pagebreak()
