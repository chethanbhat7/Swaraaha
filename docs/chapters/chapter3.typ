#import "../lib.typ": *

// --- Chapter 3: Analysis and Requirement Specification ---
#chapter_heading([ANALYSIS AND REQUIREMENT SPECIFICATION])

== Introduction
The main aim of the proposed system is to build an automated method for stuttering detection and localization to overcome the disadvantages of the traditional subjective assessment methods adopted by Speech-Language Pathologists (SLPs).
The conventional methods depend on the manual observation and interpretation, which can be time-consuming, inconsistent, and dependent on the evaluator’s expertise.
To address these issues, the system proposes an objective and data-driven approach to analyze speech dysfluencies.

The system is designed to perform two primary functions.
First, it classifies the type of dysfluency in speech which includes prolongation, block, repetition of sound, repetition of word, and interjection.
Second, it identifies and localises the exact temporal locations of these dysfluencies in the audio signal.
This dual capability enhances diagnostic accuracy and interpretability, rendering the system valuable for clinical and assistive applications.

The overall workflow of the system is a processing pipeline from start to end.
The first step is the recording or the upload of an audio sample, which is preprocessed with different techniques for the analysis.
For feature extraction, we leverage Wav2Vec 2.0 embeddings which captures rich acoustic and contextual information of the speech signal.
The features are then passed through five parallel binary classifiers, each responsible for detecting a specific type of dysfluency.
The per-classifier outputs are aggregated into a multi-label result that reports the presence probability of every dysfluency type along with a summary of the detected classes.

The system utilizes the Whisper model to convert speech to text and temporal segmentation to generate the transcriptions and the timestamp information.
We apply Connectionist Temporal Classification (CTC) based time alignment to accurately align detected dysfluencies to their corresponding places in the audio.
The final output is delivered through an visualization interface and a detailed report.

== Existing System Analysis
Current approaches to stuttering detection and analysis reveal several limitations, particularly in terms of accuracy, scalability, and usability.
Traditionally, diagnosis is performed manually by Speech-Language Pathologists (SLPs).
While this method benefits from expert knowledge, it is inherently subjective, often time-consuming, and difficult to scale.
Moreover, assessments can vary significantly between experts, leading to inconsistencies in diagnosis.

Earlier computational methods relied on traditional machine learning techniques such as Support Vector Machines (SVM) and Hidden Markov Models (HMM), combined with handcrafted features like MFCC, pitch, and Linear Predictive Coding (LPC).
Although these approaches introduced automation, they depend heavily on manual feature engineering.
This makes them less adaptable and often results in poor generalization when applied to different speakers, accents, or recording environments.

In recent years, several commercial speech-therapy applications have emerged.
However, most of these platforms focus primarily on providing exercises and training rather than accurate diagnosis.
They are typically subscription-based, require continuous internet connectivity, and lack the ability to precisely identify where dysfluencies occur within speech.

Some deep learning-based systems have improved classification performance by analyzing entire audio recordings.
Despite this progress, they generally treat speech as a whole and fail to pinpoint the exact word or segment where a dysfluency occurs.
This limits their effectiveness in detailed clinical analysis and feedback.

Overall, existing systems suffer from multiple shortcomings, including subjectivity in evaluation, reliance on handcrafted features, limited generalization capability, lack of precise localization, minimal visualization and interpretability, dependence on internet connectivity, and absence of robust offline support.

== Functional Requirements
The functional requirements of the proposed system define the core features and operations necessary for automated stuttering detection and analysis.
These requirements are structured as a set of functional units, each corresponding to a specific capability within the system.
- *Speech Recording:*
  The system shall allow users to record speech directly using a microphone.
  For the web application, this is implemented using MediaRecorder or getUserMedia APIs, while the desktop application utilizes system sound device interfaces.
- *Audio Upload:*
  The system shall support uploading of prerecorded audio files in multiple formats, including WAV, MP3, FLAC, and M4A, ensuring flexibility for users.
- *Audio Preprocessing:*
  The system shall preprocess input audio by converting it to 16 kHz mono format, removing DC offset, applying peak normalization (up to 0.95), and trimming silence segments.
  This ensures consistency and improves model performance.
- *Feature Extraction:*
  The system shall generate high-level speech representations using Wav2Vec 2.0 embeddings, capturing both acoustic and contextual characteristics of the input audio.
- *Dysfluency Detection:*
  The system shall detect five types of dysfluencies-prolongation, block, sound repetition, word repetition, and interjection-using five parallel binary classification models.
- *Multi-Label Aggregation:*
  The system shall aggregate the outputs of the five individual classifiers into a multi-label result that reports the probability of each dysfluency type and summarizes the detected classes and the primary dysfluency.
- *Speech Transcription:*
  The system shall generate a timestamped transcript of the input audio using the Whisper model, supporting multiple languages such as English, Kannada, and Hindi.
- *Dysfluency Localization:*
  The system shall identify the exact temporal locations of dysfluencies within the audio using CNN-based spectrogram analysis and Wav2Vec2 temporal attention, followed by alignment with corresponding words or syllables.
- *Visualization:*
  The system shall provide visual representations of the analysis, including waveform displays with dysfluency overlays, spectrograms, and prediction probability graphs to enhance interpretability.
- *Report Generation:*
  The system shall generate a detailed analysis report and maintain user history using local storage mechanisms such as LocalStorage or IndexedDB.
- *Model Management:*
  The system shall include a model registry to ensure consistent loading and management of trained model checkpoints across both web and desktop platforms.

== Non-Functional Requirements
The non-functional requirements define the quality attributes and operational constraints of the system.
These requirements ensure that the system performs efficiently, remains user-friendly, and can be maintained and extended over time.
- *Performance:*
  The system is designed to deliver analysis results within a few seconds. This is achieved through optimized processing techniques such as lazy loading and caching of models, efficient audio conversion using FFmpeg, and standardizing input to a fixed duration of 10 seconds at 16 kHz.
- *Accuracy:*
  The system aims to provide reliable and consistent classification of dysfluencies. To address class imbalance, techniques such as Focal Loss are employed during training. Performance is evaluated using metrics like AUROC, AUPRC, and F1-score to ensure robustness.
- *Usability:*
  The interface is designed to be intuitive and accessible for both clinicians and non-technical users. Features such as a clean graphical user interface, and dark mode.
- *Reliability:*
  The system ensures stable operation over extended usage, supporting multiple recordings without failure. It also incorporates proper error handling mechanisms, particularly for scenarios such as missing or incompatible model weights.
- *Maintainability:*
  A modular architecture is adopted to simplify development and future updates. The system is organized into distinct components such as model, backend, frontend, and application layers. Additionally, a configuration-driven model registry and uniquely fingerprinted checkpoints enable efficient model management.
- *Portability:*
  The system is built to run across multiple platforms. It leverages Python for backend and desktop components, and React for the web interface. Containerization using Docker ensures consistent deployment across different environments.
- *Scalability:*
  The architecture supports easy extension to additional dysfluency classes and languages. This is achieved through independent binary classifiers and the use of language-specific adapters, allowing the system to evolve without major redesign.
- *Offline Capability:*
  The desktop application is capable of functioning without an internet connection, ensuring accessibility in environments with limited or no connectivity.
- *Data Privacy:*
  User data is handled with a strong focus on privacy. Audio recordings are stored locally using mechanisms such as IndexedDB or local file storage, eliminating the need for cloud-based data transmission.

== Software Requirements
The software requirements of the proposed system include the tools, frameworks, and technologies used for developing, deploying, and maintaining the application across different platforms.
- *Machine Learning Frameworks:*
  The system is developed using Python 3.11 as the primary programming language. Deep learning models are implemented using PyTorch, while Hugging Face Transformers are utilized for integrating pretrained models such as Wav2Vec 2.0 (base and large variants). For audio processing and numerical computations, libraries such as librosa and NumPy are employed.
- *Web Application Technologies:*
  The web-based interface is built using React 19, along with Vite for fast development and TypeScript for type safety. Tailwind CSS is used to design a responsive and modern user interface. The backend is implemented using FastAPI and served via Uvicorn, enabling efficient handling of API requests. Audio preprocessing and format conversion are supported using FFmpeg, while speech recognition is handled using the Whisper ASR model.
- *Desktop Application Technologies:*
  The desktop application is developed using PySide6, providing a native graphical interface. Audio recording and processing are managed using libraries such as sounddevice and soundfile. Additionally, pypdfium2 is used for generating and handling report documents within the application.
- *Development and Deployment Tools:*
  For deployment and environment consistency, Docker and Docker Compose are used. The application can be hosted on platforms such as Render. Code quality and consistency are maintained using ruff for linting, while pytest is used for testing, particularly for the desktop components.
- *Model Registry:*
  A centralized model registry is implemented to manage trained models efficiently. This includes a registry module (`model/registry.py`) and a configuration file (`registry.json`), which together serve as the standardized and authorized pathway for loading model checkpoints across the system.

== Hardware Requirements
The hardware requirements define the minimum and recommended system specifications necessary for efficient execution of the proposed system.
These requirements ensure smooth performance during both development and deployment phases.

- *Processor:*
  A system with at least an Intel Core i5 or AMD Ryzen 5 processor (or higher) is required to handle audio processing and model inference efficiently.
- *Memory (RAM):*
  A minimum of 8 GB RAM is required for basic functionality. However, 16 GB RAM is recommended to ensure smoother performance, especially when handling multiple recordings or running resource-intensive tasks.
- *Storage:*
  The system requires at least 10 GB of available storage to accommodate datasets, trained model weights, and application files. Additional storage may be needed depending on usage and data accumulation.
- *Graphics Processing Unit (GPU):*
  An NVIDIA GPU is recommended for training deep learning models, as it enables faster computation through features such as torch.compile, mixed precision, and TensorFloat-32 (TF32). However, a GPU is not mandatory for inference, and the system can operate on CPU for deployment purposes.
- *Audio Input Device:*
  A functional microphone is required for recording speech input within the application.

== Feasibility Study
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

== Use Case Diagram and Descriptions
The use case diagram represents the interaction between the user and the system.
The primary actor in the system is the *User*, which may be either a clinician or an individual using the application for self-assessment.

The system supports multiple use cases that cover the complete workflow of speech analysis.
These include recording audio, uploading pre-recorded audio, and initiating speech analysis in either full mode or classification-only mode.
Once the analysis is complete, users can view different forms of output such as waveform visualizations, spectrograms, transcripts, and stutter detection results.

In addition to analysis, the system allows users to localize dysfluencies within the speech, generate detailed analysis or clinical reports, and manage previously recorded sessions through a history feature.
Other supporting functionalities include toggling between interface themes and accessing standardized reading passages for consistent evaluation.

The primary flow of interaction follows a simple sequence: the user records or uploads audio, initiates analysis, reviews visualizations and results, and finally generates or saves the report.

== Activity Diagram
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

== Sequence Diagram
The sequence diagram describes the interaction between different system components during execution.
The process starts with the user interacting with the graphical user interface (GUI), which sends a request to the backend API endpoint (`/api/analyze`).

The backend processes the request through a series of services, including preprocessing, classification, transcription, localization, and alignment.
Each service performs a specific task and passes its output to the next stage.
Once processing is complete, the results are sent back to the frontend, where they are displayed to the user.
The system also stores the results for report generation and history management.

== Data Flow Description
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

== Chapter Summary
This chapter presented a detailed analysis of the system requirements, covering both functional and non-functional aspects.
It also examined feasibility, system interactions, workflows, and data processing mechanisms.
The requirements highlight the need for an accurate, interpretable, and multilingual-ready stuttering detection system that can operate efficiently in both online and offline environments.

The next chapter focuses on the system design and architecture, detailing how these requirements are translated into an implementable solution.

#pagebreak()
