#import "../lib.typ": *

// --- Chapter 3: System Design ---
#chapter_heading[SYSTEM DESIGN]

== INTRODUCTION

System design is a core stage in software development that sets out the overall structure and main functionality of a system.
It turns user and technical requirements into a detailed framework that shows how different components, modules, and interfaces will interact.
The process involves both high-level architecture, which lays out the system's major elements and their connections, and low-level design, which specifies detailed components such as data structures, algorithms, and communication flows.

System design aims to build a solution that meets user needs and still delivers strong performance, reliability, scalability, and security.
It also values maintainability and flexibility, so the system can adapt to future changes and enhancements.
Good system design needs close collaboration among developers, architects, and domain experts, so the architecture matches functional goals and business requirements.
By organizing how each part of the system works and communicates, system design lays a solid foundation for efficient implementation and stable operation over time.

== SYSTEM ARCHITECTURE AND DESIGN

The system architecture sets out the overall structure of the Stuttering Classification and Localization System, which uses deep learning.
It gives a full picture of how the different components interact to process speech audio, extract useful features, and classify utterances as normal or dysfluent.
A sound architecture keeps data flowing smoothly, stays modular and scalable, letting the model reach high diagnostic accuracy while remaining open to future changes.
It covers all the main stages of the pipeline, from preprocessing and feature extraction to model training and classification, so that performance stays stable.
It also supports smooth integration with user interfaces for audio upload, visualization, and prediction display.
The design keeps data security and patient confidentiality central, so the analysis workflow stays ethical and reliable in practice.
The architecture also leaves room for performance tuning, so stutter detection stays fast and accurate even on large speech datasets.

Figure 3.1 shows the proposed deep learning architecture for the automatic classification and localization of stuttering from speech audio.
The framework combines two model pathways: five independent Wav2Vec 2.0 binary classifiers for classification with a Wav2Vec2 frame-level pipeline for localization, which improves feature extraction and makes classification more reliable.
The complete workflow is made up of five major phases: Input, Pre-processing, Feature Extraction, Classification, and Output.
This dual-model structure learns richer representations and raises the overall detection accuracy.
The architecture also separates reusable signal preparation from task-specific prediction.
The same normalized waveform can go straight to classification, transcription, and localization, while the final fusion stage combines their outputs without repeating preprocessing work.

#add_image(align(center, image("/assets/architecture.svg", width: 70%)), caption: [System Architecture])

*Speech Audio Input:* The system starts by collecting the raw speech audio samples, which form the primary dataset for automatic stutter detection.
Audio enters the system when the user records directly or uploads a pre-recorded file through the web interface or desktop application.
The web app captures audio with MediaRecorder APIs; the desktop app uses sounddevice.
Each sample may contain one or more dysfluencies, including blocks, prolongations, word repetitions, sound repetitions, and interjections.
Recording quality matters a great deal, since differences in microphone hardware, background noise, and speaking rate can affect model reliability.
The dataset includes recordings from Project Boli, SEP-28K, and UCLASS, and the ingestion pipeline resolves the English material of all three sources.
This stage gives the diagnostic system its clean data for the steps that follow.

*Preprocessing:* Before it is fed to the deep learning model, speech audio undergoes preprocessing to improve clarity, cut noise, and keep samples consistent.
The audio is sent to the backend and converted to 16 kHz mono WAV with FFmpeg so every clip is handled the same way.
The preprocessing pipeline also includes DC offset removal, peak normalization, and silence trimming to clean the signal and remove unwanted artifacts.
Fixed-length padding truncates or pads each clip to 48,000 samples, three seconds at 16 kHz, so all inputs have the same length.
After this step the input audio is standardized and optimized, so the deep learning architecture can extract features and classify it accurately.

*Wav2Vec 2.0:* The detection framework starts with Wav2Vec 2.0, a deep self-supervised model pre-trained on large amounts of unlabeled speech.
Wav2Vec 2.0 turns the raw waveform into contextual speech representations, learning useful acoustic features without needing manually labeled data.
This structure helps the model keep important information and capture fine-grained phonetic and prosodic patterns, such as broken phonation, repetitions, and elongated sounds, which are clear markers of dysfluent speech.
Because it is pre-trained on large speech corpora, the model already understands general speech features, so adapting it to stutter detection with limited labeled data is easier.
Because it extracts contextual features at several levels, it gives a strong base for detecting stutter events with high precision.

*Detection and Localization Pipelines:* For higher accuracy, the proposed system uses a hybrid framework that combines classification and localization.
After preprocessing, the audio runs through three parallel pipelines: classification, transcription, and localization.
The classification pipeline extracts speech representations with Wav2Vec 2.0 and passes them through five independent binary classifiers, each trained for a distinct dysfluency type.
The localization pipeline finds dysfluency regions at the frame level with Wav2Vec2 frame-level feature extraction, so it can mark the exact segments of speech where dysfluencies occur.
Combining the two approaches lets the system catch fine acoustic detail and broader prosodic structure, so results stay stable overall when the recordings vary in noise, clarity, or speaking style.

*Classification:* Once feature extraction ends, the Wav2Vec 2.0 representations go to the classification stage.
Here, five independent binary classifiers give a presence probability for each dysfluency type, using decision boundaries learned during training to separate fluent from dysfluent speech.
Sigmoid outputs give per-class probabilities, and binary cross-entropy loss guides optimization during training.
Their outputs are combined into a multi-label result that reports the presence probability of each dysfluency type and summarizes the detected classes and the primary dysfluency.
A multitask shared-backbone variant was also trained for comparison, and its outputs are combined with localization at prediction time: the combiner labels each localized region with per-class saliency scores and fuses the two views into a single annotated result.
The output gives speech-language pathologists clear results they can act on when identifying stutter events.

*Transcription and Alignment:* The transcription pipeline uses the Whisper Automatic Speech Recognition (ASR) model to produce a timestamped transcript, with support for English, Kannada, and Hindi.
A Connectionist Temporal Classification (CTC) based time alignment step links the pipelines together, mapping the detected dysfluency regions to specific words or syllables in the transcript.
Language-specific adapters keep the alignment accurate for English, Kannada, and Hindi.

*Output:* In the final stage, the system presents its predictions through a simple interface.
The output typically shows the detected dysfluency types along with confidence scores that indicate how certain the model is about each decision, plus a summary of the primary dysfluency.
To make the results easier to read, the outputs appear as waveform overlays, spectrogram visualizations, timestamped transcripts, and a detailed clinical-style report.
The outputs are designed to be plain, clear, and clinically relevant, so they suit practical use in clinical or self-assessment settings.
By providing accurate and interpretable results, the system supports early assessment and can play a useful role in improving outcomes for people who stutter.

Both the web and desktop applications load trained models through the centralized model registry.
The registry reads from a registry module (the `model/registry` package) and a configuration file (`registry.json`), so checkpoints can be updated or replaced without changing application code, which keeps the system flexible and easy to maintain.

== FLOWCHART

The process starts with the input audio, recorded or uploaded by the user.
The audio is converted to 16 kHz mono WAV with FFmpeg, then padded or truncated to 48,000 samples, three seconds at 16 kHz, so all inputs have the same length.

After normalization, the audio runs through classification, transcription, and localization in parallel.

A CTC-based alignment step combines these outputs and maps the detected dysfluencies to specific words or syllables.
The final data becomes clear per-word annotations that appear in the waveform and spectrogram displays and in the analysis report.
The stages stay separate so each can be tested on its own, and every branch receives the same normalized audio.

#add_image(align(center, image("/assets/01-system-flowchart.svg", width: 85%)), caption: [Flowchart])

Figure 3.2 shows the full workflow, starting from input audio and passing through preprocessing, the three model branches, and the final report.
First, the Wav2Vec 2.0 model is used to extract high-level contextual speech features.
For better overall performance, a hybrid model combines five separate independent Wav2Vec 2.0 binary classifiers with a Wav2Vec2 frame-level localizer: the classifiers capture dysfluency-specific acoustic patterns and the localizer finds fine-grained temporal speech segments.
The outputs from both pipelines are fused through CTC-based alignment and passed through a final aggregation layer that sorts the audio samples into blocks, prolongations, sound repetitions, word repetitions, or interjections, along with the precise localized time regions.

The system also defines a structured data flow for training.
Training uses three datasets: Project Boli (sourced via Git repositories), SEP-28K (approximately 28,000 audio clips), and UCLASS (both obtained via Kaggle).
Each dataset goes through its own normalization functions.

The processed datasets merge into a unified format: a combined_labels.csv file with multi-label binary annotations, plus individual interval CSV files for each audio clip.
The merged dataset is split into training, validation, and testing subsets with an 80:10:10 split, with the training subset further re-split 80/20 during model training.
The subsets are organized with symbolic links for efficient access, and preprocessed audio files are cached to speed up training.

== USE CASE DIAGRAM AND DESCRIPTIONS

Figure 3.3 shows the use case diagram in UML: the actor stands outside the system boundary, and lines connect it to the use cases it can start.
The only actor is the User, since clinicians and people assessing their own speech follow the same path.

Upload / Record Speech is the entry point.
Audio can come from the microphone or a saved file, and the interface checks the file before sending it to the backend.

Select Language sets the language used for transcription and alignment; English, Kannada, and Hindi are supported.

Analyze Speech is the central use case.
Full mode runs preprocessing, classification, transcription, localization, and CTC-based alignment.
Classification-only mode runs the five classifiers and skips transcription and localization.
Full mode returns a confidence score for each dysfluency type, a summary of the primary dysfluency, and the time regions where events were found.

View Analysis Results shows the per-class scores, a waveform with color-coded dysfluency regions, a spectrogram, and a timestamped transcript.

Download Clinical Report writes the stored result to a PDF without rerunning the models.
Review Assessment History reopens earlier sessions from IndexedDB and localStorage, so recordings can be compared over time.
Theme switching and the standardized reading passages support the same workflow.

#add_image(align(center, image("/assets/03-use-case-diagram.svg", width: 60%)), caption: [Use Case Diagram])

Figure 3.3 groups the six use cases around the single actor.
Upload / Record Speech and Select Language supply the audio and the language setting.
Analyze Speech uses both and produces the classification scores, localized regions, and transcript.
View Analysis Results, Download Clinical Report, and Review Assessment History then read the stored result.
A report or a history entry cannot exist before an analysis has run.
Because the results page, the report, and the history all read the same payload, they show the same outcome for a given recording.

== WORKFLOW DIAGRAM

A workflow diagram is a visual representation of a sequence of operations or tasks that complete a specific process, and people often use one to analyse, design, or manage complex systems.
It uses standard symbols: rectangles to mark actions, diamonds to mark decision points, and arrows to show the direction of flow. These make the procedural steps easier for a reader to follow.
In the field of deep learning and stutter detection, the workflow diagram shown here maps out the full process for developing a stutter classification and localization system.

#add_image(align(center, image("/assets/workflow-diagram-proper.svg", width: 100%)), caption: [Workflow Diagram])

Figure 3.4 shows the full workflow of the proposed deep learning pipeline for automatic stutter detection and localization.
The process starts by collecting raw speech audio samples from the three public stuttering datasets, Project Boli, SEP-28K, and UCLASS, all converted to 16 kHz mono and standardized through DC offset removal, peak normalization, silence trimming, and fixed-length padding.
Data augmentation such as random noise injection, time stretching, pitch shifting, temporal shifting, and amplitude scaling is applied to the waveforms, along with time and frequency masking on the spectrograms, which expands the effective dataset so the models generalize well across varied speaking patterns.
The processed audio is split into training, validation, and testing sets at an 80:10:10 ratio, and the training subset is further re-split 80/20 during training.
Model development starts by fine-tuning a pre-trained Wav2Vec 2.0 backbone, then builds a hybrid architecture that combines five independent Wav2Vec 2.0 binary classifiers with a Wav2Vec2 frame-level localizer to capture both global dysfluency patterns and fine-grained temporal segments.
Hyperparameters including learning rate, backbone freeze duration, optimizer, and the number of trainable layers are tuned to improve performance, and the trained network is validated periodically to keep learning stable and prevent overfitting.
Finally, evaluation metrics such as precision, recall, F1-score, AUROC, and mean IoU are computed, and the best-performing model is selected for reliable automated assessment of speech dysfluencies.

== SEQUENCE DIAGRAM

The sequence diagram shows how the components interact during execution.
The user works through the graphical user interface (GUI), which sends a request to the backend API endpoint (`/api/analyze`).

#add_image(align(center, image("/assets/04-sequence-diagram.svg", width: 95%)), caption: [Sequence Diagram])

The backend sends the request through a series of services: preprocessing, classification, transcription, localization, and alignment.
Each service completes its step and passes the result to the next stage.
When processing finishes, the results return to the frontend, where the user can see them.
The system also saves the results for report generation and history management.
This service sequence gives each component a clear role and makes failures easier to trace.
It also lets the system reuse the stored analysis when the user opens a previous result, because the report layer receives the same structured payload that was produced during the original request.

The request lifecycle starts when the user confirms an analysis, at which point the frontend packs the audio, the selected mode, and any optional parameters into a single request.
The backend checks the payload before it calls any model, rejecting unsupported formats or empty inputs early so that expensive inference is never wasted on bad data.
Preprocessing normalizes the waveform and produces the log-mel spectrograms and frame-level features that later stages expect, and it is the one place where audio conventions are enforced.
Classification then gives probability scores for each dysfluency type, and transcription produces a timestamped transcript that the alignment stage later links to the detected events.
Localization runs in parallel with transcription where possible, since the two stages share the same preprocessed audio and do not depend on each other's output, which keeps the overall latency low.
The alignment stage joins word timings with the localized regions so that each dysfluency can be reported in the context of the spoken words around it.
If any single service fails, the backend returns a structured error instead of a partial result, and the frontend shows the failure without discarding the user's audio, so the request can be retried.
When the request succeeds, the fused payload is stored locally so that reports and session history can be rebuilt later without re-running models, which also keeps repeated report generation fast and independent of the analysis services.
This ordering also keeps the application responsive to users. Independent operations can run in parallel after preprocessing, while the final report waits only for the outputs needed for alignment and presentation.
The stored result keeps the predictions, timestamps, transcript, and derived annotations together, so the web and desktop interfaces can show the same analysis without applying different interpretation rules.

#pagebreak()
