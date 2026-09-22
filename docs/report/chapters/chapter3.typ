#import "../lib.typ": *

// --- Chapter 3: System Design ---
#chapter_heading[SYSTEM DESIGN]

== INTRODUCTION

System design is an essential stage in software development that focuses on defining the overall structure and functionality of a system.
It translates user and technical requirements into a detailed framework that outlines how different components, modules, and interfaces will interact.
The process involves creating both high-level architecture, which defines the system's major elements and their interconnections, and low-level design, which specifies detailed components such as data structures, algorithms, and communication flows.

The main objective of system design is to build a solution that fulfills user needs while ensuring performance, reliability, scalability, and security.
It also emphasizes maintainability and flexibility so the system can adapt to future enhancements.
Effective system design requires collaboration among developers, architects, and domain experts to ensure the architecture aligns with functional goals and business requirements.
By carefully organizing how each part of the system operates and communicates, system design establishes a strong foundation for efficient implementation and long-term stability.

== SYSTEM ARCHITECTURE AND DESIGN

The system architecture defines the overall structure and organization of the Stuttering Classification and Localization System using deep learning.
It provides a comprehensive framework describing how different components interact to process speech audio, extract discriminative features, and classify utterances into normal and dysfluent categories.
A well-structured architecture ensures efficient data flow, modularity, and scalability, allowing the model to deliver high diagnostic accuracy while remaining adaptable to future advancements.
The architecture incorporates critical stages such as preprocessing, feature extraction, model training, and classification to ensure robust performance.
Furthermore, it supports seamless integration with user interfaces for audio upload, visualization, and prediction display.
Emphasis is placed on maintaining data security and patient confidentiality, ensuring ethical and reliable operation throughout the analysis workflow.
Additionally, the architecture allows for performance optimization, enabling fast and accurate stutter detection even when deployed on large speech datasets.

Figure 3.1 illustrates the proposed deep learning architecture designed for the automatic classification and localization of stuttering using speech audio.
The framework integrates two model pathways, five independent Wav2Vec 2.0 binary classifiers for classification and a Wav2Vec2 frame-level pipeline for localization, to improve feature extraction, multi-perspective learning, and classification robustness.
The complete workflow consists of five major phases: Input, Pre-processing, Feature Extraction, Classification, and Output.
This dual-model structure enables richer representation learning and enhances overall detection accuracy.

#add_image(align(center, image("/assets/architecture-verticle.png", height: 32.5%)), caption: [System Architecture])

*Speech Audio Input:* The system begins with the collection of speech audio samples, which serve as the primary dataset for stutter detection.
Audio enters the system when the user records directly or uploads a pre-recorded file through the web interface or desktop application.
The web app captures audio with MediaRecorder APIs; the desktop app uses sounddevice.
Each sample may contain one or more dysfluencies, including blocks, prolongations, word repetitions, sound repetitions, and interjections.
Ensuring adequate recording quality is crucial, as variations in microphone hardware, background noise, and speaking rate can affect model reliability.
The dataset includes recordings from Project Boli, SEP-28K, and UCLASS captured across English, Kannada, and Hindi.
This stage establishes the foundation of the diagnostic system by providing high-quality data for subsequent processing.

*Preprocessing:* Before being fed into the deep learning model, speech audio undergoes preprocessing to enhance clarity, reduce noise, and ensure consistency across samples.
The audio is sent to the backend and converted to 16 kHz mono WAV with FFmpeg so every clip is handled the same way.
The preprocessing pipeline also includes DC offset removal, peak normalization, and silence trimming to clean the signal and remove unwanted artifacts.
Fixed-length padding truncates or pads each clip to 48,000 samples, three seconds at 16 kHz, so all inputs have the same length.
This step ensures that the input audio is standardized, optimized, and ready for accurate feature extraction and classification by the deep learning architecture.

*Wav2Vec 2.0:* The first part of the detection framework is Wav2Vec 2.0, a deep self-supervised model pre-trained on large amounts of unlabeled speech.
Wav2Vec 2.0 quantizes the raw waveform into contextualized speech representations, learning robust acoustic features without requiring manually labeled data.
This structure helps the model retain important information and capture fine-grained phonetic and prosodic patterns, such as broken phonation, repetitions, and elongated sounds, which are key indicators of dysfluent speech.
Since it is pre-trained on large speech corpora, the model already understands general speech features, which makes it easier to adapt to stutter detection with limited labeled data.
Its ability to extract multi-level, contextual features makes it a powerful foundation for detecting stutter events with high precision.

*Detection and Localization Pipelines:* To achieve higher accuracy, the proposed system uses a hybrid detection framework that combines classification and localization.
After preprocessing, the audio runs through three parallel pipelines: classification, transcription, and localization.
The classification pipeline extracts speech representations with Wav2Vec 2.0 and passes them through five independent binary classifiers, each trained for a distinct dysfluency type.
The localization pipeline identifies dysfluency regions at the frame level using Wav2Vec2 frame-level feature extraction, so it can mark the precise segments of speech where dysfluencies occur.
By integrating both approaches, the system captures fine acoustic details and broader prosodic structures in the speech signal, ensuring robust results even when recordings vary in noise, clarity, or speaking style.

*Classification:* Once the feature extraction is complete, the Wav2Vec 2.0 representations are passed to the classification stage.
Here, five independent binary classifiers assign the presence probability of each dysfluency type, using decision boundaries learned during training to separate fluent and dysfluent speech.
Sigmoid outputs generate per-class probabilities, while binary cross-entropy loss ensures efficient optimization during training.
Their outputs are aggregated into a multi-label result that reports the presence probability of each dysfluency type and summarizes the detected classes and the primary dysfluency.
A multitask shared-backbone variant was also trained for comparison, and its outputs are combined with localization at prediction time: the combiner labels each localized region with per-class saliency scores and fuses the two views into a single annotated result.
This stage translates complex audio data into clear diagnostic outcomes that can assist speech-language pathologists in identifying stutter events.

*Transcription and Alignment:* The transcription pipeline uses the Whisper Automatic Speech Recognition (ASR) model to produce a timestamped transcript, with support for English, Kannada, and Hindi.
A Connectionist Temporal Classification (CTC) based time alignment step ties the pipelines together, mapping the detected dysfluency regions to specific words or syllables in the transcript.
Language-specific adapters keep the alignment accurate for English, Kannada, and Hindi.

*Output:* In the final stage, the system presents its predictions through a user-friendly interface.
The output typically shows the detected dysfluency types along with confidence scores that indicate how certain the model is about each decision, and a summary of the primary dysfluency.
To make the results more interpretable, the outputs surface as waveform overlays, spectrogram visualizations, timestamped transcripts, and a detailed clinical-style report.
The outputs are designed to be simple, clear, and clinically relevant, making them suitable for practical use in clinical or self-assessment settings.
By providing accurate and interpretable results, the system supports early assessment and can play a valuable role in improving outcomes for people who stutter.

Both the web and desktop applications load trained models through the centralized model registry.
The registry reads from a registry module (the `model/registry` package) and a configuration file (`registry.json`), so checkpoints can be updated or replaced without changing application code, which keeps the system flexible and easy to maintain.

== FLOWCHART
The data flow design follows the audio from input acquisition to final output.

The process begins with the input audio, recorded or uploaded by the user.
The audio is converted to 16 kHz mono WAV with FFmpeg, then padded or truncated to 48,000 samples, three seconds at 16 kHz, so all inputs have the same length.

After normalization, the audio runs through three parallel pipelines: classification, transcription, and localization.
Classification produces dysfluency probabilities, transcription a timestamped transcript, and localization frame-level dysfluency regions.

A CTC-based alignment step combines these outputs and maps the detected dysfluencies to specific words or syllables.
The final data is turned into per-word annotations, shown in waveform and spectrogram displays, and written into the analysis report.

The system also defines a structured data flow for training.
Training uses three datasets: Project Boli (sourced via Git repositories), SEP-28K (approximately 28,000 audio clips), and UCLASS (both obtained via Kaggle).
Each dataset goes through its own normalization functions.

The processed datasets merge into a unified format: a combined_labels.csv file with multi-label binary annotations, plus individual interval CSV files for each audio clip.
The merged dataset is split into training, validation, and testing subsets with an 80:10:10 split, with the training subset further re-split 80/20 during model training.
The subsets are organized with symbolic links for efficient access, and preprocessed audio files are cached to speed up training.

#add_image(align(center, image("/assets/Flowchart.png", height: 38%)), caption: [Flowchart])

The flowchart shown in Figure 3.2 illustrates the workflow for stutter detection using the proposed hybrid deep learning approach.
The process begins with the collection of speech audio recordings, followed by preprocessing steps such as conversion to 16 kHz mono WAV, DC offset removal, peak normalization, silence trimming, and fixed-length padding to improve signal quality and variability.
The refined audio is then split into training, validation, and testing datasets with an 80:10:10 split.
Initially, the Wav2Vec 2.0 model is used to extract high-level contextual speech features.
To enhance performance, a hybrid model combining five independent Wav2Vec 2.0 binary classifiers with a Wav2Vec2 frame-level localizer is developed, where the classifiers capture dysfluency-specific acoustic patterns and the localizer identifies fine-grained temporal segments.
The outputs from both pipelines are fused through CTC-based alignment and passed through a final aggregation layer that categorizes the audio samples as containing blocks, prolongations, sound repetitions, word repetitions, or interjections, along with the precise localized regions.
This flowchart provides a clear overview of the systematic steps involved in the proposed framework for stutter detection.

== USE CASE DIAGRAM AND DESCRIPTIONS

The use case diagram shows how the user interacts with the system.
The primary actor is the *User*, either a clinician or an individual using the application for self-assessment.

The main workflow covers recording audio, uploading pre-recorded audio, and running speech analysis in either full mode or classification-only mode.
After analysis, users can view waveform visualizations, spectrograms, transcripts, and stutter detection results.

Users can also localize dysfluencies within the speech, generate analysis or clinical reports, and revisit previously recorded sessions through a history feature.
Supporting actions include toggling between interface themes and accessing standardized reading passages for consistent evaluation.

The typical flow is: record or upload audio, initiate analysis, review the visualizations and results, and generate or save the report.

#add_image(align(center, image("/assets/UseCaseDiagram.jpeg", height: 35%)), caption: [Use Case Diagram])

The use case diagram shown in Figure 3.3 illustrates the interactions between the user and the stutter detection system.
The user can register or log in, then either record audio through the microphone or upload a pre-recorded speech file, which is analyzed through the full or classification-only pipeline.
The system displays the classification results with confidence scores, localized dysfluency regions, and a timestamped transcript, and allows the user to generate and download analysis or clinical reports.
Additional use cases include viewing the session history, switching between light and dark themes, and accessing standardized reading passages for consistent assessment.
This diagram provides a clear overview of the functional interactions supported by the proposed framework.

== WORKFLOW DIAGRAM

A workflow diagram is a structured visual representation of a sequence of operations or tasks carried out to complete a specific process, typically used to analyse, design, or manage complex systems.
It employs standardized symbols such as rectangles to denote actions, diamonds for decision points, and arrows to indicate the direction of flow, thereby offering clarity and insight into the procedural steps involved.
In the context of deep learning and stutter detection, the workflow diagram illustrated here systematically maps out the process for developing a stutter classification and localization system.

#add_image(align(center, image("/assets/workflow-diagram.png", width: 92%)), caption: [Workflow Diagram])

The workflow diagram illustrates the workflow of the proposed deep learning pipeline for stutter detection and localization.
The process begins with the collection of speech audio samples from three public stuttering datasets, Project Boli, SEP-28K, and UCLASS, all converted to 16 kHz mono and standardized through DC offset removal, peak normalization, silence trimming, and fixed-length padding.
Data augmentation techniques such as random noise injection, time stretching, pitch shifting, temporal shifting, and amplitude scaling are applied to the waveforms, along with time and frequency masking on the spectrograms, expanding the effective dataset so the models generalize well across varied speaking patterns.
The processed audio is then split into training, validation, and testing sets using an 80:10:10 ratio, with the training subset further re-split 80/20 during training.
Model development begins with fine-tuning a pre-trained Wav2Vec 2.0 backbone, followed by a hybrid architecture that combines five independent Wav2Vec 2.0 binary classifiers with a Wav2Vec2 frame-level localizer to capture both global dysfluency patterns and fine-grained temporal segments.
Hyperparameters including learning rate, backbone freeze duration, optimizer, and the number of trainable layers are tuned to enhance performance, and the trained network is periodically validated to ensure stable learning and prevent overfitting.
Finally, evaluation metrics such as precision, recall, F1-score, AUROC, and mean IoU are computed, and the best-performing model is selected for reliable automated assessment of speech dysfluencies.

== SEQUENCE DIAGRAM

The sequence diagram shows how the components interact during execution.
The user works through the graphical user interface (GUI), which sends a request to the backend API endpoint (`/api/analyze`).

The backend runs the request through a chain of services: preprocessing, classification, transcription, localization, and alignment.
Each service does its part and passes its output to the next stage.
When processing completes, the results go back to the frontend, where the user sees them.
The system also stores the results for report generation and history management.

#add_image(align(center, image("/assets/SequenceDiagram.jpeg", height: 35%)), caption: [Sequence Diagram])

The sequence diagram shown in Figure 3.5 illustrates the interaction flow between the user, the frontend, the backend services, and the stored results during a speech analysis session.
The user initiates the analysis through the graphical user interface, which sends a request to the backend API endpoint.
The backend sequentially invokes the preprocessing, classification, transcription, localization, and alignment services, each consuming the output of the previous stage.
Once the processing completes, the fused results are returned to the frontend for display and saved in the local storage for reports and the session history.
This diagram provides a clear overview of the message exchange that drives the proposed framework.

#pagebreak()
