#import "../lib.typ": *
#import "../meta.typ": *

// --- Chapter 2: Requirement Specification and Analysis ---
#chapter_heading(size: 18pt)[REQUIREMENT SPECIFICATION AND ANALYSIS]

== INTRODUCTION

A deep learning framework for stuttering classification and localization must meet clinical and technical expectations at the same time.
Before implementation, the project therefore fixes all its functional and non-functional requirements, the system components, and the workflow that connects them.
The main goal is to replace manual speech-language assessment with a model that detects dysfluencies early and reports them objectively.

The system takes an audio recording, extracts features, and sorts it into the five stuttering types.
It also locates each event in the signal.
That should cut the time manual analysis takes and reduce variation between observers, without making the output hard to read.
Wav2Vec 2.0 handles feature extraction, a Wav2Vec2 frame-level model handles localization, and Whisper produces the transcript.
The system must run on standard hardware, cope with larger datasets as they arrive, offer a simple interface, and keep patient data secure and confidential.

== FUNCTIONAL REQUIREMENTS

The functional requirements describe what the system must do to detect and classify stuttering events.
Clinicians, speech-language pathologists, and people who stutter can upload a recording or record audio through the interface.
The supported formats are WAV, MP3, FLAC, and M4A.

The system processes the audio and sorts it into five dysfluency types: prolongation, block, sound repetition, word repetition, and interjection.
For each detected class, the interface shows the result with a confidence score, so the user can see how far to trust the prediction.
The system also returns the timestamps of localized events.
Waveforms, spectrograms, and overlays mark the regions that shaped the model's decision.
Users can download the full analysis report, and uploaded audio and results are stored securely for later reference.

=== User Requirements

User requirements describe what all end users must be able to do.
They reflect how clinicians, speech-language pathologists, researchers, and people who stutter will use the system while they assess or monitor fluency.
Those users need an interface that is easy to navigate, that shows classification results, visual interpretations, and confidence levels without technical complexity, and that works with little manual effort.
For a speech analysis platform, the general requirements are:

- #keep[The user should be able to upload or record a speech sample for analysis.]
- #keep[The user should be able to see classification results as soon as the analysis finishes.]
- #keep[The system should provide detailed analysis reports.]

=== System Requirements

System requirements cover the software and hardware capabilities the system needs.
The stuttering detection system has to handle the computational load of deep learning speech analysis while delivering the operations described above.
Hardware guidance appears later in this chapter.

These requirements include:

- #keep[The system must accept audio uploads securely and classify them.]
- #keep[Users need a way to save previous analysis results and open them again.]
- #keep[Adequate storage must be available for managing large speech datasets.]

== NON-FUNCTIONAL REQUIREMENTS

Non-functional requirements set the quality, performance, and usability standards the system has to meet.
Performance comes first, because the models run deep learning and audio processing and must return results quickly.
GPU acceleration and memory management keep the recordings moving through the pipeline.
Reliability means consistent accuracy across datasets and fault tolerance when a run is interrupted.
Scalability allows any larger datasets or extra dysfluency types without hurting overall performance.
Usability means a simple interface for technical and non-technical users alike, with clear upload, analysis, and visualisation steps.
Maintainability comes from modular code that is easy to update and retrain.
Portability and availability keep the system running across platforms, with high uptime and regular backups.

=== Performance

- #keep[The system should process and analyse a speech sample within 3 to 5 seconds.]
- #keep[The classification results should keep false negatives and false positives low.]
- #keep[Performance metrics should be monitored regularly.]
- #keep[The system should handle audio uploads without significant delay.]

=== Scalability

- The system should handle growing speech datasets efficiently.
- #keep[It should scale across multiple clinics and research centres.]
- #keep[The system should adapt to improvements in deep learning models.]
- #keep[The system should handle a higher volume of concurrent users.]

=== Availability

- The system should stay accessible at all times, with minimal downtime during clinical use.
- #keep[Components should tolerate faults so a single failure has less impact.]
- #keep[The platform should keep serving users during updates or model improvements.]

=== Usability

- #keep[The interface should be easy for clinicians, speech-language pathologists, and non-technical users.]
- #keep[Audio upload and results should be easy to find and navigate.]
- #keep[The visualisation outputs should help users understand the model's predictions for the analysed speech.]

== Software Requirements

The software requirements cover the tools used to develop, train, validate, and deploy the stutter detection models.

- #keep[*Operating System:* the system runs on Linux and Windows, including Windows 10 and above and common Linux distributions. The PySide6 desktop application and the web application run on either platform, and training and inference are supported on CPU or an NVIDIA GPU.]
- #keep[*Deep Learning Framework:* PyTorch is the primary framework for building and training the models. It provides GPU support through torch.compile, mixed precision, and TensorFloat-32 (TF32), and it scales to the audio classification tasks in this project. Hugging Face Transformers supplies the pretrained speech models, Wav2Vec 2.0 for feature extraction and Whisper for timestamped transcription.]
- *Libraries:* a range of Python libraries supports different stages of the project:
  - #keep[librosa: Used for audio preprocessing and manipulation, including loading, resampling, mel-spectrogram computation, and silence trimming.]
  - #keep[NumPy: Provides support for numerical operations and array handling.]
  - #keep[Pandas: managing structured data such as labels or metadata.]
  - #keep[Scikit-learn: Used for data splitting, model evaluation metrics, and traditional machine learning algorithms.]

== Hardware Requirements

The hardware requirements give the minimum and recommended specifications for running the models.

- #keep[*Processor (CPU):* a multi-core processor such as an Intel Core i5 (8th generation or newer) or AMD Ryzen 5 and above. It provides enough power for audio preprocessing and training workflows.]
- #keep[*Storage (Hard Disk):* at least 10 GB of free disk space for datasets, checkpoints, logs, and temporary files. An SSD is recommended for faster read and write speeds.]
- #keep[*Memory (RAM):* at least 8 GB of RAM to run the training scripts and handle datasets, with 16 GB or more recommended during training and validation, when data loading and augmentation are active.]
- #keep[*Graphics Processing Unit (GPU):* an NVIDIA GPU with CUDA support is recommended for faster training. Cards such as the RTX 2060 or higher (RTX 3060, RTX 3080) cut training time and handle larger datasets and deeper networks. CUDA and cuDNN must be installed for full compatibility with PyTorch.]
- *Audio Input Device:* a working microphone is required for recording speech in the application.

The minimum configuration handles inference on short clips.
Fine-tuning the larger pretrained backbones benefits from the recommended configuration, and a GPU with at least 8 GB of memory is preferable, though classification-only mode runs comfortably on a CPU.

#pagebreak()
