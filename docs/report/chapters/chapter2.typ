#import "../lib.typ": *
#import "../meta.typ": *

// --- Chapter 2: Requirement Specification and Analysis ---
#chapter_heading[REQUIREMENT SPECIFICATION AND ANALYSIS]

== INTRODUCTION

The development of a deep learning-based framework for stuttering classification and localization requires a clear understanding of both clinical and technical aspects.
Before implementation, it is important to define specific functional and non-functional requirements, system components, and workflow to ensure accuracy and clinical reliability.
The main goal is to transform the manual process of speech-language assessment into an efficient, automated computational model that supports early and objective dysfluency detection.
The system should effectively process audio recordings, extract key features, and classify them into distinct stuttering types while localizing dysfluency events within the signal.
It must address challenges like time-consuming manual analysis and inter-observer variation, emphasizing automation, precision, and interpretability.
Technically, the framework integrates Wav2Vec 2.0 feature extraction with Wav2Vec2 frame-level localization and Whisper transcription for robust detection and localization.
In addition, the system should operate efficiently on standard hardware, support scalability for future datasets, and offer a user-friendly interface for clinicians and end users.
Data security and confidentiality are essential to maintain ethical and legal standards.
Overall, the requirement specification and analysis phase ensures that the framework is both technically strong and clinically valuable, bridging the gap between healthcare needs and AI-driven solutions.

== FUNCTIONAL REQUIREMENTS

The functional requirements describe the key operations that the system must perform to support automated detection and classification of stuttering events using deep learning models.
The application allows users, such as clinicians, speech-language pathologists, or individuals who stutter, to upload a speech recording or capture audio directly through the interface in common formats such as WAV, MP3, FLAC, and M4A.
Once the audio is provided, the system processes the signal and classifies it into five dysfluency types: prolongation, block, sound repetition, word repetition, and interjection.
The classification result is displayed on the interface along with a confidence score for each detected class, helping users understand how reliable every prediction is.
The system also localizes dysfluency events within the audio and returns their exact temporal positions as timestamps, with visualization through waveforms, spectrograms, and dysfluency overlays that highlight the regions that influenced the model's decision, improving interpretability.
Additionally, users can download the detailed analysis report for record-keeping.
The framework ensures that uploaded audio and prediction results are stored securely for future reference.

=== User Requirements

User requirements specify the functionalities and services that the end users should be able to access and utilize.
These typically reflect how users will interact with the system.
The proposed system is primarily designed for clinicians, speech-language pathologists, researchers, and individuals who stutter involved in the assessment and monitoring of speech fluency.
The users of this system require an intuitive, reliable, and efficient platform that can process speech recordings and provide accurate diagnostic insights with minimal manual effort.
The users expect the interface to be simple, interactive, and easy to navigate, allowing them to view classification results, visual interpretations, and confidence levels without technical complexity.

For a speech analysis platform, general user requirements include:

- The user should be able to upload or record a speech sample for analysis.
- The user should be able to access real-time classification results.
- The system should provide detailed analysis reports.

=== System Requirements

System requirements define the essential capabilities of software and hardware needed to support system functionality.
The proposed stuttering detection system requires components that can efficiently handle the computational demands of deep learning-based speech analysis while delivering the core functional operations described above.
Hardware guidelines appear in the hardware requirements section later in this chapter.

These requirements include:

- The system should support secure audio upload and classification.
- Users should be able to store and retrieve previous analysis results.
- The system should support proper storage for managing large speech datasets.

== NON-FUNCTIONAL REQUIREMENTS

The non-functional requirements define the overall quality, performance, and usability standards that ensure the system operates efficiently and reliably.
The key requirement is performance efficiency, as the models handle deep learning and audio processing tasks that must deliver fast and accurate results with minimal delay.
GPU acceleration and optimized memory management help achieve smooth processing of speech recordings.
Reliability is equally important: the system should maintain consistent accuracy across datasets and handle interruptions through fault tolerance.
It must also be scalable, allowing expansion for larger datasets or additional dysfluency types without compromising performance.
In terms of usability, the interface should be simple, intuitive, and accessible to both technical and non-technical users for easy audio upload, analysis, and visualization.
Maintainability should be ensured through modular, well-structured code that supports easy updates and retraining.
Finally, portability and availability ensure the system runs smoothly across different platforms with high uptime and regular backups for uninterrupted access.

=== Performance

- The system should process and analyse a speech sample within 3-5 seconds.
- The classification results should minimize false positives and false negatives.
- Performance metrics should be monitored regularly.
- The system should handle processing of audio uploads without significant delay.

=== Scalability

- The system should efficiently handle increasing speech datasets.
- It should be scalable for integration across multiple clinics and research centres.
- The system should be adaptable for improvements in deep learning models.
- The system should be capable of handling a higher volume of concurrent users.

=== Availability

- The system should remain accessible to users at all times, ensuring minimal downtime during clinical use.
- System components are designed to be fault-tolerant to reduce the impact of failures.
- The platform should support uninterrupted service even during updates or model improvements.

=== Usability

- The user interface should be user-friendly for clinicians, speech-language pathologists, and non-technical users.
- The system should provide clear navigation for audio uploads and results.
- The visualization outputs shall help users easily understand the model's predictions for the analysed speech.

== Software Requirements

The software requirements are essential for developing, training, validating, and deploying deep learning models aimed at detecting stuttering events from speech audio.
The system must support various tools and libraries that enable efficient model development, audio processing, and performance evaluation.

- *Operating System:* the proposed system is developed and executed on the Linux and Windows platforms. It is fully compatible with Windows 10 and above and common Linux distributions, offering stability, wide software support, and seamless integration with deep learning frameworks and GPU drivers. This environment ensures smooth performance, easy setup, and reliable execution of the proposed models, with both the web application and the PySide6 desktop application running on either platform and training and inference supported on CPU or NVIDIA GPU.
- *Deep Learning Framework:* the proposed system is built using PyTorch as the primary deep learning framework for building and training the machine learning models. PyTorch provides the required computational efficiency, GPU support through features such as torch.compile, mixed precision, and TensorFloat-32 (TF32), and scalability for handling complex audio classification tasks effectively. The Hugging Face Transformers library supplies the pretrained speech models, Wav2Vec 2.0 for feature extraction and Whisper for timestamped transcription.
- *Libraries:* a range of Python libraries is required to support different stages of the project:
  - librosa: Used for audio preprocessing and manipulation, including loading, resampling, mel-spectrogram computation, and silence trimming.
  - NumPy: Provides support for numerical operations and array handling.
  - Pandas: Essential for managing and analysing structured data such as labels or metadata.
  - Scikit-learn: Used for data splitting, model evaluation metrics, and traditional machine learning algorithms.

== Hardware Requirements

The hardware requirements outline the minimum and recommended specifications needed to efficiently run the deep learning models used in this project.
Proper hardware ensures smooth data processing, model training, and inference performance, which is critical when working with speech audio and deep learning pipelines.

- *Processor (CPU):* a multi-core processor such as Intel Core i5 (8th generation or newer) or AMD Ryzen 5 and above is required. These processors provide adequate computational power for audio preprocessing and managing training workflows.
- *Storage (Hard Disk):* at least 10 GB of free disk space is required to store datasets, trained model checkpoints, logs, and temporary files. SSDs are recommended for faster read/write speeds and improved data handling.
- *Memory (RAM):* a minimum of 8 GB RAM is necessary to run the training scripts and handle datasets. However, 16 GB or more is recommended for better performance, especially during the training and validation phases when data loading and augmentation are involved.
- *Graphics Processing Unit (GPU):* for faster model training and efficient computation, an NVIDIA GPU with CUDA support is highly recommended. GPUs such as the NVIDIA RTX 2060 or higher (e.g., RTX 3060, RTX 3080) can greatly reduce training time and handle larger datasets and complex neural network architectures effectively. Proper installation of CUDA and cuDNN is required to ensure full compatibility with PyTorch.
- *Audio Input Device:* a functional microphone is required for recording speech input within the application.

#pagebreak()