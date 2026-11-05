#import "../lib.typ": *
#import "../meta.typ": *

// --- Chapter 4: System Implementation ---
#chapter_heading[SYSTEM IMPLEMENTATION]

== INTRODUCTION

System implementation is a broad and demanding task: it builds and maintains a reliable software architecture that can handle many user interactions, data processing tasks, and model inference operations.
Careful design and coding practices keep the application scalable, reliable, and secure, so it can still handle changing workloads and protect sensitive patient data at every stage.
Iterative testing and debugging are a central part of the implementation process, so any issues or bugs that arise can be found and fixed before release.
This repeated cycle gives users a smooth experience, with few disruptions or errors in their daily use.
After deployment, regular monitoring and maintenance check system performance and respond to new requirements as they arise.
This steady upkeep keeps the system tuned and still responsive to user needs and expectations as conditions change over time, which supports lasting success and user satisfaction.

== ALGORITHM
The inference process follows a fixed sequence of steps that turns raw speech input into a complete, annotated analysis result.
Each step has a defined input and output, which limits the amount of hidden state passed between services.
This structure is important for repeatable analysis because the same conversion, normalization, model execution, and reporting stages are used for both recorded and uploaded speech.

- #keep[Step 1, Input Acquisition: the system acquires speech input either through real-time recording or by uploading an audio file.]
- #keep[Step 2, Audio Conversion: the input audio is converted into a 16 kHz mono WAV format using FFmpeg to maintain consistency across all inputs.]
- #keep[Step 3, Preprocessing: the audio is cleaned by removing the DC offset, applying peak normalization to a target amplitude, and trimming silent segments.]
- #keep[Step 4, Length Normalization: the processed audio is adjusted to a fixed length of 48,000 samples (three seconds at 16 kHz) by padding or truncating as required.]
- #keep[
  Step 5, Parallel Processing: the system processes the audio simultaneously through three independent branches:
  #enum[
    Classification: Wav2Vec 2.0 embeddings are generated and passed through five binary classifiers, one per dysfluency type, and the outputs are aggregated into a multi-label result with a probability score for each dysfluency type.
  ][
    Transcription: the Whisper model generates a timestamped transcript of the speech in English, Kannada, or Hindi.
  ][
    Localization: the Wav2Vec2 frame-level localizer identifies the frame-level regions within the audio where dysfluencies occur.
  ]
]
- #keep[Step 6, Alignment: the detected dysfluency regions are mapped to the corresponding words or syllables in the transcript using CTC-based time alignment.]
- #keep[Step 7, Combination and Severity: the combiner labels each localized region with per-class saliency scores from the classifier and fuses the classification and localization views, and the severity module computes a stutter index from the ratio of dysfluent speech duration to total speech duration.]
- #keep[Step 8, Visualization and Report Generation: the system displays the results through waveform overlays, spectrograms, transcripts, and confidence scores, and generates a detailed clinical-style report.]

== IMPLEMENTATION REQUIREMENTS

This section lists the implementation requirements for detecting and localizing stuttering with a hybrid deep learning model that combines Wav2Vec 2.0 feature extraction, five independent Wav2Vec 2.0 binary classifiers, a Wav2Vec2 frame-level localizer, and Whisper automatic speech recognition.
The main goal is to build a reliable, efficient system that finds speech dysfluencies and localizes them within the audio.
The hybrid model uses the strengths of both architectures: Wav2Vec 2.0 supplies self-supervised, contextual speech representations that capture fine-grained phonetic and prosodic patterns, while Whisper provides sequence-level transcription that helps map localized dysfluency regions to specific words and syllables.

The implementation relies on the Python ecosystem, which provides an interactive environment for model development, experimentation, and analysis.
Python is the main language used for backend development because it is versatile and also supports a very wide range of scientific and deep learning libraries such as PyTorch, the Hugging Face Transformers library, and NumPy, which help with model construction, training, and evaluation.
For the frontend, React 19 and PySide6 build a simple web interface and desktop application where users can record or upload speech audio and view classification and localization results.
A central model registry manages all trained models. It is configured through a `registry.json` file that maps each task to its checkpoint and per-class thresholds, so checkpoints can be updated or replaced without changing application code.
Python also handles matrix manipulation, data visualization, algorithm implementation, and smooth integration with other programming environments, which makes it a good platform for developing and testing the hybrid model for stuttering classification and localization.

=== Train the Model

Training is a major step in developing the system for stutter classification and localization.
The speech audio dataset is first collected from Project Boli, SEP-28K, and UCLASS, then preprocessed through conversion to 16 kHz mono WAV, DC offset removal, peak normalization, silence trimming, and fixed-length padding. Data augmentation also improves audio quality and reduces overfitting risk.
The Wav2Vec 2.0 model is then trained to extract high-level contextual speech features, using its self-supervised pretraining to learn strong representations.
To raise classification accuracy further, the hybrid model combines five independent Wav2Vec 2.0 binary classifiers with a Wav2Vec2 frame-level localizer.
The localizer captures fine-grained temporal features at the frame level, adding detail beyond segment-level extraction of the classifiers.
CTC-based time alignment then combines the outputs of both models into one feature representation.
Training uses the AdamW optimizer to adjust learning rates adaptively and keep convergence stable.
It also uses mini-batches of audio, and backpropagation updates the weights from the computed loss.
Validation runs periodically to monitor accuracy and prevent overfitting.
After training, the system saves optimized parameters and weights, so the models can predict on new speech audio without retraining.

=== Test the Model

Testing comes after training to check how well the proposed model detects stutter events in unseen speech audio.
When a user records or uploads a sample through the web interface or desktop application, it goes through the very same preprocessing as training: conversion to 16 kHz mono WAV, DC offset removal, peak normalization, silence trimming, and fixed-length padding. This keeps input uniform for accurate analysis.
The trained Wav2Vec 2.0 model then extracts high-level contextual speech features.
For higher accuracy, the audio also passes through the hybrid model, which combines five independent Wav2Vec 2.0 binary classifiers with a Wav2Vec2 frame-level localizer and captures both segment-level dysfluency patterns and fine temporal detail.
CTC-based time alignment combines the features from both models into one representation. The final aggregation layer then produces a classification result that names the dysfluency types present (blocks, prolongations, sound repetitions, word repetitions, or interjections) and the exact regions involved.
The interface shows the outcome with waveform overlays, spectrograms, a timestamped transcript, and confidence scores.
The system also compares the model predictions with ground truth labels to measure accuracy and generalization.
This phase shows that the hybrid model performs well on new and unseen samples, which points to reliable stutter classification and localization in real clinical use.

=== Model Evaluation
Model evaluation is essential for assessing the overall performance and reliability of the proposed system.
Once the models are trained and tested, several quantitative metrics measure their classification performance:

- #keep[Accuracy: measures the proportion of correctly predicted samples.]
- #keep[Precision: the share of predicted positive instances that are actually positive.]
- #keep[Recall (Sensitivity): measures how effectively the model identifies all relevant positive samples.]
- #keep[F1-Score: provides a harmonic mean of precision and recall to balance both metrics.]

The five Wav2Vec 2.0 binary classifiers, one per dysfluency type, are further measured with AUROC (Area Under the Receiver Operating Characteristic Curve), AUPRC (Area Under the Precision-Recall Curve), and specificity, summarized per class and as a macro average across the five dysfluency classes.
The Wav2Vec2 frame-level localizer is measured with frame-level precision, recall, and F1, detection accuracy, mean Intersection over Union (IoU) between predicted and ground-truth regions, and the false alarm rate per minute.

A confusion matrix for each of the five binary classifiers shows the distribution of predictions, including true positives, false positives, true negatives, and false negatives.
This evaluation shows where each class is strong or weak; interjections, for example, are detected most reliably, while blocks are the hardest.

All reported results use the held-out test set of 3,715 clips, which is never touched during training or threshold selection, together with a cross-corpus Boli set that checks how well the models transfer to unseen speakers.

== USER INTERFACE
The web-based interface connects the user to the proposed stutter detection and localization system.
Users record speech or upload audio files for analysis.
Once audio is provided, the backend system carries out the preprocessing steps, followed by feature extraction, classification, transcription, and localization using the trained models.
Initially, the Wav2Vec 2.0 model is used to extract high-level contextual speech features, and to achieve higher accuracy, the hybrid pipeline combining five independent Wav2Vec 2.0 binary classifiers with the Wav2Vec2 frame-level localizer is employed.
Within a few seconds, the system presents the predicted result on the screen for each of the five dysfluency types, along with confidence scores, the localized regions, and the timestamped transcript.
The tool is designed for clinicians, speech-language pathologists, researchers, and individuals who stutter.

The system provides two interfaces to the same analysis pipeline: a web application and a desktop application.
The web frontend is built with React 19, TypeScript, and Tailwind CSS, with a sidebar layout covering the input page, results page, history, and reading passages.
The input page supports audio file upload (WAV, MP3, FLAC, M4A, OGG, WMA) and direct microphone recording through the MediaRecorder API, with client-side validation before the file is sent to the backend as multipart form data.
The results page shows classification results with confidence scores, an interactive waveform with color-coded dysfluency region overlays, a timestamped transcript, and a button to generate and download a PDF report.
The backend runs on FastAPI served through Uvicorn, exposing the `/api/classify`, `/api/localize`, `/api/analyze`, and `/api/report` endpoints.
Assessment history stays in the browser: audio files go into IndexedDB for offline access and analysis metadata into localStorage.

The desktop application is built with PySide6 (Qt for Python) for a native cross-platform interface.
It includes modules for audio recording and playback (sounddevice), file handling (soundfile), and model inference, calling the shared model package directly with no separate server, so it works fully offline.
PDF report generation uses Typst with pypdfium2 for viewing.

Both interfaces share the same model package, preprocessing conventions, and result schema, so a recording analysed in the browser and the same recording analysed on the desktop produce identical predictions and reports.
The web client keeps long requests responsive by streaming progress updates and blocking any duplicate submissions while an analysis runs, while the desktop client runs inference in a worker thread so the window still stays interactive during transcription and localization.
Theme switching, keyboard navigation, and consistent layout across window sizes are handled at the component level, which keeps the interface usable for clinicians who move between machines and screen sizes.
The client checks uploaded audio before it ever reaches the backend, and the results view keeps the waveform, transcript, and per-class confidence table in sync so that selecting a timestamp jumps to the exact matching part of the recording.
Report generation is kept separate from analysis: the stored result payload is enough to rebuild the PDF, so a report can still be regenerated with a different layout or header without repeating the expensive model inference.
This shared design also makes maintenance easier, because a change to the result schema or the report template reaches both interfaces from one single source instead of being implemented twice.

#pagebreak()