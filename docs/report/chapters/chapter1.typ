#import "../lib.typ": *

// --- Chapter 1: Introduction ---
#chapter_heading[INTRODUCTION]

== INTRODUCTION //TO THE PROJECT
Speech fluency is an essential aspect of human communication, allowing individuals to express thoughts and emotions clearly.
However, many people face interruptions in their speech flow due to a disorder known as stuttering.
It is a common speech disorder characterized by involuntary repetitions, prolongations, or blocks during speech production.
This condition affects individuals across all age groups and can significantly impact their confidence, communication ability, and social interactions.
Early identification of stuttering patterns is crucial for effective therapy and management.

Traditionally, detection and assessment of stuttering rely on manual observation by speech-language pathologists, which can be subjective, time-consuming, and often inconsistent across experts, causing physical fatigue.
Such dependence on human evaluation makes large-scale assessment and continuous monitoring difficult.
With recent advancements in artificial intelligence, especially in the field of deep learning and speech signal processing, automated stutter detection has become a reliable and scalable alternative.
The project aims to develop an intelligent system that can automatically detect, classify, and analyze types of stuttering in speech, including repetitions, prolongations, blocks, and fillers, using deep learning techniques.

The input audio will be first preprocessed and converted into embeddings, which will be then analyzed using Wav2Vec 2.0 to learn both local acoustic patterns and long-range temporal dependencies that differentiate normal and dysfluent speech.
Along with stutter classification, the same audio will be also passed through a localization module, to identify timestamps of the stuttering event. // and the detected stutter timestamps will be hard-aligned with the transcript timestamps to identify the exact word, syllable, or speech segment where the dysfluency occurs.
The final model will be integrated into a user-friendly interface that allows users to record or upload speech samples and receive analysis, timestamped detection results, and classification output.

Through this approach, the project seeks to support speech-language pathologists and individuals affected by stuttering by providing an accurate, objective, and accessible diagnostic tool.
It also contributes to the broader vision of integrating artificial intelligence into healthcare to enhance diagnosis, therapy, and quality of life.

== LITERATURE REVIEW
#include "/contents/literature_survey.typ"

== EXISTING SYSTEM
Existing stutter detection systems primarily rely on manual identification or basic signal processing methods.
Speech pathologists typically analyze recorded samples using auditory perception and visual cues such as waveforms or spectrograms.
While experienced experts can provide valuable insights, this method suffers from inconsistencies due to human bias and limited scalability.

Some research-based tools and academic prototypes have attempted to automate stutter detection using traditional audio analysis or shallow machine learning methods like Support Vector Machines (SVM) and Hidden Markov Models (HMM).
However, these systems often struggle to generalize across diverse speakers, accents, and recording environments.
Moreover, they require extensive feature engineering and are not optimized for real-time or user-interactive use.
Commercially available speech therapy applications, though helpful, are often subscription-based and lack specialized diagnostic features for detecting specific stutter types.
Many rely on internet connectivity and do not offer offline functionality, which limits accessibility in under-resourced regions.

These limitations highlight the need for an advanced, reliable, and accessible stutter detection system.
A deep learning–based approach using Wav2Vec2 embeddings, and audio spectrogram analysis can automatically learn complex acoustic patterns and temporal dependencies that distinguish normal speech from dysfluent speech, enabling accurate classification and localization of various stuttering types.

== PROBLEM STATEMENT // PROPOSED SYSTEM
Stuttering assessment is largely manual, subjective, and time-consuming, leading to pathologist fatigue and limited scalability.
Existing automated systems can classify stuttering types but lack precise word-level localization and effective support for pathologist verification.
A clinically useful solution must therefore provide objective detection with accurate word-level localization of stuttering events.
/*
The proposed system introduces an intelligent and automated framework for detecting and classifying stuttering patterns from speech recordings.
It will leverages deep learning, specifically a multi-model binary classification architecture, to identify various dysfluency types effectively.
The workflow will begin with capturing or uploading speech recordings.
The audio will be then pre-processed and converted into embeddings, which will be analyzed using a CNN and Transformer-based model to capture both local acoustic patterns and long-range temporal dependencies.
These learned representations will be used to detect five independent stuttering behaviors: repetitions, prolongations, blocks, interjections, and word-level dysfluencies.

Along with stutter detection, the same audio will also passed through a transcription model to generate timestamped speech output.
The detected stutter timestamps will be then hard-aligned with the transcript timestamps to identify the exact word, syllable, or speech segment associated with the dysfluency.
Each model will output a probability score indicating the presence of a specific stutter class, and the results are compiled into a final classification report.
The trained model will be integrated into a desktop-based PyQt5 interface, allowing users to record speech, upload audio files, and visualize results in real time.
The interface will also provides playback controls and waveform displays for better interpretability.
By combining advanced speech signal processing, deep learning-based classification, and timestamp alignment with transcription, this system aims to deliver a complete end-to-end solution that will not only detect stuttering automatically but also support clinicians and individuals in monitoring progress and planning effective therapy interventions.
*/

== OBJECTIVES OF THE PROJECT
- *Preprocessing Pipeline:* To preprocess and standardize speech audio by reducing noise, normalizing volume, and preparing it for accurate stutter detection.
- *Classification Model:* To develop an automated stutter detection system that accurately identifies and classifies five types of stuttering: Prolongation, Block, Sound Repetition, Word Repetition, and Interjection.
- *Localization Model:* To implement a temporal localization module capable of identifying the exact timestamps of stuttering events, the affected region, and the duration within a speech sample.

/*
The primary goal of this project is to create an automated and accurate system for detecting and classifying stuttering in speech recordings using deep learning techniques.  The key objectives are:
- *Speech Representation Pipeline*: To develop a robust pipeline using the Wave2Vec framework to extract deep, context-aware embeddings from raw audio without manual feature engineering.
- *Hybrid CNN + Transformer Architecture*: To design and train a model that leverages both local acoustic patterns and global temporal context for accurate stutter detection and classification.
- *Specialized Classifiers For Each Stutter Class*: To build individual models for Prolongation, Block, Sound Repetition, Word Repetition, and Interjection to maximize class-wise accuracy.
- *Temporal Localization Module*: To identify the exact timestamps of stuttering events, the affected region, and the duration and frequency of occurrences within a speech sample.
- *Visual Annotation System*: To mark stutter-indicative regions on the waveform/spectrogram for an interpretable and clinician-friendly analysis interface.

== SCOPE OF THE PROJECT
This project focuses on developing a deep learning-based speech analysis system that etects and classifies stuttering behaviors from recorded or live speech input.
The system caters to both research and clinical applications by providing accurate, objective, and real-time dysfluency detection.

The project scope includes:
- Designing and training a hybrid CNN + Transformer model to classify speech dysfluencies into five categories: prolongation, block, sound repetition, word repetition, and interjection.
- Implementing a speech representation pipeline using Wav2Vec embeddings to extract deep acoustic features from raw audio without manual feature engineering.
- Developing a temporal localization and hard-alignment pipeline to map detected stutter timestamps with transcription timestamps and identify the exact word, syllable, or speech segment affected.
- Creating a cross-platform, user-friendly GUI using PyQt5 that enables users to record, upload, analyze, and visualize speech data interactively.
- Generating analytical outputs including probability scores, timestamped detections, waveform/spectrogram annotations, and diagnostic summaries.
- Supporting both real-time audio capture and offline file-based analysis to accommodate users with varying technical resources.
- Providing an intelligent and accessible tool that supports automation of speech pathology assessments and assists in early identification, monitoring, and therapy tracking for individuals with stuttering.

== ORGANIZATION OF THE REPORT
This report is organized into two chapters, each presenting a structured explanation of the project’s goal and the research carried out to support the future implementation of the project.
- *Chapter 1* introduces the project, its motivation, objectives, and overall structure. It outlines the problem of stuttering, the need for automation, and the proposed system’s benefits.
- *Chapter 2* presents the Literature Survey, reviewing previous work, existing speech analysis tools, and limitations in current methods.
- *Chapter 3* explains the System Requirements, including both hardware and software specifications, as well as functional and non-functional requirements.
- *Chapter 4* focuses on System Design, providing architectural diagrams, data flow representations, and a detailed explanation of the CNN-based classification process.
*/

#pagebreak()
