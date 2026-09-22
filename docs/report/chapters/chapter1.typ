#import "../lib.typ": *

// --- Chapter 1: Introduction ---
#chapter_heading[INTRODUCTION]

== INTRODUCTION TO THE PROJECT
Speech fluency lets people express thoughts and emotions clearly.
Stuttering, a common speech disorder, interrupts that flow with involuntary repetitions, prolongations, or blocks in speech production.
It affects people of all ages and can hurt their confidence, their ability to communicate, and their social interactions.
Identifying stuttering patterns early helps with therapy and management.

Traditionally, detection and assessment rely on manual observation by speech-language pathologists.
The results can be subjective, time-consuming, and inconsistent from one expert to another, and the work takes a physical toll on the pathologist.
That dependence on human evaluation makes large-scale assessment and continuous monitoring difficult.
With recent advances in deep learning and speech signal processing, automated stutter detection has become a reliable, scalable alternative.
This project develops a system that automatically detects, classifies, and analyzes stuttering types (prolongation, block, sound repetition, word repetition, and interjection) using deep learning.

Input audio is preprocessed and converted into embeddings, then analyzed with Wav2Vec 2.0, which captures both local acoustic patterns and long-range temporal dependencies that separate normal from dysfluent speech.
Alongside classification, the same audio passes through a localization module that returns the timestamps of the stuttering events, and the detected stutter timestamps are hard-aligned with the transcript timestamps so that only the affected speech segments are identified.
The finished model is integrated into a simple interface where users record or upload speech samples and receive an analysis with timestamped detection results and classification output.

The project aims to give speech-language pathologists and people who stutter an accurate, objective, and accessible diagnostic tool.
The system also contributes to the growing use of artificial intelligence in healthcare, supporting diagnosis, therapy, and an improved quality of life for people who stutter.

== PROPOSED METHOD
The proposed method aims to develop an efficient deep learning-based system for the early detection and classification of speech dysfluencies from audio recordings.
The process begins with comprehensive audio preprocessing that includes noise reduction, normalization, and enhancement of acoustic clarity across all samples.
Segmentation isolates the meaningful regions of speech, eliminating irrelevant silence and background, and data augmentation such as noise injection, time and pitch shifting, and scaling simulates variations in speaking rate and recording conditions.
The system accurately categorizes the input into five dysfluency types: prolongation, block, sound repetition, word repetition, and interjection, and localizes the segments where these occur, supporting early diagnosis and effective treatment planning.
This automated approach reduces manual workload, minimizes observer bias, and enhances diagnostic reliability, making it a valuable tool in computer-aided speech assessment.

== LITERATURE SURVEY
#include "/contents/literature_survey.typ"

#pagebreak()