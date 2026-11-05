#import "../lib.typ": *

// --- Chapter 1: Introduction ---
#chapter_heading[INTRODUCTION]

== INTRODUCTION TO THE PROJECT
Speech fluency lets a person put thoughts and feelings into words.
Stuttering breaks that flow through involuntary repetitions, prolongations, or blocks.
It affects people at every age and can wear down confidence, which makes conversation and social contact more difficult.
Spotting the patterns early helps therapy and day-to-day management.

Manual detection and assessment still depend heavily on a speech-language pathologist who listens to recordings and marks what they hear.
Two clinicians can disagree about the same sample, the work is slow, and repeating it tires them.
That reliance on human listening also makes large-scale assessment and continuous monitoring hard to organise.
Deep learning and speech signal processing now offer a scalable alternative.
This project builds a system that detects, classifies, and analyzes five stuttering types: prolongations, blocks, sound repetitions, word repetitions, and interjections.

The pipeline preprocesses input audio, turns it into embeddings, and analyzes it with Wav2Vec 2.0, which captures fine acoustic detail and dependencies across longer stretches of speech.
A second module localizes the stuttering events and returns their timestamps.
Those timestamps line up with the transcript, so the system flags only the affected segments.
The finished model sits behind a simple interface where a user records or uploads a sample and gets timestamped detections, class labels, and confidence scores.

The aim is an accurate, objective, and accessible diagnostic tool for speech-language pathologists and people who stutter.
The work also supports the use of artificial intelligence in healthcare, where the same analysis can directly aid diagnosis, therapy, and a better quality of life.

== PROPOSED METHOD
The proposed method is a deep learning system that detects and classifies speech dysfluencies early, using audio recordings as input.
Preprocessing removes noise, balances loudness, and sharpens acoustic detail across all samples.
Segmentation isolates the regions that carry speech so silence and background noise drop out.
Augmentation such as noise injection, time and pitch shifting, and amplitude scaling exposes the model to different speaking rates and recording conditions.
The system then puts each input into one of five dysfluency types and marks where the event occurred, which supports early diagnosis and treatment planning.
Automating this work cuts manual effort, reduces observer bias, and makes the assessment more reliable.

== LITERATURE SURVEY
#include "/contents/literature_survey.typ"

== PROBLEM STATEMENT

Stuttering assessment in clinical practice still relies on a speech-language pathologist who listens to recordings and marks dysfluencies by hand. The process is slow, subjective, and hard to reproduce, and disagreement between raters makes a patient's progress difficult to track. Most automated tools stop at binary detection: they report that a recording contains stuttering but not which type occurred or exactly where in the audio it occurred, which limits their use for therapy planning.

This project closes that gap with a single pipeline that takes an unconstrained speech recording and produces a structured analysis: whether stuttering is present, which of the five dysfluency types are involved, and the exact time regions where each event occurs. The system must also work without a large annotated corpus from the target speaker and must run fast enough for interactive assessment instead of offline batch processing.

== OBJECTIVES

The project is built around four objectives, and each one supplies one stage of the final pipeline.

+ #block[*Speech preprocessing:* clean and standardise the input audio so every later model gets consistent data. This covers resampling to a fixed rate, DC offset removal, peak normalization, silence trimming, fixed-length padding, and augmentation such as noise injection and time or pitch shifting, which together limit the effect of varying recording conditions and speaker loudness.]

+ #block[*Stutter detection:* decide whether a sample contains dysfluent speech. The binary output controls the rest of the analysis, so the model is trained to stay reliable across different speakers, languages, and background noise, not merely on clean studio recordings alone.]

+ #block[*Stutter classification:* identify which dysfluency types are present among prolongation, block, sound repetition, word repetition, and interjection. Because these events often co-occur, the system decides each type independently and gives it a separate confidence score instead of forcing one mutually exclusive label.]

+ #block[*Stutter localization:* find the time regions where detected events occur and align them with the corresponding words in the transcript. Clinicians get usable timestamps, so they can go straight to the affected segment and compare the detected events with what was said.]

The objectives are ordered so each stage builds on the one before it. Preprocessing yields a normalized waveform and spectrogram for detection and classification. Detection narrows the analysis to samples that contain dysfluencies. Type-level scores come from classification, and localization maps events back onto the time axis and the transcript.
Together these stages turn a recording into evidence that can be inspected at clip level and at event level. The classification scores summarise what was detected, while the localized intervals and aligned transcript show where it happened, which makes the result easier to review.



#pagebreak()
