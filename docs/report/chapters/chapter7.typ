#import "../lib.typ": *
#import "../meta.typ": *

// --- Chapter 7: Conclusion and Future Work ---
#chapter_heading([CONCLUSION AND FUTURE WORK])

== INTRODUCTION
This chapter wraps up the #project_title. It restates what was built, what the evaluation showed, where the system falls short, and what could come next.

== SUMMARY OF WORK
The goal was to build an automated system for detecting, classifying, and localizing speech dysfluencies in recorded speech. The system is meant to help speech-language pathologists and people in therapy by giving them an objective, consistent, and accessible assessment tool.

What was built:

- A preprocessing pipeline that converts input speech to 16 kHz mono with DC offset removal, peak normalization, silence trimming, and fixed-length padding.
- A multitask classification pipeline using Wav2Vec 2.0 embeddings with five binary heads, detecting prolongation, block, sound repetition, word repetition, and interjection.
- A localization pipeline using CNN spectrogram analysis and Wav2Vec2 frame-level features to find where dysfluencies occur in the audio.
- A transcription pipeline using Whisper ASR for English, Kannada, and Hindi, with word-level timestamps.
- A combiner that merges localization regions with classifier saliency to assign dysfluency types to time segments.
- A severity module that maps the stutter index to clinical severity levels (Fluent, Mild, Moderate, Severe).
- PDF report generation using Typst.
- A model registry for consistent checkpoint management across web and desktop apps.
- A web interface (React 19) and a desktop app (PySide6), both using the same ML backend.

== KEY FINDINGS
The multitask Wav2Vec2 classifier scored 0.490 macro F1 at the default threshold, rising to 0.533 after per-class threshold tuning. Interjection was the strongest class (F1=0.741, AUROC=0.932). Filler words like "um" and "uh" have acoustic patterns that the pretrained model picks up well. Block was the weakest (F1=0.243, improving to 0.401 with tuning). Silent pauses and hesitations are hard to tell apart from normal speech pauses.

The architecture comparison showed that Wav2Vec2 models do best on in-distribution data, but lighter CNN models (0.27M to 0.54M parameters vs. 94M to 97M) generalize better on the Boli cross-corpus set. CNN-LSTM hit 0.521 on Boli, the best cross-corpus result. This matters for deployment where compute is limited or multiple languages are needed.

The Wav2Vec2 localizer had high frame-level precision (0.676) and mean IoU (0.751), so the regions it finds overlap well with ground truth. The recall was low (0.065), meaning it misses most events. For clinical use, low false-alarm rate matters more than catching every event, so the trade-off is reasonable.

The training process turned up several practical lessons: BCEWithLogitsLoss with a single logit causes a zero-gradient collapse; gradient clipping can silently suppress learning; waveform and spectrogram augmentation must be handled separately; and freezing the backbone for 3 epochs beats longer freeze durations.

The project also produced an open-source, modular framework that combines classification, localization, transcription, and reporting in one pipeline.

== LIMITATIONS
The system has several shortcomings:

- Classification accuracy is still low for some classes. A macro F1 of 0.533 means the system makes a lot of wrong predictions, especially for block and word repetition. It is not reliable enough to serve as a standalone diagnostic tool.
- The localizer's recall (0.065) means most dysfluency events go undetected. The detected regions are accurate, but coverage is poor.
- Cross-corpus generalization is weak. The multitask model's F1 drops from 0.522 on the in-distribution test set to 0.160 on Boli. The model struggles with unseen speakers and languages.
- The training data is mostly podcast recordings (SEP-28K), which may not represent clinical or spontaneous speech.
- Crowdsourced annotations in some datasets introduce label noise.
- The system analyzes audio only. Stuttering also shows up in facial tension, eye blinking, and body movement, none of which are captured.
- The system does not support continuous real-time streaming analysis.

== FUTURE WORK
Ways to improve the system:

- Better classification: try larger pretrained models (Wav2Vec2-large, HuBERT-large), combine audio with transcript features, or use contrastive learning.
- Better localization: use curriculum learning, harder negative mining, and boundary-aware loss functions to improve recall without losing precision.
- More languages: use multilingual models like MMS or XLS-R to add languages beyond English, Kannada, and Hindi.
- Multimodal input: add facial expression and lip movement features through transformer-based fusion. Research like MMSD-Net shows that visual cues help with blocks and prolongations.
- Real-time streaming: process audio in overlapping windows for continuous monitoring during therapy sessions.
- Therapy tracking: analyze dysfluency patterns over time so therapists can measure treatment progress.
- Clinical validation: run trials with speech-language pathologists to test diagnostic accuracy and usability in real clinical settings.
- Cloud deployment: optimize for cloud inference so the system is accessible from low-resource devices through a web browser.
- Synthetic data: use LLMs and TTS systems to generate dysfluent speech for training, addressing the data scarcity problem.
- Explainability: add attention visualization and saliency maps so clinicians can see why the model made a particular prediction.

== CHAPTER SUMMARY
This chapter summarized the project's work, findings, and limitations. The system shows that Wav2Vec 2.0 pretrained representations work for automated stutter detection and localization. Classification accuracy, localization coverage, and cross-corpus generalization all need improvement, but the modular design gives a base to build on. The directions above would make it more useful for clinicians and people who stutter.

#pagebreak()
