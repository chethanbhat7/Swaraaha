#import "../lib.typ": *
#import "../meta.typ": *

// --- Chapter 7: Conclusion and Scope for Future Enhancements ---
#chapter_heading([CONCLUSION AND SCOPE FOR FUTURE #linebreak()ENHANCEMENTS])

== CONCLUSION

The stuttering classification and localization system gives speech-language pathologists an accurate and objective way to assess speech dysfluencies from audio.
Combining Wav2Vec 2.0 feature extraction with Wav2Vec2 frame-level localization and Whisper transcription gives it better features and better classification.
The pipeline handles data preprocessing, classification of five dysfluency types with five independent Wav2Vec 2.0 binary classifiers, event localization, transcription in English, Kannada, and Hindi, severity estimation, and PDF report generation.

The shipped default of independent classifiers reached a macro F1 score of 0.573, while the multitask shared-backbone variant scored 0.542 at the default threshold and 0.556 after per-class threshold tuning, with interjection the strongest class (F1 = 0.776, AUROC = 0.935) and block the weakest.
The Wav2Vec2 localizer produced balanced frame-level results (precision 0.805, recall 0.738, F1 0.770) with a mean IoU of 0.779.
In total, the system reaches a detection accuracy of 0.586 for localizing dysfluency events, alongside a frame-level F1 of 0.770 and a macro F1 of up to 0.573 for classification.

The system gives clinicians, speech-language pathologists, and people who stutter consistent predictions. These cut the manual analysis burden, limit diagnostic variability, and support timely assessment.
Classification accuracy for sparse classes such as block and word repetition (macro F1 0.556) is not yet sufficient for a standalone diagnostic tool, and cross-corpus generalization weakens outside podcast-style data (multitask F1 drops from 0.556 on the test set to 0.238 on the Boli corpus).
Despite these limits, the modular architecture combines classification, localization, transcription, and reporting in one pipeline and gives a practical base for real-world speech assessment.
The main contribution is the integration of these stages into one single usable workflow.
A clinician can move from an uploaded recording to class-level probabilities, localized regions, aligned words, and a saved report without manually joining outputs from separate tools.
The modular design also leaves room for later improvements to individual models without requiring the interface or reporting layer to be redesigned.

== SCOPE FOR FUTURE ENHANCEMENTS

Future work on the stuttering detection system could focus on adding useful new functions and on improving detection accuracy.
Larger and more diverse datasets from several different clinical sources would help the model generalize better across different speakers, stuttering styles, and languages.
Additional new deep learning architectures, such as larger pretrained models like Wav2Vec2-large and HuBERT-large, and better localization techniques such as curriculum learning, harder negative mining, and boundary-aware loss functions could further improve both the detection and the localization of dysfluency events.
A simple web and mobile interface for clinicians could also make the system easier to use in real-time therapy settings.
The system could also add explainable AI features, such as attention visualization and saliency maps, to show visual insights into how predictions are made and build greater trust and transparency in clinical decision-making.
Multilingual models like MMS and XLS-R could greatly extend coverage beyond English, Kannada, and Hindi, while synthetic dysfluent speech from LLMs and text-to-speech systems could help ease the data scarcity problem during training.
Finally, linking the system to clinic databases and electronic health records could support more routine clinical use, helping doctors and speech-language pathologists with early assessment and personalised treatment planning for stuttering.
Any such deployment should keep the current focus on privacy, clear confidence values, and reviewable evidence.
Future versions can improve coverage and accuracy and should keep showing how each result relates to the waveform, transcript, and detected dysfluency type.

#pagebreak()