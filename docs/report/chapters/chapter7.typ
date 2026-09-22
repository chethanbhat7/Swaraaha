#import "../lib.typ": *
#import "../meta.typ": *

// --- Chapter 7: Conclusion and Scope for Future Enhancements ---
#chapter_heading([CONCLUSION AND SCOPE FOR FUTURE #linebreak()ENHANCEMENTS])

== CONCLUSION

In conclusion, the proposed stuttering classification and localization system demonstrates strong potential in supporting accurate and objective assessment of speech dysfluencies through automated analysis of speech audio.
By integrating Wav2Vec 2.0 feature extraction with Wav2Vec2 frame-level localization and Whisper transcription, the framework significantly strengthens feature extraction and improves classification performance.
The systematic pipeline ranges from data preprocessing and classification of five dysfluency types using five independent Wav2Vec 2.0 binary classifiers, to event localization, transcription across English, Kannada, and Hindi, severity estimation, and PDF report generation.

The shipped default of independent classifiers reached a macro F1 score of 0.573, while the multitask shared-backbone variant scored 0.542 at the default threshold and 0.556 after per-class threshold tuning, with interjection the strongest class (F1 = 0.776, AUROC = 0.935) and block the weakest.
The Wav2Vec2 localizer delivered balanced frame-level results (precision 0.805, recall 0.738, F1 0.770) with a mean IoU of 0.779, confirming the effectiveness of the approach.
Overall, the complete system achieves a detection accuracy of 0.586 for localizing dysfluency events, alongside a frame-level F1 of 0.770 and a macro F1 of up to 0.573 for classification.

Through its systematic pipeline, the system delivers consistent and reliable predictions that can help clinicians, speech-language pathologists, and people who stutter reduce the burden of manual analysis, minimize diagnostic variability, and make timely assessment decisions.
Classification accuracy for sparse classes such as block and word repetition (macro F1 0.556) is not yet sufficient for a standalone diagnostic tool, and cross-corpus generalization weakens outside podcast-style data (multitask F1 drops from 0.556 on the test set to 0.238 on the Boli corpus).
Despite these limitations, the modular architecture combines classification, localization, transcription, and reporting in one pipeline, providing a practical foundation for real-world speech assessment.

== SCOPE FOR FUTURE ENHANCEMENTS

Looking ahead, future enhancements of the stuttering detection system could focus on expanding its functionality and improving detection accuracy.
Incorporating larger and more diverse datasets from multiple clinical sources would help the model generalize better across different speakers, stuttering styles, and languages.
The integration of additional deep learning architectures, such as larger pretrained models like Wav2Vec2-large and HuBERT-large, and advanced localization techniques such as curriculum learning, harder negative mining, and boundary-aware loss functions, could further refine the detection and localization of dysfluency events.
Moreover, developing a user-friendly web and mobile interface for clinicians could make the system more accessible in real-time therapy settings.
The system could also be enhanced with explainable AI features, such as attention visualization and saliency maps, to provide visual insights into how predictions are made, increasing trust and transparency in clinical decision-making.
Multilingual models like MMS and XLS-R could extend language coverage beyond English, Kannada, and Hindi, while synthetic dysfluent speech generated with LLMs and text-to-speech systems could address the data scarcity problem during training.
Finally, integrating this system with clinic databases and electronic health record systems could enable seamless clinical deployment, supporting doctors and speech-language pathologists in early assessment and personalized treatment planning for stuttering.

#pagebreak()