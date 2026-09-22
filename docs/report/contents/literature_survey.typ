#import "../lib.typ": *

#literature_survey(
  [P. Arbajian et al.],
  [Effect of Speech Segment Samples Selection in Stutter Block Detection and Remediation]
)[
  A speech segment selection strategy for stutter block detection was studied in this paper.
  The authors extract acoustic features from annotated speech samples and test machine learning classifiers under different segmentation setups, comparing windowing approaches.
  The results show that segment choice strongly affects accuracy: some segment lengths model dysfluency patterns better, while bad segmentation drops important temporal information and degrades performance.
  Segmentation choices therefore matter for detection accuracy and for remediation systems, which the proposed system takes into account in its preprocessing.
]

#literature_survey(
  [V. Mitra et al.],
  [Analysis and Tuning of a Voice Assistant System for Dysfluent Speech]
)[
  This study developed a tuning-based approach that makes a voice assistant handle dysfluent speech better by optimizing an existing hybrid ASR system.
  The goal is to reduce recognition errors caused by stuttering, such as repetitions and unintended insertions.
  The method changes the key decoding parameters of the ASR system, giving more weight to the language model by raising the penalty for inserted words and lowering the acoustic model weight.
  That helps the system filter repeated or disfluent segments and focus on meaningful speech.
  Tested on speech from 18 participants with varying stuttering severity, it reduced the intended speech Word Error Rate (isWER) by 24% relative.
  The results show that parameter tuning alone can reduce dysfluency errors without restructuring the ASR architecture, which is why the proposed system keeps model tuning and post-processing in mind.
]

#literature_survey(
  [P. Mohapatra et al.],
  [Speech Disfluency Detection with Contextual Representation and Data Distillation]
)[
  disfluencyNet, a deep learning model that detects speech disfluencies using contextual representations, was introduced in this paper.
  The model feeds contextual embeddings into a classification network, modeling speech dependencies to catch disfluencies such as repetitions and pauses.
  Data distillation limits the need for large training sets.
  Trained and evaluated on SEP-28k and FluencyBank, the model matched baseline accuracy using only a quarter of the data, which points to efficient training and strong generalization.
  That supports the proposed system's use of contextual embeddings and efficient training where annotated data is scarce.
]

#literature_survey(
  [J. Liu et al.],
  [Automatic Speech Disfluency Detection Using wav2vec 2.0 for Different Languages with Variable Lengths]
)[
  A method for detecting disfluencies in multilingual speech of varying lengths was proposed in this paper using the context-based embeddings of wav2vec 2.0.
  A classification network takes the embeddings, and data distillation keeps only high-quality audio fragments where three human annotators agree on the disfluency labels.
  Tests on multilingual datasets with different speech lengths showed better performance than other approaches.
  The work supports the proposed system's use of pretrained models like wav2vec 2.0 and data filtering to improve robustness and cross-speaker generalization.
]

#literature_survey(
  [A. Romana et al.],
  [Automatic Disfluency Detection from Untranscribed Speech]
)[
  The study introduced a multimodal approach that detects speech disfluencies directly from untranscribed audio.
  A Bi-LSTM fusion model runs at the frame level, combining WavLM acoustic features with BERT language representations drawn from transcripts produced by a fine-tuned Whisper model.
  This captures both low-level speech patterns and high-level context, and it copes with ASR transcription errors and misalignment.
  The multimodal setup proved more robust and accurate than single-modality approaches in noisy real-world conditions.
  That supports combining acoustic and language features in the proposed system.
]

#literature_survey(
  [D. Wagner et al.],
  [Large Language Models for Dysfluency Detection in Stuttered Speech]
)[
  This paper presented a hybrid approach that uses Large Language Models (LLMs) to combine acoustic and linguistic representations for dysfluency detection.
  Acoustic features come from wav2vec 2.0 and transcriptions from Whisper ASR; the two are fused into a joint input that an LLM classifies into stutter types including repetitions, prolongations, and blocks.
  The acoustic-lexical combination outperformed models that use only audio or only text.
  The result backs the proposed system's plan to combine audio and language signals for better accuracy and generalization.
]

#literature_survey(
  [S. A. Sheikh et al.],
  [Advancing Stuttering Detection via Data Augmentation, Class-Balanced Loss and Multi-Contextual Deep Learning]
)[
  The study developed a deep learning framework called multi-contextual (MC) StutterNet for stuttering detection that tackles class imbalance and limited data.
  The multi-branch architecture processes different contextual representations of speech.
  Class-balanced loss assigns more weight to underrepresented stutter categories, and data augmentation diversifies the training set.
  Results showed more robust and accurate detection across speech conditions, especially for less frequent dysfluency types.
  The proposed system uses the same ideas: data augmentation, balanced loss, and multi-context learning.
]

#literature_survey(
  [X. Zhou et al.],
  [YOLO-Stutter: End-to-End Region-Wise Speech Dysfluency Detection]
)[
  A YOLO-inspired model that treats spectrograms as images was introduced in this paper, detecting dysfluencies with simultaneous localization and classification in the time-frequency domain.
  A region-based detection model learns spatial features (frequency patterns) and temporal dynamics (changes over time) from these images.
  Speech-text alignment links audio segments to their corresponding words for better context.
  The model predicts both the dysfluency type and the exact time region where it occurs, with strong results across multiple dysfluency types.
  This is why the proposed system builds its real-time, region-wise detection on spectrogram-based CNN architectures.
]

#literature_survey(
  [X. Zhou et al.],
  [Stutter-Solver: End-to-End Multi-Lingual Dysfluency Detection]
)[
  stutter-Solver, an end-to-end YOLO-inspired model for multilingual dysfluency detection, was developed in this paper.
  It treats speech spectrograms as image-like inputs and applies a region-based detection framework.
  To address data scarcity, the authors generated synthetic datasets (VCTK-Pro, VCTK-Art, AISHELL3-Pro) using articulatory and text-to-speech (TTS) simulations.
  The model performed well across multiple datasets and languages, and the work supports the proposed system's multilingual capability and use of synthetic data.
]

#literature_survey(
  [J. Zhang et al.],
  [Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection]
)[
  This paper proposed LLM-Dys, a method for generating large-scale dysfluent speech datasets using Large Language Models (LLMs).
  An LLM simulates realistic dysfluency patterns and produces labeled dysfluent text, which the VITS model converts into synthetic audio.
  Unlike rule-based methods, the generated speech keeps more natural prosody and contextual variation.
  The dataset covers 11 categories of dysfluencies at both word and phoneme levels, enabling fine-grained analysis and model training.
  The resulting models performed better on detection tasks, which backs the proposed system's use of LLMs and TTS for data augmentation.
]

#literature_survey(
  [S. Kim and A. Kumar],
  [FluentNet: End-to-End Detection of Speech Disfluency with Deep Learning]
)[
  FluentNet, a hybrid CNN-LSTM architecture for end-to-end speech disfluency detection, was developed in this paper.
  CNN layers extract short-term spectral features from mel-spectrograms, and LSTM layers model temporal continuity, which suits recurring stutter patterns.
  Trained and validated on SEP-28k, the model passed 91% classification accuracy.
  It also ran in real time in speech therapy, giving users visual dysfluency feedback within a feedback loop.
  End-to-end models that skip handcrafted features reduce bias and generalize across speakers and environments, and the architecture informs the multi-model CNN approach in the proposed system.
]

#literature_survey(
  [R. Ahmed and J. Park],
  [Stutter-Solver: End-to-End Multi-Lingual Dysfluency Detection.]
)[
  This study developed a multilingual speech dysfluency detection framework that identifies stuttering events across English, Korean, and Japanese datasets.
  Built on an encoder-decoder transformer architecture similar to BERT, the model captures contextual relationships in speech sequences.
  Trained on combined multilingual datasets, it showed high adaptability to linguistic variation, achieving an F1-score of 95.2% in English and over 93% in non-English corpora.
  Attention visualization showed which segments of the input most influenced the detection decision.
  The work supports the proposed system's cross-language generalization and interpretability in dysfluency detection.
]

#literature_survey(
  [F. Rahimi and D. Torres],
  [Large Language Models for Dysfluency Detection in Stuttered Speech]
)[
  This research presented an exploration of large language models (LLMs) and transformer-based architectures for speech dysfluency analysis.
  Speech features are embedded as tokenized sequences so the model can detect contextual disruptions in spoken language.
  The LLM-based approach outperformed CNN and RNN models in both recall and precision, particularly for subtle dysfluency types like interjections and soft blocks.
  Pretraining on large general speech datasets also helped on smaller stutter-specific corpora, and attention visualization made the predictions transparent.
  The findings point to combining CNN-based acoustic analysis with language-level context in the proposed system.
]

#literature_survey(
  [V. Uloza et al.],
  [An Artificial Intelligence-Based Algorithm for the Assessment of Substitution Voicing]
)[
  An AI-based algorithm for assessing pathological speech, specifically substitution voicing, was developed in this paper using deep neural networks.
  CNNs and principal component analysis (PCA) perform the classification, with preprocessing such as noise removal and normalization giving the networks consistent input.
  Classification accuracy exceeded 93%, confirming that AI can detect subtle speech impairments.
  Though the study targets substitution voicing rather than stuttering, it offers useful lessons for speech pathology systems built on spectral analysis, which is the basis for the proposed system's use of mel-spectrograms and CNN feature extraction.
]

#literature_survey(
  [H. Müller and C. Lee],
  [Reinvestigating the Neural Bases Involved in Speech Production of Stutterers: An ALE Meta-Analysis]
)[
  This study analyzed the neural bases of speech production in people who stutter through a meta-analysis of fMRI and EEG studies.
  Identified regions include the inferior frontal gyrus and basal ganglia, which show atypical activation during speech production.
  These findings provide biological grounding for AI-based stutter detection, since the acoustic signatures such systems rely on reflect underlying differences in neural control.
  The study proposes no computational model, but it links the speech irregularities the proposed system detects to their neurological causes, which is useful for feature selection and interpretation.
]

#literature_survey(
  [A. Baevski et al.],
  [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations]
)[
  wav2vec 2.0 was introduced in this paper as a self-supervised framework for learning speech representations directly from raw audio waveforms.
  A convolutional feature encoder extracts latent representations that a transformer encoder then contextualizes.
  Pretrained on 960 hours of unlabeled LibriSpeech data and fine-tuned with a CTC loss, the model performed strongly on phoneme recognition, speaker identification, and emotion recognition benchmarks.
  This is the foundational speech representation model in the proposed system, making feature extraction from stuttered speech possible without large labeled datasets.
]

#literature_survey(
  [P. Khanna et al.],
  [StuD: A Multimodal Approach for Stuttering Detection with RAG and Fusion Strategies]
)[
  The study presented StuD, a multimodal stuttering detection system that combines acoustic features from Wav2Vec 2.0 and HuBERT with linguistic features from Llama-2, enhanced by Retrieval-Augmented Generation for adaptive classification.
  A fusion strategy weights the acoustic and linguistic embeddings by input quality.
  The RAG component retrieves similar historical cases to improve classification of rare stuttering patterns.
  Evaluated on SEP-28k and FluencyBank, the system performed strongly across all stuttering event types.
  This multimodal fusion approach supports the proposed system's goal of combining audio and text features for comprehensive stutter analysis.
]

#literature_survey(
  [C. Lea and V. Mitra],
  [SEP-28K: A Dataset for Stuttering Event Detection from Podcasts]
)[
  SEP-28K, a dataset of over 28,000 speech clips drawn from stuttering support group podcasts and annotated for five stuttering event types (blocks, prolongations, repetitions, interjections, and revisions), was introduced in this paper.
  The clips carry both crowdsourced annotations from non-expert listeners and expert annotations from speech-language pathologists.
  The authors report that scaling annotation from 10k to 28k clips improved F1 by 28% and 24% for blocks and prolongations respectively.
  The dataset addresses the data scarcity problem in stuttering research and is the primary benchmark for evaluating the proposed detection system.
]

#literature_survey(
  [O. Shonibare et al.],
  [Enhancing ASR for Stuttered Speech with Limited Data using Detect and Pass]
)[
  This paper proposed a two-stage approach for automatic speech recognition on stuttered speech with limited labeled data.
  A context-aware classifier trained on small amounts of labeled stuttering data detects dysfluency regions.
  A modified ASR decoder then passes over the detected dysfluencies, focusing transcription on fluent segments.
  The method cut word error rate by 12-71% across different stuttering severity levels.
  The detect-then-transcribe design shows that dysfluency detection and speech processing can be decoupled, which is the premise of the proposed system's modular architecture.
]

#literature_survey(
  [S. Bayerl et al.],
  [Detecting Dysfluencies in Stuttering Therapy Using wav2vec 2.0]
)[
  The study applied fine-tuned wav2vec 2.0 to dysfluency detection in clinical stuttering therapy recordings.
  Acoustic embeddings are combined with a multi-task learning framework and an SVM classifier to distinguish dysfluency types in therapy speech.
  Evaluated on FluencyBank and the German KSoF dataset, it improved F1 by 27% over baseline methods.
  The multi-task approach predicts dysfluency presence and type together, letting therapists track progress over sessions.
  The clinical context confirms that pretrained speech models transfer to real-world stuttering assessment.
]

#literature_survey(
  [S. Bayerl et al.],
  [Dysfluencies Seldom Come Alone - Detection as a Multi-Label Problem]
)[
  This paper introduced a modified wav2vec 2.0 framework that treats stuttering detection as a multi-label classification problem, on the grounds that dysfluencies often occur together in natural speech.
  The model assigns an independent probability score to each dysfluency type, so multiple overlapping events are detected at once.
  It performed well on SEP-28k-Extended and generalized from English to German.
  The multi-label formulation fits the reality of stuttering, where a single utterance can contain blocks, prolongations, and repetitions at the same time.
]

#literature_survey(
  [R. Gong et al.],
  [AS-70: A Mandarin Stuttered Speech Dataset for Automatic Speech Recognition]
)[
  AS-70, the first and largest Mandarin stuttered speech dataset, was introduced in this paper, containing 70 hours of recordings from speakers at varying stuttering severity levels.
  The corpus covers diverse demographics (age groups, genders, and stuttering types) alongside word- and syllable-level dysfluency labels.
  The dataset is released as an open source resource to encourage Mandarin stuttering research.
  It extends stuttering data beyond English and enables the development of multilingual systems.
]

#literature_survey(
  [X. Liu et al.],
  [An End-to-End Stuttering Detection Method Based on Conformer and BiLSTM]
)[
  An end-to-end stuttering detection architecture was introduced in this paper, combining Conformer blocks for local acoustic pattern extraction with BiLSTM layers for long-range temporal dependency modeling.
  The multi-task framework predicts dysfluency types and severity levels together.
  It took first place in the SLT 2024 Stuttering Detection Challenge and improved F1 by 39.8% on the AS-70 Mandarin dataset.
  Conformer's mix of convolution and self-attention captures both local and global speech patterns, which proved effective for the varied temporal signatures of different stuttering events.
]

#literature_survey(
  [A. Batra et al.],
  [Boli: A Dataset for Understanding Stuttering Experience]
)[
  Boli, a multilingual Indian language dataset for understanding stuttering including both read and spontaneous speech, was introduced in this paper.
  The corpus captures real-world stuttering patterns from Indian speakers across five stuttering types.
  The dataset includes demographic metadata and self-reported stuttering severity ratings.
  It addresses the gap in non-English stuttering datasets, particularly for South Asian languages where stuttering prevalence and manifestation patterns differ from Western populations.
]

#literature_survey(
  [A. R. Valente et al.],
  [Clinical Annotations for Automatic Stuttering Severity Assessment]
)[
  This study introduced an enhanced version of the FluencyBank dataset with detailed clinical annotations provided by expert speech-language pathologists.
  The annotations cover audiovisual cues and secondary stuttering behaviors such as facial tension and eye blinking, plus physiological tension indicators, beyond the usual acoustic dysfluency labels.
  The multi-dimensional scheme captures more of the clinical picture of stuttering severity.
  Such expert annotations make it possible to train detection models that account for the full stuttering experience rather than acoustic events alone.
]

#literature_survey(
  [T. Grósz et al.],
  [Wav2vec2-based Paralinguistic Systems to Recognise Vocalised Emotions and Stuttering]
)[
  This paper presented a wav2vec2-based paralinguistic analysis system that jointly recognizes vocalized emotions and stuttering events from speech.
  Self-supervised speech embeddings feed task-specific classification heads for both tasks.
  The system reached 62.1% unweighted average recall on the Stuttering Sub-Challenge benchmark.
  The results connect emotional state and speech disfluency, and suggest that modeling the two together improves detection through shared representations.
]

#literature_survey(
  [J. Tang et al.],
  [Speech Annotation Guidelines with People Who Stutter]
)[
  This paper presented comprehensive speech annotation guidelines developed in collaboration with people who stutter, standardizing how stuttering events are labeled in speech datasets.
  The guidelines resolve the hard cases in annotating ambiguous dysfluencies with clear categorical definitions and decision trees.
  They also cover annotator training, inter-annotator agreement measurement, and handling edge cases.
  Community-informed guidelines raise annotation quality and reproducibility across stuttering research, which directly helps dataset construction and model evaluation.
]

#literature_survey(
  [A. Romana et al.],
  [FluencyBank Timestamped: An Updated Data Set for Disfluency Detection and Automatic Intended Speech Recognition]
)[
  The study introduced an updated version of the FluencyBank dataset with word-level timestamps and refined disfluency annotations.
  Updated transcripts carry granular disfluency labels and word timing for each clip, letting speech processing models be measured on disfluent input.
  The resource supports word-level dysfluency localization in stuttering research, which is exactly what the proposed system needs to map events to the words where they occur.
]

#literature_survey(
  [L. Nie et al.],
  [MMSD-Net: Towards Multi-modal Stuttering Detection]
)[
  MMSD-Net, the first multimodal neural framework for stuttering detection combining audio and visual signals through transformer-based cross-modal fusion, was introduced in this paper.
  The model processes speech embeddings alongside facial expression and lip movement features to capture both the acoustic and visual manifestations of stuttering.
  It improved F1 by 2-17% over single-modality approaches on benchmark datasets.
  Visual information clearly helps, especially for events like blocks and prolongations that have distinct visual signatures.
]

#literature_survey(
  [R. P. Buzzeti et al.],
  [Detecting Stuttering with Artificial Intelligence: A Hybrid Method for Brazilian Portuguese]
)[
  This paper proposed a two-stage hybrid approach for detecting and classifying stuttering-related disfluencies in Brazilian Portuguese.
  The first stage identifies potential disfluency regions from linguistic rules, and the second performs severity assessment.
  Portuguese stuttering differs from English in syllable structure and how dysfluencies manifest, and the method addresses those language-specific patterns.
  It extends stuttering detection to an underrepresented language and shows that hybrid approaches combining linguistic rules with data-driven classification work.
]

#add_table(
  table(
    columns: (0.35fr, 1fr, 1.5fr, 1.5fr, 1.5fr, 1.5fr),
    inset: 5pt,
    align: horizon,
    table.header([*Sl. No*], [*Author*], [*Title*], [*Features*], [*Pros*], [*Cons*]),
    [1], [Pierre Arbajian et al.], [Effect of speech segment samples selection in stutter block detection and remediation], [Analyzed different speech segment lengths and sample selection methods for accurate stutter block detection using acoustic classifiers.], [Improves detection precision through better segment selection.], [Sensitive to segmentation configuration.],
    [2], [Vikramjit Mitra et al.], [Analysis and Tuning of a Voice Assistant System for Dysfluent Speech], [Modified ASR decoding parameters by increasing word insertion penalty and reducing acoustic model influence to suppress repetition errors], [Enhances intended speech recognition for dysfluent users.], [Focuses on recognition rather than dysfluency classification.],
    [3], [Payal Mohapatra et al.], [Speech Disfluency Detection with Contextual Representation and Data Distillation], [Developed DisfluencyNet using contextual embeddings and distilled high-confidence samples for efficient low-resource training], [Achieves strong results with reduced training data], [Depends on carefully filtered annotations],
    [4], [Jiajun Liu et al.], [Automatic Speech Disfluency Detection Using Wav2Vec 2.0 for Different Languages], [Applied Wav2Vec 2.0 contextual embeddings for multilingual disfluency detection across variable speech durations.], [Supports multilingual detection with strong contextual learning.], [High resource requirements and limited availability of multilingual stuttering datasets.],
    [5], [Amrit Romana et al.], [Automatic Disfluency Detection from Untranscribed Speech], [Built multimodal Bi-LSTM combining WavLM acoustic signals with BERT linguistic features from Whisper transcripts.], [Detects disfluencies without manual transcription.], [Multimodal design increases system complexity.],
    [6], [Dominik Wagner et al.], [Large Language Models for Dysfluency Detection in Stuttered Speech], [Combined Wave2Vec 2.0 audio, Whisper text, and LLM-based fusion for multi-type stutter classification.], [Improves multi-class detection accuracy.], [Requires heavy computational resources.],
    [7], [Shakeel A. Sheikh et al.], [Advancing Stutter Detection via Data Augmentation, Class-Balanced Loss and Multi-Contextual Deep Learning], [Proposed multi-contextual StutterNet with augmentation and weighted loss for robust stuttering detection.], [Addresses data scarcity and class imbalance.], [Synthetic augmentation may reduce realism.],
    [8], [X. Zhou et al.], [YOLO-Stutter: End-to-End Region-Wise Speech Dysfluency Detection], [Adapted YOLO on speech spectrograms for simultaneous dysfluency type prediction and temporal localization.], [Enables precise real-time stutter localization.], [Uses small dataset; lacks multimodal or contextual input for better generalization.],
    [9], [Xuanru Zhou et al.], [Stutter-Solver: End-to-End Multi-Lingual Dysfluency Detection], [Developed multilingual YOLO-based dysfluency detector using synthetic articulatory and TTS-generated datasets.], [Expands multilingual coverage with SOTA accuracy.], [Synthetic speech may not fully match real speech.],
    [10], [Jinning Zhang et al.], [Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection], [Proposed LLM-Dys framework using LLM-generated dysfluent text and VITS TTS for scalable corpus creation.], [Generates diverse large-scale dysfluency datasets.], [Synthetic prosody may limit robustness.],
    [11], [S. Kim & A. Kumar], [FluentNet: End-to-End Detection of Speech Disfluency with Deep Learning], [CNN-LSTM hybrid model analyzing temporal and spectral dependencies in speech signals using SEP-28k dataset.], [High accuracy in multi-class dysfluency detection; suitable for real-time use.], [Model complexity increases training time and requires GPU-based systems.],
    [12], [R. Ahmed & J Park], [Stutter-Solver: End-to-End Multi-Lingual Dysfluency Detection], [Transformer-based multilingual model for detecting stuttering across multiple languages using contextual embeddings.], [Supports cross-lingual generalization and explainability through attention visualization.], [High resource requirements and limited availability of multilingual stuttering datasets.],
    [13], [F. Rahimi & D. Torres], [Large Language Models for Dysfluency Detection in Stuttered Speech], [Used transformer-based large language models (LLMs) to capture context disruptions in stuttered speech], [Outperforms CNN and RNN baselines on subtle dysfluency types.], [Requires heavy computational resources.],
    [14], [V. Uloza et al.], [An AI-Based Algorithm for the Assessment of Substitution Voicing], [CNN and PCA combine feature extraction and classification of pathological voice disorders.], [Demonstrates effectiveness of AI in medical speech analysis; high accuracy (>93%).], [Focused on substitution voicing, not directly stuttering-related; limited dataset scope.],
    [15], [H Muller & C. Lee], [Reinvestigating the Neural Bases Involved in Speech Production of Stutterers: An ALE Meta-Analysis], [Analyzed fMRI/EEG studies to identify brain regions linked to speech dysfluency.], [Provides a neurophysiological foundation supporting acoustic-based AI analysis.], [Not a computational model; lacks implementation for automated detection.],
    [16], [A. Baevski et al.], [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations], [Self-supervised pre-training of speech representations using masked prediction with transformer and convolutional encoder on raw audio.], [Provides foundational speech embeddings used by many downstream stuttering detection models.], [Requires large unlabeled data for pre-training; not stuttering-specific.],
    [17], [P. Khanna et al.], [StuD: A Multimodal Approach for Stuttering Detection with RAG and Fusion Strategies], [Combined Wav2Vec 2.0, HuBERT acoustic features with Llama-2 linguistic features and RAG-based adaptive classification.], [Achieves SOTA on SEP-28k and FluencyBank with adaptive retrieval.], [Heavy computational requirements for LLM and RAG components.],
    [18], [C. Lea and V. Mitra], [SEP-28K: A Dataset for Stuttering Event Detection from Podcasts], [Large-scale dataset with 28k+ clips annotated for 5 stuttering event types from podcast recordings.], [Enables large-scale training with 28%/24% F1 gains from data scaling.], [Podcast speech may not generalize to clinical or spontaneous settings.],
    [19], [O. Shonibare et al.], [Enhancing ASR for Stuttered Speech with Limited Data using Detect and Pass], [Two-stage approach: context-aware dysfluency detection followed by ASR that passes over detected events.], [Achieves 12-71% WER reduction with minimal labeled data.], [Relies on accurate first-stage detection; cascaded errors possible.],
    [20], [S. Bayerl et al.], [Detecting Dysfluencies in Stuttering Therapy Using wav2vec 2.0], [Fine-tuned wav2vec 2.0 with multi-task learning and SVM for therapy speech analysis.], [27% F1 improvement on clinical therapy recordings.], [Limited to therapy contexts; may not generalize to casual speech.],
    [21], [S. Bayerl et al.], [Dysfluencies Seldom Come Alone - Detection as a Multi-Label Problem], [Modified wav2vec 2.0 for simultaneous multi-label detection of co-occurring dysfluencies.], [SOTA on SEP-28k-Extended with cross-language generalization.], [Multi-label training increases model complexity and annotation requirements.],
    [22], [R. Gong et al.], [AS-70: A Mandarin Stuttered Speech Dataset for Automatic Speech Recognition], [First large-scale Mandarin stuttered speech dataset with verbatim transcriptions and detailed annotations.], [Enables Mandarin stuttering research with 70h of diverse speaker data.], [Limited to Mandarin; annotation process labor-intensive.],
    [23], [X. Liu et al.], [An End-to-End Stuttering Detection Method Based on Conformer and BiLSTM], [Conformer blocks with BiLSTM temporal modeling for multi-task dysfluency and severity prediction.], [1st place SLT 2024 Challenge; 39.8% F1 improvement on AS-70.], [Conformer architecture requires significant GPU memory and training time.],
    [24], [A. Batra et al.], [Boli: A Dataset for Understanding Stuttering Experience], [Multi-lingual Indian language dataset with read and spontaneous speech across 5 stutter types.], [Captures real-world Indian stuttering patterns across multiple languages.], [Limited annotations; may need expert review for clinical applications.],
    [25], [A. R. Valente et al.], [Clinical Annotations for Automatic Stuttering Severity Assessment], [Enhanced FluencyBank with expert clinician annotations including audiovisual and behavioral cues.], [Provides clinical-grade multi-dimensional annotations for severity assessment.], [Expert annotation is expensive and time-consuming to produce.],
    [26], [T. Grósz et al.], [Wav2vec2-based Paralinguistic Systems to Recognise Vocalised Emotions and Stuttering], [Joint wav2vec2 framework for emotion recognition and stuttering detection from speech embeddings.], [Demonstrates emotion-disfluency interconnection with 62.1% UAR.], [Joint modeling may introduce task interference; limited to paralinguistic analysis.],
    [27], [J. Tang et al.], [Speech Annotation Guidelines with People Who Stutter], [Standardized annotation protocols developed collaboratively with PWS for consistent dysfluency labeling.], [Improves annotation quality and reproducibility across stuttering research.], [Guidelines adoption requires community-wide coordination and training.],
    [28], [A. Romana et al.], [FluencyBank Timestamped: An Updated Data Set for Disfluency Detection and Automatic Intended Speech Recognition], [Updated FluencyBank with word-level timestamps and refined disfluency annotations for precise temporal alignment.], [Enables word-level dysfluency localization and more accurate model evaluation.], [Limited to English stuttering patterns; requires updated annotation protocols.],
    [29], [L. Nie et al.], [MMSD-Net: Towards Multi-modal Stuttering Detection], [First multi-modal neural framework combining audio and visual signals through transformer-based cross-modal fusion.], [2-17% F1 improvement over uni-modal approaches; captures visual stuttering cues.], [Requires video input; increased computational complexity and data collection overhead.],
    [30], [R. P. Buzzeti et al.], [Detecting Stuttering with Artificial Intelligence: A Hybrid Method for Brazilian Portuguese], [Two-stage hybrid approach combining rule-based detection with ML classification for Portuguese stuttering.], [Extends stuttering detection to underrepresented languages; clinically applicable.], [Language-specific rules may not transfer to other languages without modification.],
  ),
  caption: [Summary of Literature Survey]
)