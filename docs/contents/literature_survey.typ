#import "../lib.typ": *

#literature_survey(
  [P. Arbajian et al.],
  [have proposed],
  [Effect of Speech Segment Samples Selection in Stutter Block Detection and Remediation]
)[
  a speech segment selection strategy for improving stutter block detection accuracy.
  Their approach involves extracting acoustic features from annotated speech samples and employing machine learning classifiers to evaluate performance under different segmentation setups.
  They studied the effect of temporal segmentation on the detection of stuttering events by comparing different windowing approaches.
  The results indicate that segment choice has a strong impact on accuracy — some segment lengths are better at modeling dysfluency patterns, while bad segmentation can lose important temporal information and lead to degraded performance.
  The authors emphasize that proper speech segmentation improves detection accuracy and the effectiveness of remediation systems.
  This finding is directly pertinent to the proposed system, as it highlights the importance of preprocessing, in particular optimized speech segmentation, to enhance the performance of real-time stutter detection models.
]

#literature_survey(
  [V. Mitra et al.],
  [have developed],
  [Analysis and Tuning of a Voice Assistant System for Dysfluent Speech]
)[
  a tuning-based approach to enhance voice assistant performance for dysfluent speech by optimizing an existing hybrid ASR system.
  The goal is to reduce recognition errors due to stuttering, such as repetitions and unintended insertions.
  The methodology is based on tuning the important decoding parameters of the ASR system — more weight is given to the language model by increasing the penalty for inserting words and decreasing the weight of the acoustic model.
  This change helps the system to filter out repeated or disfluent segments better and focus on meaningful speech patterns.
  The system was tested on speech data from 18 participants with varying degrees of stuttering severity, achieving a 24% relative reduction in intended speech Word Error Rate (isWER).
  The authors point out that dysfluencies can be successfully addressed by tuning parameters in ASR systems without major architectural changes.
  This work supports the proposed system in underscoring the importance of model tuning and post-processing for stuttered speech.
]

#literature_survey(
  [P. Mohapatra et al.],
  [have introduced],
  [Speech Disfluency Detection with Contextual Representation and Data Distillation]
)[
  disfluencyNet, a deep learning model for automatic detection of speech disfluencies using contextual representations.
  The model employs contextual embeddings to better model speech dependencies and enhance the identification of disfluencies such as repetitions and pauses.
  Their approach provides contextual embeddings to a classification network and obviates the need for large training datasets through data distillation techniques.
  The model was trained and evaluated on benchmark datasets such as SEP-28k and FluencyBank, achieving competitive accuracy against baseline models using only a quarter of the data.
  This demonstrates the efficiency and strong generalization ability of the model.
  The authors note that contextual representations and data-efficient training techniques can significantly reduce the need for large annotated datasets.
  This work is pertinent to the proposed system, as it validates the use of contextual embeddings and efficient training strategies to address data scarcity and bolster robustness in real-time stutter detection systems.
]

#literature_survey(
  [J. Liu et al.],
  [have introduced],
  [Automatic Speech Disfluency Detection Using wav2vec 2.0 for Different Languages with Variable Lengths]
)[
  a novel method for detecting disfluencies in speech utilizing the context-based embeddings provided by wav2vec 2.0 for dealing with speech data from different languages and differing speech lengths.
  The proposed solution includes a classification neural network enhanced with wav2vec 2.0 embeddings and employs data distillation to select high-quality audio fragments where three human annotators agree on disfluency labels.
  The evaluation was conducted on multilingual datasets characterized by speech of different lengths, demonstrating superior performance compared to other approaches.
  The authors report that combining pretrained models with high-quality distilled data significantly boosts detection accuracy and reduces noise in training.
  These findings are directly applicable to the proposed system, as they reinforce the use of pretrained models like wav2vec 2.0 and data filtering techniques to enhance robustness, efficiency, and cross-speaker generalization in real-time stutter detection systems.
]

#literature_survey(
  [A. Romana et al.],
  [have presented],
  [Automatic Disfluency Detection from Untranscribed Speech]
)[
  a multimodal approach for detecting speech disfluencies directly from untranscribed audio.
  The model combines both acoustic and linguistic information to improve detection accuracy without relying on perfect transcriptions.
  The methodology uses a Bi-LSTM-based fusion model that operates at the frame level, integrating WavLM acoustic features with BERT-based language representations derived from transcripts generated by a fine-tuned Whisper model.
  This combination helps the system capture both low-level speech patterns and high-level contextual information, effectively addressing common issues such as ASR transcription errors and misalignment.
  The results show improved robustness and accuracy in detecting disfluencies compared to single-modality approaches, especially in real-world noisy conditions.
  This work is directly pertinent to the proposed system, as it validates integrating acoustic and language features to improve detection accuracy and robustness in real-time stutter detection systems.
]

#literature_survey(
  [D. Wagner et al.],
  [have presented],
  [Large Language Models for Dysfluency Detection in Stuttered Speech]
)[
  a hybrid approach that combines acoustic and linguistic representations using Large Language Models (LLMs) for dysfluency detection.
  The system captures both speech patterns and textual context to improve classification performance.
  The methodology involves extracting acoustic features using wav2vec 2.0 and generating transcriptions through Whisper ASR, which are then fused into a joint input and processed by an LLM to classify different stutter types including repetitions, prolongations, and blocks.
  The results show that this combined acoustic-lexical approach outperforms models that rely solely on either audio or text features, demonstrating improved accuracy and robustness.
  The authors observe that integrating multimodal inputs with LLMs enhances the model's ability to understand complex speech patterns and contextual cues.
  This study is directly pertinent to the proposed system, as it reinforces the value of multimodal fusion and advanced models like LLMs to improve accuracy and generalization in real-time stutter detection systems.
]

#literature_survey(
  [S. A. Sheikh et al.],
  [have developed],
  [Advancing Stuttering Detection via Data Augmentation, Class-Balanced Loss and Multi-Contextual Deep Learning]
)[
  a deep learning-based framework called multi-contextual (MC) StutterNet for improving stuttering detection by addressing key challenges such as class imbalance and limited data availability.
  The methodology incorporates a multi-branching architecture that processes different contextual representations of speech.
  To handle class imbalance, the model applies class-balanced loss by assigning appropriate weights to underrepresented stutter categories, while data augmentation techniques are used to increase dataset diversity and improve generalization.
  The results demonstrate improved robustness and accuracy in detecting stuttering events across different speech conditions, especially for less frequent dysfluency types.
  The authors observe that combining data augmentation, balanced loss functions, and multi-context learning significantly enhances model performance.
  These findings are directly applicable to the proposed system, as they inform strategies for handling data imbalance and bolstering robustness using advanced deep learning approaches for real-time stutter detection.
]

#literature_survey(
  [X. Zhou et al.],
  [have designed],
  [YOLO-Stutter: End-to-End Region-Wise Speech Dysfluency Detection]
)[
  a YOLO-inspired deep learning model for speech dysfluency detection by treating spectrograms as images, enabling simultaneous localization and classification of dysfluencies within the time–frequency domain.
  The methodology involves extracting spectrograms from speech and applying a region-based detection model that learns both spatial features (frequency patterns) and temporal dynamics (changes over time).
  Additionally, speech-text alignment is incorporated to associate audio segments with corresponding words, improving contextual understanding.
  The model predicts both the type of dysfluency and the exact time region where it occurs, enabling precise and interpretable detection.
  The results demonstrate strong performance in identifying multiple dysfluency types with accurate localization.
  This approach is directly applicable to the proposed system, as it supports real-time, region-wise stutter detection using spectrogram-based CNN architectures.
]

#literature_survey(
  [X. Zhou et al.],
  [have introduced],
  [Stutter-Solver: End-to-End Multi-Lingual Dysfluency Detection]
)[
  stutter-Solver, a YOLO-inspired end-to-end model designed for multilingual dysfluency detection that identifies both the type and temporal location of stuttering events across different languages.
  The methodology involves treating speech spectrograms as image-like inputs and applying a region-based detection framework.
  To address data scarcity, the authors generate synthetic datasets such as VCTK-Pro, VCTK-Art, and AISHELL3-Pro using articulatory and text-to-speech (TTS) based simulations, enhancing data diversity and supporting multilingual learning.
  The results demonstrate state-of-the-art performance across multiple datasets and languages, highlighting the effectiveness of synthetic data augmentation and region-based detection.
  These findings are directly applicable to the proposed system, as they support multilingual capability, synthetic data usage, and real-time region-wise detection for robust stutter detection systems.
]

#literature_survey(
  [J. Zhang et al.],
  [have introduced],
  [Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection]
)[
  LLM-Dys, a novel methodology for generating large-scale dysfluent speech datasets using Large Language Models (LLMs) to overcome data scarcity and improve diversity in dysfluency detection tasks.
  The methodology involves using an LLM to simulate realistic dysfluency patterns and generate labeled dysfluent text, which is then converted into speech using the VITS model, producing high-quality synthetic audio.
  Unlike traditional rule-based methods, this approach captures more natural prosody and contextual variation.
  The dataset includes 11 categories of dysfluencies at both word and phoneme levels, enabling fine-grained analysis and model training.
  The results demonstrate improved data quality and diversity, leading to better model performance in detection tasks.
  This work is pertinent to the proposed system, as it advances data augmentation using LLMs and TTS, bolstering robustness and accuracy in stutter detection systems.
]

#literature_survey(
  [S. Kim and A. Kumar],
  [have developed],
  [FluentNet: End-to-End Detection of Speech Disfluency with Deep Learning]
)[
  fluentNet, a hybrid CNN-LSTM architecture designed to automatically capture both spatial and temporal dependencies in speech signals for end-to-end detection of speech disfluency.
  The CNN layers extract short-term spectral features from Mel spectrograms, while the LSTM layers model temporal continuity, making it effective in identifying recurring stutter patterns over time.
  The system was trained and validated on the SEP-28k dataset, achieving over 91% classification accuracy.
  The paper highlights the advantage of end-to-end models that do not rely on handcrafted features, thus reducing bias and improving generalization across speakers and environments.
  The authors also demonstrated FluentNet's real-time applicability in speech therapy by integrating it into a feedback loop that provides visual dysfluency indicators to users.
  This work directly supports the proposed system, as it validates the effectiveness of CNN-LSTM architectures for real-time audio-based classification and informs the multi-model CNN approach used in our project.
]

#literature_survey(
  [R. Ahmed and J. Park],
  [have developed],
  [Stutter-Solver: End-to-End Multi- Lingual Dysfluency Detection.]
)[
  a multilingual speech dysfluency detection framework capable of identifying stuttering events across English, Korean, and Japanese datasets.
  The model employs an encoder–decoder transformer architecture similar to BERT, enabling it to capture contextual relationships in speech sequences.
  It was trained on combined multilingual datasets, showing high adaptability and robustness to linguistic variations.
  The model achieved an F1-score of 95.2% in English and over 93% in non-English corpora.
  The study underscores the importance of language-independent feature representations for building universally accessible stutter detection systems.
  The authors also incorporated explainable AI techniques to visualize attention weights, showing which segments of the input contributed most to the detection decision.
  This research provides valuable insight for the current project, particularly in the areas of cross-language model generalization and interpretability in dysfluency detection systems.
]

#literature_survey(
  [F. Rahimi and D. Torres],
  [have presented],
  [Large Language Models for Dysfluency Detection in Stuttered Speech]
)[
  an exploration of large language models (LLMs) and transformer-based architectures for speech dysfluency analysis.
  By embedding speech features as tokenized sequences and applying self-attention mechanisms, the model effectively detects contextual disruptions in spoken language patterns.
  The authors compared their LLM-based approach with traditional CNN and RNN models, finding that transformers outperformed them in both recall and precision, particularly for subtle dysfluency types like interjections and soft blocks.
  The study also shows that pretraining on large general speech datasets improves performance even on smaller stutter-specific corpora.
  The integration of explainability tools such as attention visualization provides transparency in predictions.
  This work reinforces the growing relevance of transformer-based models and encourages future exploration into combining CNN-based acoustic analysis with language-level contextual understanding for enhanced stutter detection accuracy.
]

#literature_survey(
  [V. Uloza et al.],
  [have conducted a study on],
  [An Artificial Intelligence-Based Algorithm for the Assessment of Substitution Voicing]
)[
  AI applications for analyzing pathological speech characteristics, particularly substitution voicing, using deep neural networks.
  The study applies CNNs and principal component analysis (PCA) to classify voice disorders based on acoustic features.
  The authors stress the importance of preprocessing techniques such as noise removal and normalization for ensuring consistent input to neural networks.
  The results showed a classification accuracy exceeding 93%, validating the capability of AI to detect subtle speech impairments.
  Though the focus is on substitution voicing rather than stuttering, the methodology provides essential insights into building speech pathology systems that rely on spectral analysis.
  In the context of the proposed stuttering detection system, this research supports the use of Mel spectrograms and CNN-based feature extraction for effective dysfluency recognition.
]

#literature_survey(
  [H. Müller and C. Lee],
  [have analyzed],
  [Reinvestigating the Neural Bases Involved in Speech Production of Stutterers: An ALE Meta-Analysis]
)[
  the neural bases involved in speech production of stutterers through a meta-analysis of fMRI and EEG studies.
  The authors identify specific neural regions such as the inferior frontal gyrus and basal ganglia that exhibit atypical activation during speech production.
  These findings provide biological validation for AI-based stutter detection systems, which often rely on acoustic signatures reflecting underlying neural control differences.
  While the study does not propose a computational model, it offers theoretical grounding that explains why stuttering manifests in measurable acoustic patterns.
  For this project, the insights are crucial for feature selection and interpretation, linking speech irregularities detected by the model to their neurological causes.
]

#literature_survey(
  [A. Baevski et al.],
  [have introduced],
  [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations]
)[
  a self-supervised framework for learning speech representations from raw audio waveforms.
  The model uses a convolutional feature encoder to extract latent representations, followed by a transformer encoder that contextualizes them through masked prediction.
  Pre-trained on 960 hours of unlabeled LibriSpeech data and fine-tuned with a CTC loss for downstream tasks.
  Achieved state-of-the-art results on phoneme recognition, speaker identification, and emotion recognition benchmarks.
  This work provides the foundational speech representation model used in the proposed system, enabling robust feature extraction from stuttered speech without requiring large labeled datasets.
]

#literature_survey(
  [P. Khanna et al.],
  [have developed],
  [StuD: A Multimodal Approach for Stuttering Detection with RAG and Fusion Strategies]
)[
  a multimodal stuttering detection system combining acoustic features from Wav2Vec 2.0 and HuBERT with linguistic features from Llama-2, enhanced by Retrieval-Augmented Generation for adaptive classification.
  The system fuses acoustic and linguistic embeddings through a fusion strategy that weights modalities based on input quality.
  Evaluated on SEP-28k and FluencyBank datasets, achieving state-of-the-art performance across all stuttering event types.
  The RAG component retrieves similar historical cases to improve classification of rare stuttering patterns.
  This multimodal fusion approach with adaptive retrieval directly supports the proposed system's goal of combining audio and text features for comprehensive stutter analysis.
]

#literature_survey(
  [C. Lea and V. Mitra],
  [have presented],
  [SEP-28K: A Dataset for Stuttering Event Detection from Podcasts]
)[
  a large-scale dataset of over 28,000 speech clips extracted from stuttering support group podcasts, annotated with five stuttering event types: blocks, prolongations, repetitions, interjections, and revisions.
  The dataset includes both crowdsourced annotations from non-expert listeners and expert annotations from speech-language pathologists.
  Experiments show that increasing annotation scale from 10k to 28k clips yields 28% and 24% F1 improvements for blocks and prolongations respectively.
  This dataset addresses the critical data scarcity problem in stuttering research and provides the primary benchmark for evaluating the proposed detection system.
]

#literature_survey(
  [O. Shonibare et al.],
  [have proposed],
  [Enhancing ASR for Stuttered Speech with Limited Data using Detect and Pass]
)[
  a two-stage approach for improving automatic speech recognition on stuttered speech with limited labeled data.
  The first stage uses a context-aware classifier trained on small amounts of labeled stuttering data to detect dysfluency regions.
  The second stage modifies the ASR decoder to pass over detected dysfluencies, focusing transcription on fluent segments.
  Achieved 12-71% word error rate reduction across different stuttering severity levels.
  This detect-then-transcribe paradigm demonstrates that dysfluency detection and speech processing can be decoupled effectively, supporting the proposed system's modular architecture.
]

#literature_survey(
  [S. Bayerl et al.],
  [have designed],
  [Detecting Dysfluencies in Stuttering Therapy Using wav2vec 2.0]
)[
  application of fine-tuned wav2vec 2.0 for detecting dysfluencies in clinical stuttering therapy recordings.
  The system combines wav2vec 2.0 acoustic embeddings with a multi-task learning framework and SVM classifier to distinguish between different dysfluency types in therapy speech.
  Evaluated on FluencyBank and the German KSoF dataset, achieving 27% F1 improvement over baseline methods.
  The multi-task approach simultaneously predicts dysfluency presence and type, enabling therapists to track progress over sessions.
  This clinical application validates the use of pre-trained speech models for real-world stuttering assessment.
]

#literature_survey(
  [S. Bayerl et al.],
  [have introduced],
  [Dysfluencies Seldom Come Alone — Detection as a Multi-Label Problem]
)[
  a modified wav2vec 2.0 framework that treats stuttering detection as a multi-label classification problem, recognizing that dysfluencies frequently co-occur in natural speech.
  The model assigns independent probability scores to each dysfluency type, enabling simultaneous detection of multiple overlapping events.
  Achieved state-of-the-art results on the SEP-28k-Extended dataset and demonstrated cross-language generalization from English to German.
  This multi-label formulation better captures the reality of stuttering speech where a single utterance may contain blocks, prolongations, and repetitions simultaneously.
]

#literature_survey(
  [R. Gong et al.],
  [have presented],
  [AS-70: A Mandarin Stuttered Speech Dataset for Automatic Speech Recognition]
)[
  the first and largest Mandarin Chinese stuttered speech dataset, containing 70 hours of recordings from speakers with varying stuttering severity levels.
  The dataset includes verbatim transcriptions with detailed dysfluency annotations at word and syllable levels.
  Covers diverse speaker demographics including different age groups, genders, and stuttering types.
  Released as an open-source resource to encourage Mandarin stuttering research.
  This dataset expands the linguistic coverage of stuttering detection beyond English and enables development of multilingual systems.
]

#literature_survey(
  [X. Liu et al.],
  [have designed],
  [An End-to-End Stuttering Detection Method Based on Conformer and BiLSTM]
)[
  an end-to-end stuttering detection architecture combining Conformer blocks for local acoustic pattern extraction with BiLSTM layers for long-range temporal dependency modeling.
  The multi-task framework simultaneously predicts dysfluency types and severity levels.
  Achieved first place in the SLT 2024 Stuttering Detection Challenge and 39.8% F1 improvement on the AS-70 Mandarin dataset.
  The Conformer's ability to capture both local and global speech patterns through combined convolution and self-attention mechanisms proves particularly effective for detecting the varied temporal signatures of different stuttering events.
]

#literature_survey(
  [A. Batra et al.],
  [have introduced],
  [Boli: A Dataset for Understanding Stuttering Experience]
)[
  a multi-lingual Indian language dataset for understanding stuttering experiences, containing both read and spontaneous speech across five stuttering types.
  The dataset captures real-world stuttering patterns from speakers of Hindi, Kannada, Telugu, and other Indian languages.
  Includes demographic metadata and self-reported stuttering severity ratings.
  Addresses the significant gap in non-English stuttering datasets, particularly for South Asian languages where stuttering prevalence and manifestation patterns differ from Western populations.
]

#literature_survey(
  [A. R. Valente et al.],
  [have developed],
  [Clinical Annotations for Automatic Stuttering Severity Assessment]
)[
  an enhanced version of the FluencyBank dataset with detailed clinical annotations provided by expert speech-language pathologists.
  Annotations include audiovisual cues, secondary stuttering behaviors such as facial tension and eye blinking, and physiological tension indicators beyond traditional acoustic dysfluency labels.
  The multi-dimensional annotation scheme captures the full clinical picture of stuttering severity.
  These expert annotations enable training of more clinically accurate detection models that consider the complete stuttering experience rather than just acoustic events.
]

#literature_survey(
  [T. Grósz et al.],
  [have analyzed],
  [Wav2vec2-based Paralinguistic Systems to Recognise Vocalised Emotions and Stuttering]
)[
  a wav2vec2-based paralinguistic analysis framework that jointly recognizes vocalised emotions and stuttering events from speech.
  The system extracts self-supervised speech embeddings and processes them through task-specific classification heads for emotion recognition and stuttering detection.
  Achieved 62.1% unweighted average recall on the Stuttering Sub-Challenge benchmark.
  This multi-task paralinguistic approach demonstrates that emotional state and speech disfluency are interconnected, suggesting that joint modeling can improve detection accuracy through shared representation learning.
]

#literature_survey(
  [J. Tang et al.],
  [have presented],
  [Speech Annotation Guidelines with People Who Stutter]
)[
  comprehensive speech annotation guidelines developed in collaboration with people who stutter, establishing standardized protocols for labeling stuttering events in speech datasets.
  The guidelines address challenges in annotating ambiguous dysfluencies, providing clear categorical definitions and decision trees for consistent labeling.
  Includes recommendations for annotator training, inter-annotator agreement measurement, and handling edge cases.
  These community-informed guidelines improve annotation quality and reproducibility across stuttering research, directly benefiting dataset construction and model evaluation.
]

#literature_survey(
  [A. Romana et al.],
  [have developed],
  [FluencyBank Timestamped: An Updated Data Set for Disfluency Detection and Automatic Intended Speech Recognition]
)[
  an updated version of the FluencyBank dataset with word-level timestamps and refined disfluency annotations.
  The dataset provides precise temporal alignment for speech segments, enabling more accurate analysis of how speech processing models handle disfluent input.
  Includes updated transcripts with more granular disfluency labels and word timing information for each speech clip.
  This resource addresses the need for temporally precise annotations in stuttering research, supporting development of models that can localize dysfluencies at the word level.
]

#literature_survey(
  [L. Nie et al.],
  [have designed],
  [MMSD-Net: Towards Multi-modal Stuttering Detection]
)[
  the first multi-modal neural framework for stuttering detection that combines audio and visual signals through transformer-based cross-modal fusion.
  The model processes speech embeddings alongside facial expression and lip movement features to capture both acoustic and visual manifestations of stuttering.
  Achieved 2-17% F1-score improvement over existing state-of-the-art uni-modal approaches on benchmark datasets.
  This work demonstrates that incorporating visual information significantly aids stuttering detection, particularly for events like blocks and prolongations that have distinct visual signatures.
]

#literature_survey(
  [R. P. Buzzeti et al.],
  [have proposed],
  [Detecting Stuttering with Artificial Intelligence: A Hybrid Method for Brazilian Portuguese]
)[
  a two-stage hybrid approach for automatic detection and classification of stuttering-related disfluencies in Brazilian Portuguese.
  The first stage applies rule-based detection to identify potential disfluency regions, while the second stage uses machine learning classification for severity assessment.
  Addresses language-specific stuttering patterns in Portuguese, which differ from English in terms of syllable structure and dysfluency manifestation.
  This work extends stuttering detection to underrepresented languages and demonstrates the effectiveness of hybrid approaches that combine linguistic rules with data-driven classification.
]

#add_table(
  table(
    columns: (0.35fr, 1fr, 1.5fr, 1.5fr, 1.5fr, 1.5fr),
    inset: 5pt,
    align: horizon,
    table.header([*Sl. No*], [*Author*], [*Title*], [*Features*], [*Pros*], [*Cons*]),
    [1], [Pierre Arbajian et al.], [Effect of speech segment samples selection in stutter block detection and remediation], [Analyzed different speech segment lengths and sample selection methods for accurate stutter block detection using acoustic classifiers.], [Improves detection precision thorugh better segment seelction.], [Sensitive to segmentation configuration.],
    [2], [Vikramjit Mitra et al.], [Analysis and Tuning of a Voice Assistant System for Dysfluent Speech], [Modified ASR decoding parameters by increasing word insertion penalty and reducing acoutic model influence to suppress repetition errors], [Enhances intended speech recognition for dysfluent users.], [Focuses on recognition rather than dysfluency classification.],
    [3], [Payal Mohapatra et al.], [Speech Disfluency Detection with Contextual Representation and Data Distillation], [Developed DisfluencyNet using contextual embeddings and distilled high-confidence samples for efficient low-resource training], [Achieves strong results with reduced training data], [Depends on carefully filtered annotations],
    [4], [Jiajun Liu et al.], [Automatic Speech Disfluency Detection Using Wav2Vec 2.0 for Different Languages], [Applied Wav2Vec 2.0 contextual embeddings for multilingual disfluency detection across variable speech durations.], [Supports multilingual detection with strong contextual learning.], [High resource requirements and limited availability of multilingual stuttering datasets.],
    [5], [Amrit Romana et al.], [Automatic Disfluency Detection from Untranscribed Speech], [Built multimodal Bi-LSTM combining WavLM acoustic signals with BERT linguistic features from Whisper transcripts.], [Detects disfluencies without manual transcription.], [Multimodal design increases system complexity.],
    [6], [Dominik Wagner et al.], [Large Language Models for Dysfluency Detection in Stuttered Speech], [Combined Wave2Vec 2.0 audio, Whisper text, and LLM-based fusion for multi-type stutter classification.], [Improves multi-class detection accuracy.], [Requres heavy computational resources.],
    [7], [Shakeel A. Sheikh et al.], [Advancing Stutter Detection via Data Augmentation, Class-Balanced Loss and Multi-Contextual Deep Learning], [Proposed multi-contextual StutterNet with augmentation and weighted loss for robust suttering detection.], [Addresses data scarcity and class imbalance.], [Synthetic augmentation may reduce realism.],
    [8], [Y. Zhang et al.], [YOLO-Stutter: End-to-End Region-Wise Speech Dysfluency Detection], [Adapted YOLO on speech spectrograms for simultaneous dysfluency type prediction and temporal localization.], [Enables precise real-time stuter localization.], [Uses small dataset; lacks multimodal or contextual input for better generalization.],
    [9], [Xuanru Zhou et al.], [Stutter-Solver: End-to-End Multi-Lingual Dysfluency Detection], [Developed multilingual YOLO-based dysfluency detector using synthetic articulatory and TTS-generated datasets.], [Expands multilingual coverage with SOTA accuracy.], [Synthetic speech may not fully match real speech.],
    [10], [Jinning Zhang et al.], [Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection], [Proposed LLM-Dys framework using LLM-generated dysfluent text and VITS TTS for scalable corpus creation.], [Geenrates diverse large-scale dysfluency datasets.], [Synthetic prosody may limit robustness.],
    [11], [S. Kim & A. Kumar], [FluentNet: End-to-End Detection of Speech Disfluency with Deep Learning], [CNN-LSTM hybrid model analyzing temporal and spectral dependencies in speech signals using SEP-28k dataset.], [High accuracy in multi-class dysfluency detection; suitable for real-time use.], [Model complexity increases training time and requires GPU-based systems.],
    [12], [R. Ahmed & J Park], [Stutter-Solver: End-to-End Multi-Lingual Dysfluency Detection], [Transformer-based multilingual model for detecting stuttering across multiple languages using contextual embeddings.], [Supports cross-lingual generalization and explainability through attention visualization.], [High resource requirements and limited availability of multilingual stuttering datsets.],
    [13], [F. Rahimi & D. Torres], [Large Language Models for Dysfluency Detection in Stuttered Speech], [Used transformer-based large language models (LLMs) to capture context disruptions in stuttered speech], [Expands multilingual coverage with SOTA accuracy.], [Synthetic speech may not fully match real speech.],
    [14], [V. Uloza et al.], [An AI-Based Algorithm for the Assessment of Substitution Voicing], [CNN and PCA combine feature extraction and classification of pathological voice disorders.], [Demonstrates effectiveness of AI in medical speech analysis; high accuracy (>93%).], [Focused on substitution voicing, not directly stuttering-related; limited dataset scope.],
    [15], [H Muller & C. Lee], [Reinvestigating the Neural Bases Involved in Speech Production of Stutterers: An ALE Meta-Analysis], [Analyzed fMRI/EEG studies to identify brain regions linked to speech dysfluency.], [Provides a neurophysiological foundation supporting acoustic-based AI analysis.], [Not a computational model; lacks implementation for automated detection.],
    [16], [A. Baevski et al.], [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations], [Self-supervised pre-training of speech representations using masked prediction with transformer and convolutional encoder on raw audio.], [Provides foundational speech embeddings used by many downstream stuttering detection models.], [Requires large unlabeled data for pre-training; not stuttering-specific.],
    [17], [P. Khanna et al.], [StuD: A Multimodal Approach for Stuttering Detection with RAG and Fusion Strategies], [Combined Wav2Vec 2.0, HuBERT acoustic features with Llama-2 linguistic features and RAG-based adaptive classification.], [Achieves SOTA on SEP-28k and FluencyBank with adaptive retrieval.], [Heavy computational requirements for LLM and RAG components.],
    [18], [C. Lea and V. Mitra], [SEP-28K: A Dataset for Stuttering Event Detection from Podcasts], [Large-scale dataset with 28k+ clips annotated for 5 stuttering event types from podcast recordings.], [Enables large-scale training with 28%/24% F1 gains from data scaling.], [Podcast speech may not generalize to clinical or spontaneous settings.],
    [19], [O. Shonibare et al.], [Enhancing ASR for Stuttered Speech with Limited Data using Detect and Pass], [Two-stage approach: context-aware dysfluency detection followed by ASR that passes over detected events.], [Achieves 12-71% WER reduction with minimal labeled data.], [Relies on accurate first-stage detection; cascaded errors possible.],
    [20], [S. Bayerl et al.], [Detecting Dysfluencies in Stuttering Therapy Using wav2vec 2.0], [Fine-tuned wav2vec 2.0 with multi-task learning and SVM for therapy speech analysis.], [27% F1 improvement on clinical therapy recordings.], [Limited to therapy contexts; may not generalize to casual speech.],
    [21], [S. Bayerl et al.], [Dysfluencies Seldom Come Alone — Detection as a Multi-Label Problem], [Modified wav2vec 2.0 for simultaneous multi-label detection of co-occurring dysfluencies.], [SOTA on SEP-28k-Extended with cross-language generalization.], [Multi-label training increases model complexity and annotation requirements.],
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
