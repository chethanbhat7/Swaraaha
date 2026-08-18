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
  ),
  caption: [Summary of Literature Survey]
)
