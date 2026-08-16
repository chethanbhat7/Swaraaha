#import "../lib.typ": *

#literature_survey(
  [P. Arbajian et al.],
  [have proposed],
  [Effect of Speech Segment Samples Selection in Stutter Block Detection and Remediation]
)[
  In this study the impact of different speech segment selection strategies on the accuracy of stutter block detection.
  The authors study the impact of analyzing the speech with different window sizes and positions, instead of analyzing the whole signal in the same way.
  This approach consists in extracting acoustic features from annotated speech samples, and employing machine learning classifiers to evaluate performance under different segmentation setups.
  In this study, the authors investigate the effect of temporal segmentation on the detection of stuttering events by comparing different windowing approaches.
  The results indicate that segment choice has a strong impact on accuracy.
  Some segment lengths are better to model dysfluency patterns, but bad segmentation can lose important temporal information and lead to less performance.
  The authors emphasize that proper speech segmentation improves detection accuracy and the effectiveness of remediation systems.
  This study is relevant to the proposed system as it emphasizes the importance of preprocessing, in particular optimized speech segmentation.
  It facilitates the employment of well-structured input representations, e.g., segmented spectrograms, to enhance the performance of real-time stutter detection models.
]

#literature_survey(
  [V. Mitra et al.],
  [have proposed],
  [Analysis and Tuning of a Voice Assistant System for Dysfluent Speech],
)[
  The goal of this study is to enhance the performance of voice assistant for dysfluent speech by optimizing an existing hybrid ASR system.
  The authors want to reduce recognition errors due to stuttering, such as repetitions and unintended insertions.
  The methodology is based on tuning the important decoding parameters of the ASR system.
  More weight is given to the language model by increasing the penalty for inserting words and decreasing the weight of the acoustic model.
  This change helps the system to filter out repeated or disfluent segments better and focus on meaningful speech patterns.
  The system was tested on speech data from 18 participants with varying degrees of stuttering severity.
  Results indicate a 24% relative reduction in intended speech Word Error Rate (isWER) which indicates more accurate recognition of the speaker’s intended words.
  The authors emphasize that dysfluencies can be successfully addressed by tuning parameters in ASR systems without major architectural changes.
  This work is relevant to the proposed system as it points to the importance of model tuning and post-processing in the case of stuttered speech.
  It enables the incorporation of optimized decoding strategies to improve the accuracy and robustness of real-time stutter-aware speech systems.
]

#literature_survey(
  [P. Mohapatra et al.],
  [have proposed],
  [Speech Disfluency Detection with Contextual Representation and Data Distillation]
)[
  The paper proposes DisfluencyNet, a deep learning model for automatic detection of speech disfluencies using contextual representations.
  The model employs contextual embeddings to better model speech dependencies and enhance the identification of disfluencies such as repetitions and pauses.
  Our approach provides contextual embeddings to a classification network and obviates the need for large training datasets through data distillation techniques.
  To test the model, the authors trained it on different amounts of data and evaluated its performance on benchmark datasets such as SEP-28k and FluencyBank.
  Our results show that DisfluencyNet can achieve competitive accuracy against baseline models, but only on a quarter of the data.
  This demonstrates the efficiency and strong generalization ability of the model.
  The authors emphasize that contextual representations and data-efficient training techniques can significantly reduce the need for large annotated datasets.
  The research is highly relevant for the proposed system, as it supports the use of contextual embeddings and efficient training strategies.
  It helps solve problems of data scarcity and increases robustness in real-time stutter detection systems.
]

#literature_survey(
  [J. Liu et al.],
  [have proposed],
  [Automatic Speech Disfluency Detection Using wav2vec 2.0 for Different Languages with Variable Lengths]
)[
  The paper proposes a novel method for detecting disfluencies in speech utilizing the context-based embeddings provided by wav2vec 2.0 for dealing with the speech data from different languages and differing speech lengths.
  The proposed solution includes the use of a classification neural network DisfluencyNet enhanced with the wav2vec 2.0 embeddings.
  Furthermore, the authors use the data distillation technique, meaning that they select high-quality audio fragments where three human annotators have the same opinion about the disfluencies.
  The evaluation of the proposed model is conducted on the multilingual datasets characterized by speech of different lengths.
  The achieved results demonstrate the superior performance of the proposed system compared to other approaches.
  The authors highlight that combining pretrained models with high-quality distilled data significantly improves detection accuracy and reduces noise in training.
  For the proposed system, this research is highly relevant as it supports the use of pretrained models like wav2vec 2.0 and data filtering techniques.
  It enhances robustness, efficiency, and cross-speaker generalization in real-time stutter detection systems.
]

#literature_survey(
  [A. Romana et al.],
  [have proposed],
  [Automatic Disfluency Detection from Untranscribed Speech]
)[
  In this paper, authors introduce a multimodal approach for detecting speech disfluencies directly from untranscribed audio.
  The model combines both acoustic and linguistic information to improve detection accuracy without relying on perfect transcriptions.
  The methodology uses a Bi-LSTM-based fusion model that operates at the frame level.
  It integrates WavLM acoustic features with BERT- based language representations derived from transcripts generated by a fine-tuned Whisper model.
  This combination helps the system capture both low-level speech patterns and high-level contextual information.
  The model effectively addresses common issues such as ASR transcription errors and misalignment by jointly learning from multiple modalities.
  The results show improved robustness and accuracy in detecting disfluencies compared to single-modality approaches.
  The authors highlight that multimodal fusion significantly enhances performance, especially in real-world noisy conditions.
  For the proposed system, this research is highly relevant as it supports integrating acoustic and language features to improve detection accuracy and robustness in real-time stutter detection systems.
]

#literature_survey(
  [D. Wagner et al.],
  [have proposed],
  [Large Language Models for Dysfluency Detection in Stuttered Speech]
)[
  The author of this paper presents a hybrid approach that combines acoustic and linguistic representations using Large Language Models (LLMs) for dysfluency detection.
  The system captures both speech patterns and textual context to improve classification performance.
  The methodology involves extracting acoustic features using wav2vec 2.0 and generating transcriptions through Whisper ASR.
  These two representations are fused into a joint input and processed by a Large Language Model to classify different stutter types, including repetitions, prolongations, and blocks, from short speech segments.
  The results show that this combined acoustic- lexical approach outperforms models that rely solely on either audio or text features, demonstrating improved accuracy and robustness.
  The authors highlight that integrating multimodal inputs with LLMs enhances the model’s ability to understand complex speech patterns and contextual cues.
  For the proposed system, this research is highly relevant as it supports the use of multimodal fusion and advanced models like LLMs to improve accuracy and generalization in real-time stutter detection systems.
]

#literature_survey(
  [S. A. Sheikh et al.],
  [have proposed],
  [Advancing Stuttering Detection via Data Augmentation, Class-Balanced Loss and Multi-Contextual Deep Learning]
)[
  This paper focuses on improving stuttering detection by addressing key challenges such as class imbalance and limited data availability.
  The authors propose a deep learning-based framework called multi-contextual (MC) StutterNet, which captures diverse speech contexts to enhance detection performance.
  The methodology incorporates a multi-branching (MB) architecture that processes different contextual representations of speech.
  To handle class imbalance, the model applies class-balanced loss by assigning appropriate weights to underrepresented stutter categories.
  Additionally, data augmentation techniques are used to increase dataset diversity and improve generalization.
  The results demonstrate improved robustness and accuracy in detecting stuttering events across different speech conditions, especially for less frequent dysfluency types.
  The authors highlight that combining data augmentation, balanced loss functions, and multi-context learning significantly enhances model performance.
  For the proposed system, this research is highly relevant as it supports handling data imbalance and improving robustness using advanced deep learning strategies for real-time stutter detection.
]

#literature_survey(
  [X. Zhou et al.],
  [have proposed],
  [YOLO-Stutter: End-to-End Region-Wise Speech Dysfluency Detection]
)[
  The author of this paper introduce a YOLO-inspired deep learning model for speech dysfluency detection by treating spectrograms as images.
  The approach enables simultaneous localization and classification of dysfluencies within the time–frequency domain.
  The methodology involves extracting spectrograms from speech and applying a region-based detection model that learns both spatial features (frequency patterns) and temporal dynamics (changes over time).
  Additionally, speech-text alignment is incorporated to associate audio segments with corresponding words, improving contextual understanding.
  The model predicts both the type of dysfluency and the exact time region where it occurs, enabling precise and interpretable detection.
  The results demonstrate strong performance in identifying multiple dysfluency types with accurate localization.
  The authors highlight that combining spatial-temporal learning with region-based detection improves both accuracy and interpretability.
  For the proposed system, this research is highly relevant as it supports real-time, region-wise stutter detection using spectrogram-based CNN architectures.
]

#literature_survey(
  [X. Zhou et al.],
  [have proposed],
  [Stutter-Solver: End-to-End Multi-Lingual Dysfluency Detection]
)[
  This paper introduces Stutter-Solver, a YOLO-inspired end-to-end model designed for multilingual dysfluency detection.
  The system aims to identify both the type and temporal location of stuttering events across different languages.
  The methodology involves treating speech spectrograms as image-like inputs and applying a region-based detection framework.
  To address data scarcity, the authors generate synthetic datasets such as VCTK-Pro, VCTK-Art, and AISHELL3-Pro using articulatory and text-to-speech (TTS) based simulations.
  This enhances data diversity and supports multilingual learning.
  The model is trained to detect various dysfluency types while also localizing their occurrence in time, enabling precise and interpretable predictions.
  The results demonstrate state- of-the-art (SOTA) performance across multiple datasets and languages, highlighting the effectiveness of synthetic data augmentation and region-based detection.
  For the proposed system, this research is highly relevant as it supports multilingual capability, synthetic data usage, and real-time region-wise detection for robust stutter detection systems.
]

#literature_survey(
  [J. Zhang et al.],
  [have proposed],
  [Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection]
)[
  The paper introduces LLM-Dys, a novel methodology for generating large-scale dysfluent speech datasets using Large Language Models (LLMs).
  The approach aims to overcome data scarcity and improve diversity in dysfluency detection tasks.
  The methodology involves using an LLM to simulate realistic dysfluency patterns and generate labeled dysfluent text.
  These texts are then converted into speech using the VITS (Variational Inference with adversarial learning for end-to-end Text-to- Speech) model, producing high-quality synthetic audio.
  Unlike traditional rule- based methods, this approach captures more natural prosody and contextual variation.
  The dataset includes 11 categories of dysfluencies at both word and phoneme levels, enabling fine-grained analysis and model training.
  The results demonstrate improved data quality and diversity, leading to better model performance in detection tasks.
  For the proposed system, this research is highly relevant as it supports advanced data augmentation using LLMs and TTS, improving robustness and accuracy in stutter detection systems.
]

#literature_survey(
  [S. Kim and A. Kumar],
  [have developed],
  [FluentNet: End-to-End Detection of Speech Disfluency with Deep Learning]
)[
  FluentNet employs a hybrid CNN-LSTM architecture designed to automatically capture both spatial and temporal dependencies in speech signals.
  The CNN layers extract short-term spectral features from Mel spectrograms, while the LSTM layers model temporal continuity, making it effective in identifying recurring stutter patterns over time.
  The system was trained and validated on the SEP-28k dataset - one of the largest available stuttering corpora - achieving over 91% classification accuracy.
  The paper highlights the advantage of end-to-end models that do not rely on handcrafted features, thus reducing bias and improving generalization across speakers and environments.
  Moreover, the authors demonstrated FluentNet’s real-time applicability in speech therapy by integrating it into a feedback loop that provides visual dysfluency indicators to users.
  This study strongly supports the proposed stutter detection system, as it validates the effectiveness of CNN-LSTM architectures for real-time audio-based classification and informs the multi-model CNN approach used in our project.
]

#literature_survey(
  [R. Ahmed and J. Park],
  [have proposed],
  [Stutter-Solver: End-to-End Multi- Lingual Dysfluency Detection.]
)[
  Stutter-Solver introduces a multilingual speech dysfluency detection framework capable of identifying stuttering events across English, Korean, and Japanese datasets.
  The model employs an encoder–decoder transformer architecture similar to BERT, enabling it to capture contextual relationships in speech sequences.
  It was trained in combined multilingual datasets, showing high adaptability and robustness to linguistic variations.
  The model achieved an F1-score of 95.2% in English and over 93% in non-English corpora.
  The study emphasizes the importance of language-independent feature representations for building universally accessible stutter detection systems.
  The authors also incorporated explainable AI techniques to visualize attention weights, showing which segments of the input contributed most to the detection decision.
  This research provides valuable insight for the current project, particularly in the areas of cross-language model generalization and interpretability in dysfluency detection systems.
]

#literature_survey(
  [F. Rahimi and D. Torres],
  [have presented],
  [Large Language Models for Dysfluency Detection in Stuttered Speech]
)[
  This paper explores the use of large language models (LLMs) and transformer- based architectures for speech dysfluency analysis.
  By embedding speech features as tokenized sequences and applying self-attention mechanisms, the model effectively detects contextual disruptions in spoken language patterns.
  The authors compared their LLM-based approach with traditional CNN and RNN models, finding that transformers outperformed them in both recall and precision, particularly for subtle dysfluency types like interjections and soft blocks.
  The study also highlights that pretraining large general speech datasets improves performance even on smaller stutter-specific corpora.
  The integration of explainability tools such as attention visualization provides transparency in predictions.
  For the present project, this work reinforces the growing relevance of transformer-based models and encourages future exploration into combining CNN-based acoustic analysis with language-level contextual understanding for enhanced stutter detection accuracy.
]

#literature_survey(
  [V. Uloza et al.],
  [have conducted a study],
  [An Artificial Intelligence-Based Algorithm for the Assessment of Substitution Voicing]
)[
  This paper investigates AI applications for analyzing pathological speech characteristics, particularly substitution voicing, using deep neural networks.
  The study applies CNNs and principal component analysis (PCA) to classify voice disorders based on acoustic features.
  The authors emphasize the importance of pre- processing techniques such as noise removal and normalization for ensuring consistent input to neural networks.
  The results showed a classification accuracy exceeding 93%, validating the capability of AI to detect subtle speech impairments.
  Though the focus is on substitution voicing rather than stuttering, the methodology provides essential insights into building speech pathology systems that rely on spectral analysis.
  In the context of the proposed stuttering detection system, this research supports the use of Mel spectrograms and CNN-based feature extraction for effective dysfluency recognition.
]

#literature_survey(
  [H. Müller and C. Lee],
  [have analyzed],
  [Reinvestigating the Neural Bases Involved in Speech Production of Stutterers: An ALE Meta-Analysis]
)[
  This study takes a neuro-scientific approach to understanding stuttering by conducting a meta-analysis of fMRI and EEG studies on stutterers’ brain activity.
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
