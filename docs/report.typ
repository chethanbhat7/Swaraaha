#set document(
  title: [Major Project Phase 1 Report]
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
)

#set text(
  font: "Times New Roman",
  size: 12pt,
)

#set par(
    justify: true,
)

#set list(
  indent: 2em,
  spacing: 1.5em,
)

#set outline(
  indent: 0pt
)

#set outline.entry(
  fill: none
)

#show outline: set heading(
  outlined: true,
)

#show outline: set align(center)

#show heading: it => {
  v(0.5em)
  it
}

#let lit_survey_counter = counter("literature counter")

#show outline.entry: it => {
  v(12pt, weak: true)
  link(it.element.location())[
    #if it.level == 1 and it.element.func() == heading {
      v(0.5em)
      grid(
        columns: (1fr, auto),
        align: (left, center),
        stack(dir: ltr)[*#it.body()*], [*#it.page()*]
      )
    } else {
      grid(
        columns: (auto, 1fr, auto),
        align: (center, left, right),
        stack(dir: ltr)[
          #{
            show "Table": none
            show "Figure": none
            it.prefix()
          }
        ], [#h(1.5em) #it.body()], [#it.page()]
      )
    }
  ]
}

#show figure.where(kind: table): set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)

#let non_outlined_heading(level: 1, body) = {
  align(center)[#heading(level: level, outlined: false)[#body]]
  v(0.5em)
}

#let chapter_heading(body) = {
  counter(heading).step()
  counter(figure.where(kind: table)).update(0)

  context {
    let chap = counter(heading).get().first()
    let toc_title = [CHAPTER #chap #body]

    // tricks
    [
      #set text(size: 0pt)
      #v(-0.8em)
      #heading(numbering: none)[#toc_title]
    ]

    box(width: 100%, inset: 0pt)[
      #set text(weight: "bold", size: 16pt)
      #set align(left)
      CHAPTER #chap
      #set text(size: 18pt)
      #set align(center)
      #v(-0.5em)
      #body
      #v(0.5em)
    ]
  }
}

#let add_table(tb, caption: none) = {
  context {
    set figure(
      numbering: num => {
        let chap = counter(heading).get().first()
        numbering("1.1", chap, num)
      }
    )

    figure(
      tb,
      caption: caption
    )
  }
}

#let literature_survey(author, conjunctive, title, body) = {
  lit_survey_counter.step()

  strong(author)
  [ *[#context lit_survey_counter.display()]*]
  [
    #conjunctive _"#title."_

    #body
  ]
  v(1em)
}

#counter(heading).update(0)

// --- Acknowledgement ---
#[
  #set par(spacing: 2em)

  #non_outlined_heading[ACKNOWLEDGEMENT]
  #v(1em)

  We take this opportunity to express our deep heartfelt gratitude to all those people who have helped us in the successful completion of the project.

  First and foremost, we would like to express our sincere gratitude to our guide and  major project coordinator, Prof. *Ajay Shastry C.G.* for providing excellent guidance, encouragement and inspiration throughout the project work.
  Without his invaluable guidance, this work would never have been a successful one.

  We would like to express our sincere gratitude to the Head of the Department of Artificial Intelligence & Machine Learning, Prof. *Abhishek Kumar K*, for his guidance and inspiration.
  We would like to thank our Principal, *Dr. Mahesh Prasanna K*, for providing all the facilities and a proper environment to work in the college campus.

  We would like to thank our management for providing the necessary infrastructure to carry out the project work.

  We are thankful to all the teaching and non-teaching staff members of the Artificial Intelligence & Machine Learning Department for their help and needed support rendered throughout the project.
]

#pagebreak()

// --- Abstract ---
#[
  #set par(spacing: 2em)

  #non_outlined_heading[ABSTRACT]
  #v(2em)

  Stuttering is a speech disorder that disrupts the natural flow of communication through involuntary repetitions, prolongations, blocks, and pauses in speech.
  Nearly 1% of people worldwide are impacted, and it can significantly affect social, professional, and personal life.
  Traditional diagnosis methods rely on manual assessment by speech-language pathologists, which are subjective, time-consuming, and not easily accessible to all individuals.
  To overcome these limitations, this project presents an intelligent and automated system that leverages deep learning to detect and classify various types of stuttering events in speech recordings in real time.

  The core of the system is a multi-model deep learning-based architecture, where five parallel binary classifiers are individually trained to detect distinct stutter types including sound repetitions, word repetitions, prolongations, blocks, and interjections.
  The input audio is preprocessed and converted into embeddings, which are then analyzed using a CNN and Transformer-based architecture to capture both local acoustic patterns and long-range temporal dependencies in speech.
  Each model processes these features independently and outputs probability-based predictions for specific dysfluencies.
  The system also includes a transcription module that generates timestamped speech output from the same audio input.
  The detected stutter timestamps are then hard-aligned with the transcription timestamps to identify the exact word, syllable, or speech segment associated with the stutter event.
  The system is integrated with a user-friendly PyQt5 graphical interface that allows users to record or upload audio samples, visualize waveform and spectrograms, and receive detailed analysis and classification reports in real time.

  By combining deep learning-based speech analysis with transcript-based timestamp alignment and an interactive application interface, this system provides a reliable and accessible tool for early stutter detection and continuous therapy support.
  It reduces reliance on manual evaluation, offers objective insights into speech fluency, and can serve as an assistive aid for both speech pathologists and individuals undergoing therapy.
  This project demonstrates how artificial intelligence and audio signal processing can be effectively utilized in the healthcare domain to enhance diagnostic accuracy, improve therapy outcomes, and promote speech fluency assessment at scale.
]

#pagebreak()

// --- TOC outline ---
#non_outlined_heading[TABLE OF CONTENT]

#grid(
  columns: (1fr, auto),
  align: (left, center),
  stack(dir: ltr)[*Title*], [*Page \ No.*]
)

#outline(
  title: none,
  target: heading,
)

#pagebreak()

// Start roman page numbers
#set page(numbering: "I")
#counter(page).update(1)

// --- Tables list ---
#align(center)[= LIST OF TABLES]
#v(0.5em)

#grid(
  columns: (auto, 1fr, auto),
  align: center,
  stack(dir: ltr)[*Table \ No.*], [*Title*], [*Page \ No.*]
)
#outline(
  title: none,
  target: figure.where(kind: table),
)

#pagebreak()

// --- Abbreviations ---
#align(center)[= LIST OF ABBREVIATIONS]
#table(
  columns: (1fr, 3fr),
  stroke: none,
  align: (left, left),
  column-gutter: 10pt,
  table.header([*Abbreviation*], [*Description*]),
  [*ALE*], [Activattion Likelihood Estimation],
  [*ASR*], [Automatic Speech Recognition],
  [*BERT*], [Bidirectional Encoder Representations from Transformers],
  [*Bi-LSTM*], [Bidirectional Long Short-Term Memory],
  [*CNN*], [Convolutional Neural Network],
  [*EEG*], [Electroencephalography],
  [*fMRI*], [Functional Magnetic Resonance Imaging],
  [*GUI*], [Graphical User Interface],
  [*HMM*], [Hidden Markov Model],
  [*isWER*], [Intended Speech Word Error Rate],
  [*LLM*], [Large Langauge Model],
  [*LSTM*], [Long Short-Term Memory],
  [*MB*], [Multi-Branching],
  [*MC*], [Multi-Contextual],
  [*MFCC*], [Mel-Frequency Cepstral Coefficients],
  [*ML-CNN*], [Multi-Label Convolutional Neural Network],
  [*PCA*], [Principal Component Analysis],
  [*PyQT5*], [Python QT Framework],
  [*ReLU*], [Rectified Linear Unit],
  [*RNN*], [REcurrent Neural Network],
  [*Sep-28k*], [Stuttering Event Prediction Dataset (28,000 samples)],
  [*SLP*], [Speech-Language Pathologist],
  [*SOTA*], [State Of The Art],
  [*STFT*], [Short-Time Fourier Transform],
  [*STT*], [Speech-to-Text],
  [*SVM*], [Support Vector Machine],
  [*TTS*], [Text-to-Speech],
  [*UI*], [User Interface],
  [*VITS*], [Variational Inference With Adversarial Learning for end-to-end Text-to-Speech],
  [*WavLM*], [Wave Langauge Model],
  [*YOLO*], [You Only Look Once],
)

#pagebreak()

// Start doing header and footer
#let header_footer_line() = {
  line(length: 100%, stroke: 3pt + rgb("#5F1E1E"))
  v(-0.85em)
  line(length: 100%, stroke: 0.3pt + rgb("#5F1E1E"))
}

#set page(
  paper: "a4",
  margin: (top: 3cm, bottom: 2.5cm, x: 2.5cm),
  numbering: "1",
  
  // Header definition
  header: [
      #set text(9pt)
      “A Multi-Stage Deep Learning Framework for Syllable-Level Stuttering Localization, \
      Classification, and Remediation using Wav2Vec 2.0 and Ensembled Transformers” #h(1fr) 2025-26
      #v(-0.8em)
      #header_footer_line()
  ],
  
  // Footer definition
  footer: context [
    #set text(9pt)
    #header_footer_line()
    #v(-0.8em)
    Department of Artificial Intelligence & Machine Learning, V. C. E. T, Puttur. #h(1fr) Page #counter(page).display()
  ]
)

#counter(page).update(1)
#counter(heading).update(0)

#set heading(numbering: "1.1")

// --- Chapter 1: Introduction ---
#chapter_heading[INTRODUCTION]

== Introduction to the Project
Speech fluency is an essential aspect of human communication, allowing individuals to express thoughts and emotions clearly. However, many people face interruptions in their speech flow due to a disorder known as stuttering. It is a common speech disorder characterized by involuntary repetitions, prolongations, or blocks during speech production. This condition affects individuals across all age groups and can significantly impact their confidence, communication ability, and social interactions.

Early identification of stuttering patterns is crucial for effective therapy and
management. Traditionally, detection and assessment of stuttering rely on manual
observation by speech-language pathologists, which can be subjective, time-
consuming, and often inconsistent across experts. Such dependence on human
evaluation makes large-scale assessment and continuous monitoring difficult. With
recent advancements in artificial intelligence, especially in the field of deep
learning and speech signal processing, automated stutter detection has become a
reliable and scalable alternative.

The project aims to develop an intelligent system that can automatically detect and
classify various types of stuttering in speech, including repetitions, prolongations,
blocks, and fillers, using deep learning techniques. The input audio will be first
preprocessed and converted into embeddings, which will be then analyzed using
a CNN and Transformer-based architecture to learn both local acoustic patterns and
long-range temporal dependencies that differentiate normal and dysfluent speech.
Along with stutter classification, the same audio will be also passed through a
transcription module, and the detected stutter timestamps will be hard-aligned with
the transcript timestamps to identify the exact word, syllable, or speech segment where the dysfluency occurs. The final model will be integrated into a user-friendly
interface that allows users to record or upload speech samples and receive real-time
analysis, timestamped detection results, and classification output.

Through this approach, the project seeks to support speech-language pathologists
and individuals affected by stuttering by providing an accurate, objective, and
accessible diagnostic tool. It also contributes to the broader vision of integrating
artificial intelligence into healthcare to enhance diagnosis, therapy, and quality of
life.

== Existing System
Existing stutter detection systems primarily rely on manual identification or basic
signal processing methods. Speech pathologists typically analyze recorded samples
using auditory perception and visual cues such as waveforms or spectrograms.
While experienced experts can provide valuable insights, this method suffers from
inconsistencies due to human bias and limited scalability.

Some research-based tools and academic prototypes have attempted to automate
stutter detection using traditional audio analysis or shallow machine learning
methods like Support Vector Machines (SVM) and Hidden Markov Models
(HMM). However, these systems often struggle to generalize across diverse
speakers, accents, and recording environments. Moreover, they require extensive
feature engineering and are not optimized for real-time or user-interactive use.
Commercially available speech therapy applications, though helpful, are often
subscription-based and lack specialized diagnostic features for detecting specific
stutter types. Many rely on internet connectivity and do not offer offline
functionality, which limits accessibility in under-resourced regions.

These limitations highlight the need for an advanced, reliable, and accessible stutter
detection system. A deep learning–based approach using Convolutional Neural
Networks (CNNs) and audio spectrogram analysis can automatically learn 

== Proposed System
The proposed system introduces an intelligent and automated framework for
detecting and classifying stuttering patterns from speech recordings. It will
leverages deep learning, specifically a multi-model binary classification
architecture, to identify various dysfluency types effectively. The workflow will
begin with capturing or uploading speech recordings. The audio will be then pre-
processed and converted into embeddings, which will be analyzed using a CNN and
Transformer-based model to capture both local acoustic patterns and long-range
temporal dependencies. These learned representations will be used to detect five
independent stuttering behaviors: repetitions, prolongations, blocks, interjections,
and word-level dysfluencies.

Along with stutter detection, the same audio will also passed through a transcription
model to generate timestamped speech output. The detected stutter timestamps will
be then hard-aligned with the transcript timestamps to identify the exact word,
syllable, or speech segment associated with the dysfluency. Each model will output
a probability score indicating the presence of a specific stutter class, and the results
are compiled into a final classification report. The trained model will be integrated
into a desktop-based PyQt5 interface, allowing users to record speech, upload audio
files, and visualize results in real time. The interface will also provides playback
controls and waveform displays for better interpretability. By combining advanced
speech signal processing, deep learning-based classification, and timestamp
alignment with transcription, this system aims to deliver a complete end-to-end
solution that will not only detect stuttering automatically but also support clinicians
and individuals in monitoring progress and planning effective therapy interventions.

== Scope of the Project
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

== Objectives of the Project
The primary goal of this project is to create an automated and accurate system for detecting and classifying stuttering in speech recordings using deep learning techniques.  The key objectives are:
- *Speech Representation Pipeline*: To develop a robust pipeline using the Wave2Vec framework to extract deep, context-aware embeddings from raw audio without manual feature engineering.
- *Hybrid CNN + Transformer Architecture*: To design and train a model that leverages both local acoustic patterns and global temporal context for accurate stutter detection and classification.
- *Specialized Classifiers For Each Stutter Class*: To build individual models for Prolongation, Block, Sound Repetition, Word Repetition, and Interjection to maximize class-wise accuracy.
- *Temporal Localization Module*: To identify the exact timestamps of stuttering events, the affected region, and the duration and frequency of occurrences within a speech sample.
- *Visual Annotation System*: To mark stutter-indicative regions on the waveform/spectrogram for an interpretable and clinician-friendly analysis interface.

== Organization of the Report
This report is organized into two chapters, each presenting a structured explanation of the project’s goal and the research carried out to support the future implementation of the project.
- *Chapter 1* introduces the project, its motivation, objectives, and overall structure. It outlines the problem of stuttering, the need for automation, and
- *Chapter 2* presents the Literature Survey, reviewing previous work, existing speech analysis tools, and limitations in current methods.

#pagebreak()

#chapter_heading[LITERATURE SURVEY]

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
  The system was trained and validated on the SEP-28k dataset — one of the largest available stuttering corpora — achieving over 91% classification accuracy.
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

== Summary of Literature Survey
This section presents a summary of the reviewed literature on automated stutter
detection and classification. The collective findings establish that deep learning,
particularly CNN- and transformer-based architectures, represents the state of the art in
this domain.

Early approaches relied heavily on manual inspection or handcrafted acoustic features, whereas recent advances in embedding-based speech representations, spectrogram-driven CNNs, temporal modeling, and transfer learning have achieved significant improvements in detection accuracy and generalization. Studies such as YOLO- Stutter and FluentNet highlight the importance of region-wise and temporal feature learning, while works like Stutter-TTS and Wang et al. emphasize the impact of data augmentation and dataset balancing. Furthermore, explainability and multilingual adaptability, as demonstrated in Stutter-Solver and Ghosh et al., ensure that these systems remain transparent, interpretable, and globally applicable. Collectively, the insights summarized in Table 2.1 form the scientific and technical foundation for the proposed “A Multi-Stage Deep Learning Framework for Syllable-Level Stuttering Localization, Classification, and Remediation using Wav2Vec 2.0 and Ensembled Transformers” guiding model design, feature extraction, timestamp alignment, and evaluation strategies to develop an effective, scalable, and user-accessible speech pathology support system.

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
