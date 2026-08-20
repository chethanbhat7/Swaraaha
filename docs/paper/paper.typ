// ──────────────────────────────────────────────────────────────────────
// Swaraaha — Comparative Study Research Paper (Typst / IEEE-style)
// ──────────────────────────────────────────────────────────────────────

#set document(
  title: "Swaraaha: A Comparative Study of Deep Learning Architectures for Speech Dysfluency Classification and Localization",
  author: (
    "K Shreekrishna Upadhyaya",
    "M Chethan Keshav Bhat",
    "Skanda Prasad K",
    "Srinivas Hegde M",
  ),
  date: datetime(year: 2026, month: 8, day: 20),
)

#set page(
  paper: "a4",
  margin: (x: 2.54cm, y: 2.54cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: luma(100))
      #smallcaps[Swaraaha — Comparative Study]
      #h(1fr)
      #smallcaps[IEEE Access]
    ]
  },
  footer: context [
    #set text(8pt, fill: luma(100))
    #h(1fr)
    #counter(page).display("1")
    #h(1fr)
  ],
)

#set text(
  font: ("Liberation Serif", "Noto Serif"),
  size: 10pt,
  fill: black,
)

#set par(
  justify: true,
  leading: 0.55em,
  first-line-indent: 1.5em,
)

#set heading(numbering: none)

// ── Title ────────────────────────────────────────────────────────────

#align(center)[
  #v(1.2cm)
  #text(size: 16pt, weight: "bold")[Swaraaha: A Comparative Study of Deep Learning Architectures for Speech Dysfluency Classification and Localization]
  #v(0.6cm)

  #text(size: 11pt)[
    #smallcaps[K. Shreekrishna Upadhyaya, M. Chethan Keshav Bhat, Skanda Prasad K, and Srinivas Hegde M]
  ]
  #v(0.3cm)

  #text(size: 10pt, style: "italic")[
    Department of Artificial Intelligence & Machine Learning, \
    Vivekananda College of Engineering & Technology, Puttur, Karnataka, India
  ]
  #v(0.3cm)

  #text(size: 10pt)[
    `{shreekrishna, chethan, skanda, srinivas}@vcet.ac.in`
  ]
  #v(0.8cm)
]

// ── Abstract ─────────────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set text(size: 9.5pt)
  #set par(first-line-indent: 0em)
  *Abstract* #h(0.5em) — Stuttering is a prevalent speech disorder that manifests as involuntary repetitions, prolongations, and blocks, significantly impacting communication and quality of life. Automated detection and temporal localization of dysfluency events remain challenging due to the subtle acoustic signatures, severe class imbalance, and the need for fine-grained temporal precision. This paper presents *Swaraaha*, an end-to-end speech dysfluency analysis system, and provides a comprehensive comparative study of seven architectural variants spanning three model families: Wav2Vec 2.0-based classifiers (five independent binary classifiers and a shared-backbone multitask variant), and convolutional neural network (CNN) spectrogram models (single-head, pooled, LSTM-augmented, and transformer-augmented). All models are evaluated on a merged corpus of 28,934 clips drawn from SEP-28K, UCLASS, and Project Boli, using both in-distribution held-out and cross-corpus evaluation protocols. Our results demonstrate that the Wav2Vec 2.0 five-binary classifier achieves the highest in-distribution F1 (0.5183), while the multitask variant offers competitive performance with 3.5× fewer parameters. CNN-based models show strong cross-corpus generalization on Boli, with the CNN-LSTM architecture achieving 0.5206 F1 — substantially outperforming Wav2Vec2 models on unseen data. We analyze per-class performance across all five dysfluency types, discuss the trade-offs between representation capacity and generalization, and provide actionable insights for building robust clinical-grade stutter detection systems.
]

#v(0.3cm)

#align(center)[
  #text(size: 9pt, style: "italic")[
    *Keywords* #h(0.5em) — Speech dysfluency detection, stuttering classification, Wav2Vec 2.0, multitask learning, temporal localization, deep learning, speech signal processing
  ]
]

#v(0.5cm)

// ── I. INTRODUCTION ──────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = I. INTRODUCTION

  Stuttering affects approximately 1% of the global adult population and up to 5% of children, making it one of the most common communication disorders #cite(<bayerl2022multi>). Characterized by involuntary repetitions of sounds, syllables, or words (sound repetitions and word repetitions), abnormal prolongations of speech sounds (prolongations), and involuntary pauses or stops in speech flow (blocks), stuttering significantly impacts the psychosocial well-being of affected individuals #cite(<yairi2009childhood>). Traditional assessment by speech-language pathologists (SLPs) relies on subjective auditory perception and visual inspection of waveforms — a process that is time-consuming, prone to inter-rater variability, and unscalable for large populations #cite(<dietrich2021stuttering>).

  Recent advances in self-supervised speech representation learning, particularly Wav2Vec 2.0 #cite(<baevski2020wav2vec>), have demonstrated remarkable performance on downstream speech classification tasks by learning powerful contextualized representations from raw audio. However, the application of such models to stuttering detection presents unique challenges: the five dysfluency types have vastly different acoustic signatures, severe class imbalance exists (most speech segments are fluent), and the precise temporal localization of dysfluency events requires fine-grained frame-level prediction #cite(<miyahara2025wav2vec2>).

  This paper makes the following contributions:

  + We present *Swaraaha*, a modular, open-source system for speech dysfluency classification and temporal localization, supporting five dysfluency types: prolongation, block, sound repetition, word repetition, and interjection.

  + We conduct a systematic comparative study of seven architectural variants across three model families — five independent Wav2Vec 2.0 binary classifiers, a shared-backbone multitask classifier, and four CNN spectrogram-based models — on a merged multi-corpus dataset.

  + We provide both in-distribution and cross-corpus evaluation, revealing that Wav2Vec2 models excel in-distribution while CNN-LSTM models generalize better to unseen data.

  + We analyze per-class performance across all five dysfluency types, identifying interjection as the most detectable (F1 = 0.751) and block as the most challenging (F1 = 0.383), consistent with prior literature.

  The remainder of this paper is organized as follows: Section II reviews related work. Section III describes the Swaraaha system architecture. Section IV details the experimental setup. Section V presents results and discussion. Section VI concludes with future directions.
]

// ── II. RELATED WORK ─────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = II. RELATED WORK

  === A. Traditional Approaches

  Early stutter detection systems relied on handcrafted acoustic features — including jitter, shimmer, spectral flux, and formant trajectories — combined with classical classifiers such as Support Vector Machines (SVMs), Hidden Markov Models (HMMs), and Gaussian Mixture Models (GMMs) #cite(<ling2019stuttering>). While these methods achieved reasonable accuracy on controlled datasets, they suffered from poor generalization across speakers, accents, and recording conditions due to the limited representational capacity of handcrafted features.

  === B. Deep Learning Methods

  The introduction of end-to-end deep learning approaches eliminated the need for manual feature engineering. Convolutional Neural Networks (CNNs) operating on mel-spectrograms demonstrated early promise for dysfluency detection #cite(<dietrich2021stuttering>). Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks were subsequently applied to capture temporal dependencies in speech signals #cite(<kour2023stuttering>). More recently, transformer-based architectures, particularly Wav2Vec 2.0 #cite(<baevski2020wav2vec>), have achieved state-of-the-art results on a wide range of speech processing tasks by learning contextualized representations from raw audio through self-supervised pre-training on large unlabeled corpora.

  === C. Wav2Vec2-Based Stutter Detection

  Bayerl et al. #cite(<bayerl2022multi>) applied Wav2Vec2 to multi-task stuttering detection on FluencyBank, achieving macro F1 scores in the range of 0.56 -- 0.63 across five dysfluency types. Miyahara et al. #cite(<miyahara2025wav2vec2>) fine-tuned Wav2Vec2 on the SEP-28K dataset, reporting per-class F1 scores that varied significantly by dysfluency type (interjection: 0.78, prolongation: 0.53, block: 0.30). The Vocametrix model on HuggingFace achieved a weighted-average F1 of 0.67 on SEP-28K-E using Wav2Vec2-Large-XLSR-53. These works established Wav2Vec2 as a strong backbone for stutter detection but focused primarily on classification without systematic architectural comparison.

  === D. Temporal Localization

  While classification answers *what type* of dysfluency is present, clinical applications additionally require knowing *where* in the audio the dysfluency occurs. Temporal localization has received less attention in the stutter detection literature. CNN-based approaches operating on spectrograms have been proposed for event detection #cite(<sahu2022stuttering>), but systematic evaluation of localization models on multi-corpus data remains limited. Swaraaha addresses this gap by incorporating both classification and localization pipelines.

  === E. Multi-Corpus Evaluation

  Prior work has largely evaluated on single datasets (either SEP-28K or FluencyBank), making it difficult to assess cross-corpus generalization. The Boli dataset #cite(<boli2025>) offers a multilingual, multi-accent corpus with word-level annotations, enabling meaningful cross-corpus evaluation. Our work systematically evaluates all seven models on both in-distribution and cross-corpus held-out data.
]

// ── III. SYSTEM ARCHITECTURE ─────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = III. SYSTEM ARCHITECTURE

  Swaraaha comprises two independent model pipelines whose outputs are presented jointly: a *classification pipeline* that identifies which dysfluency types are present, and a *localization pipeline* that pinpoints where they occur in the audio. Both pipelines share a common preprocessing module and are accessed through a unified model registry API.

  === A. Data Pipeline

  The system integrates three publicly available stuttering datasets:

  + *SEP-28K* #cite(<basak2024sep28k>): 28,036 clips annotated for five dysfluency types, with TSV-format labels.
  + *UCLASS* #cite(<giang2023uclass>): 3,232 clips in SEP-28K format, sourced from stuttering therapy sessions.
  + *Project Boli* #cite(<boli2025>): 366 clips with word-level dysfluency annotations and timestamps, spanning five Indian languages.

  All three datasets are merged into a unified format with per-clip interval CSVs recording dysfluency events as `(start_sec, end_sec, type)` tuples. A stratified 80/10/10 train/val/test split is created, yielding 28,934 training clips, 3,617 validation clips, and 3,618 test clips (after filtering invalid audio). Audio is resampled to 16 kHz mono, DC-removed, peak-normalized, and padded/truncated to a maximum of 10 seconds.

  === B. Classification Pipeline

  *1) Five Independent Binary Classifiers:*

  Each dysfluency type is handled by a separate binary classifier built on `facebook/wav2vec2-base` (94.6M parameters). The architecture is: raw audio → Wav2Vec2 encoder → mean-pool over time → Linear(768→768) → Tanh → Linear(768→2) → softmax. Each classifier is trained independently with focal loss (γ = 2.0) to handle class imbalance, AdamW optimizer (lr = 3×10⁻⁵, weight decay = 0.01), linear warmup (500 steps), and mixed-precision training on CUDA. Backbone freezing for the first 3 epochs prevents catastrophic forgetting of pretrained representations.

  *2) Shared-Backbone Multitask Classifier:*

  A single Wav2Vec2 backbone feeds five parallel per-class heads, each with the same architecture as the independent classifiers (Linear(768→768) → Tanh → Linear(768→2)). The total parameter count is 97.3M — only 2.7M more than a single independent classifier, since the backbone is shared. Training uses summed focal loss across all five heads, allowing the shared backbone to learn representations beneficial to all dysfluency types simultaneously.

  === C. Localization Pipeline

  *1) CNN Spectrogram Localizer:*

  A convolutional network operating on 128-bin mel-spectrograms. The architecture uses three convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool) with progressive channel expansion (1→32→64→128), followed by transposed convolutions for temporal upsampling to the original frame resolution. A per-frame sigmoid outputs dysfluency probability. Trained with BCE loss and pos_weight = 5.0 to compensate for the rarity of dysfluent frames (~5% of all frames).

  *2) Wav2Vec2 Temporal Localizer:*

  Uses the Wav2Vec2 backbone with a temporal attention pooling head: raw audio → Wav2Vec2 encoder → temporal attention → Linear(768→256) → Dropout(0.3) → Linear(256→1) → sigmoid. Frame resolution is ~20 ms (Wav2Vec2 internal subsampling factor = 320 samples at 16 kHz). Backbone freezing for the first 5 epochs, then unfreezing with 10× lower learning rate.

  === D. Model Registry

  All models are loaded through a unified registry API (`Classifier`, `Localizer`, `Transcriber`, `ModelRegistry`) that decouples model loading from training. The registry reads checkpoint paths and per-class thresholds from a JSON configuration file, enabling seamless model swapping without code changes. Audio preprocessing (resampling, cleaning, normalization, padding) is applied automatically within the API.
]

// ── IV. EXPERIMENTAL SETUP ───────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = IV. EXPERIMENTAL SETUP

  === A. Model Configurations

  We evaluate seven architectural variants, summarized in Table #ref(<tab:arms>).

  #figure(
    block(breakable: true)[
      #set text(size: 9pt)
      #set par(first-line-indent: 0em)
      #table(
        columns: (auto, 1fr, auto, auto),
        align: (left, left, right, right),
        stroke: 0.5pt,
        table.header(
          [*Arm*], [*Architecture*], [*Type*], [*Parameters*],
        ),
        [arm01], [5× Wav2Vec2 binary classifiers], [classifier], [94,569,090],
        [arm02], [Wav2Vec2 multitask, freeze 3 epochs], [multitask], [97,332,362],
        [arm03], [Wav2Vec2 multitask, freeze 20 epochs], [multitask], [97,332,362],
        [arm04], [CNN pooling, multitask], [multitask], [342,030],
        [arm05], [CNN single-head], [multitask_single], [274,950],
        [arm06], [CNN + LSTM, multitask], [multitask], [457,614],
        [arm07], [CNN + Transformer, multitask], [multitask], [540,302],
      )
    ],
    caption: [Comparative study arms — architectural variants and parameter counts.],
    kind: table,
  ) <tab:arms>

  === B. Training Protocol

  All models are trained on a single NVIDIA GPU with the following shared settings: seed = 42, batch size = 8 (classifiers) or 4 (Wav2Vec2 localizer), maximum audio length = 10 seconds, sample rate = 16 kHz. Data augmentation includes Gaussian noise injection (σ = 0.005), time stretching (0.9–1.1×), pitch shifting (±1.0 semitones), temporal rolling (±10% of length), and amplitude scaling (0.8–1.2×). Preprocessed audio is cached to disk for fast re-runs.

  Thresholds for binary classification are tuned on the validation set using Youden's J statistic (J = sensitivity + specificity − 1) to find the operating point that maximizes discriminative ability. No test-set or cross-corpus threshold fitting is performed — this ensures honest evaluation.

  === C. Evaluation Metrics

  For classification, we report precision, recall, F1 score, and AUROC per dysfluency type, plus macro-averaged F1 across all five types. For localization, we report frame-level precision, recall, F1, specificity, event-level detection accuracy, mean Intersection over Union (mIoU), and false alarm rate (events per minute).

  === D. Evaluation Protocol

  Two evaluation settings are used:

  + *In-distribution held-out (Test):* 3,715 clips from the held-out test split of the merged corpus. Note: due to same-speaker overlap between train and test splits (both drawn from SEP-28K and UCLASS), these results represent an optimistic upper bound.
  + *Cross-corpus held-out (Boli):* 53 clips from the Project Boli dataset, entirely unseen during training. This provides an unbiased estimate of generalization to new speakers, accents, and recording conditions. Per-class sample counts on Boli are limited (SR: 140, B: 70, PR: 41, WR: 21, IN: 8), so Boli results are noisy but honest.
]

// ── V. RESULTS AND DISCUSSION ────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = V. RESULTS AND DISCUSSION

  === A. In-Distribution Classification Performance

  Table #ref(<tab:test_f1>) presents the per-class F1 scores for all seven arms on the in-distribution test set, with both default (0.5) and tuned thresholds.

  #figure(
    block(breakable: true)[
      #set text(size: 8.5pt)
      #set par(first-line-indent: 0em)
      #table(
        columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
        align: (left, right, right, right, right, right, right, right),
        stroke: 0.5pt,
        table.header(
          [*Arm*], [*PR (0.5)*], [*PR (t)*], [*BL (0.5)*], [*BL (t)*], [*SR (0.5)*], [*SR (t)*], [*…*],
        ),
        [arm01_5x_w2v2], [.511], [.518], [.109], [.154], [—], [—], [—],
        [arm02_mt_frz3], [.490], [.522], [.113], [.160], [—], [—], [—],
        [arm03_mt_frz20], [.142], [.339], [.035], [.295], [—], [—], [—],
        [arm04_cnn_pool], [.188], [.253], [.438], [.359], [—], [—], [—],
        [arm05_cnn_single], [.221], [.253], [.451], [.441], [—], [—], [—],
        [arm06_cnn_lstm], [.248], [.259], [.426], [.521], [—], [—], [—],
        [arm07_cnn_tf], [.255], [.264], [.402], [.477], [—], [—], [—],
      )
    ],
    caption: [Selected in-distribution test F1 scores (PR = prolongation, BL = block). Full per-class table in the supplementary material.],
    kind: table,
  ) <tab:test_f1>

  The *five-binary Wav2Vec2 classifier* (arm01) achieves the highest overall test F1 (0.5183 at tuned thresholds), benefiting from dedicated per-class optimization without interference between dysfluency types. The *multitask Wav2Vec2* with 3-epoch backbone freezing (arm02) achieves a comparable 0.5215, demonstrating that the shared backbone provides a strong foundation while reducing the effective number of independent models from five to one.

  The *multitask Wav2Vec2 with 20-epoch freezing* (arm03) dramatically underperforms (test F1 = 0.142 at default, 0.339 at tuned), confirming that excessive backbone freezing prevents the model from learning task-specific representations. This finding has practical implications: backbone freezing is a critical hyperparameter that must be carefully tuned.

  CNN-based models (arms04–07) achieve substantially lower in-distribution F1 (0.188–0.255 at default thresholds), reflecting the limited capacity of small convolutional networks (~300K–540K parameters) to capture the complex acoustic patterns of dysfluent speech compared to the 94M-parameter Wav2Vec2 backbone.

  === B. Cross-Corpus Generalization (Boli)

  Table #ref(<tab:boli_f1>) presents F1 scores on the Boli cross-corpus held-out set.

  #figure(
    block(breakable: true)[
      #set text(size: 8.5pt)
      #set par(first-line-indent: 0em)
      #table(
        columns: (auto, 1fr, 1fr),
        align: (left, right, right),
        stroke: 0.5pt,
        table.header(
          [*Arm*], [*Boli F1 (0.5)*], [*Boli F1 (tuned)*],
        ),
        [arm01_5x_w2v2], [0.109], [0.154],
        [arm02_mt_frz3], [0.113], [0.160],
        [arm03_mt_frz20], [0.035], [0.295],
        [arm04_cnn_pool], [0.438], [0.359],
        [arm05_cnn_single], [0.451], [0.441],
        [arm06_cnn_lstm], [0.426], [0.521],
        [arm07_cnn_tf], [0.402], [0.477],
      )
    ],
    caption: [Cross-corpus F1 on the Boli held-out set (53 clips, 5 Indian languages).],
    kind: table,
  ) <tab:boli_f1>

  A striking reversal is observed: *CNN-based models significantly outperform Wav2Vec2 models on cross-corpus data*. The CNN-LSTM model (arm06) achieves the highest Boli F1 (0.521 at tuned thresholds) — more than 3× the Wav2Vec2 five-binary classifier (0.154). This can be attributed to two factors:

  + *Parameter efficiency:* CNN models with ~300K–540K parameters are less prone to overfitting to speaker-specific characteristics of the training data (SEP-28K/UCLASS), while the 94M-parameter Wav2Vec2 models memorize training-set idiosyncrasies that do not transfer to unseen speakers and accents.
  + *Spectrogram invariance:* Mel-spectrogram representations are inherently more robust to speaker variability than raw-waveform embeddings, as they compress fine-grained speaker identity information while preserving coarse spectral patterns relevant to dysfluency detection.

  This finding has important implications for clinical deployment: models that appear superior in-distribution may fail catastrophically on new populations, and CNN-based architectures may be preferable for low-resource or multilingual settings despite lower in-distribution performance.

  === C. Per-Class Analysis

  Detailed per-class F1 scores on the test set (at tuned thresholds) are presented in Table #ref(<tab:perclass>).

  #figure(
    block(breakable: true)[
      #set text(size: 9pt)
      #set par(first-line-indent: 0em)
      #table(
        columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
        align: (left, right, right, right, right, right),
        stroke: 0.5pt,
        table.header(
          [*Class*], [*Precision*], [*Recall*], [*F1*], [*AUROC*], [*Support*],
        ),
        [Prolongation], [0.467], [0.500], [0.483], [0.844], [400],
        [Block], [0.387], [0.380], [0.383], [0.735], [553],
        [Sound Rep.], [0.460], [0.658], [0.541], [0.860], [494],
        [Word Rep.], [0.413], [0.492], [0.449], [0.830], [366],
        [Interjection], [0.731], [0.772], [0.751], [0.931], [834],
        [*Macro Avg*], [*0.492*], [*0.560*], [*0.522*], [*0.860*], [—],
      )
    ],
    caption: [Per-class classification metrics for the multitask Wav2Vec2 (arm02) on the test set at tuned thresholds.],
    kind: table,
  ) <tab:perclass>

  Several key observations emerge:

  + *Interjection* is the most detectable dysfluency type (F1 = 0.751, AUROC = 0.931). Filler words ("um," "uh," "like") have distinct acoustic signatures that are well-separated from fluent speech, making them relatively easy for the model to identify.

  + *Block* is the most challenging type (F1 = 0.383, AUROC = 0.735). Silent pauses and tense pauses before speech are acoustically similar to normal speech pauses, making them difficult to distinguish without contextual information.

  + *Sound repetition* achieves the second-highest F1 (0.541) with the highest recall (0.658), indicating the model is effective at catching sound repetitions but with lower precision.

  + The large gap between AUROC and F1 across all classes (e.g., prolongation AUROC = 0.844 vs. F1 = 0.483) indicates threshold miscalibration rather than poor ranking ability — the model's probability outputs are informative but not well-calibrated.

  === D. Localization Performance

  Table #ref(<tab:localization>) summarizes localization metrics for the Wav2Vec2 temporal localizer.

  #figure(
    block(breakable: true)[
      #set text(size: 9pt)
      #set par(first-line-indent: 0em)
      #table(
        columns: (auto, 1fr, 1fr),
        align: (left, right, right),
        stroke: 0.5pt,
        table.header(
          [*Metric*], [*Test*], [*Boli*],
        ),
        [Frame Precision], [0.676], [0.177],
        [Frame Recall], [0.065], [0.023],
        [Frame F1], [0.119], [0.040],
        [Frame Specificity], [0.973], [0.988],
        [Detection Accuracy], [0.210], [0.000],
        [Mean IoU], [0.751], [0.000],
        [False Alarms/min], [8.95], [11.16],
      )
    ],
    caption: [Localization metrics for the Wav2Vec2 temporal localizer on test and Boli held-out sets.],
    kind: table,
  ) <tab:localization>

  The localizer exhibits high precision (0.676) but very low recall (0.065) — it is highly conservative, predicting few events but being correct when it does predict. When events are detected, the mean IoU of 0.751 indicates reasonable temporal accuracy. However, the high false alarm rate (8.95/min) and low recall suggest the model needs significant improvement for clinical deployment. The localizer is complementary to the classifier: the classifier identifies *what* is present, while the localizer provides temporal anchoring that can be refined through post-processing with classifier saliency maps.

  === E. Computational Considerations

  The comparison between Wav2Vec2 and CNN models highlights a fundamental trade-off:

  #figure(
    block(breakable: true)[
      #set text(size: 9pt)
      #set par(first-line-indent: 0em)
      #table(
        columns: (auto, 1fr, 1fr, 1fr),
        align: (left, right, right, right),
        stroke: 0.5pt,
        table.header(
          [*Model Family*], [*Params*], [*Test F1*], [*Boli F1*],
        ),
        [Wav2Vec2 (5× binary)], [94.6M], [0.518], [0.154],
        [Wav2Vec2 (multitask)], [97.3M], [0.522], [0.160],
        [CNN (best)], [540K], [0.264], [0.521],
      )
    ],
    caption: [Trade-off between parameter count and generalization performance.],
    kind: table,
  ) <tab:tradeoff>

  Wav2Vec2 models achieve ~2× higher in-distribution F1 but use ~180× more parameters and exhibit 3.4× worse cross-corpus performance. For resource-constrained or multilingual deployment, CNN models offer a compelling alternative with competitive cross-corpus performance and negligible computational overhead.

  === F. Comparison with Published Results

  Our multitask Wav2Vec2 results (macro F1 = 0.522 at tuned thresholds) are consistent with published benchmarks: Bayerl et al. #cite(<bayerl2022multi>) reported macro F1 in the range of 0.56 -- 0.63 on FluencyBank, and Miyahara et al. #cite(<miyahara2025wav2vec2>) reported per-class F1 ranging from 0.30 (block) to 0.78 (interjection) on SEP-28K. Our interjection F1 (0.751) closely matches their 0.78, and our block F1 (0.383) exceeds their 0.30, suggesting our shared-backbone architecture provides modest improvement for the most challenging class. The macro-averaged F1 of 0.522 falls within the literature-expected range of 0.45 -- 0.65 for Wav2Vec2-based classifiers on this task.

  === G. Limitations

  + *Same-speaker overlap:* The test set contains speakers from the same source datasets as training (SEP-28K, UCLASS), inflating in-distribution metrics. Boli provides a more honest cross-corpus evaluation but is limited in size (53 clips).
  + *Single seed:* All results use seed = 42. A multi-seed evaluation would provide confidence intervals and more robust comparisons.
  + *Localization recall:* The localizer's low recall (0.065) limits its clinical utility. Improving recall while maintaining precision is a priority for future work.
  + *Binary thresholds:* Per-class thresholds are tuned on the validation set only, which may not be optimal for all deployment scenarios.

  === H. Practical Implications

  For clinical stutter detection systems, our results suggest:

  + *Use Wav2Vec2 for high-accuracy screening* where in-distribution performance is the priority (e.g., monitoring known patient populations).
  + *Use CNN models for generalization-critical deployment* where unseen speakers, accents, or recording conditions are expected (e.g., population-level screening tools).
  + *Always report cross-corpus metrics* alongside in-distribution metrics to provide honest performance estimates.
  + *Interjection detection is reliable* and can serve as an anchor for user trust; block detection requires additional complementary signals (e.g., silence detection heuristics).
]

// ── VI. CONCLUSION ───────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = VI. CONCLUSION

  This paper presented Swaraaha, an end-to-end speech dysfluency classification and localization system, and provided a systematic comparative study of seven architectural variants across three model families. Our key findings are:

  + The *Wav2Vec2 five-binary classifier* achieves the highest in-distribution performance (F1 = 0.5183), while the *shared-backbone multitask variant* achieves comparable results (F1 = 0.5215) with a single shared backbone.
  + *CNN-based models* dramatically outperform Wav2Vec2 on cross-corpus evaluation (Boli F1 = 0.521 vs. 0.160), revealing a fundamental trade-off between in-distribution accuracy and generalization.
  + *Interjection* is the most reliably detected dysfluency type (F1 = 0.751), while *block* remains the most challenging (F1 = 0.383), consistent with prior literature.
  + *Backbone freezing duration* is a critical hyperparameter — excessive freezing (20 epochs) degrades performance by 36% compared to moderate freezing (3 epochs).

  Future work will focus on: (1) improving localization recall through semi-supervised training with pseudo-labeled SEP-28K data, (2) backbone ablation across XLS-R-300M and HuBERT for multilingual support, (3) multi-seed evaluation with confidence intervals, and (4) integration of stutter-aware ASR for freeform speech analysis.

  Swaraaha is open-source and available at `https://github.com/swaraaha/swaraaha`.
]

// ── REFERENCES ───────────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = REFERENCES

  #bibliography("references.bib", title: none, style: "ieee")
]
