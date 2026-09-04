// ──────────────────────────────────────────────────────────────────────
// Swaraaha — Comparative Study Research Paper (Typst / IEEE-style)
// ──────────────────────────────────────────────────────────────────────

#set document(
  title: "Swaraaha: A Comparative Study of Deep Learning Architectures for Speech Dysfluency Classification",
  author: (
    "Ajay Shastry C G",
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
  #text(size: 16pt, weight: "bold")[Swaraaha: A Comparative Study of Deep Learning Architectures for Speech Dysfluency Classification]
  #v(0.6cm)

  #text(size: 11pt)[
    #smallcaps[Ajay Shastry C G, K. Shreekrishna Upadhyaya, M. Chethan Keshav Bhat, Skanda Prasad K, and Srinivas Hegde M]
  ]
  #v(0.3cm)

  #text(size: 10pt, style: "italic")[
    Department of Artificial Intelligence & Machine Learning, \
    Vivekananda College of Engineering & Technology, Puttur, Karnataka, India
  ]
  #v(0.3cm)

  #text(size: 10pt)[
    `{ajayshastrycg.ai, 4vp23ai020, 4vp23ai023, 4vp23ai051, 4vp23ai054}@vcetputtur.ac.in`
  ]
  #v(0.8cm)
]

// ── Abstract ─────────────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set text(size: 9.5pt)
  #set par(first-line-indent: 0em)
  *Abstract* #h(0.5em) -- Stuttering is a speech disorder that manifests as involuntary repetitions, prolongations, and blocks, impacting communication and quality of life. Automated detection of dysfluency events is challenging due to the subtle acoustic signatures and class imbalance across dysfluency types. This paper presents *Swaraaha*, an end-to-end speech dysfluency classification system, and compares seven experimental arms spanning three model families: Wav2Vec 2.0-based classifiers (five independent binary classifiers and a shared-backbone multitask variant), and convolutional neural network (CNN) spectrogram models (single-head, pooled, LSTM-augmented, and transformer-augmented). All models are evaluated on a merged corpus of 37,087 clips drawn from SEP-28K, UCLASS, and Project Boli, using both in-distribution held-out and cross-corpus evaluation protocols. The shared-backbone Wav2Vec 2.0 multitask classifier achieves the highest in-distribution macro F1 (0.5215), closely followed by the five independent binary classifiers (0.5183), while requiring ≈4.9× fewer parameters than five separate models. CNN-based models generalize better on Boli, with the CNN-LSTM architecture achieving 0.5206 F1, outperforming Wav2Vec2 models on unseen data. We analyze per-class performance across all five dysfluency types, discuss the trade-offs between representation capacity and generalization, and provide guidance for building clinical-grade stutter detection systems.
]

#v(0.3cm)

#align(center)[
  #text(size: 9pt, style: "italic")[
    *Keywords* #h(0.5em) -- Speech dysfluency detection, stuttering classification, Wav2Vec 2.0, multitask learning, deep learning, speech signal processing
  ]
]

#v(0.5cm)

// ── I. INTRODUCTION ──────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = I. INTRODUCTION

  Stuttering affects approximately 1% of the global adult population and up to 5% of children #cite(<bayerl2022multi>). Characterized by involuntary repetitions of sounds, syllables, or words (sound repetitions and word repetitions), abnormal prolongations of speech sounds (prolongations), and involuntary pauses or stops in speech flow (blocks), it impacts the psychosocial well-being of affected individuals #cite(<yairi2009childhood>). Traditional assessment by speech-language pathologists (SLPs) relies on subjective auditory perception and visual inspection of waveforms, a process that is time-consuming, prone to inter-rater variability, and unscalable for large populations #cite(<dietrich2021stuttering>).

  Recent advances in self-supervised speech representation learning, particularly Wav2Vec 2.0 #cite(<baevski2020wav2vec>), have achieved strong performance on downstream speech classification tasks by learning contextualized representations from raw audio. Applying such models to stuttering detection presents challenges: the five dysfluency types have different acoustic signatures, and severe class imbalance exists (most speech segments are fluent) #cite(<miyahara2025wav2vec2>).

  This paper makes the following contributions:

  + We present *Swaraaha*, a modular, open-source system for speech dysfluency classification, supporting five dysfluency types: prolongation, block, sound repetition, word repetition, and interjection.

  + We conduct a systematic comparative study of seven experimental arms across three model families: five independent Wav2Vec 2.0 binary classifiers, a shared-backbone multitask classifier, and four CNN spectrogram-based models, on a merged multi-corpus dataset.

  + We provide both in-distribution and cross-corpus evaluation, showing that Wav2Vec2 models perform better in-distribution while CNN-LSTM models generalize better to unseen data.

  + We analyze per-class performance across all five dysfluency types, identifying interjection as the most detectable (F1 = 0.751) and block as the most challenging (F1 = 0.383), consistent with prior literature.

  The remainder of this paper is organized as follows: Section II reviews related work, Section III describes the system architecture, Section IV details the experimental setup, Section V presents results and discussion, and Section VI concludes.
]

// ── II. RELATED WORK ─────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = II. RELATED WORK

  === A. Traditional Approaches

  Early stutter detection systems relied on handcrafted acoustic features, including jitter, shimmer, spectral flux, and formant trajectories, combined with classical classifiers such as Support Vector Machines (SVMs), Hidden Markov Models (HMMs), and Gaussian Mixture Models (GMMs) #cite(<ling2019stuttering>). These methods achieved reasonable accuracy on controlled datasets but generalized poorly across speakers, accents, and recording conditions due to the limited representational capacity of handcrafted features.

  === B. Deep Learning Methods

  The introduction of end-to-end deep learning approaches removed the need for manual feature engineering. Convolutional Neural Networks (CNNs) operating on mel-spectrograms showed early promise for dysfluency detection #cite(<dietrich2021stuttering>). Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks were applied to capture temporal dependencies in speech signals #cite(<kour2023stuttering>). More recently, transformer-based architectures, particularly Wav2Vec 2.0 #cite(<baevski2020wav2vec>), have achieved strong results on speech processing tasks by learning contextualized representations from raw audio through self-supervised pre-training on large unlabeled corpora.

  === C. Wav2Vec2-Based Stutter Detection

  Bayerl et al. #cite(<bayerl2022multi>) applied Wav2Vec2 to multi-task stuttering detection on FluencyBank, achieving macro F1 scores in the range of 0.56 -- 0.63 across five dysfluency types. Miyahara et al. #cite(<miyahara2025wav2vec2>) fine-tuned Wav2Vec2 on the SEP-28K dataset, reporting per-class F1 scores that varied significantly by dysfluency type (interjection: 0.78, prolongation: 0.53, block: 0.30). The Vocametrix model on HuggingFace achieved a weighted-average F1 of 0.67 on SEP-28K-E using Wav2Vec2-Large-XLSR-53.   These works established Wav2Vec2 as a strong backbone for stutter detection but focused on classification without systematic architectural comparison.

  === D. Multi-Corpus Evaluation

  Prior work has largely evaluated on single datasets (either SEP-28K or FluencyBank), making it difficult to assess cross-corpus generalization. The Boli dataset #cite(<boli2025>) offers a multilingual, multi-accent corpus with word-level annotations, enabling cross-corpus evaluation. We evaluate all seven models on both in-distribution and cross-corpus held-out data.
]

// ── III. SYSTEM ARCHITECTURE ─────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = III. SYSTEM ARCHITECTURE

  Swaraaha comprises a classification pipeline that identifies which dysfluency types are present in speech audio. The pipeline builds on a common preprocessing module and exposes models through a unified model registry API.

  === A. Data Pipeline

  The system integrates three publicly available stuttering datasets:

  + *SEP-28K* #cite(<basak2024sep28k>): 32,321 clips annotated for five dysfluency types, with TSV-format labels.
  + *UCLASS* #cite(<giang2023uclass>): 4,712 clips in SEP-28K format, sourced from stuttering therapy sessions.
  + *Project Boli* #cite(<boli2025>): 54 clips with word-level dysfluency annotations and timestamps, recorded by speakers of five Indian languages (Hindi, Marathi, Telugu, Bengali, and Assamese).

  Three datasets are merged into a unified corpus of 37,087 clips with per-clip interval CSVs recording dysfluency events as `(start_sec, end_sec, type)` tuples. Clips with missing or header-only (empty) audio files are removed, leaving 36,674 usable clips. A stratified 80/10/10 train/val/test split is created, yielding 29,296 training clips, 3,662 validation clips, and 3,716 test clips. To guarantee an honest cross-corpus evaluation, all Project Boli clips are pinned to the test split and never appear in the training or validation splits. Audio is resampled to 16 kHz mono, DC-removed, peak-normalized, silence-trimmed, and padded/truncated to a maximum of 3 seconds.

  === B. Classification Pipeline

  *1) Five Independent Binary Classifiers:*

  Each dysfluency type is handled by a separate binary classifier built on `facebook/wav2vec2-base` (94,569,090 parameters per classifier). The architecture follows the Hugging Face sequence-classification formulation: raw audio → Wav2Vec2 encoder → attention-mask-weighted mean pooling over time → Linear(768→2) → softmax. Each classifier is trained independently with focal loss (γ = 2.0) to handle class imbalance, AdamW optimizer (lr = 3×10⁻⁵, weight decay = 0.01), linear warmup (500 steps), and mixed-precision training on CUDA. Backbone freezing for the first 3 epochs prevents catastrophic forgetting of pretrained representations.

  *2) Shared-Backbone Multitask Classifier:*

  A single Wav2Vec2 encoder feeds five parallel per-class heads, each a small MLP (Linear(768→768) → Tanh → Linear(768→2)) operating on a temporal mean pool of the encoder's hidden states. The total parameter count is 97.3M, only 2.7M more than a single independent classifier, since the backbone is shared. Training uses summed focal loss across all five heads, allowing the shared backbone to learn representations useful for all dysfluency types.

  === C. Model Registry

  All models are loaded through a unified registry API (`Classifier`, `Localizer`, `Transcriber`, `ModelRegistry`) that decouples model loading from training. The registry reads checkpoint paths and per-class thresholds from a JSON configuration file, enabling model swapping without code changes. Audio preprocessing (resampling, cleaning, normalization, padding) is applied automatically within the API.
]

// ── IV. EXPERIMENTAL SETUP ───────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = IV. EXPERIMENTAL SETUP

  === A. Model Configurations

  We evaluate seven experimental arms, summarized in Table #ref(<tab:arms>). The first arm consists of five independently trained binary Wav2Vec2 classifiers, one per dysfluency type; the remaining arms are single models.

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
        [arm01], [5× Wav2Vec2 binary classifiers], [classifier], [472,845,450],
        [arm02], [Wav2Vec2 multitask, freeze 3 epochs], [multitask], [97,332,362],
        [arm03], [Wav2Vec2 multitask, freeze 20 epochs], [multitask], [97,332,362],
        [arm04], [CNN pooling, multitask], [multitask], [342,030],
        [arm05], [CNN single-head], [multitask_single], [274,950],
        [arm06], [CNN + LSTM, multitask], [multitask], [457,614],
        [arm07], [CNN + Transformer, multitask], [multitask], [540,302],
      )
    ],
    caption: [Comparative study arms: architectural variants and parameter counts.],
    kind: table,
  ) <tab:arms>

  === B. Training Protocol

  All models are trained on a single NVIDIA GPU with the following shared settings: seed = 42, 20 epochs with early stopping (patience = 5), gradient-norm clipping at 1.0, mixed precision in the classification pipelines, batch size = 8 (binary classifiers), 16 (multitask and CNN classifiers), or 4 (Wav2Vec2 localizer), maximum audio length = 3 seconds, sample rate = 16 kHz. Data augmentation includes Gaussian noise injection (σ = 0.005), time stretching (0.9–1.1×), pitch shifting (±1.0 semitones), circular temporal shifting (±0.1 s), and amplitude scaling (0.8–1.2×). Preprocessed audio is cached to disk for fast re-runs.

  Thresholds for binary classification are tuned on the validation set using Youden's J statistic (J = sensitivity + specificity − 1) to find the operating point that maximizes discriminative ability. No test-set or cross-corpus threshold fitting is performed, ensuring honest evaluation.

  === C. Evaluation Metrics

  For classification, we report precision, recall, F1 score, and AUROC per dysfluency type, plus macro-averaged F1 across all five types.

  === D. Evaluation Protocol

  Two evaluation settings are used:

  + *In-distribution held-out (Test):* 3,716 valid clips from the held-out test split of the merged corpus. Note: due to same-speaker overlap between train and test splits (both drawn from SEP-28K and UCLASS), these results represent an optimistic upper bound. The test split also contains the 53-clip Project Boli subset described below.
  + *Cross-corpus held-out (Boli):* 53 clips from the Project Boli dataset, pinned to the test split and therefore entirely unseen during training. This provides an estimate of generalization to new speakers, accents, and recording conditions. Per-class clip counts on Boli are limited (PR: 16, B: 35, SR: 34, WR: 14, IN: 9), so Boli results are noisy but honest.
]

// ── V. RESULTS AND DISCUSSION ────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = V. RESULTS AND DISCUSSION

  === A. In-Distribution Classification Performance

  Table #ref(<tab:test_f1>) presents selected per-class F1 scores (prolongation and block) for all seven arms on the in-distribution test set, with both default (0.5) and tuned thresholds.

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
    caption: [Selected in-distribution test F1 scores at default (0.5) and tuned thresholds (PR = prolongation, BL = block).],
    kind: table,
  ) <tab:test_f1>

  The *multitask Wav2Vec2* with 3-epoch backbone freezing (arm02) achieves the highest overall test macro F1 (0.5215 at tuned thresholds), showing that the shared backbone provides a strong foundation while reducing the number of independent models from five to one and using ≈4.9× fewer parameters than five separate classifiers. The *five-binary Wav2Vec2 classifier* (arm01) follows closely at 0.5183, benefiting from dedicated per-class optimization.

  The *multitask Wav2Vec2 with 20-epoch freezing* (arm03) performs notably worse (test F1 = 0.142 at default, 0.339 at tuned), confirming that excessive backbone freezing prevents the model from learning task-specific representations. Backbone freezing is a hyperparameter that must be carefully tuned.

  CNN-based models (arms04–07) achieve substantially lower in-distribution F1 (0.188–0.255 at default thresholds), reflecting the limited capacity of small convolutional networks (~300K–540K parameters) to capture the acoustic patterns of dysfluent speech compared to the 94M-parameter Wav2Vec2 backbone.

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

  To understand why CNN-LSTM wins on Boli, Table #ref(<tab:boli_perclass>) reports per-class tuned F1 on Boli for the five-binary (arm01), multitask Wav2Vec2 (arm03), and CNN-LSTM (arm06) models.

  #figure(
    block(breakable: true)[
      #set text(size: 8.5pt)
      #set par(first-line-indent: 0em)
      #table(
        columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
        align: (left, right, right, right, right, right, right),
        stroke: 0.5pt,
        table.header(
          [*Arm*], [*PR*], [*BL*], [*SR*], [*WR*], [*IN*], [*Macro*],
        ),
        [arm01_5x_w2v2], [0.240], [0.105], [0.279], [0.000], [0.000], [0.154],
        [arm02_mt_frz3], [0.191], [0.000], [0.250], [0.191], [0.167], [0.160],
        [arm06_cnn_lstm], [0.482], [0.782], [0.667], [0.383], [0.290], [0.521],
      )
    ],
    caption: [Per-class tuned F1 on the Boli held-out set (PR = prolongation, BL = block, SR = sound repetition, WR = word repetition, IN = interjection). Class support on Boli is small (PR: 16, BL: 35, SR: 34, WR: 14, IN: 9), so these values are noisy.],
    kind: table,
  ) <tab:boli_perclass>

  The per-class picture is more informative than the macro average alone. The CNN-LSTM model detects all five dysfluency types on Boli with non-zero F1, whereas the Wav2Vec2 models collapse to near-zero recall on several classes (notably block for the multitask model and word repetition and interjection for the binary classifiers, which never predict these classes on the unseen corpus). The CNN-LSTM improvement is spread across classes rather than driven by a single type. As with the overall Boli result, the small per-class support limits how much weight to place on any single value.

  CNN-based models outperform Wav2Vec2 models on cross-corpus data. The CNN-LSTM model (arm06) achieves the highest Boli F1 (0.521 at tuned thresholds), more than 3× the Wav2Vec2 five-binary classifier (0.154). Two possible explanations are:

  + *Parameter efficiency:* The higher-capacity Wav2Vec2 representations may be more sensitive to corpus-specific characteristics of the training data (SEP-28K/UCLASS) that do not transfer to unseen speakers and accents, whereas the compact CNN models (~300K–540K parameters) may rely more on coarse acoustic patterns.
  + *Spectrogram invariance:* Mel-spectrogram representations may be more robust to speaker variability than raw-waveform embeddings, since they compress fine-grained speaker identity information while preserving coarse spectral patterns relevant to dysfluency detection.

  The Boli results suggest that compact spectrogram-based models may provide better cross-corpus robustness in multilingual settings. However, the limited size of the Boli evaluation set (53 clips) prevents definitive conclusions, and models that perform well in-distribution may not transfer to new populations.

  === C. Per-Class Analysis

  Detailed per-class F1 scores on the test set (at tuned thresholds) are presented in Table #ref(<tab:perclass>), and the tuned thresholds themselves are listed in Table #ref(<tab:thresholds>). For every class, the Youden-optimal threshold on the validation set is a single operating point per class; the multitask Wav2Vec2 (arm02) converges to 0.45 for prolongation and 0.40 for the remaining classes.

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

  #figure(
    block(breakable: true)[
      #set text(size: 9pt)
      #set par(first-line-indent: 0em)
      #table(
        columns: (auto, 1fr),
        align: (left, center),
        stroke: 0.5pt,
        table.header(
          [*Class*], [*Threshold*],
        ),
        [Prolongation], [0.45],
        [Block], [0.40],
        [Sound Rep.], [0.40],
        [Word Rep.], [0.40],
        [Interjection], [0.40],
      )
    ],
    caption: [Per-class validation-tuned thresholds (Youden's J) for the multitask Wav2Vec2 (arm02).],
    kind: table,
  ) <tab:thresholds>

  Several observations follow:

  + *Interjection* is the most detectable dysfluency type (F1 = 0.751, AUROC = 0.931). Filler words ("um," "uh," "like") have distinct acoustic signatures that are well-separated from fluent speech, making them relatively easy for the model to identify.

  + *Block* is the most challenging type (F1 = 0.383, AUROC = 0.735). Silent pauses and tense pauses before speech are acoustically similar to normal speech pauses, making them difficult to distinguish without contextual information.

  + *Sound repetition* achieves the second-highest F1 (0.541) with the highest recall (0.658), indicating the model is effective at catching sound repetitions but with lower precision.

  + The large gap between AUROC and F1 across all classes (e.g., prolongation AUROC = 0.844 vs. F1 = 0.483) indicates that the model ranks dysfluent frames well but its probability outputs are not well matched to the chosen operating threshold. The gap is consistent with outputs that are not well calibrated, though we do not measure calibration directly.

  Table #ref(<tab:confmat>) reports the confusion matrices (TP, FP, TN, FN at tuned thresholds) for the five binary Wav2Vec2 classifiers (arm01) on the test set. They show the same pattern as the multitask model: interjection is detected most confidently, word repetition and block produce the most false negatives, and block yields the largest number of false positives relative to its support.

  #figure(
    block(breakable: true)[
      #set text(size: 8.5pt)
      #set par(first-line-indent: 0em)
      #table(
        columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
        align: (left, right, right, right, right, right),
        stroke: 0.5pt,
        table.header(
          [*Class*], [*TP*], [*FP*], [*TN*], [*FN*], [*Threshold*],
        ),
        [Prolongation], [219], [297], [3018], [181], [0.500],
        [Block], [218], [406], [2756], [335], [0.400],
        [Sound Rep.], [300], [297], [2924], [194], [0.400],
        [Word Rep.], [153], [235], [3114], [213], [0.450],
        [Interjection], [646], [161], [2720], [188], [0.550],
      )
    ],
    caption: [Confusion matrices for the five binary Wav2Vec2 classifiers (arm01) on the test set at tuned thresholds.],
    kind: table,
  ) <tab:confmat>

  === D. Computational Considerations

  The comparison between Wav2Vec2 and CNN models highlights a trade-off:

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
        [Wav2Vec2 (5× binary)], [472.8M], [0.518], [0.154],
        [Wav2Vec2 (multitask)], [97.3M], [0.522], [0.160],
        [CNN (best)], [540K], [0.264], [0.521],
      )
    ],
    caption: [Trade-off between parameter count and generalization performance.],
    kind: table,
  ) <tab:tradeoff>

  Wav2Vec2 models achieve ~2× higher in-distribution F1 but use ~180× more parameters and show 3.4× worse cross-corpus performance. For resource-constrained or multilingual deployment, CNN models offer an alternative with competitive cross-corpus performance and negligible computational overhead.

  === E. Comparison with Published Results

  Our multitask Wav2Vec2 results (macro F1 = 0.522 at tuned thresholds) are consistent with published benchmarks: Bayerl et al. #cite(<bayerl2022multi>) reported macro F1 in the range of 0.56 -- 0.63 on FluencyBank, and Miyahara et al. #cite(<miyahara2025wav2vec2>) reported per-class F1 ranging from 0.30 (block) to 0.78 (interjection) on SEP-28K. Our interjection F1 (0.751) matches their 0.78 closely, and our block F1 (0.383) exceeds their 0.30, suggesting our shared-backbone architecture provides improvement for the most challenging class. The macro-averaged F1 of 0.522 falls within the literature range of 0.45 -- 0.65 for Wav2Vec2-based classifiers on this task.

  === F. Limitations

  + *Same-speaker overlap:* The test set contains speakers from the same source datasets as training (SEP-28K, UCLASS), inflating in-distribution metrics. Boli provides a more honest cross-corpus evaluation but is limited in size (53 clips).
  + *Single seed:* All results use seed = 42. A multi-seed evaluation would provide confidence intervals and more robust comparisons.
  + *Binary thresholds:* Per-class thresholds are tuned on the validation set only, which may not suit all deployment scenarios.

  === G. Practical Implications

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

  This paper presented Swaraaha, an end-to-end speech dysfluency classification system, and compared seven experimental arms across three model families. Our key findings are:

  + The *shared-backbone multitask classifier* achieves the highest in-distribution performance (F1 = 0.5215), while the *five-binary variant* follows closely (F1 = 0.5183) using ≈4.9× more parameters in total.
  + *CNN-based models* outperform Wav2Vec2 on cross-corpus evaluation (Boli F1 = 0.521 vs. 0.160), showing a trade-off between in-distribution accuracy and generalization.
  + *Interjection* is the most reliably detected dysfluency type (F1 = 0.751), while *block* remains the most challenging (F1 = 0.383), consistent with prior literature.
  + *Backbone freezing duration* is a hyperparameter that matters: excessive freezing (20 epochs) degrades tuned macro F1 by ≈35% compared to moderate freezing (3 epochs).

  Future work will include: (1) backbone ablation across XLS-R-300M and HuBERT for multilingual support, (2) multi-seed evaluation with confidence intervals, and (3) integration of stutter-aware ASR for freeform speech analysis.

  Swaraaha is available to view at: `https://github.com/chethanbhat7/Swaraaha`.
]

// ── ACKNOWLEDGEMENT ─────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = ACKNOWLEDGEMENT

  The authors thank Prof. Akhilesh P M, pathologist, for his inputs on the clinical dimensions of speech dysfluency. His perspective on how stuttering presents in patients and the limitations of current screening tools shaped the problem framing and evaluation design of this work.
]

// ── REFERENCES ───────────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = REFERENCES

  #bibliography("references.bib", title: none, style: "ieee")
]
