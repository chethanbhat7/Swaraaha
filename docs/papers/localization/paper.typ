// ──────────────────────────────────────────────────────────────────────
// Swaraaha — Temporal Localization Research Paper (Typst / IEEE-style)
// ──────────────────────────────────────────────────────────────────────

#set document(
  title: "Swaraaha: Temporal Localization of Speech Dysfluency Events Using CNN and Wav2Vec2 Architectures",
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
      #smallcaps[Swaraaha — Temporal Localization]
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
  #text(size: 16pt, weight: "bold")[Swaraaha: Temporal Localization of Speech Dysfluency Events Using CNN and Wav2Vec2 Architectures]
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
    `{4vp23ai020, 4vp23ai023, 4vp23ai051, 4vp23ai054}@vcetputtur.ac.in`
  ]
  #v(0.8cm)
]

// ── Abstract ─────────────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set text(size: 9.5pt)
  #set par(first-line-indent: 0em)
  *Abstract* #h(0.5em) — Beyond identifying *which* type of dysfluency is present, clinical applications of stutter analysis require knowing *where* in the audio a dysfluency event occurs. Automated temporal localization of dysfluency events remains challenging due to the subtle acoustic signatures, the rarity of dysfluent frames, and the need for fine-grained temporal precision. This paper presents the temporal localization component of *Swaraaha*, an end-to-end speech dysfluency analysis system, and performs a comparative study of two complementary localizer architectures: a convolutional neural network (CNN) spectrogram localizer and a Wav2Vec 2.0 temporal localizer. Both models are evaluated on a merged corpus of 37,087 clips drawn from SEP-28K, UCLASS, and Project Boli, using both in-distribution and cross-corpus (Boli) evaluation protocols with frame-level and event-level metrics. Our results reveal that the two localizers occupy complementary operating points: the Wav2Vec2 localizer is conservative (high frame precision 0.676, low recall 0.065), while the CNN localizer is aggressive (high frame recall 0.690, high false-alarm rate 55.9/min). The CNN localizer generalizes better to unseen data (Boli frame F1 0.264 vs. 0.040), yet neither architecture is yet suitable for standalone clinical deployment. We analyze the precision–recall trade-off, discuss the implications of conservatism versus sensitivity for clinical use, and outline directions for building clinical-grade dysfluency localization systems.
]

#v(0.3cm)

#align(center)[
  #text(size: 9pt, style: "italic")[
    *Keywords* #h(0.5em) — Speech dysfluency localization, stuttering detection, temporal localization, Wav2Vec 2.0, convolutional neural networks, frame-level prediction, speech signal processing
  ]
]

#v(0.5cm)

// ── I. INTRODUCTION ──────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = I. INTRODUCTION

  Stuttering affects approximately 1% of the global adult population and up to 5% of children, making it one of the most common communication disorders #cite(<yairi2009childhood>). It manifests as involuntary repetitions of sounds, syllables, or words, abnormal prolongations of speech sounds, and involuntary pauses or stops in speech flow (blocks). While prior research has largely focused on *classifying* which dysfluency types are present in a speech segment, clinical applications — including automated therapy progress tracking, severity assessment, and intervention planning — additionally require knowing *where* in the audio each dysfluency event occurs #cite(<sahu2022stuttering>). This temporal localization of dysfluency events enables clinicians to review precisely the moments of dysfluent speech and to quantify the duration and frequency of specific event types over time.

  Temporal localization is substantially more challenging than clip-level classification. It requires frame-level or event-level prediction with fine-grained temporal precision, and the dysfluent frames are rare relative to fluent speech, creating a severe class imbalance — a challenge compounded by the subtle acoustic signatures that distinguish genuine dysfluencies from normal speech pauses and articulatory variation. Prior work on localization has received less attention than classification in the stutter detection literature #cite(<sahu2022stuttering>), and systematic evaluation of localization models on multi-corpus data remains limited.

  In this paper, we study two complementary localization architectures within the open-source *Swaraaha* system:

  + A *CNN spectrogram localizer* that operates on mel-spectrograms and produces per-frame dysfluency probability maps without downsampling the time dimension, and

  + A *Wav2Vec 2.0* #cite(<baevski2020wav2vec>) *temporal localizer* that uses a temporal attention pooling head over the encoder's hidden states to produce frame-level predictions at ~20 ms resolution.

  We evaluate both localizers on a merged corpus of 37,087 clips drawn from SEP-28K #cite(<basak2024sep28k>), UCLASS #cite(<giang2023uclass>), and Project Boli #cite(<boli2025>), under both in-distribution held-out and cross-corpus evaluation protocols. Our contributions are:

  + We present the *Swaraaha* temporal localization component, supporting frame-level prediction of dysfluency events across five dysfluency types.

  + We conduct a comparative study of two complementary localizer architectures — a CNN spectrogram localizer and a Wav2Vec2 temporal localizer — evaluating them with both frame-level (precision, recall, F1, specificity) and event-level (detection accuracy, mean IoU, false-alarm rate) metrics.

  + We demonstrate that the two localizers occupy complementary operating points — a conservative Wav2Vec2 variant and an aggressive CNN variant — and that the CNN localizer generalizes better to unseen speakers and accents on cross-corpus data.

  + We analyze the precision–recall trade-off for clinical deployment and outline how the Swaraaha registry fuses localizer regions with classifier saliency to mitigate each architecture's individual weakness.

  The remainder of this paper is organized as follows: Section II reviews related work on temporal localization. Section III describes the Swaraaha localization architecture. Section IV details the experimental setup. Section V presents results and discussion. Section VI concludes with future directions.
]

// ── II. RELATED WORK ─────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = II. RELATED WORK

  === A. Classification vs. Localization

  Most prior stutter detection systems address *clip-level classification*: determining which dysfluency types are present in a segment of speech. Classification answers the *what* question but not the *where*. While classification is sufficient for screening, clinical workflows require localizing the exact temporal extent of each dysfluency event for review and quantification #cite(<dietrich2021stuttering>). This requirement motivates the shift toward temporal localization frameworks.

  === B. Spectrogram-Based Event Detection

  CNN-based approaches operating on spectrograms have been proposed for speech event detection more generally and for stuttering events specifically #cite(<sahu2022stuttering>). These models treat the time–frequency representation as a structured input and learn localized patterns that correspond to acoustic events. However, systematic evaluation of such models for stutter localization on multi-corpus data remains limited, and prior work rarely reports both frame-level and event-level localization metrics together.

  === C. Wav2Vec2-Based Time and Frame Prediction

  Wav2Vec 2.0 #cite(<baevski2020wav2vec>) learns contextualized representations from raw audio through self-supervised pre-training on large unlabeled corpora. Its internal convolutional feature extractor subsamples audio to produce frame-level hidden states at ~20 ms resolution, making it a natural backbone for temporal prediction tasks. Wav2Vec2 has been applied to stuttering classification #cite(<bayerl2022multi>) and to temporally aligned downstream tasks, but dedicated temporal localization of dysfluency events with Wav2Vec2 remains relatively unexplored #cite(<miyahara2025wav2vec2>).

  === D. Multi-Corpus Evaluation

  Prior work has largely evaluated on single datasets (either SEP-28K #cite(<basak2024sep28k>) or FluencyBank), making it difficult to assess cross-corpus generalization of localization models. The Boli dataset #cite(<boli2025>) offers a multilingual, multi-accent corpus with word-level annotations and timestamps, enabling meaningful cross-corpus evaluation of temporal localization. Our work evaluates both localizers on both in-distribution and cross-corpus held-out data.
]

// ── III. SYSTEM ARCHITECTURE ─────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = III. SYSTEM ARCHITECTURE

  The Swaraaha localization component comprises a data pipeline, two localizer architectures, and a unified model registry. The localizers produce per-frame dysfluency probability maps whose results are fused with classifier saliency maps for deployment.

  === A. Data Pipeline

  The system integrates three publicly available stuttering datasets:

  + *SEP-28K* #cite(<basak2024sep28k>): 32,321 clips annotated for five dysfluency types, with interval timestamps recording each event's start and end times.
  + *UCLASS* #cite(<giang2023uclass>): 4,712 clips in SEP-28K format, sourced from stuttering therapy sessions.
  + *Project Boli* #cite(<boli2025>): 54 clips with word-level dysfluency annotations and timestamps, recorded by speakers of five Indian languages (Hindi, Marathi, Telugu, Bengali, and Assamese).

  All three datasets are merged into a unified corpus of 37,087 clips with per-clip interval CSVs recording dysfluency events as `(start_sec, end_sec, type)` tuples. A stratified 80/10/10 train/val/test split is created, yielding 29,296 training clips, 3,662 validation clips, and 3,716 test clips. To guarantee an honest cross-corpus evaluation, all Project Boli clips are pinned to the test split and never appear in the training or validation splits. Audio is resampled to 16 kHz mono, DC-removed, peak-normalized, silence-trimmed, and padded/truncated to a maximum of 3 seconds. The interval annotations are converted to frame-level labels aligned with each localizer's temporal resolution for training and evaluation.

  === B. CNN Spectrogram Localizer

  A convolutional network operating on 128-bin mel-spectrograms. The architecture uses four convolutional blocks (Conv2d(3×3) → BatchNorm → ReLU → Dropout2d(0.4)) with progressive channel expansion (1→32→64→128). Each block max-pools only along the frequency axis (kernel 2×1), preserving the time resolution throughout; the fourth block collapses the remaining frequency dimension with adaptive average pooling. A per-frame head of two 1×1 convolutions (128→64→1) with a sigmoid outputs dysfluency probability per frame — since time is never downsampled, no transposed convolutions are required. Trained with BCE loss and pos_weight = 5.0 to compensate for the rarity of dysfluent frames (~5% of all frames).

  === C. Wav2Vec2 Temporal Localizer

  Uses the Wav2Vec2 backbone #cite(<baevski2020wav2vec>) with a temporal attention pooling head: raw audio → Wav2Vec2 encoder → temporal attention → Linear(768→256) → Dropout(0.3) → Linear(256→1) → sigmoid. Frame resolution is ~20 ms (Wav2Vec2 internal subsampling factor = 320 samples at 16 kHz). Backbone freezing for the first 5 epochs, then unfreezing with 10× lower learning rate.

  === D. Model Registry and Combiner

  All models are loaded through a unified registry API (`Classifier`, `Localizer`, `Transcriber`, `ModelRegistry`) that decouples model loading from training. The registry reads checkpoint paths and per-frame thresholds from a JSON configuration file, enabling seamless model swapping without code changes. Audio preprocessing (resampling, cleaning, normalization, padding) is applied automatically within the API. The registry's combiner fuses localizer regions with the classifier's per-frame saliency maps, so that conservative temporal anchors from one localizer are complemented by class-aware evidence from the other, mitigating each architecture's individual weakness.
]

// ── IV. EXPERIMENTAL SETUP ───────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = IV. EXPERIMENTAL SETUP

  === A. Model Configurations

  We evaluate two localizer architectures:

  + *CNN Spectrogram Localizer:* a four-block convolutional network operating on 128-bin mel-spectrograms, producing per-frame dysfluency probability maps (~342,030 parameters).
  + *Wav2Vec2 Temporal Localizer:* the Wav2Vec2 backbone with a temporal attention pooling head, producing frame-level predictions at ~20 ms resolution.

  === B. Training Protocol

  Both models are trained on a single NVIDIA GPU with the following settings: seed = 42, 20 epochs with early stopping (patience = 5), gradient-norm clipping at 1.0, mixed precision, maximum audio length = 3 seconds, sample rate = 16 kHz. The CNN localizer uses batch size 16 and BCE loss with pos_weight = 5.0; the Wav2Vec2 localizer uses batch size 4 and freezes its backbone for the first 5 epochs, then unfreezes with a 10× lower learning rate. Data augmentation includes Gaussian noise injection (σ = 0.005), time stretching (0.9–1.1×), pitch shifting (±1.0 semitones), circular temporal shifting (±0.1 s), and amplitude scaling (0.8–1.2×). Preprocessed audio is cached to disk for fast re-runs. Frame-level thresholds are tuned on the validation set using Youden's J statistic; no test-set or cross-corpus threshold fitting is performed, ensuring honest evaluation.

  === C. Evaluation Metrics

  We report both frame-level and event-level metrics. Frame-level metrics — computed by comparing predicted frame labels against ground-truth frame labels — include precision, recall, F1, and specificity. Event-level metrics, computed by matching predicted events to ground-truth events, include detection accuracy, mean Intersection over Union (mIoU) between predicted and ground-truth event intervals, and false-alarm rate (events per minute).

  === D. Evaluation Protocol

  Two evaluation settings are used:

  + *In-distribution held-out (Test):* 3,715 valid clips from the held-out test split of the merged corpus. Note: due to same-speaker overlap between train and test splits, these results represent an optimistic upper bound.
  + *Cross-corpus held-out (Boli):* 53 clips from the Project Boli dataset, pinned to the test split and therefore entirely unseen during training. This provides an estimate of generalization to new speakers, accents, and recording conditions.
]

// ── V. RESULTS AND DISCUSSION ────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = V. RESULTS AND DISCUSSION

  === A. In-Distribution Localization Performance

  Table #ref(<tab:localization>) compares localization metrics for the two localizers — the CNN spectrogram localizer and the Wav2Vec2 temporal localizer — on the test and Boli held-out sets.

  #figure(
    block(breakable: true)[
      #set text(size: 8.5pt)
      #set par(first-line-indent: 0em)
      #table(
        columns: (auto, 1fr, 1fr, 1fr, 1fr),
        align: (left, right, right, right, right),
        stroke: 0.5pt,
        table.header(
          [*Metric*], [*CNN Test*], [*CNN Boli*], [*W2V2 Test*], [*W2V2 Boli*],
        ),
        [Frame Precision], [0.485], [0.154], [0.676], [0.177],
        [Frame Recall], [0.690], [0.918], [0.065], [0.023],
        [Frame F1], [0.570], [0.264], [0.119], [0.040],
        [Frame Specificity], [0.373], [0.284], [0.973], [0.985],
        [Detection Accuracy], [0.222], [0.375], [0.210], [0.000],
        [Mean IoU], [0.707], [0.614], [0.751], [0.000],
        [False Alarms/min], [55.89], [33.86], [8.95], [11.16],
      )
    ],
    caption: [Localization metrics for the CNN spectrogram localizer and the Wav2Vec2 temporal localizer on test and Boli held-out sets.],
    kind: table,
  ) <tab:localization>

  The two localizers exhibit complementary operating points. The *Wav2Vec2 temporal localizer* is highly conservative: on the test set it achieves high frame precision (0.676) but very low recall (0.065), missing most dysfluency events while producing relatively few false alarms (8.95/min); when it does detect an event, temporal accuracy is reasonable (mean IoU = 0.751). The *CNN spectrogram localizer* takes the opposite stance: substantially higher recall (0.690 test) and higher frame F1 (0.570 test), but at the cost of a high false-alarm rate (55.9/min test) and low specificity (0.373). Neither localizer is yet suitable for standalone clinical deployment: the Wav2Vec2 variant misses too many events, while the CNN variant flags far too many.

  === B. Cross-Corpus Generalization (Boli)

  On the Boli cross-corpus held-out set, the generalization gap between the two localizers widens dramatically. The *CNN localizer* achieves high frame recall (0.918) with moderate frame F1 (0.264) and a reduced false-alarm rate (33.86/min), and it still detects a reasonable number of events (detection accuracy 0.375, mIoU 0.614). The *Wav2Vec2 localizer* collapses on Boli: it detects no events at all (detection accuracy 0.000, mIoU 0.000, frame recall 0.023), although its frame precision (0.177) and specificity (0.985) remain high because it almost never predicts a dysfluent frame.

  This divergence is consistent with the classification literature: the 94M-parameter Wav2Vec2 backbone memorizes training-set (SEP-28K/UCLASS) idiosyncrasies that do not transfer to unseen speakers and accents, whereas the compact CNN operating on mel-spectrograms — which compress fine-grained speaker identity while preserving coarse spectral patterns relevant to dysfluency — is more robust to speaker variability. The finding has direct implications for clinical deployment: models that appear accurate in-distribution may fail catastrophically on new populations, and CNN-based localizers may be preferable for low-resource or multilingual settings despite their higher false-alarm rate.

  === C. Precision–Recall Trade-Off Analysis

  The two localizers represent opposite ends of the precision–recall spectrum, and each is ill-suited to clinical use on its own:

  + *Wav2Vec2 localizer (conservative):* when it flags an event, that flag is trustworthy (precision 0.676, specificity 0.973), but it misses the vast majority of dysfluency events (recall 0.065). An SLP using only this localizer would miss most dysfluencies, making per-session event counts unreliable. On unseen Boli data it is effectively silent.
  + *CNN localizer (sensitive):* it catches far more dysfluency events (recall 0.690) with reasonable temporal accuracy (mIoU 0.707), but its low precision (0.485) and specificity (0.373) mean most flagged frames are fluent, generating ~56 false alarms per minute. An SLP using only this localizer would be inundated with spurious alerts.

  The complementarity of these operating points is the motivation for the registry's *combiner*, which fuses localizer regions with the classifier's per-frame saliency maps: conservative temporal anchors from the Wav2Vec2 localizer are retained while class-aware evidence from the classifier filters out false positives from the CNN localizer, and sensitive detections from the CNN localizer recover events the conservative localizer misses. Improving the precision–recall balance of both localizers individually remains a priority for future work.

  === D. Computational Considerations

  The Wav2Vec2 temporal localizer inherits the ≈94M-parameter backbone and benefits from transfer learning learned on large unlabeled corpora, but this capacity is a liability for generalization to unseen corpora and incurs substantial computational cost. The CNN spectrogram localizer is dramatically more lightweight (~342K parameters) and runs with negligible computational overhead, making it attractive for real-time and edge deployment. This reproduces the fundamental trade-off observed in classification: representation capacity versus generalization and efficiency.

  === E. Comparison with Published Localization Results

  Published localization results for stuttering remain sparse and difficult to compare directly, as prior work differs in evaluation metrics, corpus, and event definitions #cite(<sahu2022stuttering>). Spectrogram-based CNN event detectors have demonstrated the viability of the general approach, while our results quantify the precision–recall trade-off that arises when such detectors are applied to rare dysfluency events. To our knowledge, a systematic head-to-head comparison of a CNN spectrogram localizer and a Wav2Vec2 temporal localizer under both frame-level and event-level metrics on multi-corpus data has not been previously reported.

  === F. Limitations

  + *Same-speaker overlap:* The test set contains speakers from the same source datasets as training, inflating in-distribution metrics. Boli provides a more honest cross-corpus evaluation but is limited in size (53 clips).
  + *Single seed:* All results use seed = 42. A multi-seed evaluation would provide confidence intervals and more robust comparisons.
  + *Wav2Vec2 localizer recall:* Low recall (0.065) and total failure on Boli limit its clinical utility for event discovery.
  + *CNN false-alarm rate:* The high false-alarm rate (55.9/min on test) limits standalone clinical use without downstream filtering.
  + *Threshold tuning:* Frame-level thresholds are tuned on the validation set only, which may not be optimal for all deployment scenarios.

  === G. Practical Implications

  For clinical stutter localization systems, our results suggest:

  + *Combine localizers rather than relying on a single architecture:* the conservative Wav2Vec2 localizer and the sensitive CNN localizer are complementary, and their fusion via the registry combiner is more clinically useful than either alone.
  + *Prefer CNN localizers for population-level and multilingual screening*, where unseen speakers, accents, and recording conditions are expected, accepting a higher false-alarm rate that can be filtered downstream.
  + *Always report cross-corpus localization metrics* alongside in-distribution metrics to provide honest estimates of generalization.
  + *Optimize the precision–recall balance* (e.g., via semi-supervised training or post-hoc event filtering) before deploying a localizer for autonomous clinical quantification.
]

// ── VI. CONCLUSION ───────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = VI. CONCLUSION

  This paper presented the temporal localization component of Swaraaha and provided a comparative study of two complementary localizer architectures — a CNN spectrogram localizer and a Wav2Vec2 temporal localizer — evaluated on a multi-corpus dataset under both frame-level and event-level metrics. Our key findings are:

  + The *Wav2Vec2 temporal localizer* is conservative, achieving high frame precision (0.676) but very low recall (0.065), and collapses to detecting nothing on unseen Boli data.
  + The *CNN spectrogram localizer* is sensitive, achieving high frame recall (0.690 test, 0.918 Boli) and superior cross-corpus generalization (Boli frame F1 0.264 vs. 0.040), but at the cost of a high false-alarm rate (55.9/min test).
  + Neither localizer is yet suitable for standalone clinical deployment; their *complementary operating points* motivate fusing localizer regions with classifier saliency via the registry combiner.
  + *Cross-corpus evaluation is essential*: the Wav2Vec2 localizer's apparent in-distribution utility does not transfer to unseen speakers and accents, mirroring the classification results.

  Future work will focus on: (1) improving localization recall through semi-supervised training with pseudo-labeled SEP-28K data, (2) improving precision via post-hoc event filtering and combiner tuning, (3) backbone ablation across XLS-R-300M and HuBERT for multilingual support, and (4) multi-seed evaluation with confidence intervals.

  Swaraaha is available to view at: `https://github.com/chethanbhat7/Swaraaha`.
]

// ── REFERENCES ───────────────────────────────────────────────────────

#block(inset: (left: 1.5cm, right: 1.5cm))[
  #set par(first-line-indent: 0em)
  = REFERENCES

  #bibliography("references.bib", title: none, style: "ieee")
]
