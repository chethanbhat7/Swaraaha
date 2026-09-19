#import "../lib.typ": *
#import "../meta.typ": *

// --- Chapter 6: Testing and Results ---
#chapter_heading[TESTING AND RESULTS]

== INTRODUCTION
This chapter covers the testing methodology, evaluation metrics, and results from training and evaluating the #project_title.
Both the classification and localization pipelines are evaluated using standard machine learning metrics on held-out test data.
Results are broken down by individual dysfluency class and compared across different model architectures.

== DATASET
The system was trained and evaluated on three publicly available stuttering datasets, merged into a unified format.

- Project Boli: a multilingual Indian language dataset with stuttering data across Hindi, Kannada, Telugu, and other Indian languages, sourced from GitHub. It includes both read and spontaneous speech.
- SEP-28K: roughly 28,000 audio clips from stuttering support group podcasts, annotated for five stuttering event types: blocks, prolongations, sound repetitions, word repetitions, and interjections. Obtained from Kaggle.
- UCLASS: an additional stuttering speech dataset from Kaggle, providing supplementary annotated clips.

Each dataset was normalized into a unified format with a `combined_labels.csv` file containing multi-label binary annotations and interval files for each clip. The five classes are prolongation, block, sound repetition, word repetition, and interjection.

The merged dataset was split 80:10:10 into training, validation, and testing subsets. An automated pipeline handles downloading, merging, and preprocessing (16 kHz mono WAV conversion, DC offset removal, peak normalization, silence trimming, and padding to 48,000 samples).

== EVALUATION METRICS
These metrics were used to evaluate classification and localization performance.

=== Classification Metrics
- Precision: the fraction of predicted positives that are actually positive.
- Recall: the fraction of actual positives that are correctly identified.
- F1-Score: the harmonic mean of precision and recall.
- AUROC (Area Under the Receiver Operating Characteristic Curve): the model's ability to separate classes across all threshold values. 1.0 is perfect; 0.5 is random.
- AUPRC (Area Under the Precision-Recall Curve): the precision-recall trade-off across thresholds. More informative than AUROC on imbalanced data.
- Specificity: the fraction of actual negatives correctly identified.
- Macro F1: the unweighted average of F1 across all classes.

=== Localization Metrics
- Frame-level Precision, Recall, and F1: accuracy of per-frame dysfluency predictions.
- Detection Accuracy: fraction of true dysfluency events correctly detected.
- Mean IoU (Intersection over Union): overlap between predicted and ground-truth regions. 1.0 is perfect overlap.

== CLASSIFICATION RESULTS

=== Multitask Wav2Vec2 Classifier
The primary classifier uses a single Wav2Vec2-base backbone with five shared binary heads (multitask). Training used Focal Loss (gamma=2.0), backbone freezing for 3 epochs, AdamW (LR $3 times 10^(-5)$), 500-step warm-up, and early stopping with patience of 5.

Results on 3,715 held-out test samples at the default threshold of 0.5 are in Table 4.1.

#add_table(
  table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: horizon,
    table.header([*Class*], [*Precision*], [*Recall*], [*F1*], [*AUROC*], [*AUPRC*], [*Support*]),
    [Prolongation], [0.560], [0.418], [0.479], [0.844], [0.467], [400],
    [Block], [0.639], [0.150], [0.243], [0.735], [0.395], [553],
    [Sound Repetition], [0.564], [0.490], [0.524], [0.860], [0.552], [494],
    [Word Repetition], [0.581], [0.383], [0.461], [0.830], [0.477], [366],
    [Interjection], [0.836], [0.666], [0.741], [0.932], [0.836], [834],
    [*Macro Average*], [--], [--], [*0.490*], [--], [--], [*3,715*],
  ),
  caption: [Multitask Wav2Vec2 classifier results at threshold 0.5 on the test set]
)

The macro F1 at the default threshold is 0.490. Interjection performs best (F1=0.741, AUROC=0.932), which makes sense because filler words like "um" and "uh" have distinct acoustic patterns. Block is the hardest class (F1=0.243). Silent pauses and hesitations are hard to distinguish from normal speech pauses.

#add_image(align(center, image("/assets/classifier_test_f1.png", width: 90%)), caption: [Per-class F1 scores of the multitask Wav2Vec2 classifier on the clip-level test set])

#add_image(align(center, image("/assets/classifier_test_auc.png", width: 90%)), caption: [Per-class AUROC and AUPRC of the multitask Wav2Vec2 classifier on the clip-level test set])

=== Threshold Optimization
A threshold sweep on the validation set tested thresholds from 0.1 to 0.9 in steps of 0.05. The threshold that maximized F1 for each class was stored in the model registry. The results are in Table 4.2.

#add_table(
  table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: horizon,
    table.header([*Class*], [*Default (0.5) F1*], [*Optimal Threshold*], [*Tuned F1*], [*Improvement*]),
    [Prolongation], [0.479], [0.45], [0.483], [+0.8%],
    [Block], [0.243], [0.35], [0.401], [+65.0%],
    [Sound Repetition], [0.524], [0.45], [0.552], [+5.2%],
    [Word Repetition], [0.461], [0.45], [0.475], [+3.0%],
    [Interjection], [0.741], [0.45], [0.755], [+1.9%],
    [*Macro Average*], [*0.490*], [--], [*0.533*], [*+8.8%*],
  ),
  caption: [Threshold optimization results: default vs. tuned thresholds]
)

Threshold tuning brought macro F1 from 0.490 to 0.533. The block class improved the most (0.243 to 0.401). The default threshold of 0.5 is too high for classes with strong imbalance, and per-class tuning is a cheap way to improve results.

=== Comparative Architecture Study
Seven architectures were compared. They range from five independent Wav2Vec2 classifiers to various CNN-based multitask models. Test set and cross-corpus Boli results are in Table 4.3.

#add_table(
  table(
    columns: (1.5fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: horizon,
    table.header([*Architecture*], [*Type*], [*Parameters*], [*Test F1 (0.5)*], [*Test F1 (Tuned)*], [*Boli F1 (Tuned)*]),
    [5x Wav2Vec2], [Classifier], [94.6M], [0.511], [0.518], [0.154],
    [MT Wav2Vec2 (frz3)], [Multitask], [97.3M], [0.490], [0.522], [0.160],
    [MT Wav2Vec2 (frz20)], [Multitask], [97.3M], [0.142], [0.339], [0.295],
    [CNN Pool], [Multitask], [0.34M], [0.188], [0.253], [0.359],
    [CNN Single], [Multitask], [0.27M], [0.221], [0.253], [0.441],
    [CNN-LSTM], [Multitask], [0.46M], [0.248], [0.259], [0.521],
    [CNN-Transformer], [Multitask], [0.54M], [0.255], [0.264], [0.477],
  ),
  caption: [Comparative study of seven architecture variants on test and Boli cross-corpus sets]
)

Wav2Vec2 models score highest in-distribution (F1 up to 0.522), helped by large-scale pretrained speech representations. CNN models have far fewer parameters (0.27M to 0.54M vs. 94M to 97M) but lower in-distribution scores. On the Boli cross-corpus set, the picture flips: CNN-LSTM reaches 0.521, the best cross-corpus result. Lighter models seem to generalize better to unseen speakers and languages, while heavier pretrained models do better on familiar data.

Freezing the backbone for 3 epochs (arm02) clearly beats 20 epochs (arm03). Early unfreezing matters for fine-tuning.

#add_image(align(center, image("/assets/classifier_boli_f1.png", width: 90%)), caption: [Per-class tuned F1 on the Project Boli cross-corpus set for the shared-backbone Wav2Vec2 model and the CNN-LSTM model])

#add_image(
  grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 5pt,
    row-gutter: 5pt,
    image("/assets/prolongation_confusion_matrix.png", width: 100%),
    image("/assets/block_confusion_matrix.png", width: 100%),
    image("/assets/soundrep_confusion_matrix.png", width: 100%),
    image("/assets/wordrep_confusion_matrix.png", width: 100%),
    image("/assets/interjection_confusion_matrix.png", width: 100%),
  ),
  caption: [Confusion matrices (TP, FP, TN, FN) for the five binary Wav2Vec2 classifiers (5x Wav2Vec2) on the clip-level test set. Top row: prolongation, block, and sound repetition; bottom row: word repetition and interjection.]
)

== LOCALIZATION RESULTS

=== Wav2Vec2 Localizer
The Wav2Vec2 localizer was evaluated on 3,715 test samples at threshold 0.5. Results are in Table 4.4.

#add_table(
  table(
    columns: (1.5fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: horizon,
    table.header([*Metric*], [*Value*], [*Metric*], [*Value*]),
    [Frame Precision], [0.676], [Detection Accuracy], [0.210],
    [Frame Recall], [0.065], [Mean IoU], [0.751],
    [Frame F1], [0.119], [False Alarms/min], [8.95],
    [Frame Specificity], [0.973], [Predicted Events], [2,852],
  ),
  caption: [Wav2Vec2 localizer evaluation results on the test set]
)

Frame-level precision is high (0.676): when the model predicts dysfluency, it is usually right. Recall is low (0.065), so it misses most dysfluency frames. The mean IoU of 0.751 means the regions it does find overlap well with ground truth. In short, the localizer is conservative: few false alarms, but many missed events.

#add_image(align(center, image("/assets/localizer_metrics.png", width: 90%)), caption: [Frame-level metrics of the Wav2Vec2 localizer on the test set])

=== CNN Spectrogram Localizer
The CNN localizer was evaluated on only 2 test samples because of data availability at the time. It predicted a single dysfluency event with 100% recall but zero precision. A larger annotated localization dataset is needed before drawing conclusions.

== TRAINING CHALLENGES AND LESSONS LEARNED
Several problems came up during training and were resolved:

- BCEWithLogitsLoss with a single logit per class created a zero-gradient equilibrium where the model collapsed to always predicting the majority class. Switching to CrossEntropyLoss with two logits (present and not present) fixed this.
- Focal Loss with gamma=2.0 was adopted to handle the class imbalance across dysfluency types, especially for underrepresented classes like block and word repetition.
- Gradient clipping silently killed learning by scaling classifier head gradients down. Removing it improved stability.
- Mixed-precision training with GradScaler produced NaN gradients. Using torch.autocast alone (without the GradScaler) worked.
- Spectrogram augmentation destroyed inputs when waveform-level augmentors were applied to spectrogram tensors. Separating the two augmentation pipelines fixed this.
- Freezing the backbone for 3 epochs let the classifier head stabilize before fine-tuning. Longer freeze durations hurt performance.

== CHAPTER SUMMARY
This chapter covered the testing methodology, metrics, and results for both classification and localization.
The multitask Wav2Vec2 classifier reached a macro F1 of 0.533 with tuned thresholds. Interjection was the easiest class (F1=0.755); block was the hardest (F1=0.401).
The comparative study showed Wav2Vec2 models winning in-distribution and lighter CNN models generalizing better cross-corpus.
The Wav2Vec2 localizer had high precision and IoU but low recall, so it misses many events even though the ones it finds are accurate.

The next chapter concludes the report.

#pagebreak()
