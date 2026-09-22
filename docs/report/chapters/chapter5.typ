#import "../lib.typ": *
#import "../meta.typ": *

// --- Chapter 5: System Testing ---
#chapter_heading[SYSTEM TESTING]

== INTRODUCTION
Testing is a crucial phase in the development of the proposed system for stutter detection and localization, ensuring the overall accuracy, stability, and reliability of both the individual models and the hybrid pipeline.
Roughly one percent of people worldwide are affected by stuttering, and diagnosis today still relies mostly on manual assessment by speech-language pathologists, which is subjective, time-consuming, and not available to everyone, making early and reliable detection essential.
In this research, Wav2Vec 2.0-based models were initially utilized to classify speech dysfluencies from audio.
To further enhance diagnostic precision, a hybrid framework was developed by combining five independent Wav2Vec 2.0 binary classifiers with the Wav2Vec2 frame-level localizer and Whisper automatic speech recognition, enabling classification, localization, and transcription in a single pipeline.
The testing process involved validating every stage of the pipeline, including data preprocessing, model training, classification, localization, and web-based visualization, through unit, integration, functional, and system tests.
Comprehensive testing confirmed that the proposed system satisfies all functional and non-functional requirements while maintaining reliable classification and localization.
Performance evaluation using precision, recall, F1-score, AUROC, AUPRC, specificity, and confusion matrix analysis, along with evaluation on the held-out test set and a cross-corpus set, demonstrated the robustness and generalization ability of the models.
The results are broken down by model family and dysfluency class, and the two model families, the Wav2Vec2-based models and the CNN-based models, are compared against each other and analyzed for their strengths and weaknesses.

== TYPES OF TESTS
Four types of tests were performed: unit testing, integration testing, functional testing, and system testing.
Together they verify the correctness of individual components, the interfaces between them, the complete feature set, and the behavior of the finished system.

=== Unit Testing
Unit testing isolates the smallest blocks of code, specific functions and classes, to verify they work as expected.
The project uses pytest suites organised under `model/tests`, `backend/tests`, `app/tests`, and `shared/tests`.

Unit tests verify the preprocessing steps (sample-rate conversion, DC offset removal, peak normalization, silence trimming, and fixed-length padding), dataset normalization and the merged `combined_labels.csv` format, model registry checkpoint loading, the combiner and CTC alignment logic, and the computation of the evaluation metrics.
On the desktop side, individual widgets such as the waveform view, results panel, and audio handler are tested in isolation.
Ruff enforces consistent code style across the codebase.

=== Integration Testing
Integration testing ensures that software modules, when connected, work together seamlessly.
The backend tests exercise the FastAPI endpoints (`/api/analyze`, `/api/classify`, `/api/localize`, and `/api/report`), which chain preprocessing, classification, transcription, localization, and fusion into a single analysis response.
The classification, fusion, localizer, severity, and report generator services are tested as combined units, and the model registry is checked against the classifier and localization pipelines.
The desktop application calls the shared model package directly, and its model runner is integration-tested against the same service layer.

=== Functional Testing
Functional testing validates the complete set of features against the specifications under typical and edge cases.
The workflow of recording or uploading audio, running the analysis, and viewing the results page with classification scores, waveform overlays, and the timestamped transcript is exercised end to end.
Report generation and download, severity mapping, multilingual transcription for English, Kannada, and Hindi, and history storage through IndexedDB and localStorage are all verified against the requirements defined in Chapter 2.

=== System Testing
System testing subjects the whole software product to end-to-end scenarios in conditions that resemble the user environment.
The React web application was exercised in supported browsers and the PySide6 desktop application in a standalone offline setup with no separate server, both against the same machine learning backend.
Supported audio formats (WAV, MP3, OGG, and M4A) were validated across the upload, drag-and-drop, and microphone capture paths, corrupted files were rejected with clear error messages, and analysis turnaround was confirmed to stay within a few seconds per clip as required by the performance requirements in Chapter 2.
The combined system-level suite confirmed that the application meets its functional and non-functional requirements.

== RESULT AND DISCUSSION
This section presents the evaluation of the trained models: how the Wav2Vec2-based and CNN-based model families were trained and evaluated, the hyper parameter tuning applied, and a comparative performance analysis of all architectures.

=== Model Evaluation of the Wav2Vec2 Classifiers
The model registry defines two Wav2Vec2-based classification configurations: the default `single` configuration of five independent Wav2Vec2 binary classifiers, and a `multitask` variant that shares a single backbone and five classification heads.
This section first analyzes the multitask variant in detail and then reports the shipped `single` configuration; the confusion matrices at the end of the section come from the shipped `single` configuration.

The multitask variant uses a single Wav2Vec2-base backbone with five shared binary heads.
Training used Focal Loss (gamma=2.0), backbone freezing for 3 epochs, AdamW (LR $3 times 10^(-5)$), a 500-step warm-up, and early stopping with a patience of 5.
Results on 3,715 held-out test samples at the default threshold of 0.5 are in Table 5.1.

#add_table(
  table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: horizon,
    table.header([*Class*], [*Precision*], [*Recall*], [*F1*], [*AUROC*], [*AUPRC*], [*Support*]),
    [Prolongation], [0.566], [0.440], [0.495], [0.845], [0.491], [400],
    [Block], [0.584], [0.264], [0.364], [0.768], [0.444], [553],
    [Sound Repetition], [0.705], [0.502], [0.586], [0.878], [0.624], [494],
    [Word Repetition], [0.621], [0.399], [0.486], [0.855], [0.550], [366],
    [Interjection], [0.810], [0.746], [0.776], [0.935], [0.848], [834],
    [*Macro Average*], [--], [--], [*0.542*], [--], [--], [*3,715*],
  ),
  caption: [Multitask Wav2Vec2 classifier results]
)

The per-class support counts cover positive samples; the macro row reports the total of 3,715 test clips because a clip may contain several stutter types at once.

The macro F1 at the default threshold is 0.542.
Interjection performs best (F1=0.776, AUROC=0.935), which makes sense because filler words like "um" and "uh" have distinct acoustic patterns.
Block is the hardest class (F1=0.364); silent pauses and hesitations are hard to distinguish from normal speech pauses.

#add_image(align(center, image("/assets/classifier_test_f1.png", width: 90%)), caption: [Per-class F1 of the multitask Wav2Vec2 classifier])

#add_image(align(center, image("/assets/classifier_test_auc.png", width: 90%)), caption: [Per-class AUROC and AUPRC of the multitask Wav2Vec2 classifier])

The shipped `single` configuration of five independent binary classifiers scores slightly higher in distribution, with a test F1 of 0.573 at threshold 0.5 (0.570 with tuned thresholds).

The confusion matrices below visualize the distribution of predictions of the shipped `single` configuration of five independent binary classifiers.

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
  caption: [Confusion matrices for the five binary classifiers]
)

The Wav2Vec2-based model family also provides the frame-level localizer, evaluated on the same 3,715 test samples at a threshold of 0.5; the shipped registry inference threshold for this localizer is 0.3. Results are in Table 5.2.

#add_table(
  table(
    columns: (1.5fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: horizon,
    table.header([*Metric*], [*Value*], [*Metric*], [*Value*]),
    [Frame Precision], [0.805], [Detection Accuracy], [0.586],
    [Frame Recall], [0.738], [Mean IoU], [0.779],
    [Frame F1], [0.770], [False Alarms/min], [4.28],
    [Frame Specificity], [0.847], [Predicted Events], [1,846],
  ),
  caption: [Wav2Vec2 localizer evaluation results]
)

The localizer detects most dysfluency events with good accuracy: frame precision is 0.805 and recall is 0.738, so most true events are found and most predictions are correct.
The mean IoU of 0.779 means the regions it finds overlap well with ground truth.
Of 995 true events it predicted 1,846, of which 1,263 were false alarms (4.28 per minute).

#add_image(align(center, image("/assets/localizer_metrics.png", width: 90%)), caption: [Frame-level metrics of the Wav2Vec2 localizer])

=== Wav2Vec2 Classifier Training
The training phase prepares the models that the system relies on for classification and localization.

The system is trained and evaluated on three publicly available stuttering datasets: Project Boli, SEP-28K, and UCLASS.
Project Boli is a multilingual Indian language dataset with stuttering data across Hindi, Kannada, Telugu, and other Indian languages, sourced from GitHub, and includes both read and spontaneous speech.
SEP-28K provides roughly 28,000 audio clips from stuttering support group podcasts, annotated for blocks, prolongations, sound repetitions, word repetitions, and interjections, and was obtained from Kaggle.
UCLASS is an additional stuttering speech dataset from Kaggle that provides supplementary annotated clips.
Each dataset is normalized into a unified format with a `combined_labels.csv` file of multi-label binary annotations and interval files per clip.
The merged dataset is split 80:10:10 into training, validation, and testing subsets, and during training the training subset is re-split 80/20 by a stratified split into a fine-tuning set and a validation set used for early stopping and threshold selection.

Each classifier is trained in a two-stage scheme.
The backbone stays frozen for the first three epochs so the classification head can stabilize, then it is unfrozen with a learning rate scaled to 0.1 of the head learning rate.
Focal Loss with a gamma value of 2 handles the class imbalance, and the AdamW optimizer runs at a learning rate of $3 times 10^(-5)$ with a weight decay of 0.01 and a 500-step warm-up.
Early stopping with a patience of 5 epochs prevents overfitting, and the checkpoint with the highest macro F1-score on the validation set is saved as the final model.
During training, data augmentation applies waveform-level transforms (random noise injection, time stretching, pitch shifting, temporal shifting, and amplitude scaling) and spectrogram-level transforms (time and frequency masking).
Checkpoints are stored under names that encode their training hyperparameters through a fingerprint convention, with checkpoint-based resume, an audio preprocessing cache for fast re-runs, `torch.compile` for roughly twice the training speed on CUDA, and gradient accumulation for effective batch size scaling.

Several problems encountered during training were resolved.
BCEWithLogitsLoss with a single logit per class created a zero-gradient equilibrium where the model collapsed to always predicting the majority class; switching to CrossEntropyLoss with two logits (present and not present) fixed this.
Gradient clipping in the classifier loop silently killed learning by scaling the classifier head gradients down, so it was removed from the classification loop while the localizer loops retained it.
Mixed-precision training with a GradScaler produced NaN gradients; using `torch.autocast` alone worked.
Freezing the backbone for 3 epochs let the classifier head stabilize before fine-tuning, and longer freeze durations hurt performance.

=== Model Evaluation of the CNN-based Models
The CNN-based models form the second model family, built to run classifications with far fewer parameters and to compare generalization against the pre-trained Wav2Vec2 baseline.

Four CNN-based multitask classification architectures were evaluated: CNN Pool, CNN Single, CNN-LSTM, and CNN-Transformer, ranging from 0.27M to 0.54M parameters.
On the held-out test set they score markedly lower than the Wav2Vec2 classifiers, with tuned F1 values between 0.249 and 0.271.
On the cross-corpus Boli set, however, the ranking flips: CNN Single reaches 0.476 and CNN-LSTM 0.508, the best cross-corpus result of all seven architectures, suggesting that the lighter models generalize better to unseen speakers and languages.

The CNN spectrogram localizer was evaluated on the same 3,715 test samples at a threshold of 0.5.
It is far more permissive than the Wav2Vec2 localizer: frame recall is high (0.873) but precision is much lower (0.527), and it predicted 4,483 events against 995 true events, with 4,329 false alarms (23.49 per minute).
Its detection accuracy (0.155) and specificity (0.330) are correspondingly low.
The heavier Wav2Vec2 localizer is therefore shipped as the default; the registry inference threshold is 0.3 for both localizers.

=== CNN-based Model Training
The CNN-based models are trained on the same unified dataset and with the same two-stage scheme used for the Wav2Vec2 classifiers, but they consume mel-spectrograms instead of raw audio embeddings.
CNN Pool applies global pooling over the convolutional feature maps, CNN Single processes a single spectrogram path, CNN-LSTM adds a recurrent layer to capture temporal context, and CNN-Transformer uses a transformer encoder over the spectrogram features.
Spectrogram-level augmentation, namely time and frequency masking, is applied during training alongside the waveform-level transforms.
Because these networks have only 0.27M to 0.54M parameters compared with roughly 94M to 97M for the Wav2Vec2 family, they train quickly and were used to study the trade-off between parameter count and cross-corpus generalization.

=== Hyper Parameter Tuning
A threshold sweep on the validation set tested thresholds from 0.1 to 0.9 in steps of 0.05 for each class.
The threshold that maximized F1 for each class was stored in the model registry under the multitask configuration. The results are in Table 5.3.

#add_table(
  table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: horizon,
    table.header([*Class*], [*Default (0.5) F1*], [*Optimal Threshold*], [*Tuned F1*], [*Improvement*]),
    [Prolongation], [0.495], [0.55], [0.457], [-7.8%],
    [Block], [0.364], [0.35], [0.429], [+17.8%],
    [Sound Repetition], [0.586], [0.40], [0.621], [+6.0%],
    [Word Repetition], [0.486], [0.35], [0.499], [+2.8%],
    [Interjection], [0.776], [0.45], [0.773], [-0.5%],
    [*Macro Average*], [*0.542*], [--], [*0.556*], [*+2.6%*],
  ),
  caption: [Threshold optimization results]
)

Threshold tuning raised the macro F1 from 0.542 to 0.556.
Block improved the most (0.364 to 0.429) because the default threshold of 0.5 is too high for a class with strong imbalance.
For prolongation and interjection the tuned threshold lowers F1 slightly on the test set, but per-class tuning on the validation set still yields a net gain of +2.6% overall.

Additional hyper parameters were tuned during development.
The backbone freeze duration was compared at 3 and 20 epochs; freezing for 3 epochs (MT frz3) clearly beats 20 epochs (MT frz20) on the test set, showing that early unfreezing matters for fine-tuning.
The learning rate scaling factor of 0.1 between the head and the unfrozen backbone, a 500-step warm-up, a weight decay of 0.01, and gradient accumulation for effective batch size scaling together gave the most stable training runs.

=== Performance Analysis of the Models
Seven architectures were compared in total, ranging from the shipped five independent Wav2Vec2 classifiers to various CNN-based multitask models.
Test set and cross-corpus Boli results are in Table 5.4.

#add_table(
  table(
    columns: (1.5fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: horizon,
    table.header([*Architecture*], [*Type*], [*Parameters*], [*Test F1 (0.5)*], [*Test F1 (Tuned)*], [*Boli F1 (Tuned)*]),
    [5x Wav2Vec2], [Classifier], [94.6M], [0.573], [0.570], [0.292],
    [MT Wav2Vec2 (frz3)], [Multitask], [97.3M], [0.542], [0.556], [0.238],
    [MT Wav2Vec2 (frz20)], [Multitask], [97.3M], [0.141], [0.378], [0.214],
    [CNN Pool], [Multitask], [0.34M], [0.149], [0.256], [0.332],
    [CNN Single], [Multitask], [0.27M], [0.212], [0.262], [0.476],
    [CNN-LSTM], [Multitask], [0.46M], [0.242], [0.249], [0.508],
    [CNN-Transformer], [Multitask], [0.54M], [0.269], [0.271], [0.345],
  ),
  caption: [Comparative study of seven architecture variants]
)

The shipped five-classifier Wav2Vec2 configuration scores highest in distribution (tuned F1 0.570), narrowly ahead of the multitask variant (0.556); both benefit from large pre-trained speech representations.
The CNN models have far fewer parameters but score lower on the test set.
On the Boli cross-corpus set the ranking flips: CNN-LSTM reaches 0.508, the best cross-corpus result, followed by CNN Single (0.476). Lighter models seem to generalize better to unseen speakers and languages, while heavy pre-trained models do better on familiar data.

#add_image(align(center, image("/assets/classifier_boli_f1.png", width: 90%)), caption: [Per-class tuned F1 on the Boli set])

It was concluded that the Wav2Vec2 family should be shipped for in-distribution accuracy, with the CNN-LSTM variant available as a lighter alternative that transfers best to unseen speakers and languages.

#pagebreak()