# Data Pipeline

---

## Quick Start

```bash
pip install -r model/requirements.txt
```

Set your Kaggle credentials in a `.env` file in the project root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Or export them:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

Then run the full setup:

```bash
python -m model.data.setup
```

This runs three steps sequentially:
1. **Download** — fetches all datasets from Kaggle/Git
2. **Merge** — normalizes labels and writes per-clip interval CSVs
3. **Prepare** — creates train/val/test splits with symlinks

---

## Dataset Health Report

Check dataset status without re-running setup:

```bash
python -m model.data.status
```

Shows total clips, label distribution, missing/empty files, and per-split counts.

---

## Pipeline Steps

### 1. Download

```bash
python -m model.data.download
```

Downloads into `RawData/`:

| Dataset | Source | Type |
|---------|--------|------|
| Project Boli | [GitHub](https://github.com/projectboli/Project_Boli_Dataset) | Git |
| SEP-28K | [Kaggle](https://www.kaggle.com/datasets/ikrbasak/sep-28k) | Kaggle |
| UCLASS | [Kaggle](https://www.kaggle.com/datasets/vudominhgiang/uclass-stuttered-speech-clips-sep-28k-format) | Kaggle |

Existing downloads are skipped.

### 2. Merge

```bash
python -m model.data.merge
```

Normalizes all three datasets into a single structure:

```
data/
├── combined_labels.csv     # clip_file + binary columns (0/1)
├── labels/                 # per-clip interval CSVs
│   ├── clip_001.csv        # start_sec,end_sec,dysfluency_type
│   ├── clip_002.csv
│   └── ...
```

Each label CSV records dysfluency intervals in seconds:

```
start_sec,end_sec,dysfluency_type
0.50,1.23,prolongation
2.10,2.80,soundrep
```

Datasets with differing formats are handled:
- **Project Boli**: Parses `.txt` transcripts with annotation codes (`B`, `PR`, `SR`, `WR`, `IN`)
- **SEP-28K**: Reads TSV files, handles un-padded filenames via glob lookup
- **UCLASS**: Reads `metadata.json`, handles missing `.wav` extensions via glob lookup

### 3. Prepare

```bash
python -m model.data.prepare
```

Creates train/val/test splits (80/10/10) with stratified label distribution:

```
data/
├── train/
│   ├── audio/              # symlinks to original WAV files
│   └── labels/             # copied interval CSVs
├── val/
│   ├── audio/
│   └── labels/
├── test/
│   ├── audio/
│   └── labels/
└── splits.json             # split assignments for reproducibility
```

Filters out clips with missing audio files and header-only WAV files (44-byte empty files with no audio data).

---

## Data Formats

### Label CSV

```
start_sec,end_sec,dysfluency_type
0.5,1.2,prolongation
2.1,2.8,soundrep
```

### Classification Dataset

`ClassificationDataset` aggregates intervals into a multi-hot vector of shape `(5,)`:

```python
from model.data.dataset import ClassificationDataset

dataset = ClassificationDataset(data_dir="data/train", sr=16000)
audio, labels = dataset[0]
# audio: float32 ndarray, shape (160000,)
# labels: uint8 ndarray, shape (5,) — [prolongation, block, soundrep, wordrep, interjection]
```

**Audio cache:** Preprocessed audio is automatically cached to `data/cache/{split}` (pickle). The first epoch processes every sample from scratch; subsequent runs load the cached pickles instantly, saving repeated `load_audio` + `clean_audio` overhead.

### Localization Dataset

`LocalizationDataset` converts intervals into a binary frame mask aligned to the spectrogram:

```python
from model.data.dataset import LocalizationDataset

dataset = LocalizationDataset(data_dir="data/train", sr=16000)
spectrogram, frame_labels = dataset[0]
# spectrogram: float32 ndarray, shape (1, 128, T)
# frame_labels: uint8 ndarray, shape (T,) — 1 = dysfluent
```

---

## Dysfluency Types

| Label | What it is |
|-------|------------|
| `prolongation` | Stretched-out sound (e.g., "sssssoup") |
| `block` | Silence or tense pause before a sound |
| `soundrep` | Repeating a sound/syllable (e.g., "b-b-ball") |
| `wordrep` | Repeating a whole word (e.g., "I-I-I want") |
| `interjection` | Filler words (e.g., "um", "uh", "like") |

---

## Augmentation

```python
from model.data.augmentation import AudioAugmentor, AugmentedDataset

augmentor = AudioAugmentor(
    noise_level=0.005,          # Gaussian noise
    time_stretch_range=(0.9, 1.1),
    pitch_shift_range=(-1.0, 1.0),
    shift_range=(-0.1, 0.1),    # temporal roll
    scale_range=(0.8, 1.2),     # amplitude scaling
)

# Wrap any torch Dataset
augmented_dataset = AugmentedDataset(dataset, augmentor=augmentor)
```

All transforms produce `float32` output. The augmentor is applied on-the-fly during dataloading.

---

## Preprocessing

Located in `model/data/preprocessing.py`:

| Function | Purpose |
|----------|---------|
| `load_audio()` | Load and resample audio to 16kHz |
| `clean_audio()` | DC removal + peak normalization + silence trim |
| `generate_mel_spectrogram()` | Log-mel spectrogram, shape `(n_mels, T)` |
| `normalize_spectrogram()` | Z-score or min-max normalization |
| `audio_to_spectrogram()` | End-to-end: audio → (1, n_mels, T) tensor |
| `create_frame_labels()` | Interval list → binary frame mask |
| `compute_class_weights()` | Inverse-frequency weights for imbalance |
| `check_audio_quality()` | Detect clipping, silence, DC offset |

---

## Configuration

All paths and settings live in `model/data/config.py`:

```python
RAW_DATA_DIR        # <project_root>/RawData
DATA_DIR            # <project_root>/data
COMBINED_DATASET_PATH  # <project_root>/data/combined_labels.csv
```

To add a dataset: add an entry to `DATASET_LIST` in `config.py` and write a normalizer in `merge.py`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `kaggle: command not found` | `pip install -r model/requirements.txt` |
| `KAGGLE_KEY not set` | Create a `.env` file with `KAGGLE_USERNAME` and `KAGGLE_KEY` |
| `Git clone timeout` | Increase `GIT_CLONE_TIMEOUT_SECONDS` in `config.py` |
| `Dataset not found, skipping` | Check the download succeeded in `RawData/` |
| `No valid examples found` | Verify directory structure matches expected layout |
| All labels are zero | Check that `CombinedLabels/labels/*.csv` use lowercase types (`prolongation` not `Prolongation`) |
