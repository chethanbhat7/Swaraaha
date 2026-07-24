# Dataset Setup

---

## Prerequisites

```bash
pip install -r model/requirements.txt
```

This installs everything you need, including `kaggle` for dataset downloads.

You also need **Git** (for the Project Boli dataset).

Set your Kaggle credentials:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

Get your API key from [kaggle.com/settings](https://www.kaggle.com/settings).

---

## Quick Start

```bash
python -m model.data.setup
```

This downloads all datasets and merges them into one CSV in a single step.

---

## Manual Setup

### 1. Download Datasets

```bash
python -m model.data.download
```

Downloads three datasets into `RawData/`:

| Dataset | Source | Type |
|---------|--------|------|
| Project Boli | [GitHub](https://github.com/projectboli/Project_Boli_Dataset) | Git |
| SEP-28K | [Kaggle](https://www.kaggle.com/datasets/ikrbasak/sep-28k) | Kaggle |
| UCLASS | [Kaggle](https://www.kaggle.com/datasets/vudominhgiang/uclass-stuttered-speech-clips-sep-28k-format) | Kaggle |

Existing datasets are skipped.

### 2. Merge Datasets

```bash
python -m model.data.merge
```

Produces `Dataset/combined_labels.csv`:

```
clip_file,Prolongation,Block,SoundRep,WordRep,Interjection
/path/to/clip_001.wav,1,0,0,0,0
/path/to/clip_002.wav,0,0,1,0,1
```

Each row is one audio clip with binary labels for each dysfluency type.

---

## Data Formats

Both `ClassificationDataset` and `LocalizationDataset` expect this layout:

```
data_dir/
├── audio/
│   ├── clip_001.wav
│   └── ...
└── labels/
    ├── clip_001.csv
    └── ...
```

Label CSV format:

```
start_sec,end_sec,dysfluency_type
0.5,1.2,prolongation
2.1,2.8,soundrep
```

- `ClassificationDataset` aggregates intervals into a multi-hot vector `(5,)`.
- `LocalizationDataset` converts intervals into a binary frame mask aligned to the spectrogram.

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

## Using in Training

```python
from model.data.dataset import ClassificationDataset, LocalizationDataset

# Classification
dataset = ClassificationDataset(data_dir="data/train", sr=16000)
audio, labels = dataset[0]  # audio: (160000,), labels: (5,) multi-hot

# Localization
dataset = LocalizationDataset(data_dir="data/train", sr=16000)
spectrogram, frame_labels = dataset[0]  # spectrogram: (1, 128, T), frame_labels: (T,)
```

---

## Configuration

Paths and settings live in `model/data/config.py`:

```python
RAW_DATA_DIR  # <project_root>/RawData
DATASET_DIR   # <project_root>/Dataset
```

To add a dataset: add an entry to `DATASET_LIST` in `config.py` and write a normalizer in `merge.py`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `kaggle: command not found` | `pip install -r model/requirements.txt` |
| `KAGGLE_KEY not set` | Export `KAGGLE_USERNAME` and `KAGGLE_KEY` env vars |
| `Git clone timeout` | Increase `GIT_CLONE_TIMEOUT_SECONDS` in `config.py` |
| `Dataset not found, skipping` | Check the download succeeded — look in `RawData/` |
| `No valid examples found` | Verify directory structure matches expected layout |
