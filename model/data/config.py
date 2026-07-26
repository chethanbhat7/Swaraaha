# Swaraaha - Data Pipeline Configuration
# Central configuration for dataset downloading, merging, and preparation.
# Adapted from Major-Project/config.py for Swaraaha.

import os

from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "RawData")
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

COMBINED_DATASET_FILENAME = "combined_labels.csv"
COMBINED_DATASET_PATH = os.path.join(DATASET_DIR, COMBINED_DATASET_FILENAME)

# ============================================================================
# DOWNLOAD CONFIGURATION
# ============================================================================

DOWNLOAD_MAX_WORKERS = 3
KAGGLE_DOWNLOAD_TIMEOUT_SECONDS = 3600
GIT_CLONE_TIMEOUT_SECONDS = 600

# Datasets to download (add new datasets here)
DATASET_LIST = [
    {
        "name": "Project Boli Dataset",
        "type": "git",
        "source": "https://github.com/projectboli/Project_Boli_Dataset",
    },
    {
        "name": "SEP-28K Dataset",
        "type": "kaggle",
        "source": "ikrbasak/sep-28k",
    },
    {
        "name": "UCLASS SEP-28K Format",
        "type": "kaggle",
        "source": "vudominhgiang/uclass-stuttered-speech-clips-sep-28k-format",
    },
]

# ============================================================================
# LABELS
# ============================================================================

DYSFLUENCY_LABELS = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection",
]

# Annotation code -> label mapping for Project Boli transcripts
PROJECT_BOLI_LABEL_MAP = {
    "B": "Block",
    "PR": "Prolongation",
    "SR": "SoundRep",
    "WR": "WordRep",
    "IN": "Interjection",
}

# ============================================================================
# WORKFLOW
# ============================================================================

WORKFLOW_TIMEOUT_SECONDS = 3600
