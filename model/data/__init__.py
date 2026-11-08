# Swaraaha - Data Processing
# Audio loading, preprocessing, spectrogram generation,
# dataset downloading, merging, and preparation.

from model.data.config import (
    COMBINED_DATASET_PATH,
    DATA_DIR,
    DYSFLUENCY_LABELS,
    RAW_DATA_DIR,
)

__all__ = [
    "COMBINED_DATASET_PATH",
    "DATA_DIR",
    "DYSFLUENCY_LABELS",
    "RAW_DATA_DIR",
    "Dataset",
    "download_datasets",
    "merge_datasets",
    "AugmentedDataset",
    "AudioAugmentor",
]


def __getattr__(name: str):
    if name in {"Dataset", "download_datasets"}:
        from model.data.download import Dataset, download_datasets

        globals().update({"Dataset": Dataset, "download_datasets": download_datasets})
        return globals()[name]
    if name == "merge_datasets":
        from model.data.merge import merge_datasets

        globals()["merge_datasets"] = merge_datasets
        return merge_datasets
    if name in {"AugmentedDataset", "AudioAugmentor"}:
        from model.data.augmentation import AugmentedDataset, AudioAugmentor

        globals().update(
            {"AugmentedDataset": AugmentedDataset, "AudioAugmentor": AudioAugmentor}
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
