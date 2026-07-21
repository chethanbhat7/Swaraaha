# Swaraaha - Dataset Downloader
# Downloads datasets from Kaggle and Git sources in parallel.
# Adapted from Major-Project/Scripts/download_datasets.py.

import concurrent.futures
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional

from model.data.config import (
    DATASET_LIST,
    DOWNLOAD_MAX_WORKERS,
    GIT_CLONE_TIMEOUT_SECONDS,
    KAGGLE_DOWNLOAD_TIMEOUT_SECONDS,
    RAW_DATA_DIR,
)


class DatasetType(Enum):
    GIT = "git"
    KAGGLE = "kaggle"
    UNKNOWN = "unknown"


class Dataset:
    """Represents a downloadable dataset from Kaggle or Git."""

    def __init__(self, name: str, type: str, source: str):
        self.name = name
        self.source = source
        self.dtype = DatasetType(type)

        handlers = {
            DatasetType.GIT: self._download_git,
            DatasetType.KAGGLE: self._download_kaggle,
        }
        self._handler = handlers.get(self.dtype)

    def get_path(self) -> Path:
        return Path(RAW_DATA_DIR) / self.name

    def download(self) -> tuple:
        """Download the dataset. Returns (success: bool, message: str)."""
        if self.get_path().exists():
            return True, f"{self.name} already exists at {self.get_path()}"

        if self._handler is None:
            return False, f"Unknown dataset type: {self.dtype}"

        return self._handler()

    def _download_kaggle(self) -> tuple:
        target_dir = self.get_path()
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"  Downloading {self.name} from Kaggle...")
            subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "-d", self.source,
                    "-p", str(target_dir),
                    "--unzip",
                ],
                capture_output=True,
                check=True,
                timeout=KAGGLE_DOWNLOAD_TIMEOUT_SECONDS,
            )
            return True, f"  {self.name} downloaded successfully."
        except FileNotFoundError:
            return (
                False,
                "  Kaggle CLI not found. Install with: pip install kaggle",
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
            if "Authentication" in stderr or "401" in stderr:
                return (
                    False,
                    "  Kaggle authentication failed. Set KAGGLE_USERNAME and "
                    "KAGGLE_KEY environment variables.",
                )
            return False, f"  Failed to download {self.name}: {stderr}"
        except Exception as e:
            return False, f"  Error downloading {self.name}: {e}"

    def _download_git(self) -> tuple:
        target_dir = self.get_path()
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"  Cloning {self.name}...")
            subprocess.run(
                ["git", "clone", self.source, str(target_dir)],
                capture_output=True,
                check=True,
                timeout=GIT_CLONE_TIMEOUT_SECONDS,
            )
            return True, f"  {self.name} cloned successfully."
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
            return False, f"  Failed to clone {self.name}: {stderr}"
        except Exception as e:
            return False, f"  Error cloning {self.name}: {e}"

    def __repr__(self):
        return f"Dataset(name={self.name!r}, type={self.dtype.value})"


def download_datasets(
    dataset_list: Optional[list] = None,
) -> dict:
    """
    Download all configured datasets in parallel.

    Args:
        dataset_list: Override dataset list. Defaults to config.DATASET_LIST.

    Returns:
        Dict mapping dataset name -> (success, message).
    """
    if dataset_list is None:
        dataset_list = DATASET_LIST

    datasets = [Dataset(**entry) for entry in dataset_list]

    print("Datasets to download:")
    for ds in datasets:
        print(f"  - {ds.name} ({ds.dtype.value})")

    print(f"\nStarting parallel downloads ({DOWNLOAD_MAX_WORKERS} workers)...")
    results = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=DOWNLOAD_MAX_WORKERS,
    ) as executor:
        futures = {
            executor.submit(ds.download): ds.name for ds in datasets
        }

        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                success, message = future.result()
                results[name] = (success, message)
                print(message)
            except Exception as e:
                results[name] = (False, f"Exception: {e}")
                print(f"  Exception downloading {name}: {e}")

    successful = sum(1 for ok, _ in results.values() if ok)
    print(f"\nDownload complete: {successful}/{len(results)} successful")

    return results


if __name__ == "__main__":
    download_datasets()
