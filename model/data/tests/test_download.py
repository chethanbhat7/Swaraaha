"""
Tests for dataset downloading (model/data/download.py).
"""

import pytest

from model.data import download as download_mod
from model.data.download import Dataset


@pytest.fixture
def fake_raw(tmp_path, monkeypatch):
    """Point download.RAW_DATA_DIR at a tmp_path."""
    monkeypatch.setattr(download_mod, "RAW_DATA_DIR", str(tmp_path))
    return tmp_path


def _dataset_with_handler(name, result=(True, "stubbed handler ran")):
    ds = Dataset(name=name, type="git", source="https://example.com/repo")
    ds._handler = lambda: result
    return ds


def test_download_skips_existing_non_empty_dir(fake_raw):
    """A fully populated dataset directory is reported as already present."""
    target = fake_raw / "Present Dataset"
    target.mkdir()
    (target / "data.wav").write_bytes(b"\x00" * 100)

    ds = _dataset_with_handler("Present Dataset", result=(False, "should not run"))
    ok, msg = ds.download()

    assert ok is True
    assert "already exists" in msg


def test_download_redownloads_empty_dir(fake_raw):
    """An empty directory from a failed download must NOT count as success."""
    target = fake_raw / "Partial Dataset"
    target.mkdir()

    ds = _dataset_with_handler("Partial Dataset")
    ok, msg = ds.download()

    assert ok is True
    assert msg == "stubbed handler ran"


def test_main_returns_zero_on_full_success(monkeypatch):
    """main() returns 0 when every dataset downloads successfully."""
    monkeypatch.setattr(
        download_mod,
        "download_datasets",
        lambda: {"A": (True, "ok"), "B": (True, "ok")},
    )
    assert download_mod.main() == 0


def test_main_returns_nonzero_on_any_failure(monkeypatch):
    """main() returns nonzero when at least one dataset fails to download."""
    monkeypatch.setattr(
        download_mod,
        "download_datasets",
        lambda: {"A": (True, "ok"), "B": (False, "boom")},
    )
    assert download_mod.main() == 1


def test_main_returns_nonzero_on_exception(monkeypatch):
    """main() propagates unexpected exceptions as a failure exit code."""
    def raise_error():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(download_mod, "download_datasets", raise_error)
    assert download_mod.main() == 1
