import os

import pytest

from app.core import recent_files


class FakeSettings:
    def __init__(self):
        self._data = {}

    def value(self, key, default=None):
        return self._data.get(key, default)

    def setValue(self, key, value):
        self._data[key] = value


@pytest.fixture
def settings():
    return FakeSettings()


def _touch(path):
    path.write_bytes(b"x")


def test_add_to_front(settings, tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    _touch(a)
    _touch(b)
    recent_files.update_recent_files(settings, str(a))
    recent_files.update_recent_files(settings, str(b))
    assert recent_files.load_recent_files(settings) == [str(b), str(a)]


def test_dedupe_moves_to_front(settings, tmp_path):
    a = tmp_path / "a.wav"
    _touch(a)
    recent_files.update_recent_files(settings, str(a))
    recent_files.update_recent_files(settings, str(a))
    assert recent_files.load_recent_files(settings) == [str(a)]


def test_caps_at_max(settings, tmp_path):
    for i in range(15):
        p = tmp_path / f"f{i}.wav"
        _touch(p)
        recent_files.update_recent_files(settings, str(p))
    files = recent_files.load_recent_files(settings)
    assert len(files) == recent_files.MAX_RECENT_FILES
    assert files[0].endswith("f14.wav")
    assert "f0.wav" not in files


def test_prunes_missing(settings, tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    _touch(a)
    _touch(b)
    recent_files.update_recent_files(settings, str(a))
    recent_files.update_recent_files(settings, str(b))
    os.remove(str(a))
    assert recent_files.load_recent_files(settings) == [str(b)]


def test_last_dir_roundtrip(settings, tmp_path):
    assert recent_files.last_dir(settings) is None
    recent_files.remember_last_dir(settings, str(tmp_path))
    assert recent_files.last_dir(settings) == str(tmp_path)
