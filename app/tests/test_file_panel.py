from app.core import recent_files
from app.ui.file_panel import AUDIO_EXTENSIONS, FilePanel


class FakeSettings:
    def __init__(self):
        self._data = {}

    def value(self, key, default=None):
        return self._data.get(key, default)

    def setValue(self, key, value):
        self._data[key] = value


def _touch(path):
    path.write_bytes(b"x")


def test_add_recent_updates_list_and_settings(qapp, tmp_path):
    a = tmp_path / "a.wav"
    _touch(a)
    settings = FakeSettings()
    panel = FilePanel(settings=settings)
    panel.add_recent(str(a))
    assert panel.get_recent_files() == [str(a)]
    assert panel._recent_list.count() == 1
    assert panel._recent_list.item(0).text() == "a.wav"
    assert recent_files.load_recent_files(settings) == [str(a)]


def test_recent_activation_emits_file_selected(qapp, tmp_path):
    a = tmp_path / "a.wav"
    _touch(a)
    panel = FilePanel(settings=FakeSettings())
    emitted = []
    panel.file_selected.connect(emitted.append)
    panel.add_recent(str(a))
    panel._on_recent_activated(panel._recent_list.item(0))
    assert emitted == [str(a)]


def test_tree_double_click_emits_file_selected(qapp, tmp_path):
    a = tmp_path / "a.wav"
    _touch(a)
    panel = FilePanel(settings=FakeSettings())
    panel.set_current_dir(str(tmp_path))
    emitted = []
    panel.file_selected.connect(emitted.append)
    index = panel._fs_model.index(str(a))
    panel._on_tree_double_clicked(index)
    assert emitted == [str(a)]


def test_tree_ignores_non_audio_file(qapp, tmp_path):
    txt = tmp_path / "notes.txt"
    _touch(txt)
    panel = FilePanel(settings=FakeSettings())
    panel.set_current_dir(str(tmp_path))
    emitted = []
    panel.file_selected.connect(emitted.append)
    panel._on_tree_double_clicked(panel._fs_model.index(str(txt)))
    assert emitted == []


def test_filter_hides_non_audio(qapp, tmp_path):
    txt = tmp_path / "notes.txt"
    _touch(txt)
    panel = FilePanel(settings=FakeSettings())
    panel.set_current_dir(str(tmp_path))
    assert panel._fs_model.nameFilters() == ["*.wav", "*.mp3", "*.flac"]
    assert AUDIO_EXTENSIONS == {".wav", ".mp3", ".flac"}
