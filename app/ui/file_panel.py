"""File browser panel: filtered audio file tree + recent files list."""

import os

from PySide6.QtCore import QDir, QSettings, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFileSystemModel,
    QLabel,
    QListWidget,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from app.core.recent_files import (
    last_dir,
    load_recent_files,
    remember_last_dir,
    update_recent_files,
)

AUDIO_FILTERS = ["*.wav", "*.mp3", "*.flac"]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


class FilePanel(QWidget):
    file_selected = Signal(str)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self._settings = settings if settings is not None else QSettings()
        self._setup_ui()
        self._setup_tree()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel("Recent Files")
        header.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px 0;")
        layout.addWidget(header)

        self._recent_list = QListWidget()
        self._recent_list.itemActivated.connect(self._on_recent_activated)
        layout.addWidget(self._recent_list)

        self._browse_btn = QPushButton("Browse Audio...")
        self._browse_btn.setProperty("cssClass", "secondary")
        self._browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self._browse_btn)

        self._tree = QTreeView()
        self._tree.setHeaderHidden(True)
        self._tree.doubleClicked.connect(self._on_tree_double_clicked)
        layout.addWidget(self._tree, stretch=1)

        self._load_recents()

    def _setup_tree(self):
        self._fs_model = QFileSystemModel(self)
        self._fs_model.setNameFilters(AUDIO_FILTERS)
        self._fs_model.setNameFilterDisables(False)
        self._tree.setModel(self._fs_model)
        start = last_dir(self._settings) or QDir.homePath()
        self.set_current_dir(start)

    def _load_recents(self):
        self._recent_list.clear()
        for path in load_recent_files(self._settings):
            self._recent_list.addItem(os.path.basename(path))

    def _on_tree_double_clicked(self, index):
        path = self._fs_model.filePath(index)
        if os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS:
            self.add_recent(path)
            self.file_selected.emit(path)

    def _on_recent_activated(self, item):
        files = load_recent_files(self._settings)
        row = self._recent_list.row(item)
        if row < len(files) and os.path.isfile(files[row]):
            self.file_selected.emit(files[row])

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Audio", "", "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        if path:
            self.add_recent(path)
            self.set_current_dir(os.path.dirname(path))
            self.file_selected.emit(path)

    def add_recent(self, path: str):
        update_recent_files(self._settings, path)
        self._load_recents()

    def set_current_dir(self, path: str):
        if os.path.isdir(path):
            remember_last_dir(self._settings, path)
            self._fs_model.setRootPath(path)
            self._tree.setRootIndex(self._fs_model.index(path))

    def get_recent_files(self) -> list[str]:
        return load_recent_files(self._settings)
