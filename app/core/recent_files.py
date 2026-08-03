"""Recent audio file tracking backed by QSettings. Pure logic, no Qt widget imports."""

import os

MAX_RECENT_FILES = 10
RECENT_FILES_KEY = "fileBrowser/recentFiles"
LAST_DIR_KEY = "fileBrowser/lastDir"


def load_recent_files(settings) -> list[str]:
    """Return recent audio file paths that still exist on disk."""
    raw = settings.value(RECENT_FILES_KEY, []) or []
    return [p for p in raw if os.path.isfile(p)]


def update_recent_files(settings, path: str) -> list[str]:
    """Insert path at the front of the recent list, dedupe, cap, and persist."""
    current = [p for p in (settings.value(RECENT_FILES_KEY, []) or []) if os.path.isfile(p)]
    current = [p for p in current if p != path]
    current.insert(0, path)
    current = current[:MAX_RECENT_FILES]
    settings.setValue(RECENT_FILES_KEY, current)
    return current


def remember_last_dir(settings, path: str) -> None:
    """Persist the last browsed directory."""
    settings.setValue(LAST_DIR_KEY, path)


def last_dir(settings) -> str | None:
    """Return the last browsed directory if it still exists, else None."""
    value = settings.value(LAST_DIR_KEY)
    if isinstance(value, str) and os.path.isdir(value):
        return value
    return None
