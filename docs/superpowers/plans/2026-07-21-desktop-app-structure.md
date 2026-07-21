# Swaraaha Desktop Application Structure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold and implement the Swaraaha PySide6 desktop application with a two-page layout (Home + Analysis), custom waveform rendering, PDF viewer, audio recording/playback, and Material 3 Expressive theming.

**Architecture:** Core modules (`core/`) are pure logic with no Qt UI imports. UI modules (`ui/`) import from `core/`. `main_window.py` wires everything via signals/slots. Two pages connected by `QStackedWidget`: Home (PDF panel + audio controls) → Analysis (waveform + results).

**Tech Stack:** PySide6, sounddevice, numpy, soundfile, pdfplumber

## Global Constraints

- PySide6>=6.5.0, sounddevice>=0.4.6, numpy>=1.24.0, soundfile>=0.12.0, pdfplumber>=0.10.0
- Font: Google Sans Flex (Regular, Medium, Bold) — bundled in `app/assets/fonts/`
- One audio at a time in memory
- Thread safety: audio threads must not touch Qt widgets — use signals
- Material 3 Expressive visual style
- Branch naming: `feat(app): <description>` (conventional commits)
- `core/` modules have zero Qt imports
- Audio sample rate: 16000 Hz mono

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `app/requirements.txt` | Create | Dependencies list |
| `app/ui/__init__.py` | Create | Package init |
| `app/ui/theme.py` | Create | Design tokens + QSS generation |
| `app/ui/styles.py` | Create | Full QSS stylesheet builder |
| `app/core/__init__.py` | Create | Package init |
| `app/core/audio_handler.py` | Create | sounddevice recording/playback |
| `app/core/pdf_handler.py` | Create | pdfplumber text extraction |
| `app/core/model_runner.py` | Create | Model inference stub |
| `app/ui/waveform_view.py` | Create | Custom QGraphicsView waveform |
| `app/ui/pdf_viewer.py` | Create | PDF file tree + text display |
| `app/ui/audio_controls.py` | Create | Record/Load/Play/Stop/Analyze buttons |
| `app/ui/results_panel.py` | Create | Classification table + localization timeline |
| `app/ui/home_page.py` | Create | Home page: PDF panel + audio controls |
| `app/ui/analysis_page.py` | Create | Analysis page: waveform + results |
| `app/ui/main_window.py` | Create | QMainWindow + QStackedWidget |
| `app/main.py` | Create | Entry point |
| `app/assets/fonts/` | Create dir | Google Sans Flex .ttf files |
| `app/assets/styles/default.qss` | Create | Base QSS |

---

### Task 1: Create Directory Structure + Requirements

**Files:**
- Create: `app/requirements.txt`
- Create: `app/core/__init__.py`
- Create: `app/ui/__init__.py`
- Create: `app/assets/fonts/.gitkeep`
- Create: `app/assets/styles/.gitkeep`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p app/core app/ui app/assets/fonts app/assets/styles
```

- [ ] **Step 2: Create `app/requirements.txt`**

```txt
PySide6>=6.5.0
sounddevice>=0.4.6
numpy>=1.24.0
soundfile>=0.12.0
pdfplumber>=0.10.0
```

- [ ] **Step 3: Create `app/core/__init__.py`**

```python
# Core modules — pure logic, no Qt imports
```

- [ ] **Step 4: Create `app/ui/__init__.py`**

```python
# UI modules — PySide6 widgets
```

- [ ] **Step 5: Create `.gitkeep` files**

```bash
touch app/assets/fonts/.gitkeep app/assets/styles/.gitkeep
```

- [ ] **Step 6: Verify structure**

Run: `find app -type f | sort`
Expected: all 5 files listed

- [ ] **Step 7: Commit**

```bash
git add app/
git commit -m "chore(app): scaffold directory structure and requirements"
```

---

### Task 2: Implement Theme Tokens + QSS Generation

**Files:**
- Create: `app/ui/theme.py`

**Produces:** `COLORS`, `SPACING`, `RADIUS` dicts, `generate_qss()` function, `load_font()` function

- [ ] **Step 1: Create `app/ui/theme.py`**

```python
"""Design tokens and QSS generation for Material 3 Expressive theme."""

import os
from PySide6.QtGui import QFontDatabase, QFont

COLORS = {
    "primary": "#6750A4",
    "on_primary": "#FFFFFF",
    "primary_container": "#EADDFF",
    "surface": "#FFFBFE",
    "surface_variant": "#F3EDF7",
    "on_surface": "#1C1B1F",
    "outline": "#79747E",
    "dysfluency": {
        "prolongation": "#B3261E",
        "block": "#7D5260",
        "soundrep": "#006D3F",
        "wordrep": "#0061A4",
        "interjection": "#984061",
    },
}

SPACING = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}

RADIUS = {"sm": 8, "md": 16, "lg": 24, "full": 9999}

FONT_FAMILY = "Google Sans Font"


def load_fonts():
    """Load Google Sans Flex fonts from assets/fonts/ directory."""
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts")
    font_files = [
        "GoogleSansFlex-Regular.ttf",
        "GoogleSansFlex-Medium.ttf",
        "GoogleSansFlex-Bold.ttf",
    ]
    loaded = []
    for fname in font_files:
        path = os.path.join(assets_dir, fname)
        if os.path.exists(path):
            font_id = QFontDatabase.addApplicationFont(path)
            if font_id != -1:
                loaded.append(fname)
    return loaded


def get_font(weight="regular"):
    """Return a QFont with the specified weight."""
    font = QFont(FONT_FAMILY)
    if weight == "bold":
        font.setBold(True)
    elif weight == "medium":
        font.setWeight(QFont.Weight.DemiBold)
    else:
        font.setWeight(QFont.Weight.Normal)
    return font
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.ui.theme import COLORS, SPACING, RADIUS, load_fonts; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/ui/theme.py
git commit -m "feat(app): add Material 3 design tokens and font loading"
```

---

### Task 3: Implement Stylesheet Builder

**Files:**
- Create: `app/ui/styles.py`

**Produces:** `build_stylesheet()` function returning complete QSS string

- [ ] **Step 1: Create `app/ui/styles.py`**

```python
"""QSS stylesheet builder using design tokens from theme.py."""

from app.ui.theme import COLORS, RADIUS, SPACING


def build_stylesheet():
    """Build and return the complete QSS stylesheet string."""
    return f"""
    /* === Global === */
    QWidget {{
        font-family: "Google Sans Font";
        font-size: 14px;
        color: {COLORS['on_surface']};
        background-color: {COLORS['surface']};
    }}

    /* === Main Window === */
    QMainWindow {{
        background-color: {COLORS['surface']};
    }}

    /* === Menu Bar === */
    QMenuBar {{
        background-color: {COLORS['surface']};
        border-bottom: 1px solid {COLORS['outline']}33;
        padding: {SPACING['sm']}px;
    }}
    QMenuBar::item {{
        padding: {SPACING['sm']}px {SPACING['md']}px;
        border-radius: {RADIUS['sm']}px;
    }}
    QMenuBar::item:selected {{
        background-color: {COLORS['primary_container']};
    }}

    /* === Buttons === */
    QPushButton {{
        background-color: {COLORS['primary']};
        color: {COLORS['on_primary']};
        border: none;
        border-radius: {RADIUS['lg']}px;
        padding: {SPACING['md']}px {SPACING['xl']}px;
        font-size: 16px;
        font-weight: 500;
        min-height: 48px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['primary']}DD;
    }}
    QPushButton:pressed {{
        background-color: {COLORS['primary']}BB;
    }}
    QPushButton:disabled {{
        background-color: {COLORS['outline']}44;
        color: {COLORS['outline']};
    }}

    /* === Secondary Button (e.g., Back) === */
    QPushButton[cssClass="secondary"] {{
        background-color: transparent;
        color: {COLORS['primary']};
        border: 2px solid {COLORS['outline']}44;
        border-radius: {RADIUS['lg']}px;
        padding: {SPACING['sm']}px {SPACING['md']}px;
        font-size: 14px;
        min-height: 36px;
    }}
    QPushButton[cssClass="secondary"]:hover {{
        background-color: {COLORS['primary_container']};
    }}

    /* === Record Button (special) === */
    QPushButton[cssClass="record"] {{
        background-color: #B3261E;
        border-radius: {RADIUS['full']}px;
        min-width: 120px;
        min-height: 120px;
        font-size: 18px;
    }}
    QPushButton[cssClass="record"]:hover {{
        background-color: #B3261EDD;
    }}
    QPushButton[cssClass="record"]:pressed {{
        background-color: #B3261EBB;
    }}

    /* === Panels === */
    QFrame[cssClass="panel"] {{
        background-color: {COLORS['surface_variant']};
        border-radius: {RADIUS['md']}px;
        padding: {SPACING['md']}px;
    }}

    /* === Tree View (PDF Browser) === */
    QTreeView {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['outline']}33;
        border-radius: {RADIUS['md']}px;
        padding: {SPACING['sm']}px;
        alternate-background-color: {COLORS['surface_variant']};
    }}
    QTreeView::item {{
        padding: {SPACING['sm']}px {SPACING['md']}px;
        border-radius: {RADIUS['sm']}px;
    }}
    QTreeView::item:selected {{
        background-color: {COLORS['primary_container']};
        color: {COLORS['on_surface']};
    }}

    /* === Text Edit (PDF Content) === */
    QTextEdit {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['outline']}33;
        border-radius: {RADIUS['md']}px;
        padding: {SPACING['md']}px;
        font-size: 14px;
        line-height: 1.6;
    }}

    /* === Status Bar === */
    QStatusBar {{
        background-color: {COLORS['surface_variant']};
        border-top: 1px solid {COLORS['outline']}33;
        padding: {SPACING['xs']}px {SPACING['md']}px;
        font-size: 12px;
    }}

    /* === Table Widget (Results) === */
    QTableWidget {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['outline']}33;
        border-radius: {RADIUS['md']}px;
        gridline-color: {COLORS['outline']}22;
    }}
    QTableWidget::item {{
        padding: {SPACING['sm']}px {SPACING['md']}px;
    }}
    QTableWidget::item:selected {{
        background-color: {COLORS['primary_container']};
    }}
    QHeaderView::section {{
        background-color: {COLORS['surface_variant']};
        border: none;
        border-bottom: 2px solid {COLORS['outline']}33;
        padding: {SPACING['sm']}px {SPACING['md']}px;
        font-weight: 600;
    }}

    /* === Scroll Bar === */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {COLORS['outline']}44;
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    """
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.ui.styles import build_stylesheet; s = build_stylesheet(); print(f'QSS length: {len(s)} chars')"`
Expected: `QSS length: >0 chars`

- [ ] **Step 3: Commit**

```bash
git add app/ui/styles.py
git commit -m "feat(app): add QSS stylesheet builder with Material 3 theme"
```

---

### Task 4: Implement Audio Handler (Core)

**Files:**
- Create: `app/core/audio_handler.py`

**Produces:** `AudioHandler` class with `start_recording`, `stop_recording`, `play_audio`, `stop_playback`, `save_audio`, `load_audio`

- [ ] **Step 1: Create `app/core/audio_handler.py`**

```python
"""Audio recording and playback using sounddevice. Pure logic, no Qt imports."""

import numpy as np
import sounddevice as sd
import soundfile as sf
import threading


class AudioHandler:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._frames = []
        self._stream = None
        self._playback_stream = None

    def start_recording(self):
        """Start capturing audio from the default microphone."""
        if self._recording:
            return
        self._frames = []
        self._recording = True

        def callback(indata, frame_count, time_info, status):
            if self._recording:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the captured audio as a numpy array."""
        if not self._recording:
            return np.array([], dtype=np.float32)
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._frames:
            return np.concatenate(self._frames, axis=0).flatten()
        return np.array([], dtype=np.float32)

    def play_audio(self, audio: np.ndarray):
        """Play audio through speakers in a background thread."""
        self.stop_playback()

        def _play():
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()

    def stop_playback(self):
        """Stop any ongoing playback."""
        try:
            sd.stop()
        except Exception:
            pass

    def save_audio(self, audio: np.ndarray, path: str):
        """Save audio array to a .wav file."""
        sf.write(path, audio, self.sample_rate)

    def load_audio(self, path: str) -> np.ndarray:
        """Load a .wav file and return audio as numpy array at 16kHz."""
        data, samplerate = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if samplerate != self.sample_rate:
            import librosa
            data = librosa.resample(data, orig_sr=samplerate, target_sr=self.sample_rate)
        return data
```

- [ ] **Step 2: Verify import and basic construction**

Run: `python -c "from app.core.audio_handler import AudioHandler; h = AudioHandler(); print(f'Sample rate: {h.sample_rate}')"`
Expected: `Sample rate: 16000`

- [ ] **Step 3: Commit**

```bash
git add app/core/audio_handler.py
git commit -m "feat(app): add AudioHandler with sounddevice recording/playback"
```

---

### Task 5: Implement PDF Handler (Core)

**Files:**
- Create: `app/core/pdf_handler.py`

**Produces:** `PdfHandler` class with `extract_text` and `list_pdfs`

- [ ] **Step 1: Create `app/core/pdf_handler.py`**

```python
"""PDF text extraction using pdfplumber. Pure logic, no Qt imports."""

import os
import pdfplumber


class PdfHandler:
    @staticmethod
    def extract_text(path: str) -> str:
        """Extract plain text from a PDF file."""
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)

    @staticmethod
    def list_pdfs(directory: str) -> list[str]:
        """List all .pdf files in the given directory (non-recursive)."""
        if not os.path.isdir(directory):
            return []
        return sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith(".pdf")
            ]
        )
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.core.pdf_handler import PdfHandler; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/core/pdf_handler.py
git commit -m "feat(app): add PdfHandler with pdfplumber text extraction"
```

---

### Task 6: Implement Model Runner Stub (Core)

**Files:**
- Create: `app/core/model_runner.py`

**Produces:** `ModelRunner` class with `analyze` method (placeholder)

- [ ] **Step 1: Create `app/core/model_runner.py`**

```python
"""Model inference wrapper — stub for Phase 1. Will integrate with model/ in Phase 3."""

import numpy as np


class ModelRunner:
    def __init__(self, models_dir: str = ""):
        self.models_dir = models_dir
        self._loaded = False

    def analyze(self, audio: np.ndarray) -> dict:
        """Run classification + localization on audio. Returns structured results."""
        # Placeholder: return mock results for UI development
        return {
            "classifications": {
                "prolongation": (False, 0.12),
                "block": (True, 0.87),
                "soundrep": (False, 0.08),
                "wordrep": (False, 0.05),
                "interjection": (True, 0.72),
            },
            "localizations": [
                (0.5, 1.2, 0.87),
                (3.4, 4.1, 0.72),
            ],
        }
```

- [ ] **Step 2: Verify import and mock output**

Run: `python -c "from app.core.model_runner import ModelRunner; import numpy as np; r = ModelRunner().analyze(np.zeros(16000)); print(list(r.keys()))"`
Expected: `['classifications', 'localizations']`

- [ ] **Step 3: Commit**

```bash
git add app/core/model_runner.py
git commit -m "feat(app): add ModelRunner stub with mock analysis results"
```

---

### Task 7: Implement Waveform View

**Files:**
- Create: `app/ui/waveform_view.py`

**Produces:** `WaveformView` class — custom `QGraphicsView` that draws audio waveform + colored overlays

- [ ] **Step 1: Create `app/ui/waveform_view.py`**

```python
"""Custom QGraphicsView for rendering audio waveforms with dysfluency overlays."""

import numpy as np
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

from app.ui.theme import COLORS


class WaveformView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._audio = None
        self._overlays = []  # list of (start_sec, end_sec, color)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(120)
        self.setStyleSheet("background-color: #FFFBFE; border: 1px solid #79747E33; border-radius: 16px;")

    def set_audio(self, audio: np.ndarray, sample_rate: int = 16000):
        """Set the audio data to display."""
        self._audio = audio
        self._sample_rate = sample_rate
        self._overlays = []
        self._draw()

    def set_overlays(self, overlays: list[tuple[float, float, str]]):
        """Set colored overlay regions. Each tuple: (start_sec, end_sec, color_hex)."""
        self._overlays = overlays
        self._draw()

    def clear_overlays(self):
        """Remove all overlays."""
        self._overlays = []
        self._draw()

    def _draw(self):
        """Redraw the waveform and overlays."""
        self._scene.clear()
        if self._audio is None or len(self._audio) == 0:
            return

        view_width = self.viewport().width()
        view_height = self.viewport().height()
        self._scene.setSceneRect(0, 0, view_width, view_height)

        duration = len(self._audio) / self._sample_rate
        mid_y = view_height / 2

        # Draw overlays first (behind waveform)
        for start_sec, end_sec, color_hex in self._overlays:
            x1 = (start_sec / duration) * view_width
            x2 = (end_sec / duration) * view_width
            color = QColor(color_hex)
            color.setAlpha(80)
            rect = QRectF(x1, 0, x2 - x1, view_height)
            self._scene.addRect(rect, QPen(Qt.PenStyle.NoPen), QBrush(color))

        # Downsample audio for drawing
        num_samples = len(self._audio)
        num_points = min(num_samples, view_width * 2)
        indices = np.linspace(0, num_samples - 1, num_points, dtype=int)
        samples = self._audio[indices]

        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val

        # Draw waveform
        pen = QPen(QColor(COLORS["primary"]))
        pen.setWidthF(1.5)
        amplitude = mid_y * 0.8

        path_points = []
        for i, sample in enumerate(samples):
            x = (i / num_points) * view_width
            y = mid_y - sample * amplitude
            path_points.append((x, y))

        for i in range(len(path_points) - 1):
            x1, y1 = path_points[i]
            x2, y2 = path_points[i + 1]
            self._scene.addLine(x1, y1, x2, y2, pen)

        # Draw center line
        center_pen = QPen(QColor(COLORS["outline"]))
        center_pen.setWidthF(0.5)
        self._scene.addLine(0, mid_y, view_width, mid_y, center_pen)

    def resizeEvent(self, event):
        """Redraw when the view is resized."""
        super().resizeEvent(event)
        self._draw()
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.ui.waveform_view import WaveformView; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/ui/waveform_view.py
git commit -m "feat(app): add custom WaveformView with dysfluency overlay support"
```

---

### Task 8: Implement PDF Viewer

**Files:**
- Create: `app/ui/pdf_viewer.py`

**Produces:** `PdfViewer` widget — `QTreeView` for file browsing + `QTextEdit` for text display

- [ ] **Step 1: Create `app/ui/pdf_viewer.py`**

```python
"""PDF file browser and text display widget."""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeView, QTextEdit, QLabel, QSplitter,
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFileSystemModel

from app.core.pdf_handler import PdfHandler
from app.ui.theme import COLORS


class PdfViewer(QWidget):
    pdf_selected = Signal(str)  # emits the full path of the selected PDF

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_dir = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Rainbow Passage")
        header.setStyleSheet("font-size: 16px; font-weight: 600; padding: 8px 0;")
        layout.addWidget(header)

        self._tree = QTreeView()
        self._model = QFileSystemModel()
        self._model.setRootPath("")
        self._tree.setModel(self._model)
        self._tree.setNameFilterDisables(False)
        self._tree.setAnimated(True)
        self._tree.setHeaderHidden(True)

        # Hide all columns except name
        for col in range(1, 4):
            self._tree.hideColumn(col)

        self._tree.clicked.connect(self._on_file_clicked)
        layout.addWidget(self._tree)

        self._text_display = QTextEdit()
        self._text_display.setReadOnly(True)
        self._text_display.setPlaceholderText("Select a PDF to view its content...")
        layout.addWidget(self._text_display)

    def set_directory(self, directory: str):
        """Set the root directory to browse for PDFs."""
        self._current_dir = directory
        if os.path.isdir(directory):
            self._tree.setRootIndex(self._model.index(directory))

    def _on_file_clicked(self, index):
        """Handle file click — extract and display text."""
        path = self._model.filePath(index)
        if path.lower().endswith(".pdf"):
            text = PdfHandler.extract_text(path)
            self._text_display.setPlainText(text)
            self.pdf_selected.emit(path)

    def get_text_display(self) -> QTextEdit:
        """Return the text display widget for external styling."""
        return self._text_display
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.ui.pdf_viewer import PdfViewer; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/ui/pdf_viewer.py
git commit -m "feat(app): add PdfViewer with file tree and text display"
```

---

### Task 9: Implement Audio Controls

**Files:**
- Create: `app/ui/audio_controls.py`

**Produces:** `AudioControls` widget — Record, Stop, Load, Play, Analyze buttons with signal emissions

- [ ] **Step 1: Create `app/ui/audio_controls.py`**

```python
"""Audio control buttons: Record, Stop, Load, Play, Analyze."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
)
from PySide6.QtCore import Signal


class AudioControls(QWidget):
    record_clicked = Signal()
    stop_clicked = Signal()
    load_clicked = Signal()
    play_clicked = Signal()
    analyze_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment()
        layout.setSpacing(16)

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("font-size: 12px; color: #79747E;")
        self._status_label.setAlignment()
        layout.addWidget(self._status_label)

        # Record button (large, circular)
        self._record_btn = QPushButton("Record Audio")
        self._record_btn.setProperty("cssClass", "record")
        self._record_btn.clicked.connect(self.record_clicked.emit)
        layout.addWidget(self._record_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Stop button (hidden by default)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("cssClass", "secondary")
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self._stop_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Load button
        self._load_btn = QPushButton("Load Audio")
        self._load_btn.clicked.connect(self.load_clicked.emit)
        layout.addWidget(self._load_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Play button (hidden until audio is loaded)
        self._play_btn = QPushButton("Play")
        self._play_btn.setProperty("cssClass", "secondary")
        self._play_btn.setVisible(False)
        self._play_btn.clicked.connect(self.play_clicked.emit)
        layout.addWidget(self._play_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Analyze button (hidden until audio is ready)
        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setVisible(False)
        self._analyze_btn.clicked.connect(self.analyze_clicked.emit)
        layout.addWidget(self._analyze_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def set_recording(self, recording: bool):
        """Update UI state for recording."""
        self._record_btn.setVisible(not recording)
        self._stop_btn.setVisible(recording)
        self._load_btn.setVisible(not recording)
        self._play_btn.setVisible(False)
        self._analyze_btn.setVisible(False)
        self._status_label.setText("Recording..." if recording else "Ready")

    def set_audio_loaded(self):
        """Update UI state when audio is loaded/recorded."""
        self._record_btn.setVisible(True)
        self._stop_btn.setVisible(False)
        self._load_btn.setVisible(True)
        self._play_btn.setVisible(True)
        self._analyze_btn.setVisible(True)
        self._status_label.setText("Audio loaded — ready to analyze")

    def set_playing(self, playing: bool):
        """Update UI state for playback."""
        self._play_btn.setText("Stop" if playing else "Play")
        self._status_label.setText("Playing..." if playing else "Ready")
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.ui.audio_controls import AudioControls; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/ui/audio_controls.py
git commit -m "feat(app): add AudioControls with record/load/play/analyze buttons"
```

---

### Task 10: Implement Results Panel

**Files:**
- Create: `app/ui/results_panel.py`

**Produces:** `ResultsPanel` widget — classification table + localization timeline

- [ ] **Step 1: Create `app/ui/results_panel.py`**

```python
"""Classification results table and localization timeline display."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QHeaderView, QFrame,
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

from app.ui.theme import COLORS
from app.ui.waveform_view import WaveformView


CLASS_NAMES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]
DISPLAY_NAMES = ["Prolongation", "Block", "Sound Repetition", "Word Repetition", "Interjection"]


class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Classification results (top)
        class_label = QLabel("Classification Results")
        class_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(class_label)

        self._table = QTableWidget(5, 3)
        self._table.setHorizontalHeaderLabels(["Class", "Detected", "Confidence"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setAlternatingRowColors(True)

        for i, name in enumerate(DISPLAY_NAMES):
            self._table.setItem(i, 0, QTableWidgetItem(name))
            self._table.setItem(i, 1, QTableWidgetItem("—"))
            self._table.setItem(i, 2, QTableWidgetItem("—"))

        layout.addWidget(self._table)

        # Localization timeline (bottom)
        loc_label = QLabel("Localization Timeline")
        loc_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(loc_label)

        self._waveform = WaveformView()
        self._waveform.setMinimumHeight(100)
        layout.addWidget(self._waveform)

        # Legend
        legend_layout = QHBoxLayout()
        for class_name, display_name in zip(CLASS_NAMES, DISPLAY_NAMES):
            color = COLORS["dysfluency"][class_name]
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 12px;")
            label = QLabel(display_name)
            label.setStyleSheet("font-size: 11px; color: #79747E;")
            legend_layout.addWidget(dot)
            legend_layout.addWidget(label)
            legend_layout.addSpacing(12)
        legend_layout.addStretch()
        layout.addLayout(legend_layout)

    def set_results(self, results: dict, audio: "np.ndarray" = None, sample_rate: int = 16000):
        """Update the panel with analysis results."""
        classifications = results.get("classifications", {})
        localizations = results.get("localizations", [])

        # Update classification table
        for i, class_name in enumerate(CLASS_NAMES):
            if class_name in classifications:
                detected, confidence = classifications[class_name]
                det_item = QTableWidgetItem("Yes" if detected else "No")
                conf_item = QTableWidgetItem(f"{confidence * 100:.0f}%")

                if detected:
                    det_item.setForeground(QColor(COLORS["dysfluency"][class_name]))
                    conf_item.setForeground(QColor(COLORS["dysfluency"][class_name]))
                    font = det_item.font()
                    font.setBold(True)
                    det_item.setFont(font)
                    conf_item.setFont(font)

                self._table.setItem(i, 1, det_item)
                self._table.setItem(i, 2, conf_item)

        # Update waveform with overlays
        if audio is not None and len(audio) > 0:
            self._waveform.set_audio(audio, sample_rate)
            overlays = []
            for start_sec, end_sec, conf in localizations:
                overlays.append((start_sec, end_sec, "#B3261E"))
            self._waveform.set_overlays(overlays)

    def clear_results(self):
        """Clear all results."""
        for i in range(5):
            self._table.setItem(i, 1, QTableWidgetItem("—"))
            self._table.setItem(i, 2, QTableWidgetItem("—"))
        self._waveform.clear_overlays()
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.ui.results_panel import ResultsPanel; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/ui/results_panel.py
git commit -m "feat(app): add ResultsPanel with classification table and localization timeline"
```

---

### Task 11: Implement Home Page

**Files:**
- Create: `app/ui/home_page.py`

**Produces:** `HomePage` widget — left PDF panel + center audio controls

- [ ] **Step 1: Create `app/ui/home_page.py`**

```python
"""Home page: Rainbow Passage PDF viewer + audio controls."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QSplitter
from PySide6.QtCore import Signal
from PySide6.QtCore import Qt

from app.ui.pdf_viewer import PdfViewer
from app.ui.audio_controls import AudioControls


class HomePage(QWidget):
    record_clicked = Signal()
    stop_clicked = Signal()
    load_clicked = Signal()
    play_clicked = Signal()
    analyze_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: PDF viewer
        self._pdf_viewer = PdfViewer()
        splitter.addWidget(self._pdf_viewer)

        # Right: Audio controls (centered)
        self._audio_controls = AudioControls()
        splitter.addWidget(self._audio_controls)

        # Set initial sizes (40% PDF, 60% controls)
        splitter.setSizes([400, 600])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        # Wire signals
        self._audio_controls.record_clicked.connect(self.record_clicked)
        self._audio_controls.stop_clicked.connect(self.stop_clicked)
        self._audio_controls.load_clicked.connect(self.load_clicked)
        self._audio_controls.play_clicked.connect(self.play_clicked)
        self._audio_controls.analyze_clicked.connect(self.analyze_clicked)

    def get_pdf_viewer(self) -> PdfViewer:
        return self._pdf_viewer

    def get_audio_controls(self) -> AudioControls:
        return self._audio_controls
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.ui.home_page import HomePage; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/ui/home_page.py
git commit -m "feat(app): add HomePage with PDF viewer and audio controls"
```

---

### Task 12: Implement Analysis Page

**Files:**
- Create: `app/ui/analysis_page.py`

**Produces:** `AnalysisPage` widget — full-width waveform + results panel + back button

- [ ] **Step 1: Create `app/ui/analysis_page.py`**

```python
"""Analysis page: full-width waveform + classification results + localization timeline."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
)
from PySide6.QtCore import Signal

from app.ui.waveform_view import WaveformView
from app.ui.results_panel import ResultsPanel


class AnalysisPage(QWidget):
    back_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(16)

        # Top bar with back button
        top_bar = QHBoxLayout()
        back_btn = QPushButton("← Back to Home")
        back_btn.setProperty("cssClass", "secondary")
        back_btn.clicked.connect(self.back_clicked.emit)
        top_bar.addWidget(back_btn)
        top_bar.addStretch()

        title = QLabel("Analysis Results")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        top_bar.addWidget(title)
        top_bar.addStretch()

        layout.addLayout(top_bar)

        # Full-width waveform
        self._waveform = WaveformView()
        self._waveform.setMinimumHeight(150)
        layout.addWidget(self._waveform)

        # Results panel (classification + localization)
        self._results = ResultsPanel()
        layout.addWidget(self._results)

    def set_results(self, results: dict, audio=None, sample_rate: int = 16000):
        """Update the page with analysis results."""
        self._results.set_results(results, audio, sample_rate)

        # Also update the top waveform with audio (without overlays)
        if audio is not None and len(audio) > 0:
            self._waveform.set_audio(audio, sample_rate)

    def get_waveform(self) -> WaveformView:
        return self._waveform

    def get_results_panel(self) -> ResultsPanel:
        return self._results
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.ui.analysis_page import AnalysisPage; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/ui/analysis_page.py
git commit -m "feat(app): add AnalysisPage with waveform and results display"
```

---

### Task 13: Implement Main Window

**Files:**
- Create: `app/ui/main_window.py`

**Produces:** `MainWindow` class — QMainWindow + QStackedWidget wiring Home ↔ Analysis pages

- [ ] **Step 1: Create `app/ui/main_window.py`**

```python
"""Main window with QStackedWidget for page navigation."""

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QFileDialog,
)
from PySide6.QtCore import Qt, QThread, Signal

from app.ui.home_page import HomePage
from app.ui.analysis_page import AnalysisPage
from app.core.audio_handler import AudioHandler
from app.core.model_runner import ModelRunner


class AnalysisWorker(QThread):
    finished = Signal(dict)

    def __init__(self, model_runner: ModelRunner, audio: np.ndarray):
        super().__init__()
        self._model_runner = model_runner
        self._audio = audio

    def run(self):
        results = self._model_runner.analyze(self._audio)
        self.finished.emit(results)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swaraaha — Speech Dysfluency Detection")
        self.setMinimumSize(1200, 800)

        self._audio_handler = AudioHandler()
        self._model_runner = ModelRunner()
        self._current_audio = None
        self._worker = None

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # Page 0: Home
        self._home_page = HomePage()
        self._stack.addWidget(self._home_page)

        # Page 1: Analysis
        self._analysis_page = AnalysisPage()
        self._stack.addWidget(self._analysis_page)

        # Wire home page signals
        self._home_page.record_clicked.connect(self._on_record)
        self._home_page.stop_clicked.connect(self._on_stop)
        self._home_page.load_clicked.connect(self._on_load)
        self._home_page.play_clicked.connect(self._on_play)
        self._home_page.analyze_clicked.connect(self._on_analyze)

        # Wire analysis page signals
        self._analysis_page.back_clicked.connect(self._go_home)

    def _on_record(self):
        self._audio_handler.start_recording()
        self._home_page.get_audio_controls().set_recording(True)
        self.statusBar().showMessage("Recording...")

    def _on_stop(self):
        audio = self._audio_handler.stop_recording()
        if len(audio) > 0:
            self._current_audio = audio
            self._home_page.get_audio_controls().set_recording(False)
            self._home_page.get_audio_controls().set_audio_loaded()
            self.statusBar().showMessage(f"Recorded {len(audio) / 16000:.1f}s of audio")
        else:
            self._home_page.get_audio_controls().set_recording(False)
            self.statusBar().showMessage("Ready")

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Audio", "", "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        if path:
            try:
                audio = self._audio_handler.load_audio(path)
                self._current_audio = audio
                self._home_page.get_audio_controls().set_audio_loaded()
                self.statusBar().showMessage(f"Loaded: {path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error loading file: {e}")

    def _on_play(self):
        if self._current_audio is not None:
            self._audio_handler.play_audio(self._current_audio)
            self.statusBar().showMessage("Playing...")

    def _on_analyze(self):
        if self._current_audio is None:
            self.statusBar().showMessage("No audio to analyze")
            return

        self.statusBar().showMessage("Analyzing audio...")
        self._worker = AnalysisWorker(self._model_runner, self._current_audio)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.start()

    def _on_analysis_done(self, results: dict):
        self._analysis_page.set_results(results, self._current_audio)
        self._stack.setCurrentIndex(1)
        self.statusBar().showMessage("Analysis complete")

    def _go_home(self):
        self._stack.setCurrentIndex(0)
        self.statusBar().showMessage("Ready")
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.ui.main_window import MainWindow; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/ui/main_window.py
git commit -m "feat(app): add MainWindow with QStackedWidget page navigation"
```

---

### Task 14: Implement Entry Point

**Files:**
- Create: `app/main.py`

**Produces:** Application entry point — loads fonts, applies stylesheet, shows MainWindow

- [ ] **Step 1: Create `app/main.py`**

```python
"""Swaraaha Desktop Application — Entry Point."""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from app.ui.main_window import MainWindow
from app.ui.theme import load_fonts
from app.ui.styles import build_stylesheet


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Swaraaha")
    app.setOrganizationName("Swaraaha")

    # Load custom fonts
    loaded_fonts = load_fonts()
    if loaded_fonts:
        print(f"Loaded fonts: {', '.join(loaded_fonts)}")
    else:
        print("Warning: No custom fonts loaded. Using system defaults.")

    # Apply stylesheet
    app.setStyleSheet(build_stylesheet())

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.main import main; print('Entry point OK')"`
Expected: `Entry point OK`

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat(app): add application entry point with font and style loading"
```

---

### Task 15: Create Default QSS File

**Files:**
- Create: `app/assets/styles/default.qss`

- [ ] **Step 1: Create `app/assets/styles/default.qss`**

```css
/* Swaraaha — Base QSS (augmented by theme.py at runtime) */
/* This file serves as a fallback and documentation of the stylesheet structure */
/* The actual stylesheet is generated by app/ui/styles.py */
```

- [ ] **Step 2: Remove .gitkeep (no longer needed)**

```bash
rm -f app/assets/styles/.gitkeep
```

- [ ] **Step 3: Commit**

```bash
git add app/assets/styles/default.qss
git commit -m "chore(app): add base QSS stylesheet placeholder"
```

---

### Task 16: Integration Smoke Test

**Files:** No new files. Verify all imports and app startup.

- [ ] **Step 1: Verify all imports resolve**

Run: `python -c "
from app.core.audio_handler import AudioHandler
from app.core.pdf_handler import PdfHandler
from app.core.model_runner import ModelRunner
from app.ui.theme import COLORS, SPACING, RADIUS, load_fonts
from app.ui.styles import build_stylesheet
from app.ui.waveform_view import WaveformView
from app.ui.pdf_viewer import PdfViewer
from app.ui.audio_controls import AudioControls
from app.ui.results_panel import ResultsPanel
from app.ui.home_page import HomePage
from app.ui.analysis_page import AnalysisPage
from app.ui.main_window import MainWindow
print('All imports OK')
"`
Expected: `All imports OK`

- [ ] **Step 2: Verify requirements.txt matches imports**

Run: `pip install -r app/requirements.txt --dry-run 2>&1 | head -20`
Expected: No errors, lists the 5 packages

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(app): complete Task 1.3 — scaffold app/ directory with PySide6 UI layout"
```

---

## Execution Order

Tasks 1-6 can run in parallel (no dependencies between them). Tasks 7-12 depend on Tasks 2-6 (theme, core modules). Tasks 13-14 depend on all UI modules. Task 15-16 are final integration.

**Recommended batch execution:**
1. **Batch 1:** Tasks 1-6 (foundation)
2. **Batch 2:** Tasks 7-12 (UI widgets)
3. **Batch 3:** Tasks 13-14 (wiring)
4. **Batch 4:** Tasks 15-16 (polish + verify)
