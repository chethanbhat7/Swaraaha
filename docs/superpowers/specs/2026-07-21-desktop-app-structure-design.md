# Design Spec: Swaraaha Desktop Application Structure

> **Task:** 1.3 — Scaffold `app/` Directory + Design UI Layout
> **Assignee:** Srinivas
> **Date:** 2026-07-21
> **Status:** Approved

---

## 1. Overview

The Swaraaha desktop application is a PySide6 tool for recording and analyzing speech dysfluency. It provides two pages: a **Home Page** for loading Rainbow Passage PDFs and recording/loading audio, and an **Analysis Page** for displaying classification results and localization timelines after analysis.

## 2. Technology Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | **PySide6** | Official Qt6 bindings, LGPL license, modern API |
| Waveform rendering | **Custom QPainter/QGraphicsView** | Lightweight, performant, no heavy deps |
| Theming | **QSS stylesheets + design tokens** | CSS-like styling, centralized in `theme.py` |
| File browser | **Directory-scoped** | User picks a folder; app scans for `.pdf` files |
| Audio backend | **sounddevice** | Modern, cross-platform, simple API |
| PDF rendering | **Text extraction + QTextEdit** | Lightweight, no Chromium dependency |
| Page architecture | **QStackedWidget** | Stack-based navigation, 2 pages |
| Visual style | **Material 3 Expressive** | Bold, lively, modern feel |
| Font | **Google Sans Flex** | Bundled in `assets/fonts/` |

## 3. Directory Structure

```
app/
├── main.py                    # Entry point — QApplication, load font, load style, show MainWindow
├── requirements.txt           # PySide6, sounddevice, numpy, soundfile, pdfplumber
├── ui/
│   ├── __init__.py
│   ├── main_window.py         # QMainWindow + QStackedWidget (2 pages)
│   ├── home_page.py           # Page 0: PDF panel + Record/Load buttons
│   ├── analysis_page.py       # Page 1: Waveform + results
│   ├── pdf_viewer.py          # PDF file tree + text extraction display
│   ├── audio_controls.py      # Record/Load/Play/Stop/Analyze buttons
│   ├── waveform_view.py       # Custom QGraphicsView for waveform rendering
│   ├── results_panel.py       # Classification table + localization timeline
│   ├── theme.py               # Design tokens + QSS generation
│   └── styles.py              # QSS stylesheet builder
├── core/
│   ├── __init__.py
│   ├── audio_handler.py       # sounddevice recording/playback — no Qt UI imports
│   ├── pdf_handler.py         # PDF text extraction (pdfplumber)
│   └── model_runner.py        # Model inference stub (placeholder for Phase 3)
└── assets/
    ├── fonts/
    │   ├── GoogleSansFlex-Regular.ttf
    │   ├── GoogleSansFlex-Medium.ttf
    │   └── GoogleSansFlex-Bold.ttf
    └── styles/
        └── default.qss        # Base QSS, augmented by theme.py at runtime
```

**Key principle:** `core/` modules are pure logic with no Qt UI imports. `ui/` modules import from `core/`. `main_window.py` wires them together via signals/slots.

## 4. Page Layouts

### 4a. Home Page

```
┌──────────────────────────────────────────────────────────┐
│ Menu Bar  [File] [View] [Help]                           │
├─────────────┬────────────────────────────────────────────┤
│             │                                            │
│  Rainbow    │         ┌────────────────────┐             │
│  Passage    │         │   [● Record Audio]  │             │
│  PDF        │         └────────────────────┘             │
│  Browser    │         ┌────────────────────┐             │
│  (file tree │         │  [📂 Load Audio]    │             │
│   + text    │         └────────────────────┘             │
│   display)  │                                            │
│             │                                            │
├─────────────┴────────────────────────────────────────────┤
│ Status Bar: "Ready" / "Recording..."                     │
└──────────────────────────────────────────────────────────┘
```

- **Left pane:** `QTreeView` (filtered to `.pdf`) + `QTextEdit` for rendered text
- **Center:** Two prominent buttons, vertically centered
- User reads Rainbow Passage while recording, then clicks Analyze to transition to Analysis page

### 4b. Analysis Page

```
┌──────────────────────────────────────────────────────────┐
│ Menu Bar  [File] [View] [Help]                [← Back]   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Waveform View (full width, custom drawn)        │    │
│  │  ▓▓▓▓▓▓░░░░▓▓▓▓▓▓▓░░░░░▓▓▓▓▓░░░░░░░▓▓▓▓▓▓▓▓   │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌───────────────────────┬──────────────────────────┐    │
│  │  Classification       │  Localization Timeline   │    │
│  │  ┌──────────────────┐ │  ┌────────────────────┐  │    │
│  │  │ Prolongation: No │ │  │ ▓▓▓░░░▓▓▓░░░░░▓▓  │  │    │
│  │  │ Block: Yes   87% │ │  │ red = prolongation │  │    │
│  │  │ Sound Rep: No    │ │  └────────────────────┘  │    │
│  │  │ Word Rep: No     │ │                          │    │
│  │  │ Interjection: No │ │                          │    │
│  │  └──────────────────┘ │                          │    │
│  └───────────────────────┴──────────────────────────┘    │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ Status Bar: "Analysis complete"                          │
└──────────────────────────────────────────────────────────┘
```

- **Top (full width):** Waveform with colored overlays for detected dysfluencies
- **Bottom-left:** Classification table (5 rows, Yes/No + confidence %)
- **Bottom-right:** Localization timeline (waveform + colored regions + legend)
- **Back button:** Returns to Home page

## 5. Navigation Flow

```
Home Page
  │
  ├─ [Record Audio] → recording starts → stop → "Analyze" button appears
  ├─ [Load Audio]   → file dialog → waveform loads → "Analyze" button appears
  │
  └─ [Analyze]      → ModelRunner.analyze() → transitions to Analysis Page
                         │
                         └─ [← Back] → returns to Home Page
```

## 6. Visual Design — Material 3 Expressive

### Color Palette

```python
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
    }
}
```

### Spacing & Radius

```python
SPACING = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}
RADIUS = {"sm": 8, "md": 16, "lg": 24, "full": 9999}
```

### Typography

- **Font:** Google Sans Flex (Regular, Medium, Bold) — bundled in `assets/fonts/`
- **Loaded at startup** via `QFontDatabase.addApplicationFont()`
- **Headings:** 18-20sp, Bold
- **Body:** 14sp, Regular
- **Labels/Captions:** 12sp, Medium

### Component Styling

- **Buttons:** Large, rounded (24px radius), filled with `Primary`, white text. Hover: brightness lift. Pressed: darken.
- **Cards/Surfaces:** Rounded corners (16px), subtle border (`Outline` at 10% opacity), `Surface` background.
- **Panels:** `Surface Variant` background, 8px padding, rounded corners.
- **Waveform View:** `Surface` background, waveform in `Primary`, overlays in accent colors (40% opacity).
- **Status Bar:** `Surface Variant` background, `On Surface` text.
- **Tables/Results:** Clean rows, alternating subtle backgrounds, colored dots for detection status.

### Animations

- **Page transitions:** Fade + slide (Analysis page slides in from right)
- **Button hover:** Subtle scale/color shift
- **Analysis loading:** Progress bar or pulse animation

## 7. Signal Architecture

```
Home Page:
  FilePanel.pdf_selected(path)
    -> PdfHandler.extract_text(path)
    -> display in QTextEdit

  AudioPanel.record_clicked()
    -> AudioHandler.start_recording()

  AudioPanel.stop_clicked()
    -> AudioHandler.stop_recording()
    -> store audio in MainWindow state
    -> update waveform display

  AudioPanel.load_clicked()
    -> QFileDialog -> AudioHandler.load_audio(path)
    -> store audio in MainWindow state
    -> update waveform display

  AudioPanel.analyze_clicked()
    -> ModelRunner.analyze(audio)
    -> transition to Analysis Page with results

Analysis Page:
  ResultsPanel displays classification + localization results
  Back button -> QStackedWidget.setCurrentIndex(0) (Home)
```

## 8. Core Module Interfaces

### AudioHandler (`core/audio_handler.py`)

```python
class AudioHandler:
    def __init__(self, sample_rate=16000, channels=1): ...
    def start_recording(self): ...
    def stop_recording(self) -> np.ndarray: ...
    def play_audio(self, audio: np.ndarray): ...
    def stop_playback(self): ...
    def save_audio(self, audio: np.ndarray, path: str): ...
    def load_audio(self, path: str) -> np.ndarray: ...
```

- Recording happens in a background thread to avoid UI freeze
- Returns audio as numpy array at 16kHz mono

### PdfHandler (`core/pdf_handler.py`)

```python
class PdfHandler:
    @staticmethod
    def extract_text(path: str) -> str: ...
    @staticmethod
    def list_pdfs(directory: str) -> list[str]: ...
```

- Uses `pdfplumber` for text extraction (lightweight, pure Python)
- Returns extracted text as string for display in QTextEdit

### ModelRunner (`core/model_runner.py`) — Phase 3 stub

```python
class ModelRunner:
    def __init__(self, models_dir: str): ...
    def analyze(self, audio: np.ndarray) -> dict: ...
    # Returns: {"classifications": {...}, "localizations": [...]}
```

- Placeholder for now; will integrate with `model/` in Phase 3

## 9. File-by-File Implementation Notes

| File | Responsibility | Key Imports |
|---|---|---|
| `main.py` | App entry point, font loading, style loading, MainWindow creation | PySide6.QtWidgets, theme, styles |
| `main_window.py` | QMainWindow, QStackedWidget, menu bar, status bar, page routing | PySide6.QtWidgets |
| `home_page.py` | Left PDF panel + center Record/Load buttons | pdf_viewer, audio_controls |
| `analysis_page.py` | Full-width waveform + bottom results panels | waveform_view, results_panel |
| `pdf_viewer.py` | QTreeView + QTextEdit for PDF browsing/display | QFileSystemModel, pdfplumber |
| `audio_controls.py` | Record/Load/Play/Stop/Analyze button group | PySide6.QtWidgets, audio_handler |
| `waveform_view.py` | Custom QGraphicsView, draws waveform + overlays | QPainter, QGraphicsView |
| `results_panel.py` | Classification table + localization timeline | PySide6.QtWidgets |
| `theme.py` | Design token constants, QSS generation functions | — |
| `styles.py` | Builds full QSS string from theme tokens | theme |
| `audio_handler.py` | sounddevice recording/playback, pure logic | sounddevice, numpy, soundfile |
| `pdf_handler.py` | PDF text extraction, file listing | pdfplumber |
| `model_runner.py` | Model inference wrapper (stub) | — |

## 10. Dependencies

```txt
# app/requirements.txt
PySide6>=6.5.0
sounddevice>=0.4.6
numpy>=1.24.0
soundfile>=0.12.0
pdfplumber>=0.10.0
```

Note: Google Sans Flex font files must be manually downloaded from fonts.google.com and placed in `app/assets/fonts/`.

## 11. Constraints & Gotchas

- **One audio at a time:** App holds only one loaded/recorded audio in memory. Loading a new one replaces the old.
- **PDF text extraction:** Only extracts plain text — images/formatting from PDFs are lost. Acceptable since Rainbow Passage is text-only.
- **Model integration deferred:** `model_runner.py` is a stub in Phase 1. Results panel shows placeholder content until Phase 3.
- **Font licensing:** Google Sans Flex is free to use; no license file needed in repo.
- **Thread safety:** Audio recording/playback threads must not touch Qt widgets directly — use signals to communicate back to UI.
