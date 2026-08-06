# Desktop Transcription UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add background transcription with a wait dialog, a language selection dialog (English/Kannada/Hindi), and internally-scrollable capped transcript tables to the PySide6 desktop app, and fix the jittery results-page scrolling.

**Architecture:** A `TranscriptionWorker(QThread)` (mirroring the existing `AnalysisWorker`) runs `AudioTranscriber.transcribe` off the UI thread for the home-page transcript (record-stop and load flows) while a modal `WaitDialog` shows. A new `LanguageDialog` prompts on Record/Load with the last choice pre-highlighted. `AudioTranscriber` switches from wav2vec2 to lazy per-language Whisper-tiny pipelines (mirroring `backend/services/transcriber.py`) with hallucination dedup. A new `cap_table_height` helper caps transcript tables so their internal scrollbar handles overflow; `WaveformView` debounces resize redraws.

**Tech Stack:** Python, PySide6 (Qt Widgets), transformers 5.14.1 (Whisper), numpy, pytest (offscreen), ruff.

## Global Constraints

- Branch: `desktop-transcription-ui` (already created).
- Tests run from repo root: `pytest` (pytest.ini has `pythonpath='.'`, `testpaths='app/tests'`; `QT_QPA_PLATFORM=offscreen` set in `app/tests/conftest.py`).
- **No network / no model downloads in tests** — every test that touches `AudioTranscriber` must monkeypatch `app.core.transcription.get_pipeline` or inject a fake transcriber.
- `ruff check` must pass on every touched file.
- Conventional commit messages (`feat:`, `fix:`, `perf:`, `test:`, `docs:`, `refactor:`).
- All new UI dialogs must inherit the app stylesheet (children of `MainWindow`) and reuse existing QSS (`nav_btn_active` for the pre-highlighted language button).
- Language codes are `"english" | "kannada" | "hindi"`.

---

### Task 1: `cap_table_height` helper

**Files:**
- Modify: `app/ui/table_utils.py`
- Test: `app/tests/test_table_utils.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `cap_table_height(table: QTableWidget, max_rows: int)` in `app/ui/table_utils.py` — caps a table's `maximumHeight` so rows beyond `max_rows` scroll inside the table.

- [ ] **Step 1: Write the failing test**

Create `app/tests/test_table_utils.py`:

```python
from PySide6.QtWidgets import QTableWidget

from app.ui.table_utils import cap_table_height


def test_cap_table_height_limits_max(qapp):
    table = QTableWidget(20, 3)
    table.setHorizontalHeaderLabels(["A", "B", "C"])
    table.resize(400, 800)
    table.show()
    qapp.processEvents()

    header = table.horizontalHeader().height()
    row = table.rowHeight(0)

    cap_table_height(table, 8)

    full = header + 20 * row
    assert table.maximumHeight() < full
    assert table.maximumHeight() > header
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/tests/test_table_utils.py -v`
Expected: FAIL — `ImportError: cannot import name 'cap_table_height'`.

- [ ] **Step 3: Write minimal implementation**

Append to `app/ui/table_utils.py`:

```python
def cap_table_height(table: QTableWidget, max_rows: int):
    """Cap a table's maximum height so longer contents scroll inside the table."""
    header_height = table.horizontalHeader().height() or 30
    row_height = table.rowHeight(0) if table.rowCount() > 0 else 30
    table.setMaximumHeight(header_height + max_rows * row_height + 4)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest app/tests/test_table_utils.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/ui/table_utils.py app/tests/test_table_utils.py
git commit -m "feat: add cap_table_height helper for scrollable tables"
```

---

### Task 2: Language selection dialog

**Files:**
- Create: `app/ui/language_dialog.py`
- Test: `app/tests/test_language_dialog.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LanguageDialog(QDialog)` with `__init__(self, current: str = "english", parent=None)`, `.selected() -> str`, and a `_buttons` dict `{code: QPushButton}`.

- [ ] **Step 1: Write the failing tests**

Create `app/tests/test_language_dialog.py`:

```python
from PySide6.QtWidgets import QDialog

from app.ui.language_dialog import LanguageDialog


def test_language_dialog_has_three_buttons(qapp):
    dialog = LanguageDialog()
    assert set(dialog._buttons.keys()) == {"english", "kannada", "hindi"}


def test_language_dialog_prehighlights_current(qapp):
    dialog = LanguageDialog("kannada")
    assert dialog._buttons["kannada"].property("cssClass") == "nav_btn_active"
    assert dialog._buttons["english"].property("cssClass") is None


def test_language_dialog_click_returns_code(qapp):
    dialog = LanguageDialog()
    dialog._buttons["hindi"].click()
    assert dialog.selected() == "hindi"
    assert dialog.result() == QDialog.DialogCode.Accepted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/tests/test_language_dialog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ui.language_dialog'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/ui/language_dialog.py`:

```python
"""Modal language selection dialog for transcription (English/Kannada/Hindi)."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

LANGUAGES = {
    "English": "english",
    "ಕನ್ನಡ (Kannada)": "kannada",
    "हिंदी (Hindi)": "hindi",
}


class LanguageDialog(QDialog):
    """Modal dialog asking the user to choose the transcription language."""

    def __init__(self, current: str = "english", parent=None):
        super().__init__(parent)
        self._selected = current
        self.setWindowTitle("Choose Language")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Choose the language")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._buttons = {}
        for label, code in LANGUAGES.items():
            btn = QPushButton(label)
            if code == current:
                btn.setProperty("cssClass", "nav_btn_active")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, c=code: self._choose(c))
            self._buttons[code] = btn
            layout.addWidget(btn)

    def _choose(self, code: str):
        self._selected = code
        self.accept()

    def selected(self) -> str:
        """Return the chosen language code ('english' | 'kannada' | 'hindi')."""
        return self._selected
```

Button styling matches the app: default `QPushButton` (primary fill) for all, `nav_btn_active` QSS class for the pre-highlighted one (same "selected" look as the Passage/Files segmented nav).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/tests/test_language_dialog.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint**

Run: `ruff check app/ui/language_dialog.py app/tests/test_language_dialog.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add app/ui/language_dialog.py app/tests/test_language_dialog.py
git commit -m "feat: add language selection dialog"
```

---

### Task 3: Wait dialog

**Files:**
- Create: `app/ui/wait_dialog.py`
- Test: `app/tests/test_wait_dialog.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `WaitDialog(QDialog)` with `__init__(self, message: str = "Generating transcript, please wait", parent=None)` and `.finish()`.

- [ ] **Step 1: Write the failing test**

Create `app/tests/test_wait_dialog.py`:

```python
from app.ui.wait_dialog import WaitDialog


def test_wait_dialog_message_and_finish(qapp):
    dialog = WaitDialog()
    assert "Generating transcript" in dialog._label.text()
    dialog.show()
    qapp.processEvents()
    assert dialog.isVisible()
    dialog.finish()
    assert not dialog.isVisible()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/tests/test_wait_dialog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ui.wait_dialog'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/ui/wait_dialog.py`:

```python
"""Modal wait dialog shown while a transcription is being generated."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout


class WaitDialog(QDialog):
    """Non-blocking modal dialog with an indeterminate spinner and a message."""

    def __init__(self, message: str = "Generating transcript, please wait", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Please wait")
        self.setModal(True)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        self._label = QLabel(message)
        self._label.setStyleSheet("font-size: 14px; font-weight: 500;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        spinner = QProgressBar()
        spinner.setRange(0, 0)
        spinner.setFixedSize(160, 8)
        layout.addWidget(spinner, alignment=Qt.AlignmentFlag.AlignCenter)

    def finish(self):
        """Close the dialog."""
        self.close()
```

`setModal(True)` + `show()` (NOT `exec()`) blocks input to the parent but keeps the event loop running so the worker's `finished` signal is delivered. `setRange(0, 0)` makes the `QProgressBar` an indeterminate busy indicator.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest app/tests/test_wait_dialog.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/ui/wait_dialog.py app/tests/test_wait_dialog.py
git commit -m "feat: add transcription wait dialog"
```

---

### Task 4: Background transcription worker

**Files:**
- Create: `app/ui/transcription_worker.py`
- Test: `app/tests/test_transcription_worker.py`

**Interfaces:**
- Consumes: `AudioTranscriber` (from `app.core.transcription`).
- Produces: `TranscriptionWorker(QThread)` with `__init__(self, transcriber, audio: np.ndarray, language: str = "english")`, signal `finished = Signal(dict)`. Emits the transcription dict, or `{"error": str}` on exception.

- [ ] **Step 1: Write the failing tests**

Create `app/tests/test_transcription_worker.py`:

```python
import numpy as np

from app.ui.transcription_worker import TranscriptionWorker


class _FakeTranscriber:
    def transcribe(self, audio, language="english"):
        return {"text": "hello", "words": [], "language": language}


def test_transcription_worker_emits_data(qapp):
    results = []
    worker = TranscriptionWorker(_FakeTranscriber(), np.zeros(1600, dtype=np.float32), "kannada")
    worker.finished.connect(results.append)
    worker.run()
    assert results[0]["text"] == "hello"
    assert results[0]["language"] == "kannada"


class _RaisingTranscriber:
    def transcribe(self, audio, language="english"):
        raise RuntimeError("boom")


def test_transcription_worker_emits_error(qapp):
    results = []
    worker = TranscriptionWorker(_RaisingTranscriber(), np.zeros(1600, dtype=np.float32))
    worker.finished.connect(results.append)
    worker.run()
    assert "error" in results[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/tests/test_transcription_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ui.transcription_worker'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/ui/transcription_worker.py`:

```python
"""Background transcription worker for the home page transcript."""

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.core.transcription import AudioTranscriber


class TranscriptionWorker(QThread):
    """Runs AudioTranscriber.transcribe off the UI thread and emits finished(dict)."""

    finished = Signal(dict)

    def __init__(self, transcriber: AudioTranscriber, audio: np.ndarray, language: str = "english"):
        super().__init__()
        self._transcriber = transcriber
        self._audio = audio
        self._language = language

    def run(self):
        try:
            data = self._transcriber.transcribe(self._audio, language=self._language)
            self.finished.emit(data)
        except Exception as e:
            self.finished.emit({"error": str(e)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/tests/test_transcription_worker.py -v`
Expected: PASS (2 tests). Note `worker.run()` is invoked directly (synchronous) — fine for a unit test of signal emission.

- [ ] **Step 5: Lint**

Run: `ruff check app/ui/transcription_worker.py app/tests/test_transcription_worker.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add app/ui/transcription_worker.py app/tests/test_transcription_worker.py
git commit -m "feat: add background transcription worker"
```

---

### Task 5: Multilingual `AudioTranscriber` (Whisper + dedup)

**Files:**
- Rewrite: `app/core/transcription.py`
- Test: `app/tests/test_transcription.py`

**Interfaces:**
- Consumes: `SimpleForcedAligner` (unchanged, from `model.localization.ctc_alignment`).
- Produces:
  - Module-level `WHISPER_MODELS = {"english": "openai/whisper-tiny", "kannada": "vasista22/whisper-kannada-tiny", "hindi": "collabora/whisper-tiny-hindi"}`.
  - Module-level `get_pipeline(language: str = "english")` — lazy, cached, monkeypatchable (mirrors `backend/services/transcriber.py`).
  - `AudioTranscriber.transcribe(audio, sample_rate=16000, localizations=None, passage_text=None, language="english")` — same return shape as before (`{text, words, duration_sec}`), words entries `{word, start_sec, end_sec, confidence, stutter, stutter_type}`.

- [ ] **Step 1: Update existing tests + add new tests**

Replace the entire content of `app/tests/test_transcription.py` with:

```python
import numpy as np
import pytest

from app.core.transcription import AudioTranscriber, WHISPER_MODELS
from app.ui.transcription_panel import TranscriptionPanel


@pytest.fixture
def no_network(monkeypatch):
    """Make every pipeline load fail so transcription falls back to the aligner."""

    def _fail(language):
        raise RuntimeError("no network")

    monkeypatch.setattr("app.core.transcription.get_pipeline", _fail)


def test_audio_transcriber_fallback(no_network):
    transcriber = AudioTranscriber()
    audio = np.zeros(16000, dtype=np.float32)
    res = transcriber.transcribe(audio, sample_rate=16000)
    assert "text" in res
    assert "words" in res
    assert len(res["words"]) > 0


def test_audio_transcriber_stutter_alignment(no_network):
    transcriber = AudioTranscriber()
    audio = np.zeros(32000, dtype=np.float32)
    localizations = [(0.0, 2.0, 0.9)]
    res = transcriber.transcribe(audio, sample_rate=16000, localizations=localizations)
    stutter_words = [w for w in res["words"] if w["stutter"]]
    assert len(stutter_words) > 0


def test_transcription_panel_ui(qapp, no_network):
    panel = TranscriptionPanel()
    assert panel._text_edit.toPlainText() == ""
    audio = np.zeros(16000, dtype=np.float32)
    panel.set_audio(audio)
    assert panel._text_edit.toPlainText() != ""
    assert panel._table.rowCount() > 0

    panel.clear()
    assert panel._text_edit.toPlainText() == ""
    assert panel._table.rowCount() == 0


def test_whisper_model_mapping():
    assert WHISPER_MODELS["english"] == "openai/whisper-tiny"
    assert WHISPER_MODELS["kannada"] == "vasista22/whisper-kannada-tiny"
    assert WHISPER_MODELS["hindi"] == "collabora/whisper-tiny-hindi"


def test_transcribe_dedups_whisper_repeats(monkeypatch):
    captured = {}

    class FakePipe:
        def __call__(self, audio, return_timestamps="word"):
            captured["language"] = "kannada"
            return {
                "text": "ಹಲೋ ಹಲೋ ಜಗತ್ತು",
                "chunks": [
                    {"text": "ಹಲೋ", "timestamp": (0.0, 0.4), "confidence": 0.9},
                    {"text": "ಹಲೋ", "timestamp": (0.4, 0.8), "confidence": 0.9},
                    {"text": "ಜಗತ್ತು", "timestamp": (0.8, 1.3), "confidence": 0.8},
                ],
            }

    monkeypatch.setattr("app.core.transcription.get_pipeline", lambda language: FakePipe())
    transcriber = AudioTranscriber()
    res = transcriber.transcribe(np.zeros(16000, dtype=np.float32), language="kannada")

    assert captured["language"] == "kannada"
    assert res["text"] == "ಹಲೋ ಜಗತ್ತು"
    assert len(res["words"]) == 2
    assert res["words"][0]["word"] == "ಹಲೋ"
    assert res["words"][0]["start_sec"] == 0.0
```

- [ ] **Step 2: Run tests to verify the three existing ones now fail**

Run: `pytest app/tests/test_transcription.py -v`
Expected: FAIL — `test_audio_transcriber_fallback` / `test_audio_transcriber_stutter_alignment` / `test_transcription_panel_ui` error or hang because `get_pipeline` doesn't exist yet (`ImportError`) or the real Whisper pipeline would be invoked (never — `ImportError` hits first since the module was rewritten without `get_pipeline`). The two new tests (`test_whisper_model_mapping`, `test_transcribe_dedups_whisper_repeats`) also fail with `ImportError`.

- [ ] **Step 3: Rewrite `app/core/transcription.py`**

Replace the entire file content with:

```python
"""Audio transcription pipeline for Swaraaha.

Converts speech audio into timestamped transcript text and aligns detected dysfluencies.
Supports English, Kannada, and Hindi via per-language Whisper-tiny pipelines.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from model.localization.ctc_alignment import SimpleForcedAligner

WHISPER_MODELS = {
    "english": "openai/whisper-tiny",
    "kannada": "vasista22/whisper-kannada-tiny",
    "hindi": "collabora/whisper-tiny-hindi",
}

WHISPER_LANG_CODES = {"english": "en", "kannada": "kn", "hindi": "hi"}

_pipelines = {}


def get_pipeline(language: str = "english"):
    """Lazily initialize (and cache) the Whisper ASR pipeline for a language."""
    lang = language.lower()
    if lang not in _pipelines:
        from transformers import pipeline

        model_id = WHISPER_MODELS.get(lang, WHISPER_MODELS["english"])
        pipe = pipeline("automatic-speech-recognition", model=model_id, device="cpu")

        lang_code = WHISPER_LANG_CODES.get(lang, "en")
        pipe.model.generation_config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
            language=lang_code, task="transcribe"
        )
        try:
            no_timestamps_token_id = pipe.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            pipe.model.generation_config.no_timestamps_token_id = no_timestamps_token_id
        except Exception:
            pass

        _pipelines[lang] = pipe
    return _pipelines[lang]


class AudioTranscriber:
    """
    Speech transcription pipeline wrapper.

    Generates text transcription and word-level timestamps for input audio.
    Supports overlaying stutter detection results onto word intervals.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        self.model_name = model_name

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        localizations: Optional[List[Tuple[float, float, float]]] = None,
        passage_text: Optional[str] = None,
        language: str = "english",
    ) -> Dict[str, Any]:
        """
        Transcribe input audio array into text and timestamped words.

        Args:
            audio: 1-D numpy array of audio samples (float32, 16kHz).
            sample_rate: Audio sampling rate.
            localizations: List of (start_sec, end_sec, confidence) for detected stutters.
            passage_text: Optional reference passage text.
            language: "english", "kannada", or "hindi".

        Returns:
            Dict containing:
                - "text": Full transcription string.
                - "words": List of dicts with keys (word, start_sec, end_sec, confidence, stutter, stutter_type).
                - "duration_sec": Audio duration in seconds.
        """
        if audio is None or len(audio) == 0:
            return {"text": "", "words": []}

        duration_sec = len(audio) / sample_rate

        transcript_text = None
        word_list: List[Dict[str, Any]] = []

        try:
            transcript_text, word_list = self._transcribe_with_whisper(audio, language)
        except Exception:
            transcript_text = None
            word_list = []

        if not transcript_text or not word_list:
            transcript_text, word_list = self._fallback_transcribe(audio, duration_sec, passage_text)

        if localizations and word_list:
            for w in word_list:
                w_start = w["start_sec"]
                w_end = w["end_sec"]
                for (st_start, st_end, conf) in localizations:
                    if max(w_start, st_start) < min(w_end, st_end):
                        w["stutter"] = True
                        w["stutter_type"] = "dysfluency"
                        break

        return {
            "text": transcript_text,
            "words": word_list,
            "duration_sec": round(duration_sec, 2),
        }

    def _transcribe_with_whisper(self, audio: np.ndarray, language: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Run Whisper word-level transcription and de-duplicate hallucinated repeats."""
        pipe = get_pipeline(language)
        result = pipe(audio, return_timestamps="word")

        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])

        word_list: List[Dict[str, Any]] = []
        prev_chunk_clean = None
        for chunk in chunks:
            chunk_text = chunk.get("text", "").strip()
            chunk_text_clean = chunk_text.lower().strip(".,?!;:-_\"'()[]{} ")
            if chunk_text_clean == prev_chunk_clean and chunk_text_clean != "":
                continue
            prev_chunk_clean = chunk_text_clean

            timestamp = chunk.get("timestamp", (0.0, 0.0))
            start = timestamp[0] if timestamp and timestamp[0] is not None else 0.0
            end = timestamp[1] if timestamp and timestamp[1] is not None else start + 0.3
            word_list.append({
                "word": chunk_text,
                "start_sec": round(float(start), 2),
                "end_sec": round(float(end), 2),
                "confidence": round(float(chunk.get("confidence", 0.9)), 2),
                "stutter": False,
                "stutter_type": None,
            })

        deduped_words = []
        prev_word_clean = None
        for w in text.split():
            w_clean = w.lower().strip(".,?!;:-_\"'()[]{} ")
            if w_clean == prev_word_clean and w_clean != "":
                continue
            deduped_words.append(w)
            prev_word_clean = w_clean

        return " ".join(deduped_words), word_list

    def _fallback_transcribe(
        self, audio: np.ndarray, duration_sec: float, reference_text: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate fallback transcription based on audio structure or passage text."""
        default_passage = (
            reference_text
            or "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow."
        )

        words_stamps = SimpleForcedAligner.align(
            audio, default_passage, sr=16000, max_length_seconds=max(10.0, duration_sec)
        )

        word_list = []
        full_words = []
        for ws in words_stamps:
            full_words.append(ws.word)
            word_list.append({
                "word": ws.word,
                "start_sec": ws.start_sec,
                "end_sec": ws.end_sec,
                "confidence": ws.confidence,
                "stutter": False,
                "stutter_type": None,
            })

        return " ".join(full_words), word_list
```

Note on `transformers` 5.x: if `pipe.tokenizer.get_decoder_prompt_ids` or `generation_config.no_timestamps_token_id` differ in the installed version, wrap only those lines defensively inside the existing `try/except` — the tests never download models, so this only matters at app runtime.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/tests/test_transcription.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint**

Run: `ruff check app/core/transcription.py app/tests/test_transcription.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add app/core/transcription.py app/tests/test_transcription.py
git commit -m "feat: support kannada and hindi transcription via whisper"
```

---

### Task 6: Thread language through `ModelRunner` and `AnalysisWorker`

**Files:**
- Modify: `app/core/model_runner.py`
- Modify: `app/ui/main_window.py` (only `AnalysisWorker.__init__` and `run` in this task)
- Test: `app/tests/test_main_window.py` (add a worker test)

**Interfaces:**
- Consumes: `ModelRunner` (from `app/core/model_runner.py`), `AnalysisWorker` (from `app/ui/main_window.py`).
- Produces: `ModelRunner.transcribe(audio, localizations=None, language="english")`, `ModelRunner.analyze(audio, language="english")`; `AnalysisWorker(model_runner, audio, language="english")`.

- [ ] **Step 1: Write the failing test**

Append to `app/tests/test_main_window.py`:

```python
def test_analysis_worker_passes_language(qapp):
    from app.ui.main_window import AnalysisWorker

    captured = {}

    class FakeRunner:
        def analyze(self, audio, language="english"):
            captured["language"] = language
            return {"ok": True}

    results = []
    worker = AnalysisWorker(FakeRunner(), np.zeros(1600, dtype=np.float32), "hindi")
    worker.finished.connect(results.append)
    worker.run()

    assert captured["language"] == "hindi"
    assert results[0] == {"ok": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/tests/test_main_window.py::test_analysis_worker_passes_language -v`
Expected: FAIL — `captured["language"]` is `"english"` (default) or `TypeError` (unexpected keyword) because `analyze` is called without `language`.

- [ ] **Step 3: Implement language threading**

Rewrite `app/core/model_runner.py` entirely:

```python
"""Model inference wrapper — stub for Phase 1. Will integrate with model/ in Phase 3."""

import numpy as np


from app.core.transcription import AudioTranscriber


class ModelRunner:
    def __init__(self, models_dir: str = ""):
        self.models_dir = models_dir
        self._loaded = False
        self.transcriber = AudioTranscriber()

    def transcribe(self, audio: np.ndarray, localizations=None, language: str = "english") -> dict:
        """Run transcription pipeline on audio."""
        return self.transcriber.transcribe(audio, localizations=localizations, language=language)

    def analyze(self, audio: np.ndarray, language: str = "english") -> dict:
        """Run classification + localization + transcription on audio. Returns structured results."""
        localizations = [
            (0.5, 1.2, 0.87),
            (3.4, 4.1, 0.72),
        ]
        classifications = {
            "prolongation": (False, 0.12),
            "block": (True, 0.87),
            "soundrep": (False, 0.08),
            "wordrep": (False, 0.05),
            "interjection": (True, 0.72),
        }
        transcription = self.transcribe(audio, localizations=localizations, language=language)

        return {
            "classifications": classifications,
            "localizations": localizations,
            "transcription": transcription,
        }
```

In `app/ui/main_window.py`, replace the `AnalysisWorker` class (lines 25-38):

```python
class AnalysisWorker(QThread):
    finished = Signal(dict)

    def __init__(self, model_runner: ModelRunner, audio: np.ndarray, language: str = "english"):
        super().__init__()
        self._model_runner = model_runner
        self._audio = audio
        self._language = language

    def run(self):
        try:
            results = self._model_runner.analyze(self._audio, language=self._language)
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit({"error": str(e)})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest app/tests/test_main_window.py::test_analysis_worker_passes_language -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `ruff check app/core/model_runner.py app/ui/main_window.py app/tests/test_main_window.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add app/core/model_runner.py app/ui/main_window.py app/tests/test_main_window.py
git commit -m "feat: thread selected language through analysis"
```

---

### Task 7: Cap transcript tables + language param on panels

**Files:**
- Modify: `app/ui/compact_transcript.py`
- Modify: `app/ui/transcription_panel.py`
- Test: `app/tests/test_transcription.py` (append two tests)

**Interfaces:**
- Consumes: `cap_table_height` (Task 1).
- Produces: `CompactTranscript.set_audio(audio, sample_rate=16000, localizations=None, language="english")`; `TranscriptionPanel.set_audio(audio, sample_rate=16000, localizations=None, language="english")`; both `run_transcription(localizations=None)` use the stored `self._language`. Compact table capped at `MAX_ROWS = 8`, results table at `MAX_ROWS = 10`.

- [ ] **Step 1: Write the failing tests**

Append to `app/tests/test_transcription.py`:

```python
def test_transcription_panel_table_capped(qapp, no_network):
    panel = TranscriptionPanel()
    words = [
        {"word": f"w{i}", "start_sec": float(i), "end_sec": float(i) + 0.5, "confidence": 0.9,
         "stutter": False, "stutter_type": None}
        for i in range(30)
    ]
    panel.set_transcription({"text": " ".join(w["word"] for w in words), "words": words})
    assert panel._table.rowCount() == 30
    full = (panel._table.horizontalHeader().height() or 30) + 30 * panel._table.rowHeight(0)
    assert panel._table.maximumHeight() < full
    assert panel._table.maximumHeight() > (panel._table.horizontalHeader().height() or 30)


def test_compact_transcript_table_capped(qapp):
    from app.ui.compact_transcript import CompactTranscript

    panel = CompactTranscript()
    words = [
        {"word": f"w{i}", "start_sec": float(i), "end_sec": float(i) + 0.5, "confidence": 0.9,
         "stutter": False, "stutter_type": None}
        for i in range(30)
    ]
    panel.set_transcription({"text": " ".join(w["word"] for w in words), "words": words})
    assert panel._table.rowCount() == 30
    full = (panel._table.horizontalHeader().height() or 30) + 30 * panel._table.rowHeight(0)
    assert panel._table.maximumHeight() < full
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/tests/test_transcription.py::test_transcription_panel_table_capped app/tests/test_transcription.py::test_compact_transcript_table_capped -v`
Expected: FAIL — `maximumHeight()` is still the default `16777215` (not capped).

- [ ] **Step 3: Update `app/ui/compact_transcript.py`**

Replace the entire file content with:

```python
"""Compact transcript view: text area + word-level alignment table.

Used as the bottom half of the Home Page right column once audio is loaded.
Auto-runs transcription; intentionally has no header, status pill, or action buttons.
"""

import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.transcription import AudioTranscriber
from app.ui.table_utils import cap_table_height
from app.ui.theme import COLORS

MAX_ROWS = 8


class CompactTranscript(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._transcriber = AudioTranscriber()
        self._audio = None
        self._language = "english"
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QLabel("Transcription")
        header.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(header)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setPlaceholderText("Transcription will appear here after loading audio...")
        self._text_edit.setMinimumHeight(90)
        self._text_edit.setMaximumHeight(140)
        layout.addWidget(self._text_edit)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Word", "Start (s)", "End (s)", "Confidence", "Status"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

    def set_audio(self, audio: np.ndarray, sample_rate: int = 16000, localizations=None, language: str = "english"):
        """Set audio array and automatically run transcription."""
        self._audio = audio
        self._language = language
        if audio is not None and len(audio) > 0:
            self.run_transcription(localizations=localizations)
        else:
            self.clear()

    def run_transcription(self, localizations=None):
        """Run the transcription pipeline on the loaded audio."""
        if self._audio is None or len(self._audio) == 0:
            return
        data = self._transcriber.transcribe(self._audio, localizations=localizations, language=self._language)
        self.set_transcription(data)

    def set_transcription(self, data: dict):
        """Display transcription dictionary."""
        text = data.get("text", "")
        words = data.get("words", [])

        self._text_edit.setPlainText(text)
        self._table.setUpdatesEnabled(False)
        try:
            self._table.setRowCount(len(words))

            for row, w in enumerate(words):
                word_item = QTableWidgetItem(str(w.get("word", "")))
                start_item = QTableWidgetItem(f"{w.get('start_sec', 0.0):.2f}")
                end_item = QTableWidgetItem(f"{w.get('end_sec', 0.0):.2f}")
                conf_item = QTableWidgetItem(f"{w.get('confidence', 0.0)*100:.0f}%")

                is_stutter = w.get("stutter", False)
                status_str = "Stutter Detected" if is_stutter else "Normal"
                status_item = QTableWidgetItem(status_str)

                if is_stutter:
                    red_color = QColor(COLORS["dysfluency"]["prolongation"])
                    status_item.setForeground(red_color)
                    word_item.setForeground(red_color)
                    font = word_item.font()
                    font.setBold(True)
                    word_item.setFont(font)
                    status_item.setFont(font)

                self._table.setItem(row, 0, word_item)
                self._table.setItem(row, 1, start_item)
                self._table.setItem(row, 2, end_item)
                self._table.setItem(row, 3, conf_item)
                self._table.setItem(row, 4, status_item)
        finally:
            self._table.setUpdatesEnabled(True)
            cap_table_height(self._table, MAX_ROWS)

    def clear(self):
        """Clear transcription display."""
        self._audio = None
        self._text_edit.clear()
        self._table.setRowCount(0)
        self._table.setMinimumHeight(0)
        self._table.setMaximumHeight(16777215)
```

- [ ] **Step 4: Update `app/ui/transcription_panel.py`**

Apply these edits:

(4a) Replace the import (line 22):

```python
from app.ui.table_utils import cap_table_height
```

(4b) After `from app.ui.theme import COLORS`, add a module constant:

```python
MAX_ROWS = 10
```

(4c) In `__init__` (after `self._transcription_data = {"text": "", "words": []}`), add:

```python
        self._language = "english"
```

(4d) Replace `set_audio` (lines 108-115):

```python
    def set_audio(self, audio: np.ndarray, sample_rate: int = 16000, localizations=None, language: str = "english"):
        """Set audio array and automatically run transcription."""
        self._audio = audio
        self._language = language
        if audio is not None and len(audio) > 0:
            self._status_label.setText(f"Audio Ready ({len(audio)/sample_rate:.1f}s)")
            self.run_transcription(localizations=localizations)
        else:
            self.clear()
```

(4e) In `run_transcription`, replace the transcribe call (line 124):

```python
        data = self._transcriber.transcribe(self._audio, localizations=localizations, language=self._language)
```

(4f) In `set_transcription`, wrap population in `setUpdatesEnabled` and swap `resize_table_to_contents(self._table)` (line 162) for `cap_table_height(self._table, MAX_ROWS)`. The method becomes:

```python
    def set_transcription(self, data: dict):
        """Display transcription dictionary."""
        self._transcription_data = data
        text = data.get("text", "")
        words = data.get("words", [])

        self._text_edit.setPlainText(text)
        self._table.setUpdatesEnabled(False)
        try:
            self._table.setRowCount(len(words))

            for row, w in enumerate(words):
                word_item = QTableWidgetItem(str(w.get("word", "")))
                start_item = QTableWidgetItem(f"{w.get('start_sec', 0.0):.2f}")
                end_item = QTableWidgetItem(f"{w.get('end_sec', 0.0):.2f}")
                conf_item = QTableWidgetItem(f"{w.get('confidence', 0.0)*100:.0f}%")

                is_stutter = w.get("stutter", False)
                status_str = "Stutter Detected" if is_stutter else "Normal"
                status_item = QTableWidgetItem(status_str)

                if is_stutter:
                    red_color = QColor(COLORS["dysfluency"]["prolongation"])
                    status_item.setForeground(red_color)
                    word_item.setForeground(red_color)
                    font = word_item.font()
                    font.setBold(True)
                    word_item.setFont(font)
                    status_item.setFont(font)

                self._table.setItem(row, 0, word_item)
                self._table.setItem(row, 1, start_item)
                self._table.setItem(row, 2, end_item)
                self._table.setItem(row, 3, conf_item)
                self._table.setItem(row, 4, status_item)
        finally:
            self._table.setUpdatesEnabled(True)
            cap_table_height(self._table, MAX_ROWS)
```

(4g) In `clear()` (line 170), after `self._table.setMinimumHeight(0)`, add:

```python
        self._table.setMaximumHeight(16777215)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest app/tests/test_transcription.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Lint**

Run: `ruff check app/ui/compact_transcript.py app/ui/transcription_panel.py app/tests/test_transcription.py`
Expected: no errors. (`resize_table_to_contents` import must be gone from `transcription_panel.py` — the import was replaced in 4a.)

- [ ] **Step 7: Commit**

```bash
git add app/ui/compact_transcript.py app/ui/transcription_panel.py app/tests/test_transcription.py
git commit -m "feat: cap transcript tables for internal scrolling"
```

---

### Task 8: Debounce `WaveformView` resize redraw

**Files:**
- Modify: `app/ui/waveform_view.py`
- Test: `app/tests/test_waveform_view.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `WaveformView` with `_resize_timer` (single-shot `QTimer`) created lazily in `resizeEvent`; `set_audio`/`set_overlays`/`clear_overlays` still call `_draw()` immediately.

- [ ] **Step 1: Write the failing tests**

Create `app/tests/test_waveform_view.py`:

```python
import numpy as np

from app.ui.waveform_view import WaveformView


def test_waveform_set_audio_draws(qapp):
    view = WaveformView()
    view.resize(400, 200)
    view.set_audio(np.zeros(1600, dtype=np.float32), 16000)
    assert len(view.scene().items()) > 0


def test_waveform_resize_schedules_debounced_redraw(qapp):
    view = WaveformView()
    view.resize(400, 200)
    view.set_audio(np.zeros(1600, dtype=np.float32), 16000)
    qapp.processEvents()
    assert view._resize_timer is not None
    assert view._resize_timer.isActive()
```

- [ ] **Step 2: Run tests to verify the second fails**

Run: `pytest app/tests/test_waveform_view.py -v`
Expected: `test_waveform_set_audio_draws` PASS; `test_waveform_resize_schedules_debounced_redraw` FAIL — `AttributeError: 'WaveformView' object has no attribute '_resize_timer'`.

- [ ] **Step 3: Update `app/ui/waveform_view.py`**

(3a) Replace the QtCore import (line 5):

```python
from PySide6.QtCore import Qt, QRectF, QTimer
```

(3b) Replace `resizeEvent` (lines 96-99):

```python
    def resizeEvent(self, event):
        """Debounce redraws during resize drags."""
        super().resizeEvent(event)
        if not hasattr(self, "_resize_timer") or self._resize_timer is None:
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._draw)
        self._resize_timer.start(50)
```

`set_audio`, `set_overlays`, `clear_overlays` keep calling `_draw()` directly (path rebuilt only on explicit audio/overlay changes).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/tests/test_waveform_view.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint**

Run: `ruff check app/ui/waveform_view.py app/tests/test_waveform_view.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add app/ui/waveform_view.py app/tests/test_waveform_view.py
git commit -m "perf: debounce waveform redraw on resize"
```

---

### Task 9: `MainWindow` integration (prompt + wait dialog + background transcription)

**Files:**
- Rewrite: `app/ui/main_window.py`
- Modify: `app/ui/results_panel.py` (language param on `set_results`)
- Modify: `app/ui/analysis_page.py` (language param on `set_results`)
- Test: `app/tests/test_main_window.py`

**Interfaces:**
- Consumes: `LanguageDialog` (Task 2), `WaitDialog` (Task 3), `TranscriptionWorker` (Task 4), `AudioTranscriber` via `ModelRunner.transcriber` (Task 5), `AnalysisWorker` (Task 6).
- Produces: `MainWindow._prompt_language() -> str` (seam, modal via `LanguageDialog.exec`), `MainWindow._start_home_transcription(audio, language)` (seam, spawns worker + wait dialog), `MainWindow._on_home_transcription_done(data, worker)`; `ResultsPanel.set_results(results, audio=None, sample_rate=16000, language="english")`; `AnalysisPage.set_results(results, audio=None, sample_rate=16000, language="english")`.

- [ ] **Step 1: Update the three existing MainWindow tests + add new tests**

Replace the entire content of `app/tests/test_main_window.py` with:

```python
import numpy as np
import pytest
import soundfile as sf

from app.ui.main_window import MainWindow


@pytest.fixture
def app_name(qapp):
    qapp.setOrganizationName("SwaraahaTests")
    qapp.setApplicationName("SwaraahaTests")
    return qapp


def _make_wav(path, sr=16000):
    sf.write(str(path), np.zeros(sr, dtype=np.float32), sr)


def _no_background_transcription(win, monkeypatch):
    monkeypatch.setattr(win, "_start_home_transcription", lambda audio, language: None)


def test_load_path_sets_audio_and_state(qapp, app_name, tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    _make_wav(path)
    win = MainWindow()
    _no_background_transcription(win, monkeypatch)
    win._load_path(str(path))
    assert win._current_audio is not None
    assert "Loaded" in win.statusBar().currentMessage()
    assert not win._home_page.get_audio_controls()._analyze_btn.isHidden()
    assert not win._home_page.get_transcription_panel().isHidden()


def test_load_path_reports_error(qapp, app_name, tmp_path):
    win = MainWindow()
    win._load_path(str(tmp_path / "missing.wav"))
    assert win._current_audio is None
    assert "Error loading file" in win.statusBar().currentMessage()


def test_file_selected_wires_to_load(qapp, app_name, tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    _make_wav(path)
    win = MainWindow()
    _no_background_transcription(win, monkeypatch)
    win._home_page.file_selected.emit(str(path))
    assert win._current_audio is not None


def test_drop_accepts_only_audio(qapp, app_name, tmp_path, monkeypatch):
    from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent

    win = MainWindow()
    _no_background_transcription(win, monkeypatch)
    wav = tmp_path / "a.wav"
    _make_wav(wav)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(wav))])
    enter = QDragEnterEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
    )
    win.dragEnterEvent(enter)
    assert enter.isAccepted()

    bad = tmp_path / "notes.txt"
    bad.write_bytes(b"x")
    mime2 = QMimeData()
    mime2.setUrls([QUrl.fromLocalFile(str(bad))])
    enter2 = QDragEnterEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, mime2,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
    )
    win.dragEnterEvent(enter2)
    assert not enter2.isAccepted()


def test_analysis_worker_passes_language(qapp):
    from app.ui.main_window import AnalysisWorker

    captured = {}

    class FakeRunner:
        def analyze(self, audio, language="english"):
            captured["language"] = language
            return {"ok": True}

    results = []
    worker = AnalysisWorker(FakeRunner(), np.zeros(1600, dtype=np.float32), "hindi")
    worker.finished.connect(results.append)
    worker.run()

    assert captured["language"] == "hindi"
    assert results[0] == {"ok": True}


def test_start_home_transcription_populates_transcript(qapp, app_name):
    win = MainWindow()

    def fake_transcribe(audio, sample_rate=16000, localizations=None, passage_text=None, language="english"):
        return {
            "text": "hello world",
            "words": [
                {"word": "hello", "start_sec": 0.0, "end_sec": 0.4, "confidence": 0.9,
                 "stutter": False, "stutter_type": None},
                {"word": "world", "start_sec": 0.4, "end_sec": 0.8, "confidence": 0.8,
                 "stutter": False, "stutter_type": None},
            ],
        }

    win._model_runner.transcriber.transcribe = fake_transcribe
    win._start_home_transcription(np.zeros(1600, dtype=np.float32), "english")

    assert win._wait_dialog is not None
    worker = win._transcription_worker
    worker.wait(5000)
    qapp.processEvents()

    assert win._wait_dialog is None
    panel = win._home_page.get_transcription_panel()
    assert panel._text_edit.toPlainText() == "hello world"
    assert panel._table.rowCount() == 2


def test_home_transcription_error_is_handled(qapp, app_name):
    win = MainWindow()

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    win._model_runner.transcriber.transcribe = _raise
    win._start_home_transcription(np.zeros(1600, dtype=np.float32), "english")
    win._transcription_worker.wait(5000)
    qapp.processEvents()

    assert win._wait_dialog is None
    assert "Transcription failed" in win.statusBar().currentMessage()


def test_prompt_language_uses_dialog(qapp, app_name, monkeypatch):
    win = MainWindow()

    class FakeDialog:
        def __init__(self, current, parent=None):
            self._current = current

        def exec(self):
            return 0

        def selected(self):
            return "hindi"

    monkeypatch.setattr("app.ui.main_window.LanguageDialog", FakeDialog)
    assert win._prompt_language() == "hindi"


def test_on_load_prompts_then_loads(qapp, app_name, tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    sf.write(str(path), np.zeros(1600, dtype=np.float32), 16000)
    win = MainWindow()
    monkeypatch.setattr("app.ui.main_window.QFileDialog.getOpenFileName", lambda *a, **k: (str(path), ""))
    monkeypatch.setattr(win, "_prompt_language", lambda: "kannada")
    _no_background_transcription(win, monkeypatch)
    win._on_load()
    assert win._current_language == "kannada"
    assert win._current_audio is not None
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest app/tests/test_main_window.py -v`
Expected: FAIL — `AttributeError: 'MainWindow' object has no attribute '_prompt_language'` / `_start_home_transcription`, and `test_load_path_sets_audio_and_state` errors because `_start_home_transcription` was monkeypatched but never called (that one would actually FAIL via the assertion since nothing sets the panel visible — no: `set_transcript_visible(True)` is still called, so it passes only after integration). In short, expect a mix of AttributeErrors.

- [ ] **Step 3: Rewrite `app/ui/main_window.py`**

Replace the entire file content with:

```python
"""Main window with QStackedWidget for page navigation."""

import os

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.audio_handler import AudioHandler
from app.core.model_runner import ModelRunner
from app.ui.analysis_page import AnalysisPage
from app.ui.home_page import HomePage
from app.ui.language_dialog import LanguageDialog
from app.ui.styles import build_stylesheet
from app.ui.theme import is_dark_mode, set_theme
from app.ui.transcription_worker import TranscriptionWorker
from app.ui.wait_dialog import WaitDialog

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


class AnalysisWorker(QThread):
    finished = Signal(dict)

    def __init__(self, model_runner: ModelRunner, audio: np.ndarray, language: str = "english"):
        super().__init__()
        self._model_runner = model_runner
        self._audio = audio
        self._language = language

    def run(self):
        try:
            results = self._model_runner.analyze(self._audio, language=self._language)
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit({"error": str(e)})


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swaraaha — Speech Dysfluency Detection")
        self.setMinimumSize(1200, 800)

        self._audio_handler = AudioHandler()
        self._model_runner = ModelRunner()
        self._current_audio = None
        self._worker = None
        self._current_language = "english"
        self._transcription_worker = None
        self._wait_dialog = None

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._home_page = HomePage()
        self._stack.addWidget(self._home_page)

        self._analysis_page = AnalysisPage()
        self._stack.addWidget(self._analysis_page)

        self._home_page.record_clicked.connect(self._on_record)
        self._home_page.stop_clicked.connect(self._on_stop)
        self._home_page.load_clicked.connect(self._on_load)
        self._home_page.play_clicked.connect(self._on_play)
        self._home_page.analyze_clicked.connect(self._on_analyze)
        self._home_page.file_selected.connect(self._on_file_selected)

        self._analysis_page.back_clicked.connect(self._go_home)

        theme_action = self.menuBar().addAction("Toggle Dark Mode")
        theme_action.triggered.connect(self._toggle_theme)

        self.setAcceptDrops(True)

    def _on_file_selected(self, path: str):
        self._load_path(path)

    def _load_path(self, path: str):
        try:
            audio = self._audio_handler.load_audio(path)
            self._current_audio = audio
            self._home_page.get_audio_controls().set_audio_loaded()
            self._home_page.get_transcription_panel().clear()
            self._home_page.set_transcript_visible(True)
            self._start_home_transcription(audio, self._current_language)
            self.statusBar().showMessage(f"Loaded: {path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading file: {e}")

    def _on_record(self):
        self._current_language = self._prompt_language()
        self._audio_handler.start_recording()
        self._home_page.get_audio_controls().set_recording(True)
        self.statusBar().showMessage("Recording...")

    def _on_stop(self):
        audio = self._audio_handler.stop_recording()
        if len(audio) > 0:
            self._current_audio = audio
            self._home_page.get_audio_controls().set_recording(False)
            self._home_page.get_audio_controls().set_audio_loaded()
            self._home_page.get_transcription_panel().clear()
            self._home_page.set_transcript_visible(True)
            self._start_home_transcription(audio, self._current_language)
            self.statusBar().showMessage(f"Recorded {len(audio) / self._audio_handler.sample_rate:.1f}s of audio")
        else:
            self._home_page.get_audio_controls().set_recording(False)
            self.statusBar().showMessage("Ready")

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Audio", "", "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        if path:
            self._current_language = self._prompt_language()
            self._load_path(path)

    def _prompt_language(self) -> str:
        """Ask the user which language to transcribe in; keeps the last choice on cancel."""
        dialog = LanguageDialog(self._current_language, self)
        dialog.exec()
        return dialog.selected()

    def _start_home_transcription(self, audio: np.ndarray, language: str):
        """Run home-page transcription in the background with a wait dialog."""
        worker = TranscriptionWorker(self._model_runner.transcriber, audio, language)
        self._transcription_worker = worker
        self._wait_dialog = WaitDialog(self)
        self._wait_dialog.show()
        worker.finished.connect(lambda data, w=worker: self._on_home_transcription_done(data, w))
        worker.start()

    def _on_home_transcription_done(self, data: dict, worker):
        if worker is not self._transcription_worker:
            return
        if self._wait_dialog is not None:
            self._wait_dialog.finish()
            self._wait_dialog = None
        if "error" in data:
            self.statusBar().showMessage(f"Transcription failed: {data['error']}")
            return
        self._home_page.get_transcription_panel().set_transcription(data)
        self.statusBar().showMessage("Transcript ready")

    def _on_play(self):
        if self._current_audio is not None:
            self._audio_handler.play_audio(self._current_audio)
            self._home_page.get_audio_controls().set_playing(True)
            self.statusBar().showMessage("Playing...")

    def _on_analyze(self):
        if self._current_audio is None:
            self.statusBar().showMessage("No audio to analyze")
            return
        if self._worker and self._worker.isRunning():
            self.statusBar().showMessage("Analysis already in progress...")
            return

        self.statusBar().showMessage("Analyzing audio...")
        self._worker = AnalysisWorker(self._model_runner, self._current_audio, self._current_language)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.start()

    def _on_analysis_done(self, results: dict):
        if "error" in results:
            self.statusBar().showMessage(f"Analysis failed: {results['error']}")
            return
        self._analysis_page.set_results(results, self._current_audio, language=self._current_language)
        self._stack.setCurrentIndex(1)
        self.statusBar().showMessage("Analysis complete")

    def _go_home(self):
        self._stack.setCurrentIndex(0)
        self.statusBar().showMessage("Ready")

    def _toggle_theme(self):
        set_theme(not is_dark_mode())
        self.setStyleSheet(build_stylesheet())

    def dragEnterEvent(self, event):
        if self._first_audio_path(event.mimeData()) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._first_audio_path(event.mimeData())
        if path:
            self._load_path(path)
            self._home_page.get_file_panel().add_recent(path)

    @staticmethod
    def _first_audio_path(mime) -> str | None:
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile():
                local = url.toLocalFile()
                if os.path.splitext(local)[1].lower() in _AUDIO_EXTENSIONS:
                    return local
        return None
```

- [ ] **Step 4: Add language param to `app/ui/results_panel.py`**

(4a) Replace the `set_results` signature (line 77):

```python
    def set_results(self, results: dict, audio: np.ndarray = None, sample_rate: int = 16000, language: str = "english"):
```

(4b) Replace the fallback transcribe call (line 112):

```python
            self._transcription_panel.set_audio(audio, sample_rate, localizations=localizations, language=language)
```

- [ ] **Step 5: Add language param to `app/ui/analysis_page.py`**

Replace `set_results` (lines 70-72):

```python
    def set_results(self, results: dict, audio: np.ndarray = None, sample_rate: int = 16000, language: str = "english"):
        """Update the page with analysis results."""
        self._results.set_results(results, audio, sample_rate, language=language)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest app/tests/test_main_window.py -v`
Expected: PASS (9 tests).

- [ ] **Step 7: Run the full suite to confirm nothing regressed**

Run: `pytest`
Expected: ALL PASS (~28 tests). If the real `AnalysisWorker`/transcription tests are unaffected, this should be green.

- [ ] **Step 8: Lint**

Run: `ruff check app/ui/main_window.py app/ui/results_panel.py app/ui/analysis_page.py app/tests/test_main_window.py`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add app/ui/main_window.py app/ui/results_panel.py app/ui/analysis_page.py app/tests/test_main_window.py
git commit -m "feat: background transcription with wait dialog and language prompt"
```

---

### Task 10: Final verification

**Files:** none modified.

- [ ] **Step 1: Run the full test suite**

Run: `pytest`
Expected: ALL PASS (~28 tests).

- [ ] **Step 2: Lint the whole app package**

Run: `ruff check app/`
Expected: no errors.

- [ ] **Step 3: Confirm git state**

Run: `git status --short` and `git log --oneline -12`
Expected: working tree clean on `desktop-transcription-ui` with the feature commits listed.

- [ ] **Step 4: Manual smoke (optional, desktop only)**

Run: `python -m app.main` from repo root, record a short clip or load a WAV, confirm the language dialog appears, the "Generating transcript, please wait" dialog shows and dismisses, the transcript table scrolls internally, and the results page scrolls smoothly.

---

## Self-Review Notes

- **Spec coverage:** wait dialog (Tasks 3, 9), scrollable home transcript (Tasks 1, 7), capped results table (Tasks 1, 7), language prompt on Record/Load (Tasks 2, 9), background transcription (Tasks 4, 9), multilingual ASR with dedup (Task 5), jittery-scroll fix (Tasks 7, 8), button-consistency requirement (Task 2 uses default `QPushButton` + `nav_btn_active`), language threading into results fallback (Tasks 6, 9 steps 4-5).
- **No placeholders:** every step has real code and exact commands.
- **Type consistency:** `cap_table_height(table, max_rows)` (Task 1) is used in Task 7; `TranscriptionWorker(transcriber, audio, language)` (Task 4) and `LanguageDialog(current, parent)` (Task 2) are used in Task 9; `AudioTranscriber.transcribe(..., language=...)` (Task 5) is used in Tasks 4, 7, 9; `set_results(..., language=...)` added in Task 9 steps 4-5.
- **No network in tests:** `no_network` fixture monkeypatches `app.core.transcription.get_pipeline`; MainWindow tests inject a fake `transcriber.transcribe` and patch `_start_home_transcription` / `_prompt_language` / `QFileDialog`.
