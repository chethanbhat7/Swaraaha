# Desktop Transcription Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the deferred Minor findings from the desktop transcription UI reviews (branch `desktop-transcription-cleanup`, stacked on `desktop-transcription-ui`).

**Architecture:** Small, surgical cleanups across the already-merged feature code: table-helper hardening, dialog input validation, dead-code removal, a shared table-population helper, a color-only QSS class for the language dialog highlight, and test-strengthening. No behavior changes to the happy paths.

**Tech Stack:** PySide6, Python 3, pytest (offscreen via `app/tests/conftest.py` `qapp` fixture), ruff.

## Global Constraints

- Branch: `desktop-transcription-cleanup` (already created from `desktop-transcription-ui` HEAD `c04fd9a`).
- Run pytest from repo root: `pytest app/tests -v` (QT offscreen set in `app/tests/conftest.py`).
- NO network/model downloads in tests — never construct `AudioTranscriber()` in a test that reaches `get_pipeline` without monkeypatching; use the existing `no_network` fixture or fakes.
- Run `ruff check` on every touched file and leave it clean.
- Conventional commit messages (`fix:`, `refactor:`, `test:`, `style:`).
- The `lang_btn_active` QSS class must be color-only (background/color/border) so the highlighted language button keeps the default 48px button height; do NOT copy nav metrics.
- Language codes stay `english|kannada|hindi`; `selected()` returns a code from that set.
- Do not touch `resize_table_to_contents`; the classification table keeps full height.

---

### Task 1: Harden `cap_table_height` (max_rows clamp + empty-table test)

**Files:**
- Modify: `app/ui/table_utils.py:14-18`
- Test: `app/tests/test_table_utils.py`

**Interfaces:**
- Consumes: existing `cap_table_height(table: QTableWidget, max_rows: int)`.
- Produces: `cap_table_height` now clamps `max_rows` to a minimum of 1 (negative/zero values no longer silently un-cap a table via Qt's negative-max-height semantics). Signature unchanged; later tasks consume it unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_cap_table_height_empty_table_uses_fallback(qapp):
    table = QTableWidget(0, 3)
    table.show()
    qapp.processEvents()
    cap_table_height(table, 5)
    header = table.horizontalHeader().height() or 30
    assert table.maximumHeight() == header + 5 * 30 + 4


def test_cap_table_height_clamps_max_rows(qapp):
    table = QTableWidget(20, 3)
    table.show()
    qapp.processEvents()
    header = table.horizontalHeader().height() or 30
    row = table.rowHeight(0)
    cap_table_height(table, 0)
    assert table.maximumHeight() == header + 1 * row + 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/tests/test_table_utils.py -v`
Expected: the clamp test fails (a `max_rows` of 0 currently yields `header + 0 + 4`, not `header + row + 4`); the fallback test may pass (fallback `or 30` already works) — that is acceptable, its purpose is regression coverage.

- [ ] **Step 3: Implement the clamp**

In `app/ui/table_utils.py`, change `cap_table_height`:

```python
def cap_table_height(table: QTableWidget, max_rows: int):
    """Cap a table's maximum height so longer contents scroll inside the table."""
    header_height = table.horizontalHeader().height() or 30
    row_height = table.rowHeight(0) if table.rowCount() > 0 else 30
    table.setMaximumHeight(header_height + max(1, max_rows) * row_height + 4)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/tests/test_table_utils.py -v`
Expected: 3 passed (2 new + 1 existing).

- [ ] **Step 5: Lint and commit**

```bash
ruff check app/ui/table_utils.py app/tests/test_table_utils.py
git add app/ui/table_utils.py app/tests/test_table_utils.py
git commit -m "fix: clamp cap_table_height max rows to a positive minimum"
```

---

### Task 2: Validate `LanguageDialog.current`

**Files:**
- Modify: `app/ui/language_dialog.py:16-35`
- Test: `app/tests/test_language_dialog.py`

**Interfaces:**
- Consumes: `LanguageDialog.__init__(current="english", parent=None)`, `LANGUAGES` dict.
- Produces: invalid `current` codes are clamped to `"english"` so `.selected()` always returns a valid code and exactly one button is highlighted. The `cssClass` property value is asserted loosely (`is not None`) in the new test because Task 9 renames it.

- [ ] **Step 1: Write the failing test**

```python
def test_language_dialog_clamps_invalid_current(qapp):
    dialog = LanguageDialog("french")
    assert dialog.selected() == "english"
    assert dialog._buttons["english"].property("cssClass") is not None
    assert all(
        dialog._buttons[c].property("cssClass") is None
        for c in ("kannada", "hindi")
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/tests/test_language_dialog.py::test_language_dialog_clamps_invalid_current -v`
Expected: FAIL — `selected()` returns `"french"` and no button has a `cssClass`.

- [ ] **Step 3: Implement the clamp**

In `app/ui/language_dialog.py`, change `__init__`:

```python
    def __init__(self, current: str = "english", parent=None):
        super().__init__(parent)
        self._selected = current if current in LANGUAGES.values() else "english"
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
            if code == self._selected:
                btn.setProperty("cssClass", "nav_btn_active")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, c=code: self._choose(c))
            self._buttons[code] = btn
            layout.addWidget(btn)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/tests/test_language_dialog.py -v`
Expected: 4 passed (3 existing + 1 new).

- [ ] **Step 5: Lint and commit**

```bash
ruff check app/ui/language_dialog.py app/tests/test_language_dialog.py
git add app/ui/language_dialog.py app/tests/test_language_dialog.py
git commit -m "fix: clamp invalid language dialog selection to english"
```

---

### Task 3: WaitDialog test hygiene + QProgressBar styling

**Files:**
- Modify: `app/tests/test_wait_dialog.py`
- Modify: `app/ui/styles.py:236-255` (insert a progress-bar rule near the table rules)

**Interfaces:**
- Consumes: `WaitDialog` (unchanged behavior).
- Produces: `QProgressBar` / `QProgressBar::chunk` QSS rules so the WaitDialog spinner renders in dark mode. Test no longer reaches into the private `_label`.

- [ ] **Step 1: Write the failing test + QSS rule**

Replace `app/tests/test_wait_dialog.py`:

```python
from PySide6.QtWidgets import QLabel

from app.ui.wait_dialog import WaitDialog


def test_wait_dialog_message_and_finish(qapp):
    dialog = WaitDialog()
    label = dialog.findChild(QLabel)
    assert label is not None
    assert "Generating transcript" in label.text()
    dialog.show()
    qapp.processEvents()
    assert dialog.isVisible()
    dialog.finish()
    assert not dialog.isVisible()
```

In `app/ui/styles.py`, add after the `QHeaderView::section` block (line 255):

```python
    /* === Progress Bar (Wait Dialog Spinner) === */
    QProgressBar {{
        background-color: {COLORS['surface_variant']};
        border: none;
        border-radius: 4px;
    }}
    QProgressBar::chunk {{
        background-color: {COLORS['primary']};
        border-radius: 4px;
    }}
```

- [ ] **Step 2: Run test to verify the new assertion passes**

Run: `pytest app/tests/test_wait_dialog.py -v`
Expected: PASS (the `findChild(QLabel)` assertion holds). This task is primarily hygiene + styling, so red-first is not required for the style rule.

- [ ] **Step 3: Lint and commit**

```bash
ruff check app/tests/test_wait_dialog.py app/ui/styles.py
git add app/tests/test_wait_dialog.py app/ui/styles.py
git commit -m "style: style wait dialog spinner and decouple test from internals"
```

---

### Task 4: Strengthen worker error test

**Files:**
- Modify: `app/tests/test_transcription_worker.py:25-30`

**Interfaces:**
- Consumes: `TranscriptionWorker` (unchanged) — emits `{"error": str(e)}` on exception.
- Produces: assertion that the error message value propagates, not just the key.

- [ ] **Step 1: Edit the test**

```python
def test_transcription_worker_emits_error(qapp):
    results = []
    worker = TranscriptionWorker(_RaisingTranscriber(), np.zeros(1600, dtype=np.float32))
    worker.finished.connect(results.append)
    worker.run()
    assert results[0]["error"] == "boom"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest app/tests/test_transcription_worker.py -v`
Expected: 2 passed.

- [ ] **Step 3: Lint and commit**

```bash
ruff check app/tests/test_transcription_worker.py
git add app/tests/test_transcription_worker.py
git commit -m "test: assert worker error message value"
```

---

### Task 5: `AudioTranscriber` cleanup (duration_sec, dead param, unused var)

**Files:**
- Modify: `app/core/transcription.py`
- Test: `app/tests/test_transcription.py`

**Interfaces:**
- Consumes: `AudioTranscriber.transcribe(audio, sample_rate=16000, localizations=None, passage_text=None, language="english")`.
- Produces: empty-audio early return now includes `"duration_sec": 0.0` (matching the documented `{text, words, duration_sec}` shape); `AudioTranscriber.__init__()` takes no `model_name` param; the localization loop no longer unpacks an unused `conf`. Signature changes do not affect any caller (all construct `AudioTranscriber()` with no args; verified via grep).

- [ ] **Step 1: Write the failing test**

Append to `app/tests/test_transcription.py`:

```python
def test_transcribe_empty_audio_includes_duration_sec(no_network):
    transcriber = AudioTranscriber()
    res = transcriber.transcribe(np.array([], dtype=np.float32))
    assert res["text"] == ""
    assert res["words"] == []
    assert res["duration_sec"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest app/tests/test_transcription.py::test_transcribe_empty_audio_includes_duration_sec -v`
Expected: FAIL — `KeyError: 'duration_sec'`.

- [ ] **Step 3: Implement the changes**

In `app/core/transcription.py`:

```python
        if audio is None or len(audio) == 0:
            return {"text": "", "words": [], "duration_sec": 0.0}
```

Remove the `__init__` method entirely (the class keeps its docstring; `AudioTranscriber()` falls back to the default constructor). Delete lines 55-56.

Change the localization loop (line 103):

```python
                for (st_start, st_end, _conf) in localizations:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/tests/test_transcription.py -v`
Expected: all tests pass (existing 8 + 1 new).

- [ ] **Step 5: Lint and commit**

```bash
ruff check app/core/transcription.py app/tests/test_transcription.py
git add app/core/transcription.py app/tests/test_transcription.py
git commit -m "fix: return duration_sec for empty audio and drop dead transcriber param"
```

---

### Task 6: Extract shared transcript table population helper

**Files:**
- Modify: `app/ui/table_utils.py` (add `populate_transcript_table` + `MAX_HEIGHT_UNCAP`)
- Modify: `app/ui/compact_transcript.py`
- Modify: `app/ui/transcription_panel.py`
- Test: `app/tests/test_transcription.py` (add one helper test; existing cap tests must stay green)

**Interfaces:**
- Consumes: `cap_table_height(table, max_rows)` (Task 1).
- Produces: `populate_transcript_table(table: QTableWidget, words: list[dict]) -> None` — fills a word-level table (columns Word/Start/End/Confidence/Status) with stutter styling, wrapped in `setUpdatesEnabled(False/True)`. `MAX_HEIGHT_UNCAP = 16777215` named constant. Both panels' `set_transcription` now delegate to the helper; their `clear()` methods use `MAX_HEIGHT_UNCAP`.

- [ ] **Step 1: Add helper to table_utils.py**

`app/ui/table_utils.py` becomes:

```python
"""Shared table sizing helpers."""

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from app.ui.theme import COLORS

MAX_HEIGHT_UNCAP = 16777215


def resize_table_to_contents(table: QTableWidget):
    """Set a table's minimum height so every row is visible without scrolling."""
    height = table.horizontalHeader().height()
    for row in range(table.rowCount()):
        height += table.rowHeight(row)
    table.setMinimumHeight(height + 4)


def cap_table_height(table: QTableWidget, max_rows: int):
    """Cap a table's maximum height so longer contents scroll inside the table."""
    header_height = table.horizontalHeader().height() or 30
    row_height = table.rowHeight(0) if table.rowCount() > 0 else 30
    table.setMaximumHeight(header_height + max(1, max_rows) * row_height + 4)


def populate_transcript_table(table: QTableWidget, words: list) -> None:
    """Populate a word-level transcript table with rows and stutter styling.

    words: list of dicts with keys word, start_sec, end_sec, confidence, stutter.
    """
    table.setUpdatesEnabled(False)
    try:
        table.setRowCount(len(words))
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

            table.setItem(row, 0, word_item)
            table.setItem(row, 1, start_item)
            table.setItem(row, 2, end_item)
            table.setItem(row, 3, conf_item)
            table.setItem(row, 4, status_item)
    finally:
        table.setUpdatesEnabled(True)
```

- [ ] **Step 2: Rewrite CompactTranscript**

`app/ui/compact_transcript.py`: remove the `QColor`, `QTableWidgetItem`, `COLORS` imports; import `cap_table_height, populate_transcript_table, MAX_HEIGHT_UNCAP` from `app.ui.table_utils`. Replace `set_transcription` and `clear`:

```python
    def set_transcription(self, data: dict):
        """Display transcription dictionary."""
        text = data.get("text", "")
        words = data.get("words", [])
        self._text_edit.setPlainText(text)
        populate_transcript_table(self._table, words)
        cap_table_height(self._table, MAX_ROWS)

    def clear(self):
        """Clear transcription display."""
        self._audio = None
        self._text_edit.clear()
        self._table.setRowCount(0)
        self._table.setMinimumHeight(0)
        self._table.setMaximumHeight(MAX_HEIGHT_UNCAP)
```

- [ ] **Step 3: Rewrite TranscriptionPanel**

`app/ui/transcription_panel.py`: remove `QColor` and `QTableWidgetItem` from imports (keep `QTableWidget`, `QTableWidgetItem`? — remove the `QTableWidgetItem` import since the helper owns it now; keep `QTableWidget`); import `cap_table_height, populate_transcript_table, MAX_HEIGHT_UNCAP` from `app.ui.table_utils`. Replace `set_transcription` and `clear`:

```python
    def set_transcription(self, data: dict):
        """Display transcription dictionary."""
        self._transcription_data = data
        text = data.get("text", "")
        words = data.get("words", [])
        self._text_edit.setPlainText(text)
        populate_transcript_table(self._table, words)
        cap_table_height(self._table, MAX_ROWS)

    def clear(self):
        """Clear transcription display."""
        self._audio = None
        self._transcription_data = {"text": "", "words": []}
        self._text_edit.clear()
        self._table.setRowCount(0)
        self._table.setMinimumHeight(0)
        self._table.setMaximumHeight(MAX_HEIGHT_UNCAP)
        self._status_label.setText("No Audio Loaded")
```

- [ ] **Step 4: Add a helper test + remove redundant fixture**

Append to `app/tests/test_transcription.py`:

```python
def test_populate_transcript_table_styles_stutter_rows(qapp):
    from PySide6.QtWidgets import QTableWidget

    from app.ui.table_utils import populate_transcript_table

    table = QTableWidget(0, 5)
    table.setHorizontalHeaderLabels(["Word", "Start (s)", "End (s)", "Confidence", "Status"])
    words = [
        {"word": "ok", "start_sec": 0.0, "end_sec": 0.3, "confidence": 0.9,
         "stutter": False, "stutter_type": None},
        {"word": "stut", "start_sec": 0.3, "end_sec": 0.9, "confidence": 0.8,
         "stutter": True, "stutter_type": "dysfluency"},
    ]
    populate_transcript_table(table, words)
    assert table.rowCount() == 2
    assert table.item(0, 0).text() == "ok"
    assert table.item(1, 4).text() == "Stutter Detected"
```

Also change `test_transcription_panel_table_capped(qapp, no_network)` to `test_transcription_panel_table_capped(qapp)` (remove the redundant `no_network` fixture — the test only calls `set_transcription`, never the pipeline).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest app/tests/test_transcription.py -v`
Expected: all tests pass.

- [ ] **Step 6: Lint and commit**

```bash
ruff check app/ui/table_utils.py app/ui/compact_transcript.py app/ui/transcription_panel.py app/tests/test_transcription.py
git add app/ui/table_utils.py app/ui/compact_transcript.py app/ui/transcription_panel.py app/tests/test_transcription.py
git commit -m "refactor: extract shared transcript table population helper"
```

---

### Task 7: Initialize `_resize_timer` in WaveformView `__init__`

**Files:**
- Modify: `app/ui/waveform_view.py`
- Test: `app/tests/test_waveform_view.py`

**Interfaces:**
- Consumes: existing debounced `resizeEvent`.
- Produces: `self._resize_timer = None` initialized in `__init__`; `resizeEvent` checks `is None` instead of `hasattr`. Tests unchanged.

- [ ] **Step 1: Edit the implementation**

In `app/ui/waveform_view.py`, add to `__init__` (near `self._overlays = []`):

```python
        self._resize_timer = None
```

Replace `resizeEvent`:

```python
    def resizeEvent(self, event):
        """Debounce redraws during resize drags."""
        super().resizeEvent(event)
        if self._resize_timer is None:
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._draw)
        self._resize_timer.start(50)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest app/tests/test_waveform_view.py -v`
Expected: 2 passed.

- [ ] **Step 3: Lint and commit**

```bash
ruff check app/ui/waveform_view.py app/tests/test_waveform_view.py
git add app/ui/waveform_view.py app/tests/test_waveform_view.py
git commit -m "refactor: initialize waveform resize timer in constructor"
```

---

### Task 8: Remove dead code (CompactTranscript.set_audio/run_transcription, ResultsPanel elif branch)

**Files:**
- Modify: `app/ui/compact_transcript.py`
- Modify: `app/ui/results_panel.py`
- Test: none required (covered by existing suite; verified dead via grep — no external callers of `CompactTranscript.set_audio`/`run_transcription`, and `ModelRunner.analyze` always returns a truthy transcription dict so the `elif audio is not None` branch is unreachable).

**Interfaces:**
- Consumes: `CompactTranscript.clear()` and `set_transcription(data)` (kept); `TranscriptionPanel.set_audio` (kept — still called from `results_panel.py`).
- Produces: `CompactTranscript` no longer exposes `set_audio`/`run_transcription`. `ResultsPanel.set_results` drops the `elif audio is not None` fallback branch.

- [ ] **Step 1: Edit CompactTranscript**

In `app/ui/compact_transcript.py`, delete the `set_audio` and `run_transcription` methods (and the now-unused `numpy` import if nothing else uses it — `_audio` is only compared with `is not None`/`len` and `np.ndarray` is not referenced as a type after removal, so drop the `import numpy as np` and the `QTableWidgetItem`/`QColor`/`COLORS` imports already removed in Task 6).

- [ ] **Step 2: Edit ResultsPanel**

In `app/ui/results_panel.py`, replace:

```python
        if transcription:
            self._transcription_panel.set_transcription(transcription)
        elif audio is not None:
            self._transcription_panel.set_audio(audio, sample_rate, localizations=localizations, language=language)
```

with:

```python
        if transcription:
            self._transcription_panel.set_transcription(transcription)
```

(Note: `TranscriptionPanel.set_audio` remains used elsewhere — results_panel.py is no longer one of its callers, but `analysis_page.py`/tests still exercise it; do NOT remove `TranscriptionPanel.set_audio`.)

- [ ] **Step 3: Run full suite to verify nothing breaks**

Run: `pytest app/tests -v`
Expected: all tests pass.

- [ ] **Step 4: Lint and commit**

```bash
ruff check app/ui/compact_transcript.py app/ui/results_panel.py
git add app/ui/compact_transcript.py app/ui/results_panel.py
git commit -m "refactor: remove dead transcription methods and unreachable fallback"
```

---

### Task 9: Color-only `lang_btn_active` QSS class

**Files:**
- Modify: `app/ui/styles.py`
- Modify: `app/ui/language_dialog.py`
- Test: `app/tests/test_language_dialog.py`

**Interfaces:**
- Consumes: `LanguageDialog` `_buttons` with `cssClass` property (set in `__init__`).
- Produces: `QPushButton[cssClass="lang_btn_active"]` color-only QSS rule (surface background, primary text, primary border, radius inherited from the default `QPushButton` rule) so the highlighted button keeps default 48px height — fixing the nav metrics (`min-height: 36px`) that made the highlighted button smaller than siblings. `LanguageDialog` sets `cssClass="lang_btn_active"`; the pre-highlight test asserts the exact class name.

- [ ] **Step 1: Add the QSS rule + update dialog + update test**

In `app/ui/styles.py`, after the `nav_btn_active` block (line 65), add:

```python
    QPushButton[cssClass="lang_btn_active"] {{
        background-color: {COLORS['surface_variant']};
        color: {COLORS['primary']};
        border: 2px solid {COLORS['primary']}66;
    }}
    QPushButton[cssClass="lang_btn_active"]:hover {{
        background-color: {COLORS['primary_container']};
    }}
```

In `app/ui/language_dialog.py`, change the highlight line:

```python
                btn.setProperty("cssClass", "lang_btn_active")
```

In `app/tests/test_language_dialog.py`, update the pre-highlight test:

```python
def test_language_dialog_prehighlights_current(qapp):
    dialog = LanguageDialog("kannada")
    assert dialog._buttons["kannada"].property("cssClass") == "lang_btn_active"
    assert dialog._buttons["english"].property("cssClass") is None
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest app/tests/test_language_dialog.py -v`
Expected: 4 passed.

- [ ] **Step 3: Lint and commit**

```bash
ruff check app/ui/styles.py app/ui/language_dialog.py app/tests/test_language_dialog.py
git add app/ui/styles.py app/ui/language_dialog.py app/tests/test_language_dialog.py
git commit -m "style: color-only highlight for language dialog keeps button height"
```

---

### Task 10: Positively assert stale-analysis-signal delivery

**Files:**
- Modify: `app/tests/test_main_window.py:102-124`

**Interfaces:**
- Consumes: `MainWindow._on_analyze`, `AnalysisWorker`, `_on_analysis_done(results, worker)` (identity-guarded).
- Produces: `test_stale_analysis_signal_does_not_clear_current_worker` now records every `_on_analysis_done` invocation and asserts the stale worker's signal was actually delivered (`stale in seen`) — so a future refactor that drops the `finished` connection would fail this test instead of passing silently.

- [ ] **Step 1: Rewrite the test**

Replace `test_stale_analysis_signal_does_not_clear_current_worker`:

```python
def test_stale_analysis_signal_does_not_clear_current_worker(qapp, app_name):
    from app.ui.main_window import AnalysisWorker

    class FakeRunner:
        def analyze(self, audio, language="english"):
            return {"ok": True}

    win = MainWindow()
    win._model_runner = FakeRunner()
    win._current_audio = np.zeros(1600, dtype=np.float32)
    win._current_language = "english"

    win._on_analyze()
    stale = win._worker
    stale.wait(5000)

    seen = []
    original = win._on_analysis_done

    def spy(results, worker):
        seen.append(worker)
        return original(results, worker)

    win._on_analysis_done = spy

    fresh = AnalysisWorker(FakeRunner(), np.zeros(1600, dtype=np.float32), "english")
    win._worker = fresh

    qapp.processEvents()

    assert stale in seen
    assert win._worker is fresh
    assert win._stack.currentIndex() == 0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest app/tests/test_main_window.py::test_stale_analysis_signal_does_not_clear_current_worker -v`
Expected: PASS.

- [ ] **Step 3: Lint and commit**

```bash
ruff check app/tests/test_main_window.py
git add app/tests/test_main_window.py
git commit -m "test: assert stale analysis signal is actually delivered"
```

---

### Task 11: Final verification

**Files:** none.

- [ ] **Step 1: Full suite**

Run: `pytest app/tests -v`
Expected: all tests pass (44 prior + 2 new table-utils + 1 new language-dialog + 1 new transcription duration = 48).

- [ ] **Step 2: Ruff**

Run: `ruff check app/`
Expected: only the 8 pre-existing errors in untouched files remain; every file touched by this branch is clean.

- [ ] **Step 3: Git state**

Run: `git status`
Expected: clean working tree on `desktop-transcription-cleanup`; `git log --oneline` shows the 10 cleanup commits stacked on `c04fd9a`.

- [ ] **Step 4: Update the SDD ledger**

Append the 10 task completions (commits) to `.superpowers/sdd/progress.md` under a new `## Cleanup plan (desktop-transcription-cleanup)` section.
