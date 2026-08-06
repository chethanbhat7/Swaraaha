# Design Spec: Desktop Transcription UI Changes

> **Task:** Transcription UX improvements for the desktop app
> **Assignee:** Srinivas
> **Date:** 2026-08-06
> **Status:** Approved

---

## 1. Overview

Five UX changes to the PySide6 desktop app (`app/`):

1. **Wait screen** — transcription runs in the background; a modal *"Generating transcript, please wait"* dialog with a spinner is shown from Stop and Load until the transcript arrives.
2. **Scrollable home transcript** — the compact transcript table on the home page scrolls internally instead of growing past the visible area.
3. **Capped results table** — the word-level table on the Analysis Results page shows only a few rows at a time; the user scrolls inside the table for the rest (same behavior as the home page). This also fixes the sluggish, jittery results-page scrolling.
4. **Language selection** — a *"Choose the language"* modal with **English / ಕನ್ನಡ (Kannada) / हिंदी (Hindi)** buttons appears whenever the user clicks **Record Audio** or **Load Audio**. The transcript can now be Kannada or Hindi, not just English.
5. **Performance** — the results page is jittery to scroll; the root cause is the full-height word table (`resize_table_to_contents`), fixed by #3, plus targeted redraw debouncing.

## 2. Problem Statement

- When the user stops recording or loads an audio file, the UI blocks while `AudioTranscriber.transcribe` runs synchronously on the main thread. There is no feedback that transcription is in progress, and long files freeze the UI.
- `resize_table_to_contents` (in `app/ui/table_utils.py`) sets each table's **minimum height** to header + sum of *all* row heights, which (a) makes the home transcript non-scrollable and (b) makes the results page extremely long and jittery to scroll for long audio.
- Transcription is English-only. The user wants Kannada and Hindi support via a language prompt on Record/Load.
- `WaveformView.resizeEvent` redraws the full waveform on every resize event (a known jitter source).

## 3. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| ASR engine | Per-language Whisper-tiny pipelines, lazily cached, mirroring `backend/services/transcriber.py`: `english → openai/whisper-tiny`, `kannada → vasista22/whisper-kannada-tiny`, `hindi → collabora/whisper-tiny-hindi`; `forced_decoder_ids` from `tokenizer.get_decoder_prompt_ids(language, task="transcribe")`; `return_timestamps="word"`; backend's consecutive-duplicate dedup | User: "there's already an existing multi-lingual pipeline which the web app uses for the transcriptions." Reuse it instead of inventing a new one |
| Language dialog | Modal `QDialog`, heading "Choose the language", three buttons (**English**, **ಕನ್ನಡ (Kannada)**, **हिंदी (Hindi)**); last selection pre-highlighted for one-click confirm | User picked modal over segmented/inline; pre-highlight keeps repeated recordings fast |
| When dialog shows | Every **Record Audio** and **Load Audio** click; last selection pre-selected. Recent-file click and drag-drop reuse the last stored language without prompting | User: "whenever the user clicks on 'Record Audio' or 'Load Audio'"; avoids interrupting passive flows |
| Wait screen | Modal dialog with spinner animation + "Generating transcript, please wait"; auto-dismisses when transcription completes | User asked for exactly this; shown on Stop and Load flows |
| Transcription threading | New `TranscriptionWorker(QThread)` mirroring the existing `AnalysisWorker` pattern; owned by `MainWindow`; panels keep synchronous `set_transcription(data)` so existing panel tests remain valid | Async layer lives only at the `MainWindow` level |
| Home transcript scroll | New `cap_table_height(table, max_rows)` helper; `CompactTranscript` capped at ~8 rows | QTableWidget's built-in scrollbar handles overflow once max height is set |
| Results table scroll | Same helper; `TranscriptionPanel` capped at ~10 rows | Same behavior as home page, per user request |
| Classification table | Keeps full-height `resize_table_to_contents` (always 5 rows) | Constant small size; no scroll needed |
| Perf fix | `setUpdatesEnabled(False/True)` around table population; debounce `WaveformView.resizeEvent` redraw with a single-shot `QTimer`; rebuild the waveform path only on `set_audio`/`set_overlays` | Fixes the jittery results-page scroll |
| Model downloads | First use downloads the Whisper model for the chosen language (lazy); not cached in tests | Avoids bundling models |

## 4. Architecture & Components

### 4a. `app/ui/transcription_worker.py` (new)

```
TranscriptionWorker(QThread)   # mirrors AnalysisWorker in main_window.py
  __init__(transcriber: AudioTranscriber, audio, language: str)
  run(): data = transcriber.transcribe(audio, language=language)
         finished.emit(data)  # or {"error": str} on exception
```

- One-shot worker per transcription; `MainWindow` discards results from a stale worker if a newer transcription started.

### 4b. `app/ui/language_dialog.py` (new)

```
LanguageDialog(QDialog)
  __init__(current: str = "english", parent=None)
  selected() -> str   # "english" | "kannada" | "hindi"
```

- Modal dialog; heading "Choose the language"; three buttons, the current one pre-highlighted (active style). Clicking a button accepts and returns its code.

### 4c. `app/ui/wait_dialog.py` (new)

```
WaitDialog(QDialog)
  __init__(message: str = "Generating transcript, please wait", parent=None)
  finish()   # closes/accepts the dialog
```

- Frameless or window-modal dialog with an animated spinner (`QProgressBar` in indeterminate mode or a rotating indicator) and the message label. Auto-dismissed from `MainWindow` when the worker emits `finished`.

### 4d. `app/core/transcription.py` (modified)

- `AudioTranscriber` gains a `language` parameter to `transcribe(audio, sample_rate=16000, localizations=None, passage_text=None, language="english")`.
- Lazy per-language pipeline cache (module-level dict, mirroring the backend): Whisper pipeline + `forced_decoder_ids` for `en`/`kn`/`hi`, `return_timestamps="word"`.
- Dedup consecutive repeated words/chunks (backend's Whisper hallucination guard).
- Fallback (`SimpleForcedAligner`, English rainbow passage) remains best-effort for any language when ASR fails.
- Stutter overlay logic unchanged.

### 4e. `app/core/model_runner.py` (modified)

- `analyze(audio, language="english")` and `transcribe(audio, localizations=None, language="english")` thread the language through to `AudioTranscriber`.
- `AnalysisWorker` gains a `language` argument.

### 4e2. `app/ui/results_panel.py` & `app/ui/transcription_panel.py` (modified)

- `ResultsPanel.set_results(results, audio, sample_rate, language="english")` and `AnalysisPage.set_results(results, audio, sample_rate, language="english")` gain a `language` param; `MainWindow._on_analysis_done` threads `_current_language`.
- `ResultsPanel` fallback re-transcription passes `language` into `transcription_panel.set_audio(audio, sample_rate, localizations=localizations, language=language)`.
- `TranscriptionPanel.set_audio(..., language="english")` and `CompactTranscript.set_audio(..., language="english")` accept the language so any remaining synchronous call site matches the chosen language.

### 4f. `app/ui/main_window.py` (modified)

- New state: `_current_language` (default `"english"`), `_transcription_worker`, `_wait_dialog`.
- `_on_record`: prompt language first, then start recording.
- `_on_stop`: stop recording; if audio produced, start background transcription via `TranscriptionWorker` + show `WaitDialog`; on `finished` populate `CompactTranscript.set_transcription(data)` and dismiss dialog.
- `_on_load`: prompt language first (but NOT for recent-file click / drag-drop — reuse `_current_language`), then `_load_path` which now starts background transcription instead of synchronous `set_audio`.
- `_on_file_selected` (recent-file) and drop path: reuse `_current_language`, no prompt.
- `_prompt_language()` seam method wraps `LanguageDialog.exec()` so tests can monkeypatch it.
- `_on_analyze`: thread `_current_language` into `AnalysisWorker`.

### 4g. Table height helpers (`app/ui/table_utils.py`, modified)

- Keep `resize_table_to_contents(table)`.
- Add `cap_table_height(table, max_rows)` — sets `table.setMaximumHeight(header_height + max_rows * row_height + 4)` so the table's internal scrollbar handles overflow.
- `CompactTranscript` (home, ~8 rows) and `TranscriptionPanel` (results, ~10 rows) call `cap_table_height` after population; classification table keeps full height.

### 4h. `app/ui/waveform_view.py` (modified)

- `resizeEvent` schedules a single-shot `QTimer(0)` redraw instead of calling `_draw()` directly; `set_audio`/`set_overlays` still rebuild immediately.

## 5. Data Flow

```
Record/Load click
  → _prompt_language()  (modal; last choice pre-highlighted)
  → audio obtained (recording stopped / file loaded)
  → _current_language set
  → TranscriptionWorker(transcriber, audio, language).start()
  → WaitDialog shown (modal, spinner)
  → worker.finished(data)  →  CompactTranscript.set_transcription(data)
  → WaitDialog.finish()
```

Analysis flow threads `_current_language` so the results-page transcript matches what the user chose.

## 6. Error Handling

- `TranscriptionWorker` catches exceptions and emits `{"error": str}`; `MainWindow` shows the error (status/`QMessageBox`) and dismisses the wait dialog.
- Stale worker results (a newer transcription started) are discarded.
- ASR pipeline failure falls back to the English `SimpleForcedAligner` (best-effort).

## 7. Testing

- Keep all 22 existing tests passing (`pytest` with `QT_QPA_PLATFORM=offscreen`; `ruff` on touched files).
- New tests:
  - `LanguageDialog`: clicking each button returns the matching code.
  - `TranscriptionWorker`: fake transcriber; emits `finished(dict)`; emits `{"error": ...}` on exception.
  - `AudioTranscriber` language mapping + dedup: monkeypatched fake Whisper pipeline — **no network/model download in tests**.
  - `cap_table_height`: caps `setMaximumHeight` correctly.
  - `MainWindow` record/load flow: `_prompt_language()` monkeypatched so modal dialogs don't block tests; asserts background transcription populates the compact transcript.
  - Existing tests that call `set_audio` synchronously (e.g. `test_transcription_panel_ui`) keep working since panels retain synchronous `set_transcription`.

## 8. Out of Scope

- React web frontend (`frontend/`).
- Classification/language adapters under `model/localization/`.
- Bundling or pre-caching Whisper models.
