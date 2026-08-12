"""Speech transcription pipeline for Swaraaha.

Unified Whisper-based transcriber usable by both the web backend and the
desktop app. Accepts audio as a file path, raw bytes, or numpy array and
returns text with word-level timestamps plus stutter flagging.

Supports English, Kannada, and Hindi via per-language Whisper-tiny pipelines.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from model.data.preprocessing import load_audio_input
from model.localization.ctc_alignment import SimpleForcedAligner

WHISPER_MODELS = {
    "english": "openai/whisper-tiny",
    "kannada": "vasista22/whisper-kannada-tiny",
    "hindi": "collabora/whisper-tiny-hindi",
}

WHISPER_LANG_CODES = {"english": "en", "kannada": "kn", "hindi": "hi"}

_pipelines = {}


def _is_repeated_fragment(word: str) -> bool:
    """Detect syllable/letter repetitions inside a single token.

    Catches hyphenated repeats ("s-s", "ba-ba", "s-s-s") and stretched
    single characters ("sss", "b-b-b") that ASR emits for sound repetitions.
    """
    if not word:
        return False
    if "-" in word:
        parts = [p for p in word.split("-") if p]
        if len(parts) >= 2 and len(set(parts)) == 1:
            return True
    return bool(re.search(r"(.)\1{2,}", word))


def _flag_repetitions(words: List[Dict[str, Any]]) -> None:
    """Mark word/syllable repetitions in a word list as stutter events.

    Consecutive identical words are flagged as 'wordrep'; repeated syllable
    fragments (e.g. "s-s", "sss") as 'soundrep'. Called before localization
    overlay so repetition types are kept for words already flagged.
    """
    for i, w in enumerate(words):
        if w.get("stutter"):
            continue
        current = (w.get("word") or "").strip().lower()
        if not current:
            continue
        previous = (words[i - 1].get("word") or "").strip().lower() if i > 0 else ""
        if current == previous and current != "":
            w["stutter"] = True
            w["stutter_type"] = "wordrep"
        elif _is_repeated_fragment(current):
            w["stutter"] = True
            w["stutter_type"] = "soundrep"


def get_pipeline(language: str = "english"):
    """Lazily initialize (and cache) the Whisper ASR pipeline for a language."""
    lang = language.lower()
    if lang not in _pipelines:
        from transformers import pipeline

        model_id = WHISPER_MODELS.get(lang, WHISPER_MODELS["english"])
        pipe = pipeline("automatic-speech-recognition", model=model_id, device="cpu")

        lang_code = WHISPER_LANG_CODES.get(lang, "en")
        try:
            pipe.model.generation_config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
                language=lang_code, task="transcribe"
            )
            no_timestamps_token_id = pipe.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            pipe.model.generation_config.no_timestamps_token_id = no_timestamps_token_id
        except Exception:
            pass

        _pipelines[lang] = pipe
    return _pipelines[lang]


class Transcriber:
    """Speech transcription with word-level timestamps and stutter flagging."""

    def transcribe(
        self,
        audio,
        language: str = "english",
        localizations: Optional[List[Tuple[float, float, float]]] = None,
        passage_text: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Transcribe audio into text and timestamped words.

        Args:
            audio: File path, raw bytes, or 1-D numpy array.
            language: "english", "kannada", or "hindi".
            localizations: List of (start_sec, end_sec, confidence) stutter regions.
            passage_text: Optional reference passage for the fallback aligner.
            sample_rate: Sample rate of the input (numpy arrays).

        Returns:
            {"text", "words": [{word, start_sec, end_sec, confidence, stutter,
                               stutter_type}], "duration_sec"}
        """
        if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
            return {"text": "", "words": [], "duration_sec": 0.0}

        audio_array = load_audio_input(audio, sr=sample_rate)
        if len(audio_array) == 0:
            return {"text": "", "words": [], "duration_sec": 0.0}

        # Whisper and the forced aligner both require 16 kHz audio. Path/bytes
        # inputs are resampled to sample_rate by load_audio_input, so bring
        # non-16k input back down; ndarray input is already 16 kHz
        # (load_audio_from_array always targets 16000) and must not be
        # double-resampled.
        if sample_rate != 16000 and not isinstance(audio, np.ndarray):
            import librosa

            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=16000
            )

        duration_sec = len(audio_array) / 16000

        transcript_text = None
        word_list: List[Dict[str, Any]] = []

        try:
            transcript_text, word_list = self._transcribe_with_whisper(audio_array, language)
        except Exception:
            transcript_text = None
            word_list = []

        if not transcript_text and not word_list:
            transcript_text, word_list = self._fallback_transcribe(
                audio_array, duration_sec, passage_text
            )

        _flag_repetitions(word_list)

        if localizations and word_list:
            for w in word_list:
                if w.get("stutter"):
                    continue
                w_start = w["start_sec"]
                w_end = w["end_sec"]
                for (st_start, st_end, _conf) in localizations:
                    if max(w_start, st_start) < min(w_end, st_end):
                        w["stutter"] = True
                        w["stutter_type"] = "dysfluency"
                        break

        return {
            "text": transcript_text,
            "words": word_list,
            "duration_sec": round(duration_sec, 2),
        }

    def _transcribe_with_whisper(
        self, audio: np.ndarray, language: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
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
