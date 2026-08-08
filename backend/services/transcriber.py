"""Service layer for speech transcription and language detection using Whisper."""

import io

import numpy as np
import soundfile as sf

from backend.services.audio_utils import convert_to_wav

_pipelines = {}


def get_pipeline(language: str = "english"):
    """Lazily initialize the Whisper ASR pipeline for the chosen language."""
    global _pipelines
    lang_lower = language.lower()
    if lang_lower not in _pipelines:
        from transformers import pipeline
        
        # Decide the model ID based on selected language
        if lang_lower == "kannada":
            model_id = "vasista22/whisper-kannada-tiny"
        elif lang_lower == "hindi":
            model_id = "collabora/whisper-tiny-hindi"
        else:
            model_id = "openai/whisper-tiny"
            
        print(f"Loading Whisper model '{model_id}' for language '{language}'...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device="cpu"
        )
        
        # Setup the token ids correctly to enable native script transcription
        lang_code = "kn" if lang_lower == "kannada" else ("hi" if lang_lower == "hindi" else "en")
        pipe.model.generation_config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
            language=lang_code, task="transcribe"
        )
        
        # Dynamically find notimestamps token ID and set it in generation_config
        try:
            no_timestamps_token_id = pipe.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            pipe.model.generation_config.no_timestamps_token_id = no_timestamps_token_id
        except Exception as e:
            print(f"Failed to set no_timestamps_token_id: {e}")
            
        _pipelines[lang_lower] = pipe
        
    return _pipelines[lang_lower]


def _collapse_runs(items, clean, max_run=3):
    """Drop items that repeat consecutively more than max_run times.

    Real stutters repeat a word or syllable 2-3 times; longer identical runs
    are almost always Whisper hallucination loops. Keeps the first `max_run`
    occurrences of a run, drops the rest.
    """
    result = []
    prev_clean = None
    run = 0
    for item in items:
        c = clean(item)
        if c != "" and c == prev_clean:
            run += 1
            if run > max_run:
                continue
        else:
            prev_clean = c
            run = 1
        result.append(item)
    return result


def _clean(token: str) -> str:
    return token.lower().strip(".,?!;:-_\"'()[]{} ")


def transcribe_audio_bytes(audio_bytes: bytes, language: str = "english") -> dict:
    """
    Transcribe audio bytes using a user-selected language.

    Args:
        audio_bytes: Raw audio file bytes.
        language: User-selected language ("english", "kannada", "hindi").

    Returns:
        A dict containing:
            - "text": The full transcribed text.
            - "language": The selected language.
            - "chunks": A list of dicts with:
                - "text": Chunk text.
                - "start": Start timestamp (seconds).
                - "end": End timestamp (seconds).
                - "language": Selected language.
    """
    print(f"DEBUG: transcriber received language: '{language}'")
    try:
        # Convert audio bytes to standard 16kHz mono WAV via ffmpeg
        wav_bytes = convert_to_wav(audio_bytes)
        
        # Read the WAV audio data
        audio_data, sr = sf.read(io.BytesIO(wav_bytes))
        
        # Resample to 16kHz if needed (should already be 16kHz via convert_to_wav)
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Normalize to float32 mono array in [-1.0, 1.0]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / (np.abs(audio_data).max() + 1e-8)

        # Get language-specific pipeline
        pipe = get_pipeline(language)

        # Call the pipeline with the required settings
        result = pipe(
            audio_data,
            return_timestamps=True
        )

        # Extract transcription and chunks
        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])

        # Standardize language string
        selected_lang_str = str(language).capitalize()
        ALLOWED_LANGUAGES = {"English", "Hindi", "Kannada"}
        if selected_lang_str not in ALLOWED_LANGUAGES:
            selected_lang_str = "English"

        # Collapse consecutive repeated chunks, keeping genuine repetitions
        # (up to 3) while dropping Whisper hallucination loops.
        chunks = _collapse_runs(chunks, lambda c: _clean(c.get("text", "")), max_run=3)
        formatted_chunks = []

        for chunk in chunks:
            timestamp = chunk.get("timestamp")
            start = timestamp[0] if timestamp else 0.0
            end = timestamp[1] if timestamp else 0.0

            if start is None:
                start = 0.0
            if end is None:
                end = 0.0

            formatted_chunks.append({
                "text": chunk.get("text", "").strip(),
                "start": round(start, 2),
                "end": round(end, 2),
                "language": selected_lang_str
            })

        # Deduplicate individual consecutive words in the final text string,
        # keeping genuine stuttered repetitions (up to 3).
        words = _collapse_runs(text.split(), _clean, max_run=3)
        deduped_text = " ".join(words)

        return {
            "text": deduped_text,
            "language": selected_lang_str,
            "chunks": formatted_chunks
        }

    except Exception as e:
        print(f"Transcription service error: {e}")
        return {
            "text": "",
            "language": "Unknown",
            "chunks": []
        }
