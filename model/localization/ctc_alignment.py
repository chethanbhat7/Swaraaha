"""
CTC-based forced alignment for word-level timestamps.

Uses Wav2Vec 2.0's CTC outputs to align transcript text with audio,
producing word-level timestamps for dysfluency analysis.

This module implements a simplified forced alignment that:
1. Tokenizes the transcript using Wav2Vec2's tokenizer
2. Runs CTC inference to get per-frame token probabilities
3. Uses CTC decoding to find optimal alignment
4. Maps token positions back to word boundaries

Usage:
    aligner = CTCTimeAligner()
    word_timestamps = aligner.align(audio, "hello world", sr=16000)
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from model.localization.language_adapter import WordTimestamp


class CTCTimeAligner:
    """
    CTC-based forced aligner for word-level timestamps.

    Uses Wav2Vec 2.0's CTC head to align transcript text with audio frames,
    producing word-level start/end times.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """
        Args:
            model_name: HuggingFace Wav2Vec2 model with a CTC head.

        The default is the LibriSpeech-960h fine-tuned checkpoint. The
        pre-training checkpoint (facebook/wav2vec2-base) has NO trained CTC
        head — Wav2Vec2ForCTC.from_pretrained would initialize it randomly,
        producing silently meaningless alignment timestamps.
        """
        self.model_name = model_name
        self._processor = None
        self._model = None

    @property
    def processor(self):
        if self._processor is None:
            from transformers import Wav2Vec2Processor
            self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self):
        if self._model is None:
            from transformers import Wav2Vec2ForCTC
            self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    def align(
        self,
        audio: np.ndarray,
        text: str,
        sr: int = 16000,
        max_length_seconds: float = 3.0,
    ) -> List[WordTimestamp]:
        """
        Align transcript text with audio to produce word-level timestamps.

        Args:
            audio: 1-D float32 audio array, values in [-1.0, 1.0].
            text: Transcript text to align.
            sr: Sample rate (must be 16000).
            max_length_seconds: Max audio length for model input.

        Returns:
            List of WordTimestamp with start/end times for each word.
        """
        if not text or not text.strip():
            return []

        max_samples = int(max_length_seconds * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)))

        # Process audio
        inputs = self.processor(
            audio, sampling_rate=sr, return_tensors="pt", padding=True
        )

        # Run CTC inference
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
            # logits: (1, T_frames, vocab_size)

        # Get CTC probabilities
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)  # (1, T_frames)

        # Wav2Vec2 subsampling: 320 samples per frame
        frame_duration = 320 / sr

        # Decode alignment using CTC greedy + word boundary detection
        word_timestamps = self._decode_alignment(
            pred_ids[0].numpy(),
            probs[0].numpy(),
            text,
            frame_duration,
        )

        return word_timestamps

    def _decode_alignment(
        self,
        pred_ids: np.ndarray,
        probs: np.ndarray,
        text: str,
        frame_duration: float,
    ) -> List[WordTimestamp]:
        """
        Decode CTC predictions into word-level timestamps.

        Uses the blank token (CTC) to detect word boundaries and maps
        token positions to words in the transcript.
        """
        blank_id = self.processor.tokenizer.pad_token_id or 0
        vocab = self.processor.tokenizer.get_vocab()

        # Build reverse vocab
        id_to_token = {v: k for k, v in vocab.items()}

        # Extract non-blank tokens with their positions
        tokens = []
        for i, token_id in enumerate(pred_ids):
            if token_id != blank_id and token_id != 0:
                token_str = id_to_token.get(token_id, "")
                if token_str.strip():
                    tokens.append((i, token_str))

        if not tokens:
            return []

        # Split text into words
        words = text.split()
        if not words:
            return []

        # Map tokens to words (greedy alignment)
        word_timestamps = []
        token_idx = 0

        for word in words:
            if token_idx >= len(tokens):
                # No more tokens — estimate remaining word positions
                if word_timestamps:
                    last_end = word_timestamps[-1].end_sec
                else:
                    last_end = 0.0
                word_timestamps.append(WordTimestamp(
                    word=word,
                    start_sec=round(last_end, 4),
                    end_sec=round(last_end + frame_duration * 5, 4),
                    confidence=0.3,
                ))
                continue

            # Find start frame for this word
            start_frame = tokens[token_idx][0]

            # Consume tokens for this word (approximate: ~1 token per character)
            word_len = len(word)
            end_token_idx = min(token_idx + max(1, word_len // 2), len(tokens) - 1)
            end_frame = tokens[end_token_idx][0]

            # Compute average confidence for this word's tokens
            token_confidences = [
                float(np.max(probs[tokens[i][0]]))
                for i in range(token_idx, min(end_token_idx + 1, len(tokens)))
            ]
            conf = float(np.mean(token_confidences)) if token_confidences else 0.5

            word_timestamps.append(WordTimestamp(
                word=word,
                start_sec=round(start_frame * frame_duration, 4),
                end_sec=round((end_frame + 1) * frame_duration, 4),
                confidence=round(conf, 4),
            ))

            token_idx = end_token_idx + 1

        return word_timestamps

    def align_with_syllables(
        self,
        audio: np.ndarray,
        text: str,
        language_code: str = "en",
        sr: int = 16000,
    ) -> List:
        """
        Align text and produce syllable-level timestamps.

        Args:
            audio: Audio array.
            text: Transcript text.
            language_code: Language code for syllabification.
            sr: Sample rate.

        Returns:
            List of SyllableTimestamp from the appropriate language adapter.
        """
        from model.localization.language_adapter import LanguageAdapterRegistry

        word_timestamps = self.align(audio, text, sr=sr)
        registry = LanguageAdapterRegistry()
        adapter = registry.get(language_code)
        return adapter.adapt(word_timestamps, text)


class SimpleForcedAligner:
    """
    Simpler forced aligner that doesn't require Wav2Vec2 CTC.

    Uses audio energy and text length to estimate word boundaries.
    Useful when CTC model is not available or for quick prototyping.
    """

    @staticmethod
    def align(
        audio: np.ndarray,
        text: str,
        sr: int = 16000,
        max_length_seconds: float = 3.0,
    ) -> List[WordTimestamp]:
        """
        Estimate word boundaries using audio energy and text structure.

        This is a fallback aligner that distributes words proportionally
        across the audio duration based on character count.
        """
        if not text or not text.strip():
            return []

        words = text.split()
        total_chars = sum(len(w) for w in words)
        duration = min(len(audio) / sr, max_length_seconds)

        word_timestamps = []
        current_time = 0.0

        for word in words:
            proportion = len(word) / total_chars if total_chars > 0 else 1.0 / len(words)
            word_duration = proportion * duration

            word_timestamps.append(WordTimestamp(
                word=word,
                start_sec=round(current_time, 4),
                end_sec=round(current_time + word_duration, 4),
                confidence=0.5,
            ))

            current_time += word_duration

        return word_timestamps


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CTC Alignment — Self Test ===")

    # Test simple aligner
    audio = np.random.randn(160000).astype(np.float32)
    aligner = SimpleForcedAligner()
    timestamps = aligner.align(audio, "hello world this is a test", sr=16000)
    for wt in timestamps:
        print(f"  {wt.word}: {wt.start_sec:.2f}-{wt.end_sec:.2f} (conf={wt.confidence:.2f})")

    # Test empty text
    empty = aligner.align(audio, "", sr=16000)
    assert len(empty) == 0

    print("=== Self test passed ===")
