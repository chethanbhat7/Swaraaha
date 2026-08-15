"""
Tests for CTCTimeAligner model selection (model/localization/ctc_alignment.py).

The CTC head must come from a checkpoint that was fine-tuned for ASR.
Loading a pre-training-only checkpoint (e.g. facebook/wav2vec2-base) via
Wav2Vec2ForCTC produces a randomly initialized CTC head whose alignment
timestamps are garbage.
"""

from model.localization.ctc_alignment import CTCTimeAligner, SimpleForcedAligner


def test_ctc_aligner_default_is_ctc_finetuned_model():
    aligner = CTCTimeAligner()
    assert aligner.model_name == "facebook/wav2vec2-base-960h"


def test_ctc_aligner_default_not_pretraining_only_checkpoint():
    aligner = CTCTimeAligner()
    assert aligner.model_name != "facebook/wav2vec2-base"


def test_ctc_aligner_accepts_explicit_model_override():
    aligner = CTCTimeAligner(model_name="custom/ctc-aligner")
    assert aligner.model_name == "custom/ctc-aligner"


def test_simple_forced_aligner_still_available():
    assert SimpleForcedAligner.align(
        __import__("numpy").zeros(16000, dtype="float32"), "hello world", sr=16000
    )
