"""Tests for Whisper hallucination-loop collapsing in the transcriber service."""

import pytest

from backend.services.transcriber import _clean, _collapse_phrase_runs


def test_single_word_run_collapses_to_max_run():
    tokens = ["nervous"] * 10
    result = _collapse_phrase_runs(tokens, _clean, max_run=3)
    assert result == ["nervous"] * 3


def test_phrase_loop_collapses_to_max_run():
    phrase = ["I", "was", "nervous"]
    tokens = phrase * 10
    result = _collapse_phrase_runs(tokens, _clean, max_run=3)
    assert result == phrase * 3


def test_short_stutter_is_preserved():
    tokens = ["I", "was", "nervous"] * 2
    result = _collapse_phrase_runs(tokens, _clean, max_run=3)
    assert result == ["I", "was", "nervous"] * 2


def test_no_repetition_unchanged():
    tokens = ["I", "like", "tea"]
    result = _collapse_phrase_runs(tokens, _clean, max_run=3)
    assert result == tokens


def test_phrase_loop_with_punctuation_collapses():
    phrase = ["I", "was", "nervous,"]
    tokens = phrase * 8
    result = _collapse_phrase_runs(tokens, _clean, max_run=3)
    assert result == phrase * 3


def test_words_after_loop_are_kept():
    tokens = ["I", "was", "nervous"] * 6 + ["then", "I", "calmed"]
    result = _collapse_phrase_runs(tokens, _clean, max_run=3)
    assert result == ["I", "was", "nervous"] * 3 + ["then", "I", "calmed"]


def test_repeating_word_inside_phrase_is_not_overcollapsed():
    tokens = ["I", "I", "I", "like", "I", "I", "like"]
    result = _collapse_phrase_runs(tokens, _clean, max_run=3)
    assert result == ["I", "I", "I", "like", "I", "I", "like"]
