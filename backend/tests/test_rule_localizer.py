from backend.services.rule_localizer import regions_from_words


def test_no_regions_when_no_stutters():
    words = [
        {"text": "the", "start": 0.0, "end": 0.2},
        {"text": "cat", "start": 0.2, "end": 0.5},
    ]
    assert regions_from_words(words) == []


def test_word_repetition_single_merged_region():
    words = [
        {"text": "I", "start": 0.0, "end": 0.2},
        {"text": "I", "start": 0.2, "end": 0.4},
        {"text": "like", "start": 0.4, "end": 0.8},
    ]
    regions = regions_from_words(words)
    assert len(regions) == 1
    assert regions[0]["type"] == "wordrep"
    assert regions[0]["start"] == 0.0
    assert regions[0]["end"] == 0.4


def test_sound_repetition_fragment_flagged():
    words = [{"text": "s-s", "start": 1.0, "end": 1.5}]
    regions = regions_from_words(words)
    assert len(regions) == 1
    assert regions[0]["type"] == "soundrep"
    assert regions[0]["confidence"] == 1.0


def test_adjacent_regions_merge():
    words = [
        {"text": "b-b", "start": 0.0, "end": 0.3},
        {"text": "b-b", "start": 0.3, "end": 0.6},
    ]
    regions = regions_from_words(words)
    assert len(regions) == 1
    assert regions[0]["end"] == 0.6


def test_skips_words_without_timestamps():
    words = [
        {"text": "I", "start": 0.0, "end": 0.2},
        {"text": "I"},
        {"text": "I", "start": 0.5, "end": 0.7},
    ]
    regions = regions_from_words(words)
    assert len(regions) == 1
    assert regions[0]["start"] == 0.5


def test_word_repetition_inside_multitoken_chunk():
    chunks = [
        {"text": "I I like tea", "start": 0.0, "end": 2.0},
    ]
    regions = regions_from_words(chunks)
    assert len(regions) == 1
    assert regions[0]["type"] == "wordrep"
    assert regions[0]["start"] == 0.0
    assert regions[0]["end"] == 1.0


def test_word_repetition_spans_chunks():
    chunks = [
        {"text": "I I", "start": 0.0, "end": 1.0},
        {"text": "like tea", "start": 1.0, "end": 2.0},
    ]
    regions = regions_from_words(chunks)
    assert len(regions) == 1
    assert regions[0]["start"] == 0.0
    assert regions[0]["end"] == 1.0


def test_repetition_in_middle_of_chunk():
    chunks = [
        {"text": "the the the boy", "start": 0.0, "end": 2.0},
    ]
    regions = regions_from_words(chunks)
    assert len(regions) == 1
    assert regions[0]["type"] == "wordrep"
    assert regions[0]["start"] == 0.0
    assert regions[0]["end"] == 1.5
