"""
Tests for dataset normalization and merging (model/data/merge.py).

Uses a fake RawData tree on a tmp_path to exercise the SEP-28K, UCLASS and
Project Boli normalizers plus merge_datasets output generation.
"""

import os

import pytest

from model.data import merge as merge_mod

SEP28K_CLIP = "SEP-28K Dataset/clips/stuttering-clips/clips"


@pytest.fixture
def fake_raw(tmp_path, monkeypatch):
    """Build a minimal fake RawData tree and point merge.RAW_DATA_DIR at it."""
    raw = tmp_path / "RawData"
    raw.mkdir()

    sep_clips = raw / SEP28K_CLIP
    sep_clips.mkdir(parents=True)
    (sep_clips / "FluencyBank_010_0.wav").write_bytes(b"\x00" * 100)

    labels_csv = raw / "SEP-28K Dataset" / "fluencybank_labels.csv"
    labels_csv.write_text(
        "Show,EpId,ClipId,Start,Stop,Prolongation,Block,SoundRep,WordRep,"
        "Interjection,NoStutteredWords\n"
        "FluencyBank,010,0,88960,136960,0,2,0,0,0,2\n"
    )

    uclass_clips = raw / "UCLASS SEP-28K Format" / "clips" / "clips"
    uclass_clips.mkdir(parents=True)
    (uclass_clips / "M_0030_16y4m_1_dysfluent_000.wav").write_bytes(b"\x00" * 100)
    (uclass_clips / "M_0031_16y4m_1_fluent_000.wav").write_bytes(b"\x00" * 100)

    meta = raw / "UCLASS SEP-28K Format" / "clips" / "metadata.json"
    meta.write_text(
        "["
        '{"clip_id": "M_0030_16y4m_1_dysfluent_000", "duration": 3.0, '
        '"end_time": 3.0, "file_path": "clips/clips/M_0030_16y4m_1_dysfluent_000.wav", '
        '"is_fluent": false, "start_time": 0.0, '
        '"labels": {"Block": 1, "Interjection": 0, "NoStutteredWords": 0, '
        '"Prolongation": 0, "SoundRep": 1, "WordRep": 0}},'
        '{"clip_id": "M_0031_16y4m_1_fluent_000", "duration": 3.0, '
        '"end_time": 3.0, "file_path": "clips/clips/M_0031_16y4m_1_fluent_000.wav", '
        '"is_fluent": true, "start_time": 0.0, '
        '"labels": {"Block": 0, "Interjection": 0, "NoStutteredWords": 1, '
        '"Prolongation": 0, "SoundRep": 0, "WordRep": 0}}'
        "]"
    )

    monkeypatch.setattr(merge_mod, "RAW_DATA_DIR", str(raw))
    return raw


def test_normalize_sep28k_writes_full_clip_weak_intervals(fake_raw):
    """SEP-28K Start/Stop are episode-relative sample boundaries, not event
    intervals: each present label must become a full-clip weak interval."""
    df, intervals = merge_mod.normalize_sep28k()

    assert len(df) == 1
    clip = os.path.join(fake_raw, SEP28K_CLIP, "FluencyBank_010_0.wav")
    assert clip in intervals
    assert intervals[clip] == [(0.0, 3.0, "Block")]
    assert "source" in df.columns
    assert df.iloc[0]["source"] == "sep28k"
    assert df.iloc[0]["Block"] == 1


def test_normalize_sep28k_never_uses_start_stop_as_seconds(fake_raw):
    """Even if Start/Stop contain huge episode-relative sample values, the
    generated intervals must stay within clip bounds."""
    _, intervals = merge_mod.normalize_sep28k()
    for clip, clip_intervals in intervals.items():
        for start, end, _ in clip_intervals:
            assert 0.0 <= start < end <= 3.0


def test_normalize_uclass_writes_centered_intervals(fake_raw):
    """UCLASS clips are centered on the stutter event (~1.5s into the 3s clip):
    each present label becomes a centered interval."""
    df, intervals = merge_mod.normalize_uclass()

    assert len(df) == 2
    dysfluent = os.path.join(
        fake_raw, "UCLASS SEP-28K Format", "clips", "clips",
        "M_0030_16y4m_1_dysfluent_000.wav",
    )
    fluent = os.path.join(
        fake_raw, "UCLASS SEP-28K Format", "clips", "clips",
        "M_0031_16y4m_1_fluent_000.wav",
    )
    assert dysfluent in intervals
    assert fluent not in intervals
    # Block and SoundRep present; both centered at ~1.5s.
    assert intervals[dysfluent] == [(1.25, 1.75, "Block"), (1.25, 1.75, "SoundRep")]
    assert df[df["clip_file"] == dysfluent].iloc[0]["source"] == "uclass"
    assert df[df["clip_file"] == fluent].iloc[0]["source"] == "uclass"


def test_normalize_sep28k_single_annotator_vote_is_negative(fake_raw):
    """A lone 'yes' vote must not flip the clip positive (SEP-28K majority is
    >=2 of 3 annotators); previously any single vote counted."""
    labels_csv = fake_raw / "SEP-28K Dataset" / "fluencybank_labels.csv"
    labels_csv.write_text(
        "Show,EpId,ClipId,Start,Stop,Prolongation,Block,SoundRep,WordRep,"
        "Interjection,NoStutteredWords\n"
        "FluencyBank,010,0,88960,136960,0,1,0,0,0,2\n"
    )
    df, intervals = merge_mod.normalize_sep28k()
    assert len(df) == 1
    assert df.iloc[0]["Block"] == 0
    assert intervals == {}


def test_normalize_sep28k_majority_votes_are_positive(fake_raw):
    """Two+ 'yes' votes flip the clip positive (SEP-28K majority rule)."""
    labels_csv = fake_raw / "SEP-28K Dataset" / "fluencybank_labels.csv"
    labels_csv.write_text(
        "Show,EpId,ClipId,Start,Stop,Prolongation,Block,SoundRep,WordRep,"
        "Interjection,NoStutteredWords\n"
        "FluencyBank,010,0,88960,136960,0,2,0,0,0,2\n"
    )
    df, intervals = merge_mod.normalize_sep28k()
    assert len(df) == 1
    assert df.iloc[0]["Block"] == 1
    clip = os.path.join(fake_raw, SEP28K_CLIP, "FluencyBank_010_0.wav")
    assert intervals[clip] == [(0.0, 3.0, "Block")]


def test_normalize_sep28k_skips_malformed_episode_id_without_aborting(fake_raw):
    """Rows whose Episode/Clip ID cannot be parsed must be skipped, not crash
    the whole merge (int(EpId) previously raised ValueError)."""
    labels_csv = fake_raw / "SEP-28K Dataset" / "fluencybank_labels.csv"
    labels_csv.write_text(
        "Show,EpId,ClipId,Start,Stop,Prolongation,Block,SoundRep,WordRep,"
        "Interjection,NoStutteredWords\n"
        "BadShow,Ep1,0,88960,136960,0,2,0,0,0,2\n"
        "FluencyBank,010,0,88960,136960,0,2,0,0,0,2\n"
    )
    df, _ = merge_mod.normalize_sep28k()
    assert len(df) == 1
    assert df.iloc[0]["clip_file"].endswith("FluencyBank_010_0.wav")


def test_merge_datasets_adds_source_column(fake_raw, tmp_path):
    """combined_labels.csv must carry a 'source' column identifying the
    originating dataset."""
    out = tmp_path / "merged" / "combined_labels.csv"
    combined = merge_mod.merge_datasets(output_path=str(out))

    assert combined is not None
    assert "source" in combined.columns
    sources = set(combined["source"])
    assert sources == {"sep28k", "uclass"}

    reloaded = combined.to_csv(index=False)
    assert "source" in reloaded.splitlines()[0]


def test_merge_datasets_regenerates_stale_label_csvs_without_force(fake_raw, tmp_path):
    """combined_labels.csv is always overwritten, so per-clip CSVs must be
    regenerated whenever their content diverges from the merged output — even
    without --force. Otherwise classification/localization labels silently
    disagree with the combined CSV."""
    out = tmp_path / "merged" / "combined_labels.csv"
    labels_dir = out.parent / "labels"
    labels_dir.mkdir(parents=True)
    stale = labels_dir / "FluencyBank_010_0.csv"
    stale.write_text("start_sec,end_sec,dysfluency_type\n0.000,0.100,block\n")

    merge_mod.merge_datasets(output_path=str(out), force=False)
    content = stale.read_text()
    assert "0.000,3.000,Block" in content


def test_merge_datasets_does_not_rewrite_unchanged_csvs(fake_raw, tmp_path):
    """A re-merge with unchanged output must not rewrite identical per-clip
    CSVs (avoids gratuitous mtime churn that would invalidate caches)."""
    out = tmp_path / "merged" / "combined_labels.csv"
    merge_mod.merge_datasets(output_path=str(out), force=False)
    label_path = out.parent / "labels" / "FluencyBank_010_0.csv"
    mtime = label_path.stat().st_mtime_ns

    merge_mod.merge_datasets(output_path=str(out), force=False)
    assert label_path.stat().st_mtime_ns == mtime


def test_parse_project_boli_transcript_returns_seconds_and_skips_word_lines(tmp_path):
    """Boli transcript times are in SECONDS; each event is two lines (code line
    + stuttered-word line). The word lines must not be parsed as events, and
    seconds must not be divided by 1000."""
    transcript = tmp_path / "10_727253_EI.txt"
    transcript.write_text(
        "5.526907\t6.871290\tSR\n"
        "5.526907\t6.871290\ts s sunset\n"
        "8.723551\t9.739307\tB\n"
        "8.723551\t9.739307\tmy . . name\n"
        "11.770818\t12.607323\tWR\n"
        "11.770818\t12.607323\tI I am\n"
    )
    labels, intervals = merge_mod._parse_project_boli_transcript(transcript)
    assert labels == {"SoundRep", "Block", "WordRep"}
    assert intervals == [
        (5.526907, 6.871290, "SoundRep"),
        (8.723551, 9.739307, "Block"),
        (11.770818, 12.607323, "WordRep"),
    ]


@pytest.mark.parametrize(
    "transcript_stem, expected_audio",
    [
        ("10_727253_EI", "727253_english_image_blob.wav"),
        ("1_14446_E1_1", "14446_english_1_paragraph_blob.wav"),
        ("2_862202_E3_9", "862202_english_3_paragraph_blob.wav"),
        ("6_31038_EI_2", "31038_english_image_blob.wav"),
        ("16_976172_E1", "976172_english_1_paragraph_blob.wav"),
        ("60_727253_E1", "727253_english_1_paragraph_blob.wav"),
        ("5_596658_E2_7", "596658_english_2_paragraph_blob.wav"),
    ],
)
def test_find_matching_audio_maps_boli_transcript_to_audio(
    tmp_path, transcript_stem, expected_audio
):
    """Boli transcript stems are '{nEvents}_{speaker}_{task}' but audio files
    are '{speaker}_english_{task}_blob.wav'; matching must map between them."""
    audios = tmp_path / "Audios"
    audios.mkdir()
    (audios / expected_audio).write_bytes(b"\x00" * 10)
    assert merge_mod._find_matching_audio(audios, transcript_stem) == audios / expected_audio


def test_normalize_project_boli_ingests_real_named_clips(tmp_path, monkeypatch):
    """End-to-end: a Boli clip with real-world names + seconds timestamps must
    produce a labeled row with its precise intervals."""
    base = tmp_path / "RawData" / "Project Boli Dataset"
    (base / "Transcripts").mkdir(parents=True)
    (base / "Audios").mkdir()
    (base / "Transcripts" / "10_727253_EI.txt").write_text(
        "5.526907\t6.871290\tSR\n"
        "5.526907\t6.871290\ts s sunset\n"
    )
    (base / "Audios" / "727253_english_image_blob.wav").write_bytes(b"\x00" * 10)
    monkeypatch.setattr(merge_mod, "RAW_DATA_DIR", str(tmp_path / "RawData"))

    df, intervals = merge_mod.normalize_project_boli()

    assert df is not None
    assert len(df) == 1
    row = df.iloc[0]
    assert row["SoundRep"] == 1
    assert row["source"] == "boli"
    assert intervals[str(row["clip_file"])] == [(5.526907, 6.871290, "SoundRep")]
