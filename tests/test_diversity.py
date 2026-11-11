"""Tests for the shared diversity guard's same-record hardening (live F3,
2026-11-05: «музыка для бега» served Hilary Duff «Come Clean» four times
in one card under three different diversity keys).

Contract under test (services/pipelines/_diversity.py):
  - normalized_title_key: the order-insensitive, junk-stripped record
    identity — the exported helper for the handler's miss chain (rank+1
    with the same key is the same song, not a new candidate);
  - artist_key validation chain: a valid "Artist - Track" split -> artist
    bucket; a REVERSED split (the channel names the right side) -> the
    channel fallback; a junk-handle channel (username-like single word)
    -> the title-hash bucket, so every copy of one record shares a key;
  - the guard: 3+ copies of the SAME RECORD in the top window keep the
    best 1 (any buckets the copies landed in); artist buckets still keep
    2 — two DIFFERENT songs by one artist are legitimate neighbours;
  - the exact live shape: the official upload, a reversed split and an
    agnusia89 re-upload of Come Clean plus one other artist -> the card
    keeps ONE Come Clean and the other artist.

The keep-2 artist semantics for the mood/similar callers stay covered by
the existing tests (test_mood_pipeline.py section 4,
test_similar_pipeline.py section 4); this file pins the F3 hardening.
"""
import logging

from services.pipelines._diversity import (
    DIVERSITY_KEEP_RECORD,
    artist_diversity_guard,
    artist_key,
    normalized_title_key,
)


def _cand(vid: str, title: str, channel: str, views: int = 1000) -> dict:
    """A fake yt-dlp candidate (the guard reads title/channel/views only)."""
    return {"video_id": vid, "title": title, "channel": channel,
            "duration": 240, "views": views}


# ════ 1. normalized_title_key: the exported record identity ════

def test_normalized_title_key_order_insensitive_and_junk_stripped():
    """The official upload, its reversed 4K twin and a bare Topic-style
    title clean to ONE key whatever the word order or decoration; a
    different song by the same artist keeps its own key."""
    official = normalized_title_key("Hilary Duff - Come Clean (Official Video) [HD]")
    reversed_ = normalized_title_key("Come Clean - Hilary Duff (Official Music Video) 4K")
    assert official == reversed_ == "clean come duff hilary"
    # the bare track title (auto-generated "- Topic" uploads) is its own
    # key — a strict subset of the full one, matched by the guard, not here
    assert normalized_title_key("Come Clean") == "clean come"
    # a different song by the same artist never collides
    assert normalized_title_key("Hilary Duff - Fly") == "duff fly hilary"
    # deterministic: same input, same key (safe to store/exchange)
    assert normalized_title_key("Come Clean") == normalized_title_key("come clean")


# ════ 2. artist_key: split validation and the junk fallback chain ════

def test_artist_key_valid_split_artist_bucket():
    """A clean 'Artist - Track' title keys by the artist token set, exactly
    as before the hardening."""
    assert artist_key(_cand("v1", "Hilary Duff - Come Clean (Official Video) [HD]",
                            "Hilary Duff - Topic")) == ("artist", frozenset({"hilary", "duff"}))


def test_artist_key_reversed_split_caught_via_channel():
    """'Come Clean - Hilary Duff (...)' on a Hilary Duff channel: the
    channel clearly names the RIGHT side of the dash, so the reversed
    split is rejected and the channel (minus '- Topic') keys the bucket —
    the copy collapses with the official upload instead of spawning a
    bogus {come clean} artist."""
    assert artist_key(_cand("v1", "Come Clean - Hilary Duff (Official Music Video) 4K",
                            "Hilary Duff - Topic")) == ("artist", frozenset({"hilary", "duff"}))


def test_artist_key_junk_channel_falls_back_to_title_hash():
    """The F3 fallback chain: no usable split + a username-like channel
    ('agnusia89', a letter+digit mix) or a random-ish >= 12-char single
    word -> the title-hash bucket, so copies of one record share ONE key
    instead of hiding under a channel handle."""
    assert artist_key(_cand("v1", "Come Clean", "agnusia89")) == \
        ("title", frozenset({"come", "clean"}))
    assert artist_key(_cand("v2", "Come Clean (Official Video)", "fancentral23")) == \
        ("title", frozenset({"come", "clean"}))
    assert artist_key(_cand("v3", "come clean", "xqzvmlkprtwsb")) == \
        ("title", frozenset({"come", "clean"}))


def test_artist_key_split_validation_junk_sides():
    """Mirrored splits (artist side == track side), too-short artist sides
    and username-like artist sides fail validation and fall to the next
    step of the chain."""
    # mirrored: "Come Clean - Come Clean" -> channel fallback
    assert artist_key(_cand("v1", "Come Clean - Come Clean", "Music Channel")) == \
        ("artist", frozenset({"music", "channel"}))
    # no artist token of length >= 3 -> channel fallback
    assert artist_key(_cand("v2", "a - Come Clean", "Hilary Duff - Topic")) == \
        ("artist", frozenset({"hilary", "duff"}))
    # a username-like single word cannot be the artist side -> title bucket
    assert artist_key(_cand("v3", "agnusia89 - Come Clean", "")) == \
        ("title", frozenset({"agnusia89", "come", "clean"}))


def test_artist_key_vevo_channel_is_not_a_reversal_witness():
    """A squashed VEVO handle ('JethroTullVEVO') must not vote in the
    reversal check: a legit 'Jethro Tull - Bungle in the Jungle' keeps its
    title-split artist key (the false reversal would break the s3 case)."""
    assert artist_key(_cand("v1", "Jethro Tull - Bungle in the Jungle",
                            "JethroTullVEVO")) == ("artist", frozenset({"jethro", "tull"}))


# ════ 3. the guard: same-record collapse vs artist keep-2 ════

def test_live_shape_four_copies_keep_one_come_clean(caplog):
    """The exact live shape (q123): the official upload, a reversed split,
    the bare '- Topic' title and an agnusia89 re-upload of Come Clean
    plus one other artist -> the card keeps ONE Come Clean (the best by
    the filter's raw ordering) and the other artist; the enriched list
    drops the same video_ids; one same-record log line."""
    filtered = [
        _cand("off1", "Hilary Duff - Come Clean (Official Video) [HD]",
              "Hilary Duff - Topic", 900),
        _cand("rev1", "Come Clean - Hilary Duff (Official Music Video) 4K",
              "Music Videos HD", 800),
        _cand("top1", "Come Clean", "Hilary Duff - Topic", 700),
        _cand("agn1", "Come Clean", "agnusia89", 600),
        _cand("oth1", "Other Artist - Other Song", "Other Artist - Topic", 500),
    ]
    enriched = [dict(c) for c in filtered]

    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        kept, enr = artist_diversity_guard(filtered, enriched, label="mood")

    come_clean = [c for c in kept if "come clean" in c["title"].lower()]
    assert len(come_clean) == 1 and come_clean[0]["video_id"] == "off1", \
        "4 copies of one record must collapse to the single best copy"
    assert [c["video_id"] for c in kept] == ["off1", "oth1"]
    assert {c["video_id"] for c in enr} == {"off1", "oth1"}
    assert "mood diversity" in caplog.text
    assert "same record" in caplog.text
    assert f"kept {DIVERSITY_KEEP_RECORD} (same song)" in caplog.text


def test_same_record_group_collapses_even_inside_one_artist_bucket():
    """Three copies that ALL key to the same artist bucket ({hilary, duff}
    via the split, the channel fallback and the reversal fix) still keep
    only 1: the same-record pass runs first, so the artist keep-2 cannot
    resurrect a second copy of the same song."""
    filtered = [
        _cand("v1", "Hilary Duff - Come Clean", "Hilary Duff - Topic", 500),
        _cand("v2", "Come Clean", "Hilary Duff - Topic", 400),
        _cand("v3", "Come Clean - Hilary Duff (Official Music Video) 4K",
              "Hilary Duff - Topic", 300),
        _cand("o1", "Other Artist - Other Song", "Other Artist - Topic", 200),
    ]
    kept, enr = artist_diversity_guard(filtered, [dict(c) for c in filtered],
                                       label="similar")
    assert [c["video_id"] for c in kept] == ["v1", "o1"]
    assert {c["video_id"] for c in enr} == {"v1", "o1"}


def test_artist_bucket_keeps_two_different_songs():
    """Artist semantics unchanged by the hardening: three DIFFERENT songs
    by one artist (distinct title keys — no same-record subset matching)
    keep the best 2; the trigger, the window and the identity contract
    (no trigger -> the same list objects) are the original ones."""
    filtered = [
        _cand("v1", "Hilary Duff - Come Clean", "Hilary Duff - Topic", 500),
        _cand("v2", "Hilary Duff - So Yesterday", "HilaryDuffVEVO", 400),
        _cand("v3", "Hilary Duff - Fly", "Hilary Duff Official", 300),
        _cand("o1", "Other Artist - Other Song", "Other Artist - Topic", 200),
    ]
    kept, enr = artist_diversity_guard(filtered, [], label="mood")
    assert [c["video_id"] for c in kept] == ["v1", "v2", "o1"], \
        "two different songs by one artist survive; only the third drops"


def test_junk_channel_copies_collapse_via_title_hash_bucket():
    """Copies that all degrade to the title-hash bucket (junk channels,
    no usable split) group by their identical title key: 3+ trigger, best
    1 survives, the other artist is untouched."""
    filtered = [
        _cand("j1", "Come Clean", "agnusia89", 500),
        _cand("j2", "Come Clean (Official Video)", "fancentral23", 400),
        _cand("j3", "come clean", "xqzvmlkprtwsb", 300),
        _cand("o1", "Other Artist - Other Song", "Other Artist - Topic", 200),
    ]
    kept, _ = artist_diversity_guard(filtered, [dict(c) for c in filtered],
                                     label="mood")
    assert [c["video_id"] for c in kept] == ["j1", "o1"]


def test_two_same_record_copies_below_trigger_untouched():
    """The same-record rule triggers at 3+ like the artist rule: two copies
    of one record pass through untouched (identity preserved)."""
    filtered = [
        _cand("v1", "Hilary Duff - Come Clean", "Hilary Duff - Topic", 500),
        _cand("v2", "Come Clean", "Hilary Duff - Topic", 400),
        _cand("o1", "Other Artist - Other Song", "Other Artist - Topic", 300),
        _cand("o2", "Third Artist - Tune", "Third Artist - Topic", 200),
    ]
    enriched = [dict(c) for c in filtered[:2]]
    kept, enr = artist_diversity_guard(filtered, enriched, label="mood")
    assert kept is filtered and enr is enriched
    assert len(kept) == 4


def test_same_record_subset_needs_two_tokens():
    """A one-word title never subset-matches ('hello' is not every song
    with 'hello' in it); only EQUAL one-word keys group. Two 'Love' songs
    by different artists with decorated titles stay separate records."""
    filtered = [
        _cand("v1", "Love", "Adele - Topic", 500),          # {love}
        _cand("v2", "Pigeons In Love", "Some Band - Topic", 400),  # {pigeons,in,love}
        _cand("v3", "Love (Remix)", "Other - Topic", 300),  # {love, remix}
        _cand("o1", "Other Artist - Other Song", "Other Artist - Topic", 200),
    ]
    kept, _ = artist_diversity_guard(filtered, [], label="mood")
    assert [c["video_id"] for c in kept] == ["v1", "v2", "v3", "o1"], \
        "a bare one-word title must not swallow decorated titles by subset"


def test_same_record_drops_reach_beyond_the_window():
    """Like the artist rule, a crowded record's copies drop from the WHOLE
    pool, not just the top-6 window: three copies inside the window crowd
    the record and a fourth copy beyond position 6 drops with them."""
    filtered = [
        _cand("v1", "Hilary Duff - Come Clean (Official Video) [HD]",
              "Hilary Duff - Topic", 900),
        _cand("v2", "Come Clean", "Hilary Duff - Topic", 800),
        _cand("v3", "Come Clean - Hilary Duff (Official Music Video) 4K",
              "Music Videos HD", 700),
    ] + [_cand(f"f{i}", f"Filler Band {i} - Filler Song {i}",
               f"Filler Band {i} - Topic", 600 - i) for i in range(4)] \
      + [_cand("deep1", "come clean", "agnusia89", 100)]
    kept, _ = artist_diversity_guard(filtered, [], label="mood")
    vids = [c["video_id"] for c in kept]
    assert "v1" in vids and "deep1" not in vids, \
        "the beyond-window copy of the crowded record drops too"
