"""Variant modifiers (owner decision 16.11): a NAMED edit version
("the way i are slowed") is part of the user's wish, not noise.

Covers the three core pieces of services/llm_search.py:
  1. the shared detector contract (VARIANT_WORDS / is_variant_upload —
     imported by from_source.py / match_score.py, FROZEN);
  2. the extract contract (target.variant in SYSTEM_EXTRACT);
  3. the choke-point variant guard in search_candidates (after LLM#2,
     next to the official-first guard) + the «Вариант-запрос:» rank note.

Conventions follow tests/test_llm_search.py: BothubClient.chat_json is
mocked with a side_effect returning ALREADY parsed dicts;
services.llm_search.search_metadata is patched; no network. The this_track
confident-match shortcut is disabled for the rank-path tests (its own
tests live in test_llm_search.py) and exercised as the incident path in
its own test below.
"""
import logging
from unittest.mock import patch

import pytest

from services.llm_search import (
    SYSTEM_EXTRACT,
    VARIANT_WORDS,
    _canonical_variant,
    is_variant_upload,
    search_candidates,
)


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Enrich step offline (same as tests/test_llm_search.py): the top-5
    enrich would be a real yt-dlp call; the stub returns None."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers ════

def _extract_json(intent="this_track", target=None, queries=None):
    """LLM#1 extract — a fake already-parsed answer."""
    return {
        "intro": "Ищу редакт-версию трека...",
        "intent": intent,
        "signals": {"language": "en", "era": None, "vocal": None,
                    "genre": None, "mood": None},
        "target": target or {"artist": None, "track": None, "variant": None},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries or ["timbaland the way i are slowed"],
    }


def _rank_json(scores, best_index=0):
    """LLM#2 rank — scores is a list of (video_id, score) in ranked order."""
    return {
        "ranked": [{"video_id": vid, "score": sc, "rationale": "ok"}
                   for vid, sc in scores],
        "best_index": best_index,
        "confidence": "high",
        "verdict": "Timbaland — The Way I Are (2007).",
        "need_clarify": False,
        "clarify_question": "",
    }


def _way_i_are_pool():
    """The incident pool: the official original plus slowed/sped-up fan
    edits of the same record (distinct video_ids, plain channels)."""
    return [
        {"video_id": "v1", "title": "Timbaland - The Way I Are",
         "channel": "TimbalandVEVO", "duration": 250, "views": 90_000_000},
        {"video_id": "v2", "title": "Timbaland - The Way I Are (Slowed)",
         "channel": "Chill Vibes", "duration": 260, "views": 500_000},
        {"video_id": "v3", "title": "Timbaland - The Way I Are (Sped Up)",
         "channel": "Fan Edits", "duration": 200, "views": 300_000},
    ]


# ════ 1. the detector contract (VARIANT_WORDS / is_variant_upload) ════

@pytest.mark.parametrize("title,channel,want", [
    # the canonical positive set
    ("Timbaland - Give It To Me (slowed)", "", "slowed"),
    ("Timbaland - Give It To Me slowed down", "", "slowed"),
    ("Timbaland - Give It To Me sped up", "", "sped_up"),
    ("Timbaland - Give It To Me (Speed Up)", "", "sped_up"),
    ("The Way I Are (Nightcore)", "", "nightcore"),
    ("The Way I Are 8D Audio", "", "nightcore"),
    ("The Way I Are (Reverb)", "", "reverb"),
    ("The Way I Are (Remix)", "", "remix"),
    ("The Way I Are (Remixed)", "", "remix"),
    ("The Way I Are (Acoustic Version)", "", "acoustic"),
    ("Live at Wembley (Queen)", "", "live"),
    ("Song (Live)", "", "live"),
    ("Song (Live Version)", "", "live"),
    ("Song Live Session", "", "live"),
    ("The Way I Are (Instrumental)", "", "instrumental"),
    ("The Way I Are (Extended Mix)", "", "extended"),
    # 'slowed' outranks its standard '+ reverb' pairing (dict precedence)
    ("Timbaland - The Way I Are (slowed + reverb)", "", "slowed"),
    # the marker may sit in the CHANNEL instead of the title
    ("Timbaland - The Way I Are", "Slowed Music", "slowed"),
    # cheap RU pairs (stems prefix-match the morphology)
    ("Timbaland - The Way I Are (замедленная версия)", "", "slowed"),
    ("Timbaland - The Way I Are (ускоренная версия)", "", "sped_up"),
    # FALSE positives must not fire (a false positive demotes originals)
    ("Live and Let Die", "", None),
    ("Living in a Ghost Town", "", None),
    ("Slow Dance", "", None),
    ("Revolution", "", None),
    # hostile: marker glued inside an artist name (word boundary holds)
    ("Slowdive - Alizon", "", None),
    ("Song (18d Bass Boosted)", "", None),
])
def test_is_variant_upload(title, channel, want):
    assert is_variant_upload(title, channel) == want


def test_variant_words_contract_shape():
    """The FROZEN contract another agent imports: canonical variant keys ->
    tuples of str markers; the canonical set is exactly the prompt's set."""
    assert set(VARIANT_WORDS) == {
        "slowed", "sped_up", "nightcore", "reverb", "remix",
        "acoustic", "live", "instrumental", "extended"}
    for markers in VARIANT_WORDS.values():
        assert isinstance(markers, tuple)
        assert all(isinstance(m, str) and m for m in markers)


def test_canonical_variant_folds_llm_answers():
    """target.variant answers (free-form) fold onto the canonical set;
    junk/null normalize to '' — an unparseable variant never steers."""
    assert _canonical_variant("slowed") == "slowed"
    assert _canonical_variant("8d") == "nightcore"          # 8d -> nightcore
    assert _canonical_variant("8D Audio") == "nightcore"
    assert _canonical_variant("sped up") == "sped_up"
    assert _canonical_variant("SPED_UP") == "sped_up"
    assert _canonical_variant("remixed") == "remix"
    assert _canonical_variant(None) == ""
    assert _canonical_variant("null") == ""
    assert _canonical_variant("banana") == ""


# ════ 2. the extract contract ════

def test_system_extract_variant_lines():
    """SYSTEM_EXTRACT carries the variant field + the ВАРИАНТ-МОДИФИКАТОР
    rule (the full snapshot tripwires live in tests/test_mood_pipeline.py)."""
    assert '"variant": "slowed|sped_up|nightcore|reverb|remix|acoustic|live|instrumental|extended|null"' \
        in SYSTEM_EXTRACT
    assert "ВАРИАНТ-МОДИФИКАТОР" in SYSTEM_EXTRACT
    assert 'сохрани модификатор в queries ("the way i are slowed")' in SYSTEM_EXTRACT


# ════ 3. full this_track run: the incident pool ════

@pytest.mark.asyncio
async def test_this_track_slowed_incident_rank_path(bothub_client):
    """«timbaland the way i are slowed» (16.11): extract returns
    target.variant=slowed; the pool is [official 0.95, slowed fan 0.6,
    sped_up fan 0.5]. After the choke-point guard: slowed FIRST (its own
    score, no boost), the official pressed under the weakest matching
    (0.6 - 0.05), the foreign variant below the original, best on slowed.
    The rank input also carried the «Вариант-запрос:» note."""
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault(gen_name, []).append(user)
        if gen_name == "llm-extract":
            return _extract_json(target={"artist": "Timbaland",
                                         "track": "The Way I Are",
                                         "variant": "slowed"})
        return _rank_json([("v1", 0.95), ("v2", 0.6), ("v3", 0.5)],
                          best_index=0)

    # Disable the confident-match shortcut so the LLM#2 rank path runs with
    # the mocked scores (the shortcut path has its own test below).
    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = _way_i_are_pool()
        out = await search_candidates("timbaland the way i are slowed",
                                      bothub_client)

    assert out.signals.get("variant") == "slowed"
    # the rank note reached LLM#2 as its own line
    rank_msg = captured["llm-rank"][0]
    assert "Вариант-запрос: slowed (искомая редакт-версия, оригинал НЕ ответ)" \
        in rank_msg
    assert "variant=slowed" not in rank_msg  # kept out of the Сигналы join
    # order: slowed first, original below it, foreign variant last
    assert [c.video_id for c in out.candidates] == ["v2", "v1", "v3"]
    assert out.candidates[out.best_index].video_id == "v2"
    assert out.candidates[0].score == pytest.approx(0.6)   # matching: untouched
    assert out.candidates[1].score == pytest.approx(0.55)  # min(0.95, 0.6-0.05)
    assert out.candidates[2].score <= out.candidates[1].score


@pytest.mark.asyncio
async def test_this_track_slowed_incident_shortcut_path(bothub_client):
    """The REAL incident path: the official upload fuzzy-matches the target
    perfectly, the confident-match shortcut fires (NO LLM#2 call) and would
    deliver the original — the choke-point guard still flips the pool to
    the requested slowed edit."""
    calls = []

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        calls.append(gen_name)
        return _extract_json(target={"artist": "Timbaland",
                                     "track": "The Way I Are",
                                     "variant": "slowed"})

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = _way_i_are_pool()
        out = await search_candidates("timbaland the way i are slowed",
                                      bothub_client)

    assert calls == ["llm-extract"], "the fuzzy shortcut must skip LLM#2"
    assert [c.video_id for c in out.candidates][0] == "v2"
    assert out.candidates[out.best_index].video_id == "v2"
    assert out.candidates[out.best_index].score > \
        next(c.score for c in out.candidates if c.video_id == "v1")


# ════ 4. variant set, no matching upload ════

@pytest.mark.asyncio
async def test_variant_requested_none_found_pool_unchanged(
        bothub_client, caplog):
    """Variant=slowed but the pool holds no slowed upload: the pool stands
    EXACTLY as ranked (scores and order untouched, best untouched) and one
    log line says so."""
    entries = [
        {"video_id": "v1", "title": "Timbaland - The Way I Are",
         "channel": "TimbalandVEVO", "duration": 250, "views": 90_000_000},
        {"video_id": "v2", "title": "Timbaland - Apologize",
         "channel": "TimbalandVEVO", "duration": 210, "views": 40_000_000},
        {"video_id": "v3", "title": "Timbaland - Carry Out",
         "channel": "TimbalandVEVO", "duration": 230, "views": 20_000_000},
    ]

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(target={"artist": "Timbaland",
                                         "track": "The Way I Are",
                                         "variant": "slowed"})
        return _rank_json([("v1", 0.9), ("v2", 0.8), ("v3", 0.7)],
                          best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = entries
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out = await search_candidates("timbaland the way i are slowed",
                                          bothub_client)

    assert [c.video_id for c in out.candidates] == ["v1", "v2", "v3"]
    assert [c.score for c in out.candidates] == [0.9, 0.8, 0.7]
    assert out.best_index == 0
    assert any("variant requested, none found" in r.getMessage()
               for r in caplog.records)


# ════ 5. no variant: bit-for-bit previous behavior ════

@pytest.mark.asyncio
async def test_no_variant_official_first_still_wins(bothub_client):
    """No variant named: nothing new fires. The official-first guard (F1
    incident B) still demotes a fan copy scored ABOVE the official — the
    original wins, best moves to it, and the rank input has NO
    «Вариант-запрос:» line (this pins that the change broke nothing)."""
    captured = {}
    entries = _way_i_are_pool()[:2] + [
        {"video_id": "v4", "title": "Timbaland - Give It To Me",
         "channel": "TimbalandVEVO", "duration": 240, "views": 10_000_000},
    ]

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault(gen_name, []).append(user)
        if gen_name == "llm-extract":
            return _extract_json(target={"artist": "Timbaland",
                                         "track": "The Way I Are",
                                         "variant": None})
        # the ranker LOVES the slowed fan copy (0.98) over the official
        return _rank_json([("v2", 0.98), ("v1", 0.95), ("v4", 0.4)],
                          best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = entries
        out = await search_candidates("timbaland the way i are", bothub_client)

    assert "variant" not in out.signals
    assert "Вариант-запрос" not in captured["llm-rank"][0]
    # official-first: the fan copy clamped to 0.95 - 0.05, official first,
    # the demoted fan's best slot handed to the record's new leader
    assert [c.video_id for c in out.candidates] == ["v1", "v2", "v4"]
    assert out.candidates[1].score == pytest.approx(0.90)
    assert out.candidates[out.best_index].video_id == "v1"


# ════ 6. variant miss fallback (16.11) ════
#
# Incident 16.11: «the weeknd blinding lights sped up» — variant=sped_up
# recognized, the pool held NO sped-up upload, and the ranker (reading the
# «оригинал НЕ ответ» note) zeroed every score and answered need_clarify
# over a NON-EMPTY pool: the user got a circular «name the artist» ask
# although he HAD named the artist. Owner product rule: a variant miss is
# not a dead end — the edit may simply not exist on YouTube, so the
# original ships as the closest fallback, with an honest intro note.

def _weeknd_extract(language="ru", variant="sped_up"):
    """LLM#1 extract of the 16.11 incident; `language` steers the fallback
    note's RU/EN form (the note follows signals.language, like the other
    user-facing lines)."""
    return {
        "intro": "Ищу: The Weeknd, Blinding Lights, редакт-версия.",
        "intent": "this_track",
        "signals": {"language": language, "era": None, "vocal": None,
                    "genre": None, "mood": None},
        "target": {"artist": "The Weeknd", "track": "Blinding Lights",
                   "variant": variant},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": ["the weeknd blinding lights sped up"],
    }


def _weeknd_pool():
    """The incident pool: the official original plus a fan re-upload — NO
    sped-up upload (is_variant_upload -> None for both, no matching). The
    fan copy carries a year so the filter's _norm_title dedup keeps both
    entries (same normalized title = one candidate)."""
    return [
        {"video_id": "w1", "title": "The Weeknd - Blinding Lights",
         "channel": "TheWeekndVEVO", "duration": 200, "views": 800_000_000},
        {"video_id": "w2", "title": "The Weeknd - Blinding Lights 2020",
         "channel": "Fan Reuploads", "duration": 200, "views": 1_000_000},
    ]


def _clarify_rank_json(vids, question):
    """LLM#2 rank of the incident miss: every score zeroed (the ranker
    obeyed «оригинал НЕ ответ») and a clarify asked INSTEAD of a verdict —
    an EMPTY verdict, so the sec4 zeroed-rank guard does not eat the pool
    and the rank's own clarify reaches the choke point over a NON-EMPTY
    candidate list (the (a) form of the miss)."""
    return {
        "ranked": [{"video_id": vid, "score": 0, "rationale": "не редакт"}
                   for vid in vids],
        "best_index": 0,
        "confidence": "low",
        "verdict": "",
        "need_clarify": True,
        "clarify_question": question,
    }


@pytest.mark.asyncio
async def test_variant_miss_incident_clarify_converted_to_delivery(
        bothub_client, caplog):
    """INCIDENT (a): rank zeroed all scores + need_clarify=true over a
    non-empty pool -> the fallback converts the clarify into a delivery:
    need_clarify=False, question dropped, candidates kept in their order
    (the originals are already on top), honest RU note in the intro."""
    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _weeknd_extract(language="ru")
        return _clarify_rank_json(
            ["w1", "w2"], "Уточни исполнителя или название трека.")

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = _weeknd_pool()
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out = await search_candidates(
                "the weeknd blinding lights sped up", bothub_client)

    # converted to a delivery, candidates returned (the handler's preview
    # takes it from here)
    assert out.need_clarify is False
    assert out.clarify_question == ""
    assert [c.video_id for c in out.candidates] == ["w1", "w2"]
    # honest note appended after the extract's intro
    assert out.intro.startswith("Ищу: The Weeknd, Blinding Lights, редакт-версия.")
    assert "Точной sped_up-версии не нашёл — показываю ближайшее." in out.intro
    assert any("variant miss fallback: delivered 2 candidates, "
               "note added (sped_up)" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_variant_miss_note_english(bothub_client):
    """EN form: signals.language=en -> the note is the EN line, not RU."""
    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _weeknd_extract(language="en")
        return _clarify_rank_json(
            ["w1", "w2"], "Name the artist or the track.")

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = _weeknd_pool()
        out = await search_candidates(
            "the weeknd blinding lights sped up", bothub_client)

    assert out.need_clarify is False
    assert "No exact sped_up version found - showing the closest." in out.intro
    assert "не нашёл" not in out.intro


@pytest.mark.asyncio
async def test_variant_miss_delivered_pool_gets_note(bothub_client):
    """Case (b): the pool is delivered (no clarify), no matching upload —
    the delivery stands exactly as ranked and the note is added."""
    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _weeknd_extract(language="ru")
        return _rank_json([("w1", 0.9), ("w2", 0.7)], best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = _weeknd_pool()
        out = await search_candidates(
            "the weeknd blinding lights sped up", bothub_client)

    assert out.need_clarify is False
    assert out.clarify_question == ""
    assert [c.video_id for c in out.candidates] == ["w1", "w2"]
    assert [c.score for c in out.candidates] == [0.9, 0.7]
    assert out.best_index == 0
    assert "Точной sped_up-версии не нашёл" in out.intro


@pytest.mark.asyncio
async def test_variant_match_found_no_note(bothub_client):
    """Variant found (matching exists): bit-for-bit previous behavior —
    NO fallback note (neither language), the intro is the extract's own."""
    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(target={"artist": "Timbaland",
                                         "track": "The Way I Are",
                                         "variant": "slowed"})
        return _rank_json([("v1", 0.95), ("v2", 0.6), ("v3", 0.5)],
                          best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = _way_i_are_pool()
        out = await search_candidates("timbaland the way i are slowed",
                                      bothub_client)

    assert out.need_clarify is False
    assert "не нашёл" not in out.intro
    assert "No exact" not in out.intro
    assert out.intro == "Ищу редакт-версию трека..."
    # the guard itself still pressed the original under the slowed upload
    assert [c.video_id for c in out.candidates] == ["v2", "v1", "v3"]


@pytest.mark.asyncio
async def test_no_variant_zeroed_clarify_previous_behavior(bothub_client):
    """No variant named: the fallback never fires, even on the zeroed-rank
    clarify path (sec4 with a non-empty verdict) — previous behavior."""
    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _weeknd_extract(variant=None)
        # verdict NON-empty + all scores 0: sec4 empties the pool and asks
        # the shared empty-result clarify — the pre-fallback behavior
        return {
            "ranked": [{"video_id": "w1", "score": 0, "rationale": ""},
                       {"video_id": "w2", "score": 0, "rationale": ""}],
            "best_index": 0,
            "confidence": "low",
            "verdict": "Точного совпадения нет.",
            "need_clarify": False,
            "clarify_question": "",
        }

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = _weeknd_pool()
        out = await search_candidates("the weeknd blinding lights",
                                      bothub_client)

    assert out.need_clarify is True
    assert out.candidates == []
    assert "не нашёл" not in out.intro
    assert "No exact" not in out.intro


@pytest.mark.asyncio
async def test_variant_empty_pool_previous_empty_path(bothub_client):
    """Variant requested but the pool is EMPTY: the honest empty path
    stands (f3's clarify), no note — an empty fallback delivery would be
    a lie."""
    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        return _weeknd_extract(language="ru")

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = []
        out = await search_candidates(
            "the weeknd blinding lights sped up", bothub_client)

    assert out.need_clarify is True
    assert out.candidates == []
    assert "не нашёл" not in out.intro


async def test_variant_miss_zeroed_with_verdict_keeps_pool(bothub_client):
    """(16.11 acceptance V1) sec4-обнуление С вердиктом больше не опустошает
    пул при вариант-промахе — фолбэк доносит оригинал с честной пометкой
    (инцидент blinding lights sped up шёл именно этим путём)."""
    entries = [
        {"video_id": "w1", "title": "Blinding Lights 2020",
         "channel": "The Weeknd", "duration": 200, "views": 10 ** 9},
    ]
    rank = {"ranked": [{"video_id": "w1", "score": 0, "rationale": "."}],
            "best_index": 0, "confidence": "low",
            "verdict": "Точной версии нет.", "need_clarify": False,
            "clarify_question": ""}
    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(target={"artist": "The Weeknd",
                                         "track": "Blinding Lights",
                                         "variant": "sped_up"})
        return rank
    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata", return_value=entries):
            from services.llm_search import search_candidates
            out = await search_candidates("the weeknd blinding lights sped up",
                                          bothub_client)
    assert out.candidates, "pool survived sec4 for the fallback"
    assert not out.need_clarify
    assert "sped_up" in out.intro


async def test_variant_zeroed_pool_kept_even_when_matching_exists(bothub_client):
    """(acceptance r3, V1) ранжир обнулил пул при живом sped-up кандидате —
    sec4 всё равно держит пул: вариант-гвард отсортирует его наверх."""
    entries = [
        {"video_id": "w1", "title": "Blinding Lights 2020",
         "channel": "The Weeknd", "duration": 200, "views": 10 ** 9},
        {"video_id": "v1", "title": "Blinding Lights (sped up)",
         "channel": "TikTok Tunes", "duration": 170, "views": 10 ** 6},
    ]
    rank = {"ranked": [{"video_id": "w1", "score": 0, "rationale": "."},
                       {"video_id": "v1", "score": 0, "rationale": "."}],
            "best_index": 0, "confidence": "low",
            "verdict": "Нет точного варианта.", "need_clarify": False,
            "clarify_question": ""}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(target={"artist": "The Weeknd",
                                         "track": "Blinding Lights",
                                         "variant": "sped_up"})
        return rank

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata", return_value=entries):
            from services.llm_search import search_candidates
            out = await search_candidates("the weeknd blinding lights sped up",
                                          bothub_client)
    assert out.candidates, "pool kept despite zeroed rank"
    assert out.candidates[0].video_id == "v1" or not out.need_clarify
    assert not out.need_clarify


async def test_variant_match_clears_ranker_clarify(bothub_client):
    """(acceptance r4, V5/V6) редакт найден и возглавляет пул — вопрос
    ранжира гасится: ответ существует, доставка лучше переспроса."""
    entries = [
        {"video_id": "v1", "title": "smells like teen spirit (slowed)",
         "channel": "fan edits", "duration": 250, "views": 10 ** 6},
        {"video_id": "w1", "title": "Smells Like Teen Spirit 1991",
         "channel": "Nirvana", "duration": 260, "views": 10 ** 9},
    ]
    rank = {"ranked": [{"video_id": "v1", "score": 0, "rationale": "."},
                       {"video_id": "w1", "score": 0, "rationale": "."}],
            "best_index": 0, "confidence": "low", "verdict": "Есть вариант.",
            "need_clarify": True,
            "clarify_question": "Какой именно темп вам нужен?"}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(target={"artist": "Nirvana",
                                         "track": "Smells Like Teen Spirit",
                                         "variant": "slowed"})
        return rank

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata", return_value=entries):
            from services.llm_search import search_candidates
            out = await search_candidates("nirvana smells like teen spirit slowed",
                                          bothub_client)
    assert not out.need_clarify
    assert out.candidates[0].video_id == "v1"
