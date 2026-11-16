"""Artist-only scope (incident 14.11, fixed 2026-11-14): a bare artist query
(«depeche mode» / «Depeche Mode» / «Кино») must REACH the search and return
only songs of THAT artist; a «Не то» then gives another song by the same
artist.

Contract under test:
  - gate: a bare artist (lowercase AND capitalized) is no longer clarified
    away by _pre_search_gate — it flows into LLM#1 and YouTube;
  - scope detection: intent=this_track + target.artist set + target.track
    empty -> SearchOut.scope_artist carries the artist (.strip()'ed); a
    given track, or intent similar/mood (even with an artist in target),
    keeps the scope OFF;
  - pool guarantee: non-matching candidates (covers/reactions by other
    artists) are DEMOTED below every match (score clamp, 0.05 gap under the
    weakest match) when the pool holds 1-4 matches, and DROPPED outright
    when it holds >= 5 matches;
  - matching evidence: channel minus " - Topic", enriched canonical artist,
    or the artist name in the title as whole words (word boundary,
    case-insensitive);
  - rank note: the scope artist is rendered as the «Исполнитель-запрос:»
    line of the ranker input, non-matches flagged [чужой?], and the
    transport key never leaks into the Сигналы join;
  - adversarial battery: «Кино» (RU), «The Prodigy» (stop-word "the"),
    «U2» (digit in the name — the old deterministic gate could never see
    it; the LLM target path must), «P!nk» (punctuation), «рок» (NOT an
    artist: the mood path keeps the scope off);
  - regression: a regular this_track with artist+track («nirvana smells
    like teen spirit») never activates the scope.

Mocking follows tests/test_llm_search.py: chat_json patched on the
bothub_client fixture (already-parsed dicts, dispatched by gen_name),
search_metadata patched in services.llm_search, enrich's yt-dlp layer cut
— no network.
"""
import logging

import pytest
from unittest.mock import patch

from services.llm_search import Candidate, _format_for_rank, _scope_match, search_candidates


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Same guard as tests/test_llm_search.py: the enrich step must not
    touch the network (candidates here are fake)."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers ════

def _extract_json(intent="this_track", target=None, queries=None, signals=None,
                  reference=None):
    """LLM#1 extract — a fake parsed dict. The artist-only flavor defaults
    to a bare-artist target (artist set, track None): the SYSTEM_EXTRACT
    contract for a bare artist query."""
    return {
        "intro": "Ищу песни этого исполнителя...",
        "intent": intent,
        "signals": signals or {"language": "en", "era": None, "vocal": None,
                               "genre": None, "mood": None},
        "target": target if target is not None else {"artist": "Depeche Mode", "track": None},
        "reference": reference or {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries or ["Depeche Mode best songs", "Depeche Mode Enjoy the Silence"],
    }


def _rank_json(ranked, best_index=0):
    """LLM#2 rank — a fake parsed dict over explicit (video_id, score)
    pairs; the LLM's best may be a non-match (the incident's cover win)."""
    return {
        "ranked": [{"video_id": vid, "score": score, "rationale": "."}
                   for vid, score in ranked],
        "best_index": best_index,
        "confidence": "high",
        "verdict": "Depeche Mode — британская электроник-группа.",
        "need_clarify": False,
        "clarify_question": "",
    }


def _entry(vid, title, channel, dur=250, views=10 ** 5):
    """A raw yt-dlp-style search result dict."""
    return {"video_id": vid, "title": title, "channel": channel,
            "duration": dur, "views": views}


_DM_POOL = [
    _entry("dm1", "Depeche Mode - Enjoy the Silence", "Depeche Mode - Topic", 254, 10 ** 6),
    _entry("dm2", "Depeche Mode - Personal Jesus", "Depeche Mode - Topic", 260, 9 * 10 ** 5),
    _entry("dm3", "Depeche Mode - Never Let Me Down Again", "Depeche Mode - Topic", 265, 8 * 10 ** 5),
    _entry("dm4", "Depeche Mode - Stripped", "Depeche Mode", 250, 7 * 10 ** 5),
    # title-word match on a non-artist channel
    _entry("dm5", "Depeche Mode - Walking in My Shoes", "Mute Records", 280, 6 * 10 ** 5),
    # a live cut of the SAME artist: a match (noise, but his song)
    _entry("dm_live", "Depeche Mode - Enjoy the Silence (Live)", "Depeche Mode - Topic", 300, 5 * 10 ** 5),
    # covers by OTHER artists: the incident's pollution
    _entry("cov1", "Enjoy the Silence (Cover)", "Cover Nation", 240, 4 * 10 ** 5),
    _entry("cov2", "Personal Jesus Tribute", "Tribute Channel", 250, 3 * 10 ** 5),
]


async def _run_scope_search(bothub_client, description, extract, pool,
                            rank=None, captured=None, excluded_artists=None):
    """One artist-only search with every external layer mocked. Returns the
    SearchOut; `captured` (a dict) also receives the llm-rank user message."""
    rank = rank or _rank_json([("dm1", 0.9), ("dm2", 0.8)], best_index=0)

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if isinstance(captured, dict):
            captured.setdefault("gens", []).append(gen_name)
            if gen_name == "llm-rank":
                captured["rank_input"] = user
        if gen_name == "llm-extract":
            return extract
        return rank

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = pool
            return await search_candidates(description, bothub_client,
                                           excluded_artists=excluded_artists)


# ════ 1. the gate no longer refuses a bare artist ════

@pytest.mark.asyncio
async def test_bare_artist_lowercase_reaches_search(bothub_client):
    """«depeche mode» (lowercase) — the search runs, results come back."""
    out = await _run_scope_search(
        bothub_client, "depeche mode", _extract_json(), _DM_POOL,
        rank=_rank_json([("dm1", 0.9), ("dm2", 0.8)], best_index=0))
    assert not out.need_clarify
    assert out.candidates
    assert out.scope_artist == "Depeche Mode"


@pytest.mark.asyncio
async def test_bare_artist_capitalized_reaches_search(bothub_client):
    """«Depeche Mode» (capitalized) — formerly the deterministic
    _pre_search_gate refused to search and asked for a track; now the same
    input as the lowercase variant flows into LLM#1 and YouTube."""
    out = await _run_scope_search(
        bothub_client, "Depeche Mode", _extract_json(), _DM_POOL,
        rank=_rank_json([("dm1", 0.9), ("dm2", 0.8)], best_index=0))
    assert not out.need_clarify
    assert out.candidates
    assert out.scope_artist == "Depeche Mode"


def test_pre_search_gate_passes_bare_artists():
    """Unit: the gate itself lets bare artists through — both cases,
    lowercase and capitalized (the 14.11 inconsistency is gone)."""
    from services.llm_search import _pre_search_gate
    for d in ("depeche mode", "Depeche Mode", "Кино", "AC/DC", "U2", "P!nk"):
        need, _ = _pre_search_gate(d)
        assert not need, f"a bare artist must reach the search: {d!r}"


# ════ 2. scope detection: target shape decides ════

@pytest.mark.asyncio
async def test_scope_active_for_artist_only_target(bothub_client):
    """target.artist set + target.track None -> scope_artist reported and
    mirrored into signals (the rank-note transport)."""
    out = await _run_scope_search(
        bothub_client, "depeche mode", _extract_json(), _DM_POOL)
    assert out.scope_artist == "Depeche Mode"
    assert out.signals.get("scope_artist") == "Depeche Mode"


@pytest.mark.asyncio
async def test_scope_inactive_when_track_given(bothub_client):
    """A concrete target (artist+track) is a normal this_track search — no
    scope, even though the artist is named."""
    out = await _run_scope_search(
        bothub_client, "nirvana smells like teen spirit",
        _extract_json(target={"artist": "Nirvana", "track": "Smells Like Teen Spirit"},
                      queries=["Nirvana Smells Like Teen Spirit"]),
        [_entry("n1", "Nirvana - Smells Like Teen Spirit", "Nirvana - Topic", 260, 10 ** 6),
         _entry("n2", "Nirvana - Come As You Are", "Nirvana - Topic", 240, 9 * 10 ** 5)],
        rank=_rank_json([("n1", 0.95), ("n2", 0.6)], best_index=0))
    assert out.scope_artist == ""
    assert "scope_artist" not in out.signals


@pytest.mark.asyncio
async def test_scope_inactive_for_similar_with_artist(bothub_client):
    """similar + an (erroneous) target.artist fill — the scope keys on
    intent=this_track only; similar never scopes."""
    out = await _run_scope_search(
        bothub_client, "похожее на Depeche Mode",
        _extract_json(intent="similar",
                      target={"artist": "Depeche Mode", "track": None},
                      reference={"artist": "Depeche Mode", "track": None},
                      queries=["synth-pop 80s hits"]),
        _DM_POOL[:2],
        rank=_rank_json([("dm1", 0.9), ("dm2", 0.8)], best_index=0))
    assert out.scope_artist == ""
    assert "scope_artist" not in out.signals


@pytest.mark.asyncio
@pytest.mark.parametrize("target,expected", [
    # track "" and a missing track key both count as "no track" -> scope ON
    ({"artist": "Depeche Mode", "track": ""}, "Depeche Mode"),
    ({"artist": "Depeche Mode"}, "Depeche Mode"),
    # whitespace-only artist or nothing extractable -> scope OFF
    ({"artist": "   ", "track": None}, ""),
    ({"artist": None, "track": None}, ""),
])
async def test_scope_detection_shapes(bothub_client, target, expected):
    """The exact detection rule against the real search path: scope artist
    = non-empty target.artist with an empty target track, reported
    .strip()'ed as the LLM returned it."""
    out = await _run_scope_search(
        bothub_client, "depeche mode",
        _extract_json(target=target), _DM_POOL[:2],
        rank=_rank_json([("dm1", 0.9), ("dm2", 0.8)], best_index=0))
    assert out.scope_artist == expected


@pytest.mark.asyncio
async def test_scope_inactive_for_mood_with_artist(bothub_client):
    """mood + an (erroneous) target.artist fill — the scope stays off; the
    mood pipeline runs normally."""
    out = await _run_scope_search(
        bothub_client, "рок",
        _extract_json(intent="mood",
                      target={"artist": "рок", "track": None},
                      signals={"language": "ru", "era": None, "vocal": None,
                               "genre": "rock", "mood": "бодрое"},
                      queries=["classic rock hits"]),
        _DM_POOL[:2],
        rank=_rank_json([("dm1", 0.9), ("dm2", 0.8)], best_index=0))
    assert not out.need_clarify
    assert out.candidates
    assert out.scope_artist == ""
    assert "scope_artist" not in out.signals


def test_scope_detection_unit_from_target_shape():
    """Unit on the exact detection rule as the real code states it: artist
    non-empty + track empty -> scope; track '' or None both count as
    empty; artist whitespace-only -> off. (The full-run equivalents above
    cover the same rule through search_candidates.)"""
    def _detect(tgt):
        return bool((tgt.get("artist") or "").strip()
                    and not (tgt.get("track") or "").strip())
    assert _detect({"artist": "Depeche Mode", "track": None})
    assert _detect({"artist": "U2", "track": ""})
    assert _detect({"artist": "U2"})
    assert not _detect({"artist": "Nirvana", "track": "Smells Like Teen Spirit"})
    assert not _detect({"artist": None, "track": None})
    assert not _detect({"artist": "  ", "track": None})


# ════ 3. pool guarantee: demotion and hard drop ════

@pytest.mark.asyncio
async def test_pool_with_5plus_matches_drops_covers(bothub_client, caplog):
    """8-candidate pool (6 artist matches incl. a live cut + 2 foreign
    covers), the LLM ranks a cover FIRST (the incident): >= 5 matches ->
    the covers are DROPPED outright, best moves to the top match."""
    caplog.set_level(logging.INFO, logger="bot.llm_search")
    rank = _rank_json([
        ("cov1", 0.95),  # the LLM's (wrong) best — a cover by another band
        ("dm1", 0.90),
        ("cov2", 0.85),
        ("dm2", 0.80),
        ("dm3", 0.70),
        ("dm4", 0.60),
        ("dm5", 0.50),
        ("dm_live", 0.40),
    ], best_index=0)
    out = await _run_scope_search(bothub_client, "depeche mode",
                                  _extract_json(), _DM_POOL, rank=rank)
    ids = [c.video_id for c in out.candidates]
    assert "cov1" not in ids and "cov2" not in ids, "5+ matches -> covers dropped"
    assert ids == ["dm1", "dm2", "dm3", "dm4", "dm5", "dm_live"]
    assert out.candidates[out.best_index].video_id == "dm1", \
        "the dropped cover's best slot moves to the top match"
    assert "artist scope: 6/8 candidates match 'Depeche Mode'" in caplog.text, \
        "the mandated one-line scope log"


@pytest.mark.asyncio
async def test_pool_with_2_matches_demotes_covers_below(bothub_client):
    """Only 2 matches in the pool: the covers SURVIVE (a thin pool keeps
    its fillers visible) but sit below EVERY match — the cover the LLM
    scored 0.95 is clamped under the weakest match (0.80) with the 0.05
    gap, i.e. 0.75, and best moves to the top match."""
    pool = _DM_POOL[:2] + _DM_POOL[6:8]  # dm1, dm2 + cov1, cov2
    rank = _rank_json([
        ("cov1", 0.95),
        ("dm1", 0.90),
        ("dm2", 0.80),
        ("cov2", 0.30),
    ], best_index=0)
    out = await _run_scope_search(bothub_client, "depeche mode",
                                  _extract_json(), pool, rank=rank)
    ids = [c.video_id for c in out.candidates]
    assert ids == ["dm1", "dm2", "cov1", "cov2"], \
        "covers stay, but strictly below both matches"
    by_id = {c.video_id: c for c in out.candidates}
    assert by_id["cov1"].score == pytest.approx(0.75)  # 0.80 - 0.05
    assert by_id["cov2"].score == pytest.approx(0.30)  # already below the floor
    assert out.candidates[out.best_index].video_id == "dm1"


@pytest.mark.asyncio
async def test_pool_all_matches_untouched(bothub_client):
    """An all-match pool: no drop, no demotion, the LLM's order stands."""
    pool = _DM_POOL[:3]
    rank = _rank_json([("dm2", 0.9), ("dm1", 0.8), ("dm3", 0.7)], best_index=0)
    out = await _run_scope_search(bothub_client, "depeche mode",
                                  _extract_json(), pool, rank=rank)
    assert [c.video_id for c in out.candidates] == ["dm2", "dm1", "dm3"]
    assert out.candidates[out.best_index].video_id == "dm2"
    assert out.scope_artist == "Depeche Mode"


# ════ 4. matching evidence (unit on _scope_match) ════

def test_scope_match_channel_topic_suffix():
    """«Depeche Mode - Topic» == scope «Depeche Mode» (the YouTube Music
    suffix is stripped before the comparison)."""
    c = Candidate("v1", "Enjoy the Silence", "Depeche Mode - Topic", 254, 100)
    assert _scope_match(c, "Depeche Mode")
    assert _scope_match({"video_id": "v1", "title": "Enjoy the Silence",
                         "channel": "Depeche Mode - Topic"}, "depeche mode")


def test_scope_match_canonical_artist():
    """A compilation channel is fine when the ENRICHED canonical artist
    names the scope artist (dicts carry canonical fields)."""
    c = {"video_id": "v1", "title": "Enjoy the Silence",
         "channel": "Compilation Shelf", "canonical_artist": "Depeche Mode"}
    assert _scope_match(c, "Depeche Mode")


def test_scope_match_fan_channel_needs_title_words():
    """A fan-cover channel does not match by channel; with the artist name
    in the TITLE as whole words it does match."""
    no = {"video_id": "v1", "title": "Enjoy the Silence (Cover)",
          "channel": "Cover Nation"}
    assert not _scope_match(no, "Depeche Mode")
    yes = {"video_id": "v2", "title": "depeche mode - Enjoy the Silence (Cover)",
           "channel": "Cover Nation"}
    assert _scope_match(yes, "Depeche Mode"), \
        "the artist named in the title (any case) IS his song's upload"


def test_scope_match_word_boundary_not_substring():
    """The artist name must appear as whole words — never a glued substring
    («Кино» vs «киношный», «U2» vs «v2rocket»)."""
    kino = {"video_id": "v1", "title": "киношный вайб", "channel": "Some Guy"}
    assert not _scope_match(kino, "Кино")
    u2 = {"video_id": "v2", "title": "v2rocket song", "channel": "Some Guy"}
    assert not _scope_match(u2, "U2")
    assert _scope_match({"video_id": "v3", "title": "U2 - Beautiful Day",
                         "channel": "Island Records"}, "U2")


# ════ 5. rank note (the LLM#2 input) ════

@pytest.mark.asyncio
async def test_rank_note_and_alien_marker_in_rank_input(bothub_client):
    """The ranker SEES the scope: the «Исполнитель-запрос:» line (next to
    «Источник:»), [чужой?] on every non-match, and the transport key never
    leaks into the Сигналы join."""
    captured = {}
    pool = _DM_POOL[:1] + _DM_POOL[6:8]  # dm1 + cov1, cov2
    await _run_scope_search(bothub_client, "depeche mode", _extract_json(),
                            pool, rank=_rank_json([("dm1", 0.9), ("cov1", 0.5)],
                                                  best_index=0),
                            captured=captured)
    ri = captured["rank_input"]
    assert "Исполнитель-запрос: Depeche Mode" in ri
    assert "все кандидаты должны быть его песнями, РАЗНЫЕ песни" in ri
    assert "[чужой?]" in ri
    dm_line = next(l for l in ri.splitlines() if "video_id=dm1" in l)
    cov_line = next(l for l in ri.splitlines() if "video_id=cov1" in l)
    assert "[чужой?]" not in dm_line
    assert "[чужой?]" in cov_line
    sig_line = next(l for l in ri.splitlines() if l.startswith("Сигналы:"))
    assert "scope_artist" not in sig_line


def test_format_for_rank_no_scope_line_without_scope():
    """Without signals['scope_artist'] the ranker message keeps its exact
    pre-14.11 shape (no note, no markers)."""
    out = _format_for_rank("песня из рекламы", {"genre": "rock"}, [
        {"video_id": "v1", "title": "Song", "channel": "Band",
         "duration": 200, "views": 10}])
    assert "Исполнитель-запрос:" not in out
    assert "[чужой?]" not in out


# ════ 6. adversarial battery ════

@pytest.mark.asyncio
async def test_russian_artist_kino_full_run(bothub_client):
    """«Кино» (RU, capitalized — formerly gated into clarify) searches and
    scopes; the RU pool matches by channel/title."""
    pool = [
        _entry("k1", "Кино - Группа крови", "Кино - Topic", 280, 10 ** 6),
        _entry("k2", "Кино - Звезда по имени Солнце", "Кино - Topic", 300, 9 * 10 ** 5),
        _entry("x1", "Группа крови кавер", "Кавер Студия", 260, 10 ** 5),
    ]
    extract = _extract_json(target={"artist": "Кино", "track": None},
                            signals={"language": "ru", "era": None, "vocal": None,
                                     "genre": None, "mood": None},
                            queries=["Кино лучшие песни"])
    rank = _rank_json([("x1", 0.9), ("k1", 0.85), ("k2", 0.7)], best_index=0)
    out = await _run_scope_search(bothub_client, "Кино", extract, pool, rank=rank)
    assert not out.need_clarify
    assert out.scope_artist == "Кино"
    ids = [c.video_id for c in out.candidates]
    assert ids[:2] == ["k1", "k2"], "matches lead the foreign cover"
    assert out.candidates[out.best_index].video_id == "k1"


def test_the_prodigy_stopword_the():
    """The stop-word "the" stays part of the identity: «The Prodigy - Topic»
    matches by channel; a bare «Prodigy» channel still matches via the
    title words."""
    assert _scope_match({"video_id": "v1", "title": "Firestarter",
                         "channel": "The Prodigy - Topic"}, "The Prodigy")
    assert _scope_match({"video_id": "v2",
                         "title": "The Prodigy - Firestarter",
                         "channel": "Prodigy"}, "The Prodigy")


@pytest.mark.asyncio
async def test_u2_digit_in_name_via_target(bothub_client):
    """«U2» — a digit in the name the OLD deterministic gate could never
    catch (digits were track markers); the LLM target path must: scope
    detected from target, pool matched, scope reported."""
    pool = [
        _entry("u1", "U2 - Beautiful Day", "U2 - Topic", 280, 10 ** 6),
        _entry("u2", "U2 - With or Without You", "U2 - Topic", 300, 9 * 10 ** 5),
        _entry("x1", "Beautiful Day Cover", "Acoustic Covers", 240, 10 ** 5),
    ]
    extract = _extract_json(target={"artist": "U2", "track": None},
                            queries=["U2 best songs"])
    rank = _rank_json([("x1", 0.9), ("u1", 0.85), ("u2", 0.7)], best_index=0)
    out = await _run_scope_search(bothub_client, "U2", extract, pool, rank=rank)
    assert out.scope_artist == "U2"
    ids = [c.video_id for c in out.candidates]
    assert ids[:2] == ["u1", "u2"]
    assert out.candidates[out.best_index].video_id == "u1"


def test_pink_punctuation_via_target_detect():
    """«P!nk» — punctuation in the name; _ref_key normalizes both sides to
    «p nk», so the channel and the title word-run both match."""
    assert _scope_match({"video_id": "v1", "title": "So What",
                         "channel": "P!nk - Topic"}, "P!nk")
    assert _scope_match({"video_id": "v2", "title": "P!nk - So What",
                         "channel": "Fan Channel"}, "P!nk")


# ════ 9. the target artist survives its own exclusion list (web port) ════

@pytest.mark.asyncio
async def test_scope_never_bans_target_artist_via_exclusions(bothub_client, caplog):
    """(2026-11-14, web port) The web frontend passes the rejected cards'
    artists back verbatim as excluded_artist — in artist-only mode every
    card's artist IS the target band. The core strips the scope key from
    excluded_artists before the pool filter: the band survives its own
    «Не то» chain while a foreign rejected channel stays banned."""
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out = await _run_scope_search(
            bothub_client, "depeche mode", _extract_json(), _DM_POOL,
            rank=_rank_json([("dm1", 0.9), ("dm2", 0.8)], best_index=0),
            excluded_artists={"depeche mode", "cover nation"})
    # the band is NOT banned by its own key: its songs survive
    assert {c.video_id for c in out.candidates} & {"dm1", "dm2", "dm3", "dm4"}
    # the foreign ban still bites: both covers are gone
    assert not {c.video_id for c in out.candidates} & {"cov1", "cov2"}
    # the guard logged its decision
    assert "target artist kept out of the channel ban" in caplog.text
