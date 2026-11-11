"""Tests for the SimilarPipeline (Stage 2a of the pipeline refactor).

Contract under test:
  - routing: "similar" and its normalized variants reach ONE shared,
    stateless SimilarPipeline; from_source, None and unknown intents stay
    on the shared LegacyPipeline;
  - behavior-preserving move from LegacyPipeline.run: the _resolve_reference
    block and the getSimilar-neighbors query injection (with the ref_key
    self-exclusion) now live in SimilarPipeline; discovery/filter/rank reuse
    the shared legacy helpers — the similar flow keeps its SearchOut shape
    (ranked candidates, the reference banned to the tail, the LLM queries
    preserved after the neighbors);
  - the LLM#2 rank call receives SYSTEM_RANK_SIMILAR as the system prompt
    (Stage 2a prompt split; selected by normalized intent, so a variant
    spelling like "similar_tracks" ranks under the same prompt), while
    LLM#1 keeps SYSTEM_EXTRACT;
  - the shared artist diversity guard (mood's «не три трека одного
    исполнителя» rule, acceptance s3: four Jethro Tull tracks in a top-5):
    3+ of the top-6 filtered candidates from one canonical artist — the
    "Artist - Track" title split and the "Artist - Topic" channel fallback
    unify into ONE key — keep only the best 2 before ranking, enriched
    trimmed consistently; two same-artist candidates never trigger it.

Mocking follows tests/test_llm_search.py: BothubClient.chat_json patched on
the bothub_client fixture (already-parsed dicts, dispatched by gen_name),
search_metadata patched in services.llm_search, no network. The canned
fixtures mirror the similar tests of test_llm_search.py.
"""
import logging

import pytest
from unittest.mock import patch

from services.llm_search import (
    SYSTEM_EXTRACT,
    SYSTEM_RANK_SIMILAR,
    SYSTEM_RANK_STRICT,
    search_candidates,
)
from services.pipelines._diversity import artist_diversity_guard
from services.pipelines import route
from services.pipelines.legacy import LegacyPipeline
from services.pipelines.similar import SimilarPipeline


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Same guard as test_llm_search.py: the enrich step must not touch the
    network (candidates here are fake)."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers (canned fixtures from test_llm_search.py, similar tests) ════

def _extract_json(intent="similar", reference=None, target=None, queries=None):
    """LLM#1 extract — a fake parsed dict (similar flavor)."""
    return {
        "intro": "Ищем в этом вайбе...",
        "intent": intent,
        "signals": {"language": "ru", "era": None, "vocal": "instrumental",
                    "genre": "rock", "mood": None},
        "target": target or {"artist": None, "track": None},
        "reference": reference or {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries or ["test query 1", "test query 2"],
    }


def _rank_json(best_index=0, confidence="high", need_clarify=False):
    """LLM#2 rank — a fake parsed dict."""
    return {
        "ranked": [
            {"video_id": "v1", "score": 0.95, "rationale": "Отличный грув."},
            {"video_id": "v2", "score": 0.80, "rationale": "Близко по жанру."},
            {"video_id": "v3", "score": 0.70, "rationale": "Запасной вариант."},
        ],
        "best_index": best_index,
        "confidence": confidence,
        "verdict": "Этот трек — то что надо.",
        "need_clarify": need_clarify,
        "clarify_question": "Уточни жанр?" if need_clarify else "",
    }


def _yt_entries():
    """Fake yt-dlp results (3 candidates)."""
    return [
        {"video_id": "v1", "title": "Cool Rock Song", "channel": "Rock Band",
         "duration": 240, "views": 500000},
        {"video_id": "v2", "title": "Another Rock Track", "channel": "Other Band",
         "duration": 300, "views": 100000},
        {"video_id": "v3", "title": "Soft Rock Ballad", "channel": "Soft Band",
         "duration": 270, "views": 50000},
    ]


class _MetaSvcQ89:
    """track_search knows the track, getSimilar knows the canonical artist's
    neighbors, while LLM#1 put a non-musical context into reference.artist.
    Mirrors the fixture from test_llm_search.py (canonical: which artist the
    neighbors belong to)."""
    _lastfm_key = "k"

    def __init__(self, matches=None, neighbors=None, canonical="Bobby Vee"):
        self._matches = matches if matches is not None else [
            {"artist": "Bobby Vee", "listeners": 400000},
            {"artist": "The Carpenters", "listeners": 50000},
        ]
        self._neighbors = neighbors if neighbors is not None else [
            "Rick Nelson", "Del Shannon", "Bobby Vee"
        ]
        self._canonical = canonical
        self.calls = []

    async def track_search(self, track, limit=8):
        self.calls.append(("track_search", track))
        return self._matches

    async def similar_artists(self, artist, limit=10):
        self.calls.append(("getSimilar", artist))
        return self._neighbors if artist == self._canonical else []


def _chat_spy(captured, extract):
    """A chat_json side effect recording (gen_name, system) per call and
    answering with the canned extract / rank dicts."""
    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault("calls", []).append((gen_name, system))
        if gen_name == "llm-extract":
            return extract
        return _rank_json(best_index=0)
    return fake_chat


# ════ 1. routing contract ════

def test_route_similar_to_dedicated_pipeline():
    """'similar' (and its normalized variants) -> ONE shared SimilarPipeline;
    from_source, None and unknown intents stay on the shared legacy one."""
    p = route("similar")
    assert isinstance(p, SimilarPipeline)
    assert p.name == "similar"
    # variants normalize to the same shared instance
    assert route("Similar") is p
    assert route("similar_tracks") is p
    assert route("  SIMILAR_MUSIC ") is p
    # the non-similar keys stay legacy (one shared instance); "mood" has its
    # own pipeline since Stage 2b, so a genuinely unknown string stands in
    legacy = route(None)
    assert isinstance(legacy, LegacyPipeline)
    assert route(None) is legacy
    assert route("unknown_intent") is legacy
    assert p is not legacy


def test_similar_pipeline_is_stateless():
    """One shared instance with no mutable instance attributes: safe for
    concurrent chats (the registry contract)."""
    p = route("similar")
    mutable = {k: v for k, v in vars(p).items()
               if isinstance(v, (list, dict, set))}
    assert not mutable, "the pipeline must hold no mutable instance state"


# ════ 2. e2e: the moved flow keeps the SearchOut shape ════

@pytest.mark.asyncio
async def test_similar_e2e_shape_and_rank_prompt(bothub_client):
    """The plain similar flow (no reference fields): resolve does not start,
    the LLM queries search as-is, and the SearchOut carries the same fields
    as before the move — ranked candidates, best_index, verdict. The rank
    call gets SYSTEM_RANK_SIMILAR; LLM#1 still gets SYSTEM_EXTRACT."""
    captured = {}
    extract = _extract_json(intent="similar")

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_spy(captured, extract)):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                captured["queries"] = list(queries)
                return _yt_entries()
            mock_yt.side_effect = cap
            out = await search_candidates("Хочу рок помягче", bothub_client)

    # SearchOut shape (assertions of the pre-split test_similar_search)
    assert out.intent == "similar"
    assert out.confidence == "high"
    assert len(out.candidates) == 3
    assert not out.need_clarify
    assert out.best_index == 0
    assert out.candidates[0].match_score > 0
    assert out.verdict == "Этот трек — то что надо."
    # the LLM queries search as-is (no neighbors to inject)
    assert captured["queries"] == ["test query 1", "test query 2"]
    # prompt selection: extract untouched, rank = the similar split
    assert [s for g, s in captured["calls"] if g == "llm-extract"] == [SYSTEM_EXTRACT]
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_SIMILAR]


@pytest.mark.asyncio
async def test_reference_resolved_neighbors_injected_reference_banned(bothub_client):
    """The moved orchestration end to end (the q89 scenario from
    test_llm_search.py): the non-musical reference.artist resolves via
    track.search, the canonical artist's neighbors lead the queries (the LLM
    guesses preserved after them), and the reference itself is banned to the
    tail of the results."""
    svc = _MetaSvcQ89()
    captured = {}
    extract = _extract_json(
        intent="similar",
        reference={"artist": "Jennifer Connelly",
                   "track": "The Night Has a Thousand Eyes"},
        queries=["Mazzy Star Fade Into You", "Portishead Roads"],
    )

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_spy(captured, extract)):
        with patch("services.llm_search.search_metadata") as mock_yt:
            entries = [
                {"video_id": "v1", "title": "The Night Has a Thousand Eyes",
                 "channel": "Bobby Vee - Topic", "duration": 150, "views": 800000},
                {"video_id": "v2", "title": "Travelin' Man",
                 "channel": "Rick Nelson - Topic", "duration": 160, "views": 300000},
                {"video_id": "v3", "title": "Fade Into You",
                 "channel": "Mazzy Star", "duration": 280, "views": 9000000},
            ]

            async def cap(queries, per_query=20, on_progress=None):
                # both the resolve step (per_query=3) and discovery
                # (per_query=20) see the same pool — same as the canned mock
                # in test_llm_search.py
                if per_query == 20:
                    captured["queries"] = list(queries)
                return entries
            mock_yt.side_effect = cap
            out = await search_candidates(
                "Jennifer Connelly the night has thousands eyes найди похожий трек",
                bothub_client, metadata_svc=svc)

    # resolve ran: track.search + getSimilar on the canonical artist
    assert ("track_search", "The Night Has a Thousand Eyes") in svc.calls
    assert ("getSimilar", "Bobby Vee") in svc.calls
    # neighbors lead discovery, the LLM's guesses stay after them
    assert captured["queries"][0] in ("Rick Nelson", "Del Shannon")
    assert "Mazzy Star Fade Into You" in captured["queries"]
    # the reference (Bobby Vee) is banned from the best, pushed to the tail
    best = out.candidates[out.best_index]
    assert best.video_id != "v1"
    assert out.candidates[-1].video_id == "v1"
    # and the rank call under the similar prompt
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_SIMILAR]


@pytest.mark.asyncio
async def test_rank_payload_marks_empirical_neighbors(bothub_client):
    """Doctrine s3, layer 2 e2e: the candidates list the LLM#2 ranker sees
    carries a deterministic [neighbor?] marker on the candidates whose
    channel IS a getSimilar neighbor of the resolved reference — and only on
    them (the genre-mate and the banned reference carry no marker)."""
    svc = _MetaSvcQ89(
        matches=[{"artist": "The Moody Blues", "listeners": 900000}],
        neighbors=["Barclay James Harvest", "The Strawbs"],
        canonical="The Moody Blues",
    )
    captured = {}
    extract = _extract_json(
        intent="similar",
        reference={"artist": "The Moody Blues", "track": "Nights in White Satin"},
        queries=["Procol Harum A Whiter Shade of Pale"],
    )

    async def spy_chat(system, user, max_tokens=None, trace=None,
                       gen_name="llm", model=None):
        captured.setdefault("calls", []).append((gen_name, system, user))
        if gen_name == "llm-extract":
            return extract
        return _rank_json(best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=spy_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                if per_query == 20:
                    return [
                        {"video_id": "v1", "title": "Poor Man's Moody Blues",
                         "channel": "Barclay James Harvest - Topic", "duration": 240, "views": 50000},
                        {"video_id": "v2", "title": "A Whiter Shade of Pale",
                         "channel": "Procol Harum - Topic", "duration": 240, "views": 900000},
                        {"video_id": "v3", "title": "Nights in White Satin",
                         "channel": "The Moody Blues - Topic", "duration": 280, "views": 900000},
                    ]
                return []
            mock_yt.side_effect = cap
            await search_candidates("песня как Nights in White Satin",
                                    bothub_client, metadata_svc=svc)

    rank_payloads = [u for g, s, u in captured["calls"] if g == "llm-rank"]
    assert len(rank_payloads) == 1
    payload = rank_payloads[0]
    v1_line = next(l for l in payload.splitlines() if "video_id=v1" in l)
    v2_line = next(l for l in payload.splitlines() if "video_id=v2" in l)
    v3_line = next(l for l in payload.splitlines() if "video_id=v3" in l)
    # the empirical neighbor is marked, the marker prepended to the title
    assert "| [neighbor?] Poor Man's Moody Blues |" in v1_line
    # the genre answer (same era/genre, not a getSimilar neighbor) is not
    assert "[neighbor?]" not in v2_line
    # the reference itself is not (getSimilar never returns the reference)
    assert "[neighbor?]" not in v3_line


@pytest.mark.asyncio
async def test_neighbor_self_exclusion_from_queries(bothub_client):
    """getSimilar never returns the reference itself, but the normalization
    guard is free: a neighbor matching the resolved artist (by ref_key) is
    NOT injected into the queries."""
    svc = _MetaSvcQ89(
        matches=[{"artist": "The Moody Blues", "listeners": 900000}],
        neighbors=["The Moody Blues", "Barclay James Harvest"],
        canonical="The Moody Blues",
    )
    captured = {}
    extract = _extract_json(
        intent="similar",
        reference={"artist": "The Moody Blues", "track": "Nights in White Satin"},
        queries=["Mazzy Star Fade Into You"],
    )

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_spy(captured, extract)):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                if per_query == 20:
                    captured["queries"] = list(queries)
                    return _yt_entries()
                return []
            mock_yt.side_effect = cap
            await search_candidates("похожее на Nights in White Satin",
                                    bothub_client, metadata_svc=svc)

    assert "Barclay James Harvest" in captured["queries"]
    assert "The Moody Blues" not in captured["queries"], \
        "the reference itself must not lead discovery"


@pytest.mark.asyncio
async def test_no_reference_fields_no_resolution(bothub_client):
    """similar without reference fields: resolution does not start (neither
    track_search nor getSimilar is called) — the gating moved unchanged."""
    svc = _MetaSvcQ89()
    captured = {}
    extract = _extract_json(intent="similar",
                            reference={"artist": None, "track": None})

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_spy(captured, extract)):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                captured["queries"] = list(queries)
                return _yt_entries()
            mock_yt.side_effect = cap
            out = await search_candidates("что-нибудь dream-pop", bothub_client,
                                          metadata_svc=svc)

    assert svc.calls == []
    assert captured["queries"] == ["test query 1", "test query 2"]
    assert out.candidates[out.best_index].video_id == "v1"


# ════ 3. rank-prompt selection at the shared call site ════

@pytest.mark.asyncio
async def test_variant_intent_ranks_under_similar_prompt(bothub_client):
    """The rank prompt is selected by the NORMALIZED intent: a variant
    spelling ("similar_tracks") routes to SimilarPipeline and its LLM#2 call
    still receives SYSTEM_RANK_SIMILAR."""
    captured = {}
    extract = _extract_json(intent="similar_tracks")

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_spy(captured, extract)):
        with patch("services.llm_search.search_metadata", return_value=_yt_entries()):
            out = await search_candidates("Хочу рок помягче", bothub_client)

    assert out.intent == "similar_tracks"
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_SIMILAR]


@pytest.mark.asyncio
async def test_this_track_fallthrough_ranks_under_strict_prompt(bothub_client, monkeypatch):
    """The rank call site is shared with this_track: its fall-through path
    (no confident fuzzy shortcut) must receive SYSTEM_RANK_STRICT."""
    import services.fuzzy as fuzzy_mod
    monkeypatch.setattr(fuzzy_mod, "pair_match", lambda *a: 0.0)  # block the shortcut

    captured = {}
    extract = _extract_json(intent="this_track",
                            target={"artist": "Michael Jackson",
                                    "track": "Billie Jean"})

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_spy(captured, extract)):
        with patch("services.llm_search.search_metadata", return_value=_yt_entries()):
            out = await search_candidates("Michael Jackson Billie Jean", bothub_client)

    assert out.intent == "this_track"
    assert len(out.candidates) == 3  # the rank tail ran
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_STRICT]


# ════ 4. diversity guard (the mood rule, shared via services.pipelines._diversity) ════

def _cand_div(vid: str, title: str, channel: str, views: int = 1000) -> dict:
    """A fake candidate with an explicit title: the artist-keyed guard reads
    the 'Artist - Track' title split first, the channel minus '- Topic' as
    the fallback."""
    return {"video_id": vid, "title": title, "channel": channel,
            "duration": 240, "views": views}


def test_diversity_guard_unifies_title_split_and_channel_keys(caplog):
    """Acceptance s3: one artist held 4 of the top-6 slots — two candidates
    keyed by the 'Artist - Track' title split, two by the 'Artist - Topic'
    channel fallback. The shared artist key counts them as ONE artist: only
    the best 2 survive, the other artists stay in place, the enriched list
    drops the same video_ids, and the log line carries the similar label."""
    filtered = [
        _cand_div("v1", "Jethro Tull - Aqualung", "Jethro Tull - Topic", 500),
        _cand_div("v2", "Locomotive Breath", "Jethro Tull - Topic", 400),
        _cand_div("v3", "Other Band - Song", "Other Band - Topic", 300),
        _cand_div("v4", "Jethro Tull - Bungle in the Jungle", "JethroTullVEVO", 200),
        _cand_div("v5", "Thick as a Brick", "Jethro Tull - Topic", 100),
        _cand_div("v6", "Third Band - Tune", "Third Band - Topic", 90),
    ]
    enriched = [dict(c) for c in filtered[:5]]

    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        kept, enr = artist_diversity_guard(filtered, enriched, label="similar")

    # two title-split + two channel-fallback candidates of one artist -> the best 2
    assert [c["video_id"] for c in kept] == ["v1", "v2", "v3", "v6"], \
        "the third and fourth same-artist candidates must drop (both key paths)"
    # the enriched copies drop the same video_ids
    assert {c["video_id"] for c in enr} == {"v1", "v2", "v3"}
    assert "similar diversity" in caplog.text
    assert "kept 2 each (by artist)" in caplog.text


def test_diversity_guard_two_same_artist_below_trigger():
    """Two same-artist candidates never reach the 3+ trigger: both lists come
    back unchanged (identity — mood's no-trigger contract holds for similar
    too)."""
    filtered = [
        _cand_div("v1", "Jethro Tull - Aqualung", "Jethro Tull - Topic", 500),
        _cand_div("v2", "Other Band - Song", "Other Band - Topic", 400),
        _cand_div("v3", "Jethro Tull - Locomotive Breath", "JethroTullVEVO", 300),
        _cand_div("v4", "Third Band - Tune", "Third Band - Topic", 200),
    ]
    enriched = [dict(c) for c in filtered[:2]]
    kept, enr = artist_diversity_guard(filtered, enriched, label="similar")
    assert kept is filtered and enr is enriched
    assert len(kept) == 4


@pytest.mark.asyncio
async def test_diversity_guard_caps_crowded_artist_before_rank(bothub_client, caplog):
    """Acceptance s3, e2e: four same-artist candidates reach the filter — the
    guard trims the pool to 2 BEFORE ranking, so the LLM#2 rank payload (and
    the SearchOut) carries at most 2 of the crowded artist; the untouched
    other artist still ranks."""
    captured = {}
    extract = _extract_json(intent="similar")
    entries = [
        {"video_id": "v1", "title": "Jethro Tull - Aqualung",
         "channel": "Jethro Tull - Topic", "duration": 240, "views": 500000},
        {"video_id": "v2", "title": "Locomotive Breath",
         "channel": "Jethro Tull - Topic", "duration": 240, "views": 400000},
        {"video_id": "v3", "title": "Jethro Tull - Bungle in the Jungle",
         "channel": "JethroTullVEVO", "duration": 240, "views": 300000},
        {"video_id": "v4", "title": "Thick as a Brick",
         "channel": "Jethro Tull - Topic", "duration": 240, "views": 200000},
        {"video_id": "v5", "title": "A Whiter Shade of Pale",
         "channel": "Procol Harum - Topic", "duration": 240, "views": 100000},
    ]

    async def spy_chat(system, user, max_tokens=None, trace=None,
                       gen_name="llm", model=None):
        captured.setdefault("calls", []).append((gen_name, system, user))
        if gen_name == "llm-extract":
            return extract
        return _rank_json(best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=spy_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                return entries if per_query == 20 else []
            mock_yt.side_effect = cap
            with caplog.at_level(logging.INFO, logger="bot.llm_search"):
                out = await search_candidates("песня как Nights in White Satin",
                                              bothub_client)

    # the rank payload holds at most 2 candidate lines of the crowded artist
    rank_payloads = [u for g, s, u in captured["calls"] if g == "llm-rank"]
    assert len(rank_payloads) == 1
    tull_lines = [l for l in rank_payloads[0].splitlines()
                  if "video_id=" in l and "Jethro Tull" in l]
    assert len(tull_lines) == 2
    # the untouched other artist still reaches the ranker
    assert any("Procol Harum" in l for l in rank_payloads[0].splitlines())
    # the SearchOut carries the trimmed pool too (title OR channel: the
    # channel-keyed candidate's title need not name the artist)
    tull_out = [c for c in out.candidates
                if "Jethro Tull" in (c.title or "") or "Jethro Tull" in (c.channel or "")]
    assert len(tull_out) == 2
    # one guard log line under the similar label
    assert "similar diversity" in caplog.text


# ════ 5. miss-streak restart (live-test F2, 2026-11-05) ════

_F2_NEIGHBORS = ["Justin Hayward", "Procol Harum", "Jethro Tull",
                 "King Crimson", "The Alan Parsons Project", "Genesis"]


def _f2_svc():
    """getSimilar(The Moody Blues) -> six canned neighbors (the live shape:
    the restart must be able to slide past the first window)."""
    return _MetaSvcQ89(
        matches=[{"artist": "The Moody Blues", "listeners": 900000}],
        neighbors=list(_F2_NEIGHBORS),
        canonical="The Moody Blues",
    )


@pytest.mark.asyncio
async def test_zero_streak_keeps_first_neighbor_window(bothub_client):
    """Default restart_streak=0: the window is the FIRST _NEIGHBOR_QUERIES
    neighbors, byte-identical to today's discovery (F2 leaves the no-restart
    path untouched)."""
    svc = _f2_svc()
    captured = {}
    extract = _extract_json(
        intent="similar",
        reference={"artist": "The Moody Blues", "track": "Nights in White Satin"},
        queries=["Mazzy Star Fade Into You"],
    )
    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_spy(captured, extract)):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                if per_query == 20:
                    captured["queries"] = list(queries)
                    return _yt_entries()
                return []
            mock_yt.side_effect = cap
            await search_candidates("похожее на Nights in White Satin",
                                    bothub_client, metadata_svc=svc)

    assert captured["queries"][:3] == _F2_NEIGHBORS[:3]
    assert captured["queries"][3] == "Mazzy Star Fade Into You"


@pytest.mark.asyncio
async def test_restart_streak_rotates_neighbor_window(bothub_client):
    """F2: with restart_streak=2 (two «Не то») the discovery window slides to
    neighbors[2:2+_NEIGHBOR_QUERIES] — the NEXT three neighbors instead of
    the same first three that produced the rejected pool."""
    svc = _f2_svc()
    captured = {}
    extract = _extract_json(
        intent="similar",
        reference={"artist": "The Moody Blues", "track": "Nights in White Satin"},
        queries=["Procol Harum A Whiter Shade of Pale", "Mazzy Star Roads"],
    )
    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_spy(captured, extract)):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                if per_query == 20:
                    captured["queries"] = list(queries)
                    return _yt_entries()
                return []
            mock_yt.side_effect = cap
            await search_candidates("похожее на Nights in White Satin",
                                    bothub_client, metadata_svc=svc,
                                    restart_streak=2)

    # the window slid: neighbors[2:5] lead, the first-window names never ran
    assert captured["queries"][:3] == _F2_NEIGHBORS[2:5]
    assert "Justin Hayward" not in captured["queries"]
    assert "Procol Harum" not in captured["queries"]
    # the LLM guesses stay after the rotated neighbors (no echo exclusions here)
    assert captured["queries"][3:] == ["Procol Harum A Whiter Shade of Pale",
                                       "Mazzy Star Roads"]


@pytest.mark.asyncio
async def test_restart_echo_guard_drops_llm_query_echoes(bothub_client):
    """F2: on a restart with excluded_titles the LLM#1 guesses echoing the
    rejected song/artist are dropped before discovery — the 'Procol Harum
    A Whiter Shade of Pale' query must not re-run."""
    svc = _f2_svc()
    captured = {}
    extract = _extract_json(
        intent="similar",
        reference={"artist": "The Moody Blues", "track": "Nights in White Satin"},
        queries=["Justin Hayward", "Procol Harum", "Jethro Tull",
                 "Procol Harum A Whiter Shade of Pale", "King Crimson Epitaph"],
    )
    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_spy(captured, extract)):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                if per_query == 20:
                    captured["queries"] = list(queries)
                    return _yt_entries()
                return []
            mock_yt.side_effect = cap
            await search_candidates("похожее на Nights in White Satin",
                                    bothub_client, metadata_svc=svc,
                                    excluded_titles={
                                        "A Whiter Shade of Pale (2007 Remaster)",
                                        "PROCOL HARUM - A Whiter Shade of Pale"},
                                    excluded_artists={"procol harum", "fly records"})

    assert "Procol Harum" not in captured["queries"]
    assert "Procol Harum A Whiter Shade of Pale" not in captured["queries"]
    # untouched guesses survive
    assert "King Crimson Epitaph" in captured["queries"]
    assert "Mazzy Star" not in captured["queries"]  # sanity: no phantoms added


@pytest.mark.asyncio
async def test_restart_full_f2_scenario(bothub_client, caplog):
    """F2 end to end: streak-2 restart with title exclusions — rotated
    neighbor window, echo-guarded LLM queries, and the same-title candidates
    (other channels/uploads) dropped from the pool before ranking."""
    import logging
    svc = _f2_svc()
    captured = {}
    extract = _extract_json(
        intent="similar",
        reference={"artist": "The Moody Blues", "track": "Nights in White Satin"},
        queries=["Justin Hayward", "Procol Harum", "Jethro Tull",
                 "Procol Harum A Whiter Shade of Pale", "King Crimson Epitaph"],
    )
    entries = [
        # the same rejected song from two channels the artist/channel
        # excludes cannot catch (MrLodurr / Beat-Bootleg are not excluded)
        {"video_id": "v1", "title": "PROCOL HARUM - A Whiter Shade of Pale",
         "channel": "MrLodurr", "duration": 240, "views": 900000},
        {"video_id": "v2", "title": "A Whiter Shade of Pale",
         "channel": "Beat-Bootleg Channel", "duration": 240, "views": 800000},
        # a fresh direction candidate: survives
        {"video_id": "v3", "title": "Justin Hayward - Nights in White Satin",
         "channel": "Justin Hayward - Topic", "duration": 280, "views": 500000},
    ]
    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_spy(captured, extract)):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                if per_query == 20:
                    captured["queries"] = list(queries)
                    return entries
                return []
            mock_yt.side_effect = cap
            with caplog.at_level(logging.INFO, logger="bot.llm_search"):
                out = await search_candidates(
                    "песня как night in white sating\nНе то: A Whiter Shade of Pale",
                    bothub_client, metadata_svc=svc,
                    excluded_ids={"hxa6GUIkNac", "z0vCwGUZe1I"},
                    excluded_artists={"procol harum", "fly records"},
                    excluded_titles={"A Whiter Shade of Pale (2007 Remaster)",
                                     "PROCOL HARUM - A Whiter Shade of Pale"},
                    restart_streak=2)

    # rotated window leads, no echoes of the rejected song/artist anywhere
    assert captured["queries"][:3] == _F2_NEIGHBORS[2:5]
    assert not any("Whiter" in q or q == "Procol Harum" for q in captured["queries"])
    # the same-title uploads are gone; the fresh candidate is the pool
    assert [c.video_id for c in out.candidates] == ["v3"]
    assert "title exclusion: dropped 2 same-title candidate(s)" in caplog.text
    assert "echo guard: dropped 2 restart query(ies)" in caplog.text


# ════ 6. B1 (research П4): sonar rescue of a low-confidence channel ════

_B1_NEIGHBORS = ["David Lindley", "Lowell George", "Peter Case",
                 "Flaco Jiménez", "John Hiatt", "Jorge Calderón"]


@pytest.mark.asyncio
async def test_b1_sonar_rescue_resolves_reference_and_leads_discovery(bothub_client, caplog):
    """The live П4 case e2e («похожее на Ry Cooder - Dior wild man», a DIOR-ad
    fan upload unknown to Last.fm): Last.fm misses, the only yt-dlp channel is
    an aggregator the credibility filter rejected -> one sonar question
    rescues Ry Cooder (source="sonar"), HIS getSimilar neighbors lead
    discovery, the rescued reference is banned to the tail, and the
    low-confidence warning stays silent."""
    import logging
    bothub_client._sonar_model = "sonar"
    svc = _MetaSvcQ89(matches=[], neighbors=list(_B1_NEIGHBORS),
                      canonical="Ry Cooder")
    captured = {}
    extract = _extract_json(
        intent="similar",
        reference={"artist": "Ry Cooder", "track": "Dior wild man"},
        queries=["slide guitar blues"],
    )

    async def spy_chat(system, user, max_tokens=None, trace=None,
                       gen_name="llm", model=None):
        captured.setdefault("calls", []).append((gen_name, system, user))
        if gen_name == "llm-extract":
            return extract
        return _rank_json(best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=spy_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                if per_query == 3:
                    # the resolve probe: only an aggregator carries the title
                    # (its channel has neither "ry" nor "cooder" as a token)
                    return [{"video_id": "r1",
                             "title": "Dior wild man (Full Official Track)",
                             "channel": "BluesMen Channel - Blues Collection",
                             "duration": 91, "views": 50000}]
                captured["queries"] = list(queries)
                return [
                    {"video_id": "v1", "title": "Mercury Blues",
                     "channel": "David Lindley - Topic", "duration": 260, "views": 400000},
                    {"video_id": "v2", "title": "Dior wild man",
                     "channel": "Ry Cooder - Topic", "duration": 91, "views": 900000},
                ]
            mock_yt.side_effect = cap
            with patch("services.llm_search._sonar_ground") as sg:
                async def sg_hit(b, question, trace=None):
                    captured["sonar_q"] = question
                    return {"artist": "Ry Cooder", "track": "Dior wild man",
                            "sure": True}
                sg.side_effect = sg_hit
                with caplog.at_level(logging.INFO, logger="bot.llm_search"):
                    out = await search_candidates(
                        "похожее на Ry Cooder Dior wild man", bothub_client,
                        metadata_svc=svc)

    # the sonar rescue fired with the П4 question and resolved the reference
    assert "похожее на трек: Ry Cooder - Dior wild man" in captured["sonar_q"]
    assert "sonar rescued" in caplog.text
    assert "channel fallback kept low-confidence" not in caplog.text
    assert ("getSimilar", "Ry Cooder") in svc.calls
    # the rescued artist's neighbors lead discovery (LLM guess kept after)
    assert captured["queries"][:3] == _B1_NEIGHBORS[:3]
    assert "slide guitar blues" in captured["queries"]
    # the reference itself is banned to the tail, a neighbor wins
    best = out.candidates[out.best_index]
    assert best.video_id == "v1"
    assert out.candidates[-1].video_id == "v2"
