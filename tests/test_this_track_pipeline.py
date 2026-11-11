"""Tests for the ThisTrackPipeline (Stage 1 of the pipeline refactor).

Contract under test:
  - confident-match shortcut: a concrete target + a decisive fuzzy best
    (>= FUZZY_SHORTCUT_MIN, gap to a DIFFERENT-track_key runner-up >=
    AMBIGUITY_GAP) skips the LLM#2 chat_json call entirely;
  - no shortcut when the best fuzzy score is under the threshold or a
    different-track runner-up is within AMBIGUITY_GAP (ambiguity -> LLM#2);
  - a noise-flagged best (tribute cover) never short-cuts -> LLM#2;
  - duplicate uploads of the SAME track (same track_key) never count as a
    runner-up — the same dedup logic is_ambiguous uses;
  - artist recovery from the track string: an artist-less target whose
    track string IS the whole query («Deoche mode») recovers the artist from
    the best candidate's channel and fires the same shortcut path (noise
    and different-artist gap guards included; same-artist different-track
    candidates are agreement, not ambiguity);
  - the shortcut adds the "shortcut" trace tag next to the intent tag;
  - superlative fact-resolve: an artist-only this_track query with a
    superlative marker resolves the concrete track via sonar BEFORE query
    expansion; the sonar question asks for the fact the marker names
    (debut / last / most famous); a sonar miss or a disabled layer keeps
    the old behavior;
  - superlative FACT-CHECK (bake-off t4): when LLM#1 already filled
    target.track, the marker still fires — exactly one sonar call either
    way; sonar agreeing changes nothing, sonar's confident DIFFERENT track
    wins (corrected target + leading query + log line), a sonar miss keeps
    the LLM fill (logged);
  - route() normalization: variant spellings reach the right pipeline;
  - yt-suggest query normalization (B3): canonical spellings from YouTube's
    search-suggest join the pool right after the primary query (one suggest
    call on "{artist} {track}", similarity-gated, top-2, fail-soft — a
    down/raising/slow suggest layer never changes the search behavior).

Mocking follows tests/test_llm_search.py: chat_json patched on the
bothub_client fixture (already-parsed dicts, dispatched by gen_name),
search_metadata patched in services.llm_search, no network. Fuzzy scores:
this_track.py lazy-imports the services.fuzzy functions at call time, so
tests that need exact scores patch pair_match on services.fuzzy directly.
"""
import pytest
from unittest.mock import patch

from services.llm_search import search_candidates
from services.pipelines import route
from services.pipelines.this_track import FUZZY_SHORTCUT_MIN, ThisTrackPipeline
from services.pipelines.legacy import LegacyPipeline
from services.match_score import AMBIGUITY_GAP
from services.tracing import SearchTrace


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Same guard as test_llm_search.py: the enrich step must not touch the
    network (candidates here are fake)."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers ════

def _extract_json(intent="this_track", target=None, queries=None):
    """LLM#1 extract — a fake parsed dict (this_track flavor)."""
    return {
        "intro": "Looking for that exact track...",
        "intent": intent,
        "signals": {"language": "en", "era": None, "vocal": None, "genre": None, "mood": None},
        "target": target or {"artist": None, "track": None},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries or ["depeche mode enjoy the silence"],
    }


def _rank_json(best_index=0, confidence="high"):
    """LLM#2 rank — a fake parsed dict over the canned Depeche Mode ids."""
    return {
        "ranked": [
            {"video_id": "dm1", "score": 0.90, "rationale": "The original studio cut."},
            {"video_id": "dm2", "score": 0.70, "rationale": "A different track."},
        ],
        "best_index": best_index,
        "confidence": confidence,
        "verdict": "The original from the Violator album.",
        "need_clarify": False,
        "clarify_question": "",
    }


def _dm_entries():
    """The right candidate + a clearly different runner-up (different songs)."""
    return [
        {"video_id": "dm1", "title": "Depeche Mode - Enjoy the Silence",
         "channel": "Depeche Mode - Topic", "duration": 254, "views": 10**6},
        {"video_id": "dm2", "title": "Depeche Mode - Personal Jesus",
         "channel": "Depeche Mode - Topic", "duration": 260, "views": 9 * 10**5},
    ]


def _fuzzy_pair_by_title(monkeypatch, by_title: dict):
    """Canned fuzzy.pair_match keyed by the candidate's raw title. The
    pipeline resolves fuzzy.pair_match lazily at call time (from services
    .fuzzy), so patching the module attribute takes effect per call."""
    import services.fuzzy as fuzzy_mod

    def _pair(cand_artist, cand_title, target_artist, target_track):
        return by_title.get(cand_title, 0.0)

    monkeypatch.setattr(fuzzy_mod, "pair_match", _pair)


class _TraceTagSpy(SearchTrace):
    """Trace spy: records every tag() call. Tracing is not initialized, so
    every other SearchTrace method is a safe no-op and the spy runs through
    the whole pipeline."""

    def __init__(self):
        super().__init__("a" * 32, 1, 2, "depeche mode enjoy the silence")
        self.tags: list[str] = []

    def tag(self, tags):
        self.tags.extend(tags)
        super().tag(tags)


async def _sonar_question_for(bothub_client, description):
    """Runs an artist-only this_track search and returns the exact user
    question of the sonar-ground call (None when sonar is never called)."""
    bothub_client._sonar_model = "sonar"  # the layer is enabled
    extract = _extract_json(target={"artist": "Depeche Mode", "track": None},
                            queries=["Depeche Mode songs"])
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None, gen_name="llm", model=None):
        if gen_name == "sonar-ground":
            captured["sonar_q"] = user
            return {"artist": "Depeche Mode", "track": "Dreaming of Me", "sure": True}
        if gen_name == "llm-extract":
            return extract
        return _rank_json(best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                return _dm_entries()
            mock_yt.side_effect = cap
            await search_candidates(description, bothub_client)
    return captured.get("sonar_q")


_DM_TARGET = {"artist": "Depeche Mode", "track": "Enjoy the Silence"}


# ════ 1. confident-match shortcut ════

@pytest.mark.asyncio
async def test_shortcut_skips_llm2(bothub_client, monkeypatch):
    """Decisive fuzzy match (0.95 vs 0.30, different tracks): LLM#2 is never
    called, the SearchOut is built directly — correct best, high confidence,
    the fuzzy score carried into Candidate.score, match fields set."""
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence": 0.95,
        "Depeche Mode - Personal Jesus": 0.30,
    })
    with patch.object(bothub_client, "chat_json") as mock_chat:
        # A rank answer is queued on purpose: consuming it would mean LLM#2 ran.
        mock_chat.side_effect = [
            _extract_json(target=_DM_TARGET),
            _rank_json(best_index=0),
        ]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = _dm_entries()
            out = await search_candidates("depeche mode enjoy the silence", bothub_client)

    assert mock_chat.await_count == 1, "the shortcut must skip the LLM#2 rank call"
    mock_yt.assert_awaited_once()
    assert out.intent == "this_track"
    assert out.confidence == "high"
    assert out.best_index == 0
    assert out.candidates[out.best_index].video_id == "dm1"
    assert out.candidates[out.best_index].score == pytest.approx(0.95)
    assert out.candidates[out.best_index].match_score >= 0.65, \
        "match fields must be set exactly as the rank tail sets them"
    assert out.verdict == "Depeche Mode - Enjoy the Silence"
    assert not out.need_clarify


@pytest.mark.asyncio
async def test_no_shortcut_below_threshold(bothub_client, monkeypatch):
    """Best fuzzy score 0.75 < FUZZY_SHORTCUT_MIN: the normal legacy path —
    LLM#2 ranks, its scores and verdict land in the SearchOut."""
    assert FUZZY_SHORTCUT_MIN == 0.80  # the tuned threshold, pinned
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence": 0.75,
        "Depeche Mode - Personal Jesus": 0.30,
    })
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract_json(target=_DM_TARGET),
            _rank_json(best_index=0),
        ]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = _dm_entries()
            out = await search_candidates("depeche mode enjoy the silence", bothub_client)

    assert mock_chat.await_count == 2, "no confident match -> the rank call must run"
    assert out.candidates[out.best_index].video_id == "dm1"
    assert out.candidates[out.best_index].score == pytest.approx(0.90)  # the LLM score, not fuzzy
    assert out.verdict == "The original from the Violator album."


@pytest.mark.asyncio
async def test_ambiguous_near_duplicates_still_rank(bothub_client, monkeypatch):
    """Two uploads of the same song whose track_keys differ (the title form
    carries the artist, the bare title on a label channel doesn't), fuzzy
    gap 0.04 < AMBIGUITY_GAP: ambiguous -> LLM#2 still decides."""
    entries = [
        {"video_id": "dm1", "title": "Depeche Mode - Enjoy the Silence",
         "channel": "Depeche Mode - Topic", "duration": 254, "views": 10**6},
        {"video_id": "dm2", "title": "Enjoy the Silence",
         "channel": "Mute Records", "duration": 250, "views": 9 * 10**5},
    ]
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence": 0.90,
        "Enjoy the Silence": 0.86,
    })
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract_json(target=_DM_TARGET),
            _rank_json(best_index=0),
        ]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = entries
            out = await search_candidates("depeche mode enjoy the silence", bothub_client)

    assert mock_chat.await_count == 2, "a different-track runner-up within the gap must block the shortcut"
    assert out.candidates[out.best_index].video_id == "dm1"


@pytest.mark.asyncio
async def test_same_track_duplicates_do_not_block_shortcut(bothub_client, monkeypatch):
    """Duplicate uploads of the SAME track (same track_key: the official
    suffix is cleaned by track_key) with near-equal fuzzy scores 0.90/0.88:
    duplicates are confidence, not ambiguity — the shortcut still fires."""
    entries = [
        {"video_id": "dm1", "title": "Enjoy the Silence",
         "channel": "Depeche Mode - Topic", "duration": 254, "views": 10**6},
        {"video_id": "dm2", "title": "Enjoy the Silence (Official Music Video)",
         "channel": "Depeche Mode - Topic", "duration": 250, "views": 9 * 10**5},
    ]
    _fuzzy_pair_by_title(monkeypatch, {
        "Enjoy the Silence": 0.90,
        "Enjoy the Silence (Official Music Video)": 0.88,
    })
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [_extract_json(target=_DM_TARGET), _rank_json()]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = entries
            out = await search_candidates("depeche mode enjoy the silence", bothub_client)

    assert mock_chat.await_count == 1, "same-key duplicates must not count as a runner-up"
    assert out.confidence == "high"
    assert out.candidates[out.best_index].video_id == "dm1"


@pytest.mark.asyncio
async def test_noisy_best_falls_through_to_rank(bothub_client, monkeypatch):
    """Noise guard: a tribute-band cover is the only high-fuzzy candidate in
    the pool (is_noise=True from the "(Cover)" title) — the shortcut must
    NOT deliver it as the original with confidence high; both LLM calls run
    (LLM#1 extract + LLM#2 rank) and the rank scores land in the output."""
    entries = [
        {"video_id": "trib1", "title": "Enjoy the Silence (Cover)",
         "channel": "Depeche Mode Tribute Band", "duration": 250, "views": 10**5},
    ]
    rank = {
        "ranked": [{"video_id": "trib1", "score": 0.55, "rationale": "A tribute cover."}],
        "best_index": 0,
        "confidence": "medium",
        "verdict": "The pool only has a tribute cover.",
        "need_clarify": False,
        "clarify_question": "",
    }
    _fuzzy_pair_by_title(monkeypatch, {"Enjoy the Silence (Cover)": 0.98})
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [_extract_json(target=_DM_TARGET), rank]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = entries
            out = await search_candidates("depeche mode enjoy the silence", bothub_client)

    assert mock_chat.await_count == 2, "a noisy best must fall through to the LLM#2 rank call"
    assert out.candidates[out.best_index].video_id == "trib1"
    assert out.candidates[out.best_index].score == pytest.approx(0.55)  # the LLM score, not fuzzy
    assert out.confidence == "medium"
    assert out.candidates[out.best_index].is_noise, "the cover must stay noise-flagged"


@pytest.mark.asyncio
async def test_shortcut_tags_trace(bothub_client, monkeypatch):
    """Observability: the shortcut firing is visible in the trace — the
    "shortcut" tag lands next to the intent tag set by the search head."""
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence": 0.95,
        "Depeche Mode - Personal Jesus": 0.30,
    })
    spy = _TraceTagSpy()
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [_extract_json(target=_DM_TARGET), _rank_json()]
        with patch("services.llm_search.search_metadata", return_value=_dm_entries()):
            await search_candidates("depeche mode enjoy the silence", bothub_client, trace=spy)

    assert mock_chat.await_count == 1  # the shortcut fired
    assert spy.tags == ["this_track", "shortcut"], \
        "the shortcut tag must be added without dropping the intent tag"


@pytest.mark.asyncio
async def test_no_shortcut_no_trace_tag(bothub_client, monkeypatch):
    """Observability: a search that falls through to LLM#2 (best under the
    threshold) never gets the "shortcut" tag — the tag marks the shortcut
    path only."""
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence": 0.75,
        "Depeche Mode - Personal Jesus": 0.30,
    })
    spy = _TraceTagSpy()
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [_extract_json(target=_DM_TARGET), _rank_json()]
        with patch("services.llm_search.search_metadata", return_value=_dm_entries()):
            await search_candidates("depeche mode enjoy the silence", bothub_client, trace=spy)

    assert mock_chat.await_count == 2  # the rank path ran
    assert "shortcut" not in spy.tags
    assert spy.tags == ["this_track"]


# ════ 1b. artist recovery from the track string (the «Deoche mode» gap) ════

@pytest.mark.asyncio
async def test_artist_recovery_shortcut_fires_real_fuzzy(bothub_client):
    """Prod shape of the known LLM#1 gap: «Deoche mode» extracts target
    {artist: None, track: "Deoche mode"} and the confident-match shortcut
    (both fields required) can never fire. REAL fuzzy, nothing patched: the
    official channel "Depeche Mode" matches the track string at 0.87, two
    DIFFERENT Depeche tracks sit within AMBIGUITY_GAP of each other yet the
    recovery still fires (same-artist candidates are agreement, not
    ambiguity), the different-artist candidate is far below. LLM#2 is never
    called; the verdict names the recovered artist (the channel) and the
    cleaned title."""
    extract = _extract_json(target={"artist": None, "track": "Deoche mode"},
                            queries=["Deoche mode"])
    entries = [
        {"video_id": "dm1", "title": "Depeche Mode - Enjoy the Silence",
         "channel": "Depeche Mode - Topic", "duration": 254, "views": 10**6},
        {"video_id": "dm2", "title": "Depeche Mode - Personal Jesus",
         "channel": "Depeche Mode - Topic", "duration": 260, "views": 9 * 10**5},
        {"video_id": "md1", "title": "Molchat Doma - Sudno",
         "channel": "Molchat Doma - Topic", "duration": 210, "views": 8 * 10**5},
    ]
    with patch.object(bothub_client, "chat_json") as mock_chat:
        # A rank answer is queued on purpose: consuming it would mean LLM#2 ran.
        mock_chat.side_effect = [extract, _rank_json(best_index=0)]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = entries
            out = await search_candidates("Deoche mode", bothub_client)

    assert mock_chat.await_count == 1, "the artist-recovery shortcut must skip the LLM#2 rank call"
    assert out.confidence == "high"
    best = out.candidates[out.best_index]
    assert best.video_id in ("dm1", "dm2")
    assert best.channel == "Depeche Mode - Topic"
    # the verdict leads with the recovered artist and cleans the title
    assert out.verdict.startswith("Depeche Mode -")
    assert "official" not in out.verdict.lower()
    assert out.verdict in ("Depeche Mode - Enjoy the Silence",
                           "Depeche Mode - Personal Jesus")


@pytest.mark.asyncio
async def test_artist_recovery_canned_scores_verdict_cleaned(bothub_client, monkeypatch):
    """Canned pair scores (the repo's fuzzy-patch pattern): a decisive best
    (0.95 vs 0.30, different artists) with the real artist gate satisfied
    by the channel ("Depeche Mode" vs "Deoche mode" = 0.87) fires the SAME
    shortcut path — one LLM call, high confidence, and the verdict drops
    the junk bracket plus the leading artist run from the title."""
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence (Official Video)": 0.95,
        "Molchat Doma - Sudno": 0.30,
    })
    extract = _extract_json(target={"artist": None, "track": "Deoche mode"},
                            queries=["Deoche mode"])
    entries = [
        {"video_id": "dm1", "title": "Depeche Mode - Enjoy the Silence (Official Video)",
         "channel": "Depeche Mode - Topic", "duration": 254, "views": 10**6},
        {"video_id": "md1", "title": "Molchat Doma - Sudno",
         "channel": "Molchat Doma - Topic", "duration": 210, "views": 8 * 10**5},
    ]
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [extract, _rank_json(best_index=0)]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = entries
            out = await search_candidates("Deoche mode", bothub_client)

    assert mock_chat.await_count == 1
    assert out.confidence == "high"
    assert out.candidates[out.best_index].video_id == "dm1"
    assert out.candidates[out.best_index].score == pytest.approx(0.95)
    assert out.verdict == "Depeche Mode - Enjoy the Silence"


@pytest.mark.asyncio
async def test_artist_recovery_noise_guard_still_applies(bothub_client, monkeypatch):
    """The recovery inherits the noise guard: a cover-titled best on the
    right channel (is_noise from the "(Cover)" title) must not be delivered
    with shortcut confidence — both LLM calls run and the rank scores land."""
    _fuzzy_pair_by_title(monkeypatch, {"Enjoy the Silence (Cover)": 0.98})
    extract = _extract_json(target={"artist": None, "track": "Deoche mode"},
                            queries=["Deoche mode"])
    entries = [
        {"video_id": "trib1", "title": "Enjoy the Silence (Cover)",
         "channel": "Depeche Mode - Topic", "duration": 250, "views": 10**5},
    ]
    rank = {
        "ranked": [{"video_id": "trib1", "score": 0.55, "rationale": "A tribute cover."}],
        "best_index": 0,
        "confidence": "medium",
        "verdict": "The pool only has a tribute cover.",
        "need_clarify": False,
        "clarify_question": "",
    }
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [extract, rank]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = entries
            out = await search_candidates("Deoche mode", bothub_client)

    assert mock_chat.await_count == 2, "a noisy best must fall through to the LLM#2 rank call"
    assert out.candidates[out.best_index].score == pytest.approx(0.55)
    assert out.confidence == "medium"


@pytest.mark.asyncio
async def test_artist_recovery_blocked_by_different_artist_runner_up(bothub_client, monkeypatch):
    """Two DIFFERENT artists near-equally matching the garbled string
    (0.90 vs 0.86, gap < AMBIGUITY_GAP): the recovery cannot tell which
    artist the string means — ambiguity goes to LLM#2 exactly like the
    confident-match shortcut's gap rule."""
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence": 0.90,
        "Pet Shop Boys - West End Girls": 0.86,
    })
    extract = _extract_json(target={"artist": None, "track": "Depeche mode"},
                            queries=["Deoche mode"])
    entries = [
        {"video_id": "dm1", "title": "Depeche Mode - Enjoy the Silence",
         "channel": "Depeche Mode - Topic", "duration": 254, "views": 10**6},
        {"video_id": "psb1", "title": "Pet Shop Boys - West End Girls",
         "channel": "Pet Shop Boys - Topic", "duration": 250, "views": 9 * 10**5},
    ]
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [extract, _rank_json(best_index=0)]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = entries
            out = await search_candidates("Deoche mode", bothub_client)

    assert mock_chat.await_count == 2, "a different-artist runner-up within the gap must block the recovery"
    assert out.candidates[out.best_index].video_id == "dm1"


@pytest.mark.asyncio
async def test_artist_recovery_not_tried_when_artist_present(bothub_client, monkeypatch, caplog):
    """The recovery pass is strictly for the artist-less gap: with
    target.artist set and a weak fuzzy best, neither shortcut fires (two LLM
    calls) and no recovery log line appears — the artist-present flow is
    unchanged."""
    import logging as _logging
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence": 0.75,
        "Depeche Mode - Personal Jesus": 0.30,
    })
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [_extract_json(target=_DM_TARGET), _rank_json(best_index=0)]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = _dm_entries()
            with caplog.at_level(_logging.INFO, logger="bot.llm_search"):
                out = await search_candidates("depeche mode enjoy the silence", bothub_client)

    assert mock_chat.await_count == 2
    assert "artist recovered" not in caplog.text
    assert out.verdict == "The original from the Violator album."


# ════ 2. superlative fact-resolve ════

@pytest.mark.asyncio
async def test_superlative_resolves_concrete_track(bothub_client):
    """Artist-only target + "самый первый" -> one sonar question resolves the
    concrete track BEFORE query expansion: it leads the search phrases and
    fills target.track (the rank/match stages target the concrete track)."""
    bothub_client._sonar_model = "sonar"  # the layer is enabled
    extract = _extract_json(target={"artist": "Depeche Mode", "track": None},
                            queries=["Depeche Mode early songs"])
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None, gen_name="llm", model=None):
        captured.setdefault("gens", []).append(gen_name)
        if gen_name == "sonar-ground":
            captured["sonar_q"] = user
            return {"artist": "Depeche Mode", "track": "Dreaming of Me", "sure": True}
        if gen_name == "llm-extract":
            return extract
        return _rank_json(best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                captured["queries"] = list(queries)
                return _dm_entries()
            mock_yt.side_effect = cap
            out = await search_candidates("депеш мод самый первый их трек", bothub_client)

    assert "sonar-ground" in captured["gens"]
    assert "Depeche Mode" in captured["sonar_q"]
    # the concrete track leads discovery and the recall expansion targets it
    assert captured["queries"][0] == "Depeche Mode - Dreaming of Me"
    assert any("Dreaming of Me" in q for q in captured["queries"][1:])
    assert out.target["track"] == "Dreaming of Me"
    assert out.target["artist"] == "Depeche Mode"


@pytest.mark.asyncio
async def test_superlative_debut_marker_asks_debut_question(bothub_client):
    """A debut marker (EN here) asks for the debut fact: the sonar answer
    lands in target, so a mismatched question would resolve the wrong fact
    into everything downstream."""
    q = await _sonar_question_for(bothub_client, "depeche mode first single")
    assert q == ("Какой первый сингл/дебютный трек у артиста Depeche Mode? "
                 "Пользователь просит: depeche mode first single")


@pytest.mark.asyncio
async def test_superlative_last_marker_asks_last_question(bothub_client):
    """"последний" markers ask for the LATEST release, not the debut."""
    q = await _sonar_question_for(bothub_client, "последний сингл депеш мод")
    assert q == ("Какой последний сингл/трек у артиста Depeche Mode? "
                 "Пользователь просит: последний сингл депеш мод")


@pytest.mark.asyncio
async def test_superlative_most_famous_marker_asks_hit_question(bothub_client):
    """"самая известная" markers ask for the best-known song, not the debut."""
    q = await _sonar_question_for(bothub_client, "самая известная песня депеш мод")
    assert q == ("Какая самая известная/популярная песня у артиста Depeche Mode? "
                 "Пользователь просит: самая известная песня депеш мод")


@pytest.mark.asyncio
async def test_superlative_sonar_miss_unchanged(bothub_client):
    """Sonar answers without a track (sure=false): no resolve, the original
    LLM queries search unchanged and the target stays artist-only."""
    bothub_client._sonar_model = "sonar"
    extract = _extract_json(target={"artist": "Depeche Mode", "track": None},
                            queries=["Depeche Mode early songs"])
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None, gen_name="llm", model=None):
        captured.setdefault("gens", []).append(gen_name)
        if gen_name == "sonar-ground":
            return {"artist": "Depeche Mode", "track": "", "sure": False}
        if gen_name == "llm-extract":
            return extract
        return _rank_json(best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                captured["queries"] = list(queries)
                return _dm_entries()
            mock_yt.side_effect = cap
            out = await search_candidates("депеш мод самый первый их трек", bothub_client)

    assert "sonar-ground" in captured["gens"]          # the attempt was made
    assert captured["queries"] == ["Depeche Mode early songs"]  # nothing resolved
    assert not (out.target.get("track") or "")
    assert len(out.candidates) == 2                     # the regular path ran on


@pytest.mark.asyncio
async def test_superlative_sonar_disabled_no_call(bothub_client):
    """The layer is off (empty sonar model): not a single sonar call — the
    gate mirrors the from_source branch exactly."""
    assert not bothub_client.sonar_enabled
    extract = _extract_json(target={"artist": "Depeche Mode", "track": None},
                            queries=["Depeche Mode early songs"])
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None, gen_name="llm", model=None):
        captured.setdefault("gens", []).append(gen_name)
        if gen_name == "llm-extract":
            return extract
        return _rank_json(best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                captured["queries"] = list(queries)
                return _dm_entries()
            mock_yt.side_effect = cap
            await search_candidates("депеш мод самый первый их трек", bothub_client)

    assert "sonar-ground" not in captured["gens"]
    assert captured["queries"] == ["Depeche Mode early songs"]


@pytest.mark.asyncio
async def test_no_superlative_no_sonar_even_with_artist_only(bothub_client):
    """Artist-only target WITHOUT a superlative marker: no sonar call — the
    fact-resolve is strictly for discography-fact phrasings."""
    bothub_client._sonar_model = "sonar"
    extract = _extract_json(target={"artist": "Depeche Mode", "track": None},
                            queries=["Depeche Mode songs"])
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None, gen_name="llm", model=None):
        captured.setdefault("gens", []).append(gen_name)
        if gen_name == "llm-extract":
            return extract
        return _rank_json(best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata", return_value=_dm_entries()):
            await search_candidates("что-нибудь от депеш мод", bothub_client)

    assert "sonar-ground" not in captured["gens"]


# ════ 2b. superlative fact-check of the LLM's own track fill (bake-off t4) ════

async def _run_superlative_factcheck(bothub_client, sonar_answer):
    """Runs a superlative this_track search whose LLM#1 extract ALREADY
    filled target.track (the t4 shape: the marker fires, the fill is not
    trusted blindly). Returns (captured, out); the sonar spy answers with
    sonar_answer."""
    bothub_client._sonar_model = "sonar"  # the layer is enabled
    extract = _extract_json(target={"artist": "Depeche Mode",
                                    "track": "Just Can't Get Enough"},
                            queries=["Depeche Mode Just Can't Get Enough"])
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None, gen_name="llm", model=None):
        captured.setdefault("gens", []).append(gen_name)
        if gen_name == "sonar-ground":
            captured["sonar_q"] = user
            return sonar_answer
        if gen_name == "llm-extract":
            return extract
        return _rank_json(best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                captured["queries"] = list(queries)
                return _dm_entries()
            mock_yt.side_effect = cap
            out = await search_candidates("депеш мод самый первый их трек", bothub_client)
    return captured, out


@pytest.mark.asyncio
async def test_superlative_llm_fill_sonar_agrees_no_correction(bothub_client, caplog):
    """t4: the marker fires even though LLM#1 filled target.track — exactly
    one sonar call is made either way. Sonar returns the SAME track: no
    correction, the LLM fill, its queries and the target stay untouched."""
    import logging as _logging
    with caplog.at_level(_logging.INFO, logger="bot.llm_search"):
        captured, out = await _run_superlative_factcheck(
            bothub_client,
            {"artist": "Depeche Mode", "track": "Just Can't Get Enough", "sure": True})

    assert captured["gens"].count("sonar-ground") == 1, \
        "the fact-check call must happen even with a filled target.track"
    assert "Depeche Mode" in captured["sonar_q"]
    assert out.target["artist"] == "Depeche Mode"
    assert out.target["track"] == "Just Can't Get Enough"
    assert captured["queries"][0] == "Depeche Mode Just Can't Get Enough", \
        "agreement must not rewrite the LLM queries"
    assert "corrected by sonar" not in caplog.text
    assert "sonar confirmed llm" in caplog.text


@pytest.mark.asyncio
async def test_superlative_llm_fill_sonar_corrects(bothub_client, caplog):
    """t4: sonar answers a confident DIFFERENT track — sonar wins: the
    target, the leading discovery query and the log line all carry the
    corrected track."""
    import logging as _logging
    with caplog.at_level(_logging.INFO, logger="bot.llm_search"):
        captured, out = await _run_superlative_factcheck(
            bothub_client,
            {"artist": "Depeche Mode", "track": "Dreaming of Me", "sure": True})

    assert captured["gens"].count("sonar-ground") == 1
    assert out.target["artist"] == "Depeche Mode"
    assert out.target["track"] == "Dreaming of Me"
    assert captured["queries"][0] == "Depeche Mode - Dreaming of Me"
    # the correction log names both tracks (repr quote style varies with
    # apostrophes in the strings — assert the parts, not the exact quoting)
    assert "superlative: llm" in caplog.text
    assert "corrected by sonar" in caplog.text
    assert "Just Can't Get Enough" in caplog.text
    assert "Dreaming of Me" in caplog.text


@pytest.mark.asyncio
async def test_superlative_llm_fill_sonar_miss_keeps_fill(bothub_client, caplog):
    """t4: sonar misses (sure=false, no track) — the LLM's own track fill was
    named in the query, so it stays: target and queries unchanged, the miss
    is logged."""
    import logging as _logging
    with caplog.at_level(_logging.INFO, logger="bot.llm_search"):
        captured, out = await _run_superlative_factcheck(
            bothub_client,
            {"artist": "Depeche Mode", "track": "", "sure": False})

    assert captured["gens"].count("sonar-ground") == 1  # the check was made
    assert out.target["track"] == "Just Can't Get Enough"
    assert captured["queries"][0] == "Depeche Mode Just Can't Get Enough"
    assert "sonar miss, keeping llm fill" in caplog.text


# ════ 3. route normalization ════

def test_route_normalizes_intent_variants():
    """Variant spellings map to the right pipeline; unknown intents and None
    keep the legacy fallback; each canonical key serves ONE shared instance."""
    assert route("Similar").name == "similar"   # Stage 2a: own pipeline
    assert route("similar_tracks") is route("similar")
    assert route("This Track").name == "this_track"
    assert isinstance(route("This Track"), ThisTrackPipeline)
    assert route("this_track") is route("  THISTRACK ")
    assert route("From Source").name == "from_source"  # Stage 3: own pipeline
    # unknown and None -> the shared legacy key ("mood" is Stage 2b's own)
    assert route("totally_unknown").name == "legacy"
    assert route(None) is route("totally_unknown")
    assert isinstance(route(None), LegacyPipeline)


def test_route_this_track_pipeline_is_stateless():
    """One shared instance with no mutable instance attributes: safe for
    concurrent chats (the registry contract)."""
    p = route("this_track")
    assert p.name == "this_track"
    mutable = {k: v for k, v in vars(p).items()
               if isinstance(v, (list, dict, set))}
    assert not mutable, "the pipeline must hold no mutable instance state"


@pytest.mark.asyncio
async def test_raw_intent_variant_reaches_this_track_pipeline(bothub_client, monkeypatch):
    """End to end: LLM#1 answered intent='This Track' (a variant spelling);
    route() normalizes it and the this_track pipeline serves the query —
    the shortcut fires on a confident match (one LLM call total)."""
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence": 0.95,
        "Depeche Mode - Personal Jesus": 0.30,
    })
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [_extract_json(intent="This Track", target=_DM_TARGET),
                                 _rank_json()]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = _dm_entries()
            out = await search_candidates("depeche mode enjoy the silence", bothub_client)

    assert mock_chat.await_count == 1, "variant spelling must still reach the shortcut"
    assert out.confidence == "high"
    assert out.candidates[out.best_index].video_id == "dm1"


# ════ 4. yt-suggest query normalization (B3) ════

def _suggest_returning(suggestions, log=None):
    """Patches services.suggest.yt_suggest with a canned list; records every
    query it was asked. The pipeline resolves the module lazily per call, so
    the module-attribute patch takes effect (the conftest autouse fixture
    already cut the real HTTP — this replaces the whole helper)."""
    asked = []

    async def fake(query, timeout=4.0):
        asked.append(query)
        return list(suggestions)

    return patch("services.suggest.yt_suggest", side_effect=fake), asked


def _suggest_raising(exc):
    async def broken(query, timeout=4.0):
        raise exc

    return patch("services.suggest.yt_suggest", side_effect=broken)


@pytest.mark.asyncio
async def test_suggest_widens_pool_recovery_shortcut_fires(bothub_client, caplog):
    """B3 prod shape: «Deoche mode» extracts target {artist: None, track:
    "Deoche mode"} and suggest returns the canonical «depeche mode». The
    canonical spelling joins the pool right after the primary query (pool
    WIDER than the LLM's own queries), the wiring logs the normalization,
    and the artist-recovery shortcut fires on the candidates the canonical
    query surfaces — LLM#2 never called."""
    import logging as _logging
    extract = _extract_json(target={"artist": None, "track": "Deoche mode"},
                            queries=["Deoche mode"])
    entries = [
        {"video_id": "dm1", "title": "Depeche Mode - Enjoy the Silence",
         "channel": "Depeche Mode - Topic", "duration": 254, "views": 10**6},
        {"video_id": "dm2", "title": "Depeche Mode - Personal Jesus",
         "channel": "Depeche Mode - Topic", "duration": 260, "views": 9 * 10**5},
        {"video_id": "md1", "title": "Molchat Doma - Sudno",
         "channel": "Molchat Doma - Topic", "duration": 210, "views": 8 * 10**5},
    ]
    suggest_patch, asked = _suggest_returning(
        ["depeche mode", "depeche mode playlist", "depeche mode best of"])
    captured = {}
    with suggest_patch:
        with patch.object(bothub_client, "chat_json") as mock_chat:
            # A rank answer is queued on purpose: consuming it would mean LLM#2 ran.
            mock_chat.side_effect = [extract, _rank_json(best_index=0)]
            with patch("services.llm_search.search_metadata") as mock_yt:
                async def cap(queries, per_query=20, on_progress=None):
                    captured["queries"] = list(queries)
                    return entries
                mock_yt.side_effect = cap
                with caplog.at_level(_logging.INFO, logger="bot.llm_search"):
                    out = await search_candidates("Deoche mode", bothub_client)

    assert asked == ["Deoche mode"], "one suggest call on the target strings"
    # the pool: primary first, canonical spellings right after, capped at 2 extras
    assert captured["queries"][0] == "Deoche mode"
    assert captured["queries"][1:] == ["depeche mode", "depeche mode playlist"]
    assert len(captured["queries"]) == 3, "the pool is wider than the LLM queries"
    assert "yt suggest:" in caplog.text
    # the shortcut decides on the wider pool: recovery fires, LLM#2 skipped
    assert mock_chat.await_count == 1
    assert out.confidence == "high"
    assert out.verdict.startswith("Depeche Mode -")


@pytest.mark.asyncio
async def test_suggest_asks_artist_track_concatenation(bothub_client):
    """A full target asks for f"{artist} {track}" exactly once; an
    already-canonical answer (the top suggestion IS the raw query) adds
    nothing — the pool is the primary plus the usual recall expansion only."""
    suggest_patch, asked = _suggest_returning(["depeche mode enjoy the silence"])
    captured = {}
    with suggest_patch:
        with patch.object(bothub_client, "chat_json") as mock_chat:
            mock_chat.side_effect = [_extract_json(target=_DM_TARGET), _rank_json()]
            with patch("services.llm_search.search_metadata") as mock_yt:
                async def cap(queries, per_query=20, on_progress=None):
                    captured["queries"] = list(queries)
                    return _dm_entries()
                mock_yt.side_effect = cap
                await search_candidates("depeche mode enjoy the silence", bothub_client)

    assert asked == ["Depeche Mode Enjoy the Silence"]
    assert captured["queries"] == [
        "depeche mode enjoy the silence",
        '"Depeche Mode - Enjoy the Silence" topic',
        '"Depeche Mode - Enjoy the Silence"',
        "Depeche Mode - Enjoy the Silence -cover -live -karaoke -remix",
    ]


@pytest.mark.asyncio
async def test_suggest_unrelated_suggestions_not_added(bothub_client):
    """Gate: suggestions that don't near-match the raw query (similarity <
    0.6) are popular-query noise, not re-spellings — the pool stays exactly
    the LLM's own queries."""
    suggest_patch, _ = _suggest_returning(
        ["rick astley never gonna give you up", "smells like teen spirit"])
    extract = _extract_json(target={"artist": None, "track": "Deoche mode"},
                            queries=["Deoche mode"])
    captured = {}
    with suggest_patch:
        with patch.object(bothub_client, "chat_json") as mock_chat:
            mock_chat.side_effect = [extract, _rank_json(best_index=0)]
            with patch("services.llm_search.search_metadata") as mock_yt:
                async def cap(queries, per_query=20, on_progress=None):
                    captured["queries"] = list(queries)
                    return _dm_entries()
                mock_yt.side_effect = cap
                await search_candidates("Deoche mode", bothub_client)

    assert captured["queries"] == ["Deoche mode"]


@pytest.mark.asyncio
async def test_suggest_empty_adds_nothing(bothub_client, monkeypatch):
    """Suggest down/empty ([] — the conftest offline default shape): the
    wiring is a no-op, the confident-match shortcut path is unchanged."""
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence": 0.95,
        "Depeche Mode - Personal Jesus": 0.30,
    })
    suggest_patch, _ = _suggest_returning([])
    captured = {}
    with suggest_patch:
        with patch.object(bothub_client, "chat_json") as mock_chat:
            mock_chat.side_effect = [_extract_json(target=_DM_TARGET), _rank_json()]
            with patch("services.llm_search.search_metadata") as mock_yt:
                async def cap(queries, per_query=20, on_progress=None):
                    captured["queries"] = list(queries)
                    return _dm_entries()
                mock_yt.side_effect = cap
                out = await search_candidates("depeche mode enjoy the silence", bothub_client)

    assert mock_chat.await_count == 1, "the shortcut fires as always"
    assert captured["queries"][0] == "depeche mode enjoy the silence"
    assert len(captured["queries"]) == 4  # primary + 3 recall phrases, no extras


@pytest.mark.asyncio
async def test_suggest_raising_never_breaks_search(bothub_client, monkeypatch):
    """Fail-soft end to end: a suggest layer that RAISES (worse than a
    timeout) changes nothing — the confident-match shortcut still fires and
    the pool is the usual expansion."""
    _fuzzy_pair_by_title(monkeypatch, {
        "Depeche Mode - Enjoy the Silence": 0.95,
        "Depeche Mode - Personal Jesus": 0.30,
    })
    captured = {}
    with _suggest_raising(RuntimeError("suggest exploded")):
        with patch.object(bothub_client, "chat_json") as mock_chat:
            mock_chat.side_effect = [_extract_json(target=_DM_TARGET), _rank_json()]
            with patch("services.llm_search.search_metadata") as mock_yt:
                async def cap(queries, per_query=20, on_progress=None):
                    captured["queries"] = list(queries)
                    return _dm_entries()
                mock_yt.side_effect = cap
                out = await search_candidates("depeche mode enjoy the silence", bothub_client)

    assert mock_chat.await_count == 1, "a suggest outage must not break the shortcut"
    assert out.confidence == "high"
    assert out.candidates[out.best_index].video_id == "dm1"
    assert captured["queries"][0] == "depeche mode enjoy the silence"


# ════ wave2 w3: the shortcut verdict describes the DELIVERED upload ════

@pytest.mark.asyncio
async def test_shortcut_verdict_from_delivered_upload(bothub_client, monkeypatch):
    """The target names a remix, the shortcut matched the PLAIN official
    upload (judge w3 PARTIAL): the verdict must describe what was actually
    delivered, not the LLM target's version wording."""
    _fuzzy_pair_by_title(monkeypatch, {
        "Franz Ferdinand - Take Me Out": 0.95,
        "Franz Ferdinand - The Dark of the Matinée": 0.30,
    })
    entries = [
        {"video_id": "ff1", "title": "Franz Ferdinand - Take Me Out",
         "channel": "Franz Ferdinand - Topic", "duration": 240, "views": 10**6},
        {"video_id": "ff2", "title": "Franz Ferdinand - The Dark of the Matinée",
         "channel": "Franz Ferdinand - Topic", "duration": 260, "views": 9 * 10**5},
    ]
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract_json(target={"artist": "Franz Ferdinand",
                                  "track": "Take Me Out (Daft Punk Remix)"}),
            _rank_json(best_index=0),
        ]
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = entries
            out = await search_candidates(
                "franz ferdinand take me out daft punk remix", bothub_client)

    assert out.candidates[out.best_index].video_id == "ff1"
    assert out.verdict == "Franz Ferdinand - Take Me Out", \
        "the verdict names the delivered plain upload, not the target remix"
