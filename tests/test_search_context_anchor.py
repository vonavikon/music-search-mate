"""Clarification memory (Madagascar audit, П1+П3, prod case 12.11).

The user hunted Boston's "More Than a Feeling" via «Мадагаскар 2»: the
from_source search was followed by FRAGMENT answers to the bot's clarifying
questions («это трек 80х примерно, его там на кассете включают», «нет, ее
включали в момент когда они угнали машину»). Each fragment started a FRESH
search — the movie anchor was lost, sonar was asked literally, and it
honestly returned songs from OTHER car-heist movies (Heathens/Suicide
Squad, Bellbottoms/Baby Driver), which the ranker scored 1.0.

search_candidates now takes source_anchor (the dialog's LAST from_source
reference, supplied by the handler) and:
  - injects a «КОНТЕКСТ ИСТОЧНИКА» block into the LLM#1 user message
    (persuasion: the SYSTEM_EXTRACT rule);
  - deterministically populates an empty extract reference source from the
    anchor for from_source/this_track (guarantee: sonar/rank/YouTube always
    see the movie), mirroring it into signals — the transport the rank
    input reads (_format_for_rank renders the «Источник:» line from it).

Driving pattern mirrors test_llm_search.py: chat_json mocked with parsed
dicts, search_metadata patched, the REAL FromSourcePipeline runs end to end
(sonar is off: the fixture client carries no sonar model).
"""
import logging
from unittest.mock import patch

import pytest

from services.llm_search import _format_for_rank, search_candidates


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Enrich step offline (same isolation as test_llm_search.py): the fake
    candidates need no yt-dlp metadata pass."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers ════

_ANCHOR = "мадагаскар 2"
# Fragment answers of the 12.11 case: no source of their own, >=4 words and
# no referential markers -> _is_referential is False, so the plain user
# message is EXACTLY the description (the assertions below rely on that).
_FRAGMENT = "нет, ее включали в момент когда они угнали машину"


def _extract_json(intent="from_source", reference=None, queries=None):
    """LLM#1 extract — an already-parsed dict (chat_json mock returns it)."""
    return {
        "intro": "Ищем: саундтрек из фильма.",
        "intent": intent,
        "signals": {"language": "ru", "era": "1980s", "vocal": None,
                    "genre": None, "mood": None},
        "target": {"artist": None, "track": None},
        "reference": reference or {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries or ["Madagascar 2 soundtrack", "Madagascar 2 OST"],
    }


def _rank_json(best_index=0):
    """LLM#2 rank — an already-parsed dict."""
    return {
        "ranked": [
            {"video_id": "v1", "score": 0.95, "rationale": "Саундтрек фильма."},
            {"video_id": "v2", "score": 0.60, "rationale": "Похоже по сцене."},
            {"video_id": "v3", "score": 0.40, "rationale": "Запасной."},
        ],
        "best_index": best_index,
        "confidence": "medium",
        "verdict": "Похоже, это он.",
        "need_clarify": False,
        "clarify_question": "",
    }


def _yt_entries():
    """Fake yt-dlp results (3 clean audio-track candidates)."""
    return [
        {"video_id": "v1", "title": "Madagascar 2 Escape OST", "channel": "OST Channel",
         "duration": 240, "views": 900000},
        {"video_id": "v2", "title": "Boston - More Than a Feeling", "channel": "Boston - Topic",
         "duration": 280, "views": 120000000},
        {"video_id": "v3", "title": "Some Studio Track", "channel": "Some Artist - Topic",
         "duration": 210, "views": 40000},
    ]


class _Recorder:
    """chat_json stand-in that records (system, user) per call and answers
    extract-then-rank, the exact sequence the from_source flow makes.
    __call__ is SYNC on purpose: patch.object on an async method creates an
    AsyncMock, and an AsyncMock side_effect must be a plain callable whose
    return value is the awaited answer (the same contract the list-style
    side_effect of test_llm_search.py relies on)."""

    def __init__(self, extract, rank):
        self.calls = []
        self._answers = [extract, rank]

    def __call__(self, system, user, max_tokens=None, **kwargs):
        self.calls.append({"system": system, "user": user})
        return self._answers.pop(0) if self._answers else None


async def _run(bothub_client, extract, source_anchor=_ANCHOR, rank=None):
    """search_candidates over the REAL from_source pipeline with bothub and
    yt-dlp mocked. Returns (SearchOut, recorded chat calls)."""
    rec = _Recorder(extract, rank if rank is not None else _rank_json())
    with patch.object(bothub_client, "chat_json", side_effect=rec):
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = _yt_entries()
            out = await search_candidates(_FRAGMENT, bothub_client,
                                          source_anchor=source_anchor)
    return out, rec


class _EchoRefPipeline:
    """route() stand-in (the test_search_choke.py pattern): returns a minimal
    SearchOut whose reference IS ctx.extract's reference — asserting on it
    pins what every real pipeline receives from the mutated extract."""

    async def run(self, ctx):
        from services.llm_search import SearchOut
        return SearchOut(intent=ctx.extract.get("intent"), signals={},
                         queries=[], assoc=[], reference=ctx.extract.get("reference"),
                         target={}, candidates=[], best_index=-1,
                         confidence="low")


async def _run_stubbed(bothub_client, extract, source_anchor=_ANCHOR):
    """search_candidates with route() stubbed: exercises the head's
    inheritance block only (no pipeline internals)."""
    with patch.object(bothub_client, "chat_json", return_value=extract):
        with patch("services.llm_search.route", return_value=_EchoRefPipeline()):
            return await search_candidates(_FRAGMENT, bothub_client,
                                           source_anchor=source_anchor)


# ════ 1. anchor block in the LLM#1 prompt ════

@pytest.mark.asyncio
async def test_anchor_block_injected_into_extract_prompt(bothub_client):
    """source_anchor renders as a SEPARATE prefixed block before the user's
    words — the fragment itself stays untouched (not glued)."""
    out, rec = await _run(bothub_client, _extract_json())
    user = rec.calls[0]["user"]
    first_line, _, rest = user.partition("\n")
    assert first_line == f"КОНТЕКСТ ИСТОЧНИКА (из предыдущих запросов диалога): {_ANCHOR}"
    # the user's words are their own block, byte-identical, not merged
    assert rest.strip().splitlines()[-1] == _FRAGMENT
    assert out.candidates, "the search itself still completes"


# ════ 2. deterministic inheritance into an empty reference ════

@pytest.mark.asyncio
async def test_anchor_inherited_into_empty_reference(bothub_client, caplog):
    """Extract (from_source) leaves the reference source empty -> the anchor
    populates it deterministically: SearchOut.reference carries the movie
    even though LLM#1 never named it. The log line is the audit trail."""
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, _ = await _run(bothub_client, _extract_json())
    assert out.reference.get("source") == _ANCHOR
    assert "source anchor inherited" in caplog.text


@pytest.mark.asyncio
async def test_anchor_inherited_for_this_track_too(bothub_client):
    """this_track with an empty reference inherits as well (the fragment may
    read as "a concrete track I'm hunting"); the hunt stays anchored to the
    movie. Pinned at the search_candidates seam (route stubbed like
    test_search_choke.py) — the assert reads ctx.extract's reference, the
    exact dict every pipeline receives."""
    out = await _run_stubbed(bothub_client,
                             _extract_json(intent="this_track"))
    assert out.reference.get("source") == _ANCHOR


@pytest.mark.asyncio
async def test_no_inheritance_for_mood_or_similar(bothub_client):
    """similar/mood never borrow the anchor (КОНТЕКСТ НЕ СОЗДАЁТ РЕФЕРЕНС):
    ctx.extract's reference stays exactly what LLM#1 returned."""
    for intent in ("similar", "mood"):
        out = await _run_stubbed(bothub_client, _extract_json(intent=intent))
        assert not (out.reference.get("source") or "").strip(), intent


# ════ 3. no anchor -> behavior unchanged ════

@pytest.mark.asyncio
async def test_no_anchor_prompt_and_reference_untouched(bothub_client):
    """Without source_anchor the extract user message is byte-identical to
    the description (no context line) and the reference stays empty — the
    previous behavior exactly."""
    rec = _Recorder(_extract_json(), _rank_json())
    with patch.object(bothub_client, "chat_json", side_effect=rec):
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = _yt_entries()
            out = await search_candidates(_FRAGMENT, bothub_client)
    assert rec.calls[0]["user"] == _FRAGMENT
    assert "КОНТЕКСТ ИСТОЧНИКА" not in rec.calls[0]["user"]
    assert not (out.reference.get("source") or "").strip()


@pytest.mark.asyncio
async def test_empty_anchor_string_is_no_anchor(bothub_client):
    """A whitespace-only anchor is falsy: same byte-identical prompt."""
    rec = _Recorder(_extract_json(), _rank_json())
    with patch.object(bothub_client, "chat_json", side_effect=rec):
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = _yt_entries()
            out = await search_candidates(_FRAGMENT, bothub_client,
                                          source_anchor="   ")
    assert rec.calls[0]["user"] == _FRAGMENT
    assert not (out.reference.get("source") or "").strip()


# ════ 4. the extract's OWN source is never overridden ════

@pytest.mark.asyncio
async def test_own_reference_source_not_overridden(bothub_client, caplog):
    """The query names its own source («Такси 2») and LLM#1 returns it in
    reference.source -> the anchor must NOT clobber it, and no inheritance
    log fires."""
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, _ = await _run(
            bothub_client,
            _extract_json(reference={"artist": None, "track": None,
                                     "source": "Такси 2"}))
    assert out.reference.get("source") == "Такси 2"
    assert _ANCHOR not in str(out.reference)
    assert "source anchor inherited" not in caplog.text


# ════ 5. П3: the source reaches the rank input ════

@pytest.mark.asyncio
async def test_rank_prompt_carries_source_line(bothub_client):
    """The 12.11 root cause at the rank stage: the fragment named no movie,
    so the ranker happily scored OTHER car-heist soundtracks 1.0. The rank
    user message must now carry an explicit «Источник:» line with the
    movie, and the source must not leak into the Сигналы join."""
    out, rec = await _run(bothub_client, _extract_json())
    rank_user = rec.calls[1]["user"]
    assert "Источник:" in rank_user
    assert _ANCHOR in rank_user
    assert f"Источник: {_ANCHOR}" in rank_user
    # dedicated line only — no duplicate "source=..." fragment in Сигналы
    assert "source=" not in rank_user
    assert out.candidates and out.best_index == 0


@pytest.mark.asyncio
async def test_rank_prompt_carries_own_source_too(bothub_client):
    """A reference source LLM#1 itself returned (following the
    УНАСЛЕДОВАННЫЙ ИСТОЧНИК rule) is mirrored into the rank input the same
    way — visible whatever its origin."""
    _, rec = await _run(
        bothub_client,
        _extract_json(reference={"artist": None, "track": None,
                                 "source": "Такси 2"}))
    assert "Источник: Такси 2" in rec.calls[1]["user"]


# ════ unit: the «Источник:» rendering contract of _format_for_rank ════

def test_format_for_rank_source_from_reference_param():
    """reference.source wins and renders as its own line after Описание."""
    out = _format_for_rank("нет, другую", {"era": "1980s"}, [],
                           reference={"artist": None, "track": None,
                                      "source": "мадагаскар 2"})
    lines = out.splitlines()
    assert lines[0] == "Описание: нет, другую"
    assert lines[1] == "Источник: мадагаскар 2 (искомый трек должен быть из него)"


def test_format_for_rank_source_from_signals_transport():
    """signals["source"] (the extract-side inheritance transport — the shared
    rank tail passes only signals) renders the same line and stays OUT of
    the Сигналы join."""
    out = _format_for_rank("нет, другую", {"era": "1980s", "source": "мадагаскар 2"}, [])
    assert "Источник: мадагаскар 2" in out
    assert "Сигналы: era=1980s" in out
    assert "source=" not in out


def test_format_for_rank_no_source_line_without_source():
    """No source anywhere -> the message is exactly the pre-12.11 shape
    (byte-comparable skeleton: Описание, then Сигналы)."""
    out = _format_for_rank("песня из рекламы", {"genre": "rock"}, [])
    assert out.splitlines()[:2] == ["Описание: песня из рекламы", "Сигналы: genre=rock"]


# ════ circular-clarify sanitizer (13.11, F1 incident) ════

@pytest.mark.asyncio
async def test_circular_clarify_replaced_with_scene_question(bothub_client):
    """A from_source clarify asking for artist/title is circular (the user
    does not know it - that is why they search by scene). The guard swaps
    it for the scene question; a scene-question clarify passes through."""
    rank = {"ranked": [], "best_index": -1, "confidence": "low",
            "verdict": "", "need_clarify": True,
            "clarify_question": "Какой именно трек вы ищете? Уточните исполнителя или название."}
    out, _ = await _run(bothub_client, _extract_json(), rank=rank)
    assert out.need_clarify
    assert "исполнител" not in out.clarify_question.lower()
    assert "происходило на экране" in out.clarify_question


@pytest.mark.asyncio
async def test_scene_clarify_passes_through(bothub_client):
    rank = {"ranked": [], "best_index": -1, "confidence": "low",
            "verdict": "", "need_clarify": True,
            "clarify_question": "В какой сцене фильма звучала музыка? Опиши момент."}
    out, _ = await _run(bothub_client, _extract_json(), rank=rank)
    assert "сцене" in out.clarify_question or "сцен" in out.clarify_question


@pytest.mark.asyncio
async def test_this_track_artist_ask_not_sanitized(bothub_client):
    """this_track keeps the model's wording: there the user DOES know at
    least part of the name, asking for it is legitimate."""
    rank = {"ranked": [], "best_index": -1, "confidence": "low",
            "verdict": "", "need_clarify": True,
            "clarify_question": "Уточните исполнителя или название."}
    out, _ = await _run(bothub_client, _extract_json(intent="this_track",
                                                     queries=["Boston More Than a Feeling"]),
                        rank=rank)
    assert out.clarify_question == "Уточните исполнителя или название."
