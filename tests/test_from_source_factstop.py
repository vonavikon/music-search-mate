"""Tests for the soundtrack fact-stop (Madagascar audit П2) in
FromSourcePipeline.

Prod case 12.11: a weak from_source pool delivered twenty one pilots
"Heathens" (Suicide Squad) for a Madagascar 2 scene hunt with outcome
"ok" — the ranker scored "a soundtrack from a movie" that was not THE
movie. The contract under test (the decide point, AFTER the rank tail):

  - GATE: fires only when a reference SOURCE name is present (tolerantly
    read from whichever anchor field carries it — the exact key is the
    llm_search source-anchor work's contract, not this file's) AND the
    best pick is weak (LLM rank score < FACTSTOP_MAX_SCORE=0.4, or the
    rank tail returned low confidence). High score + non-low confidence
    pays ZERO extra sonar calls.
  - ONE verification sonar call per search (gen_name="sonar-factstop",
    the bothub sonar path, ctx-level latch): the exact RU question asks
    whether the pick is on the OFFICIAL soundtrack of THAT source.
  - in_soundtrack=true (sure) -> deliver as today ("fact-stop passed").
  - in_soundtrack=false, or unsure with score < FACTSTOP_REJECT_SCORE
    (0.25) -> NO wrong-movie delivery: the honest clarify path with the
    П2 message naming up to 3 real soundtrack songs ("fact-stop
    rejected"). Sonar garbage/exception parse as unsure; nothing raises.
  - unsure with score in [0.25, 0.4) -> deliver as today, logged
    ("fact-stop unsure, delivering").
  - SCOPE: the PLAIN flow only (the analog chain ranks in similar mode
    and delivers tracks deliberately NOT from the soundtrack); no source
    name -> gate off entirely; sonar disabled -> no verification; the
    rank tail already asking to clarify -> nothing will deliver, no call.

Mocking follows tests/test_from_source_pipeline.py: BothubClient.chat_json
patched on the bothub_client fixture (dispatched by gen_name),
search_metadata patched in services.llm_search, no network.
"""
import logging
import time
from unittest.mock import patch

import pytest

from services.pipelines.base import PipelineContext
from services.pipelines.from_source import (
    FACTSTOP_MAX_SCORE,
    FACTSTOP_REJECT_SCORE,
    FromSourcePipeline,
    _factstop_verdict,
    _factstop_verify,
    _reference_source,
)


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Same guard as test_from_source_pipeline.py: the enrich step must not
    touch the network (candidates here are fake)."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers (the test_from_source_pipeline.py patterns) ════

def _extract_json(queries=None, source=None, top_anchor=None, lang="ru"):
    """LLM#1 extract — a fake parsed dict (from_source flavor). source lands
    in extract["reference"]["source"], top_anchor in extract["source_anchor"]
    (the two anchor carry points the gate reads tolerantly)."""
    extract = {
        "intro": "Ищем: трек из фильма.",
        "intent": "from_source",
        "signals": {"language": lang, "era": None, "vocal": None,
                    "genre": None, "mood": None},
        "target": {"artist": None, "track": None},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries or ["madagascar escape africa soundtrack"],
    }
    if source is not None:
        extract["reference"]["source"] = source
    if top_anchor is not None:
        extract["source_anchor"] = top_anchor
    return extract


def _rank_json(ranked=None, best_index=0, confidence="medium",
               need_clarify=False, clarify_question=""):
    """LLM#2 rank — a fake parsed dict; defaults to a weak-pool shape."""
    if ranked is None:
        ranked = [
            {"video_id": "v1", "score": 0.35, "rationale": "Саундтрек фильма."},
            {"video_id": "v2", "score": 0.20, "rationale": "Близкий вариант."},
            {"video_id": "v3", "score": 0.10, "rationale": "Запасной."},
        ]
    return {"ranked": ranked, "best_index": best_index, "confidence": confidence,
            "verdict": "Вот эта композиция.", "need_clarify": need_clarify,
            "clarify_question": clarify_question}


def _weak_pool():
    """The prod-case pool: the WRONG-movie candidate on top of a weak field
    (enrich gives no canonical fields — the autouse patch — so the
    verification label falls back to channel/title)."""
    return [
        {"video_id": "v1", "title": "Heathens",
         "channel": "twenty one pilots - Topic", "duration": 190,
         "views": 1_900_000_000},
        {"video_id": "v2", "title": "Cool Soundtrack Theme",
         "channel": "Film Orchestra", "duration": 240, "views": 500000},
        {"video_id": "v3", "title": "Soft End Titles",
         "channel": "Piano Duo", "duration": 270, "views": 50000},
    ]


async def _run(bothub_client, description, extract, entries, rank=None,
               ground=None, factstop=None, factstop_error=None, pipeline=None,
               seed=None):
    """Run the pipeline DIRECTLY on a PipelineContext. captured holds the
    gen_name of every chat_json call, the sonar-ground question, the
    sonar-factstop question/system and its call count, the discovery
    queries and the llm-rank input. `ground`/`factstop`/`seed` are the
    canned parsed answers per sonar call (seed=None = unsure answer: the
    soundtrack seeding runs but adds nothing); factstop_error makes the
    verification call raise (the timeout/outage path)."""
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault("calls", []).append(gen_name)
        if gen_name == "sonar-ground":
            captured["ground_q"] = user
            return ground
        if gen_name == "sonar-seed":
            captured["seed_q"] = user
            return seed
        if gen_name == "sonar-ground-probe":
            # (2026-11-16) the relaxed probe fires when the sure-only seed
            # added nothing (this scenario's seed is unsure=None): a None
            # answer adds no query, the call order gains one entry.
            return None
        if gen_name == "sonar-factstop":
            captured["factstop_q"] = user
            captured["factstop_system"] = system
            captured["factstop_calls"] = captured.get("factstop_calls", 0) + 1
            if factstop_error is not None:
                raise factstop_error
            return factstop
        if gen_name == "llm-rank":
            captured["rank_input"] = user
            return rank if rank is not None else _rank_json()
        raise AssertionError(f"unexpected LLM call: {gen_name}")

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                captured["queries"] = list(queries)
                return entries
            mock_yt.side_effect = cap
            ctx = PipelineContext(
                description=description, bothub=bothub_client, extract=extract,
                t_start=time.perf_counter(), t_extract=0.05)
            out = await (pipeline or FromSourcePipeline()).run(ctx)
    return out, captured


_MAD2 = "Madagascar 2"
# The real soundtrack of the source (what sonar names back on a "no").
_MAD2_SONGS = ["Once Upon a Time in Africa - Hans Zimmer",
               "The Traveling Song - will.i.am",
               "Big and Chunky - will.i.am"]
_FACTSTOP_NO = {"in_soundtrack": False, "sure": True,
                "soundtrack_songs": _MAD2_SONGS}
_FACTSTOP_YES = {"in_soundtrack": True, "sure": True,
                 "soundtrack_songs": _MAD2_SONGS}
# The exact verification question for the prod-case pick (label from the
# channel minus "- Topic" + the title; enrich gave nothing canonical).
_EXPECTED_Q = ('Есть ли песня "twenty one pilots - Heathens" в официальном '
               'саундтреке "Madagascar 2"? Ответь строго JSON: '
               '{"in_soundtrack": true|false, "sure": true|false, '
               '"soundtrack_songs": ["песня - исполнитель", ...до 3]}')


# ════ 1. low score + sonar "no": the wrong movie must not deliver ════

@pytest.mark.asyncio
async def test_low_score_sonar_false_stops_delivery_and_names_songs(
        bothub_client, caplog):
    """The prod case: a weak pool (best llm-rank 0.35 < 0.4) topped by the
    Suicide Squad track, sonar confirms it is NOT on the Madagascar 2
    soundtrack -> NO delivery of the wrong candidate, the honest clarify
    path returns with the П2 message naming the real soundtrack songs, and
    EXACTLY ONE verification call was made."""
    bothub_client._sonar_model = "sonar"
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, captured = await _run(
            bothub_client, "песня из мадагаскар 2 где шницели падали на алекса",
            _extract_json(source=_MAD2), _weak_pool(),
            rank=_rank_json(ranked=[
                {"video_id": "v1", "score": 0.35, "rationale": "Саундтрек фильма."},
                {"video_id": "v2", "score": 0.20, "rationale": "Близкий вариант."},
                {"video_id": "v3", "score": 0.10, "rationale": "Запасной."},
            ], confidence="medium"),
            factstop=_FACTSTOP_NO)

    # exactly one verification call, with the exact П2 question
    assert captured["factstop_calls"] == 1
    assert captured["factstop_q"] == _EXPECTED_Q
    assert "официальном саундтреке" in captured["factstop_system"]
    # the call order: ground first (a miss), the soundtrack seed (unsure —
    # adds nothing, 2026-11-14 F1 layer), rank, then the verification
    assert captured["calls"] == ["sonar-ground", "sonar-seed",
                                 "sonar-ground-probe", "llm-rank",
                                 "sonar-factstop"]
    # the honest clarify shape (the same mechanism as the escalation return)
    assert out.need_clarify
    assert not out.candidates and out.best_index == -1
    assert out.confidence == "low" and out.intent == "from_source"
    # the wrong-movie candidate never reaches the user
    assert all(c.video_id != "v1" for c in out.candidates)
    # the message names the source and the real soundtrack songs
    q = out.clarify_question
    assert f"Не могу подтвердить, что это из «{_MAD2}»." in q
    for song in _MAD2_SONGS:
        assert song in q
    assert "Скажи, какая из них, или опиши сцену ещё раз." in q
    assert any("fact-stop rejected" in r.message for r in caplog.records)


# ════ 2. low score + sonar "yes": delivered as today ════

@pytest.mark.asyncio
async def test_low_score_sonar_true_delivers(bothub_client, caplog):
    """Sonar confirms the pick IS on the official soundtrack -> the pipeline
    returns the ranked result exactly as today (candidates, best, no
    clarify), one verification call, "fact-stop passed" logged."""
    bothub_client._sonar_model = "sonar"
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, captured = await _run(
            bothub_client, "песня из мадагаскар 2 где шницели падали на алекса",
            _extract_json(source=_MAD2), _weak_pool(),
            rank=_rank_json(confidence="medium"), factstop=_FACTSTOP_YES)

    assert captured["factstop_calls"] == 1
    assert out.candidates and out.candidates[out.best_index].video_id == "v1"
    assert not out.need_clarify
    assert out.confidence == "medium"
    assert any("fact-stop passed" in r.message for r in caplog.records)


# ════ 3. high score: zero verification calls ════

@pytest.mark.asyncio
async def test_high_score_no_verification_call(bothub_client):
    """score 0.95 >= FACTSTOP_MAX_SCORE with non-low confidence: the happy
    path is unchanged and pays NOTHING — the verification sonar call count
    is zero (the ground call is a different gen_name and still allowed)."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, "песня из мадагаскар 2 где шницели падали на алекса",
        _extract_json(source=_MAD2), _weak_pool(),
        rank=_rank_json(ranked=[
            {"video_id": "v1", "score": 0.95, "rationale": "Тот самый трек."},
            {"video_id": "v2", "score": 0.80, "rationale": "Близкий вариант."},
            {"video_id": "v3", "score": 0.70, "rationale": "Запасной."},
        ], confidence="high"),
        factstop=_FACTSTOP_NO)  # would reject — must never be asked

    assert captured.get("factstop_calls", 0) == 0
    assert "sonar-factstop" not in captured["calls"]
    assert out.candidates and out.candidates[out.best_index].video_id == "v1"
    assert not out.need_clarify


# ════ 4. garbage / exception: the unsure rules by score ════

@pytest.mark.asyncio
async def test_garbage_answer_low_score_honest_path(bothub_client):
    """Sonar garbage (chat_json -> None) parses as unsure; with score 0.20
    < FACTSTOP_REJECT_SCORE the wrong-movie candidate does NOT deliver —
    the honest clarify path (no songs to name) returns instead."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, "песня из мадагаскар 2 где шницели падали на алекса",
        _extract_json(source=_MAD2), _weak_pool(),
        rank=_rank_json(ranked=[
            {"video_id": "v1", "score": 0.20, "rationale": "Саундтрек фильма."},
            {"video_id": "v2", "score": 0.10, "rationale": "Близкий вариант."},
        ], confidence="medium"),
        factstop=None)

    assert captured["factstop_calls"] == 1
    assert out.need_clarify and not out.candidates
    # no songs from a garbage answer: the re-describe variant of the message
    assert out.clarify_question == \
        f"Не могу подтвердить, что это из «{_MAD2}». Опиши сцену ещё раз."


@pytest.mark.asyncio
async def test_exception_mid_score_delivers_with_unsure_log(
        bothub_client, caplog):
    """A raising verification call (timeout/outage shape) never breaks the
    search and parses as unsure; score 0.30 in [0.25, 0.4) -> delivered as
    today with the "fact-stop unsure, delivering" log."""
    bothub_client._sonar_model = "sonar"
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, captured = await _run(
            bothub_client, "песня из мадагаскар 2 где шницели падали на алекса",
            _extract_json(source=_MAD2), _weak_pool(),
            rank=_rank_json(ranked=[
                {"video_id": "v1", "score": 0.30, "rationale": "Саундтрек фильма."},
                {"video_id": "v2", "score": 0.20, "rationale": "Близкий вариант."},
            ], confidence="medium"),
            factstop_error=RuntimeError("sonar circuit open"))

    assert captured["factstop_calls"] == 1
    assert out.candidates and out.candidates[out.best_index].video_id == "v1"
    assert not out.need_clarify
    assert any("fact-stop unsure, delivering" in r.message
               for r in caplog.records)


# ════ 5. no reference source: the gate is off entirely ════

@pytest.mark.asyncio
async def test_no_reference_source_gate_skipped(bothub_client):
    """No source name anywhere (today's extract contract for from_source):
    even a very weak pool (0.20, confidence low) keeps today's behavior —
    no verification call, the ranked result returns as-is."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, "песня из фильма со шницелями",
        _extract_json(), _weak_pool(),
        rank=_rank_json(ranked=[
            {"video_id": "v1", "score": 0.20, "rationale": "Саундтрек фильма."},
            {"video_id": "v2", "score": 0.10, "rationale": "Близкий вариант."},
        ], confidence="low"),
        factstop=_FACTSTOP_NO)  # would reject — must never be asked

    assert captured.get("factstop_calls", 0) == 0
    assert "sonar-factstop" not in captured["calls"]
    assert out.candidates and out.candidates[out.best_index].video_id == "v1"
    assert not out.need_clarify


# ════ 6. the gate's trigger and anchor tolerance ════

@pytest.mark.asyncio
async def test_low_confidence_triggers_at_high_score(bothub_client):
    """The rank tail's own notion: confidence "low" fires the gate even at
    score 0.50 >= 0.4 (the design's "or the rank tail returns low conf").
    Sonar says yes -> delivered, one call."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, "песня из мадагаскар 2 где шницели падали на алекса",
        _extract_json(source=_MAD2), _weak_pool(),
        rank=_rank_json(ranked=[
            {"video_id": "v1", "score": 0.50, "rationale": "Саундтрек фильма."},
            {"video_id": "v2", "score": 0.40, "rationale": "Близкий вариант."},
        ], confidence="low"),
        factstop=_FACTSTOP_YES)

    assert captured["factstop_calls"] == 1
    assert out.candidates and out.candidates[out.best_index].video_id == "v1"


@pytest.mark.asyncio
async def test_source_anchor_read_from_any_carry_point(bothub_client):
    """The gate does not depend on WHICH anchor field carries the source
    (the llm_search source-anchor work owns the exact key): a top-level
    extract["source_anchor"] fires the gate exactly like
    reference["source"] does."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, "song from the second madagascar movie",
        _extract_json(top_anchor=_MAD2), _weak_pool(),
        rank=_rank_json(confidence="medium"), factstop=_FACTSTOP_NO)

    assert captured["factstop_calls"] == 1
    assert 'саундтреке "Madagascar 2"' in captured["factstop_q"]
    assert out.need_clarify and not out.candidates


@pytest.mark.asyncio
async def test_provenance_source_value_is_not_a_source(bothub_client):
    """reference["source"] == "llm"/"sonar" is resolution PROVENANCE
    (ref_resolved's contract), never a film title: the gate stays off."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, "песня из мадагаскар 2",
        _extract_json(source="llm"), _weak_pool(),
        rank=_rank_json(confidence="low"), factstop=_FACTSTOP_NO)

    assert captured.get("factstop_calls", 0) == 0
    assert out.candidates


def test_reference_source_reads_a_ctx_level_anchor():
    """If the anchor ever lands directly on the context (a PipelineContext
    field), _reference_source finds it too — pure unit check."""
    ctx = PipelineContext(description="d", extract={})
    ctx.source_anchor = "Fallout"
    assert _reference_source(ctx) == "Fallout"
    assert _reference_source(PipelineContext(description="d", extract={})) == ""


@pytest.mark.asyncio
async def test_sonar_disabled_skips_verification(bothub_client):
    """No sonar layer -> no verification possible: today's behavior (the
    weak pool delivers as the rank tail decided), no sonar-factstop call."""
    assert not bothub_client.sonar_enabled
    out, captured = await _run(
        bothub_client, "песня из мадагаскар 2",
        _extract_json(source=_MAD2), _weak_pool(),
        rank=_rank_json(confidence="low"))

    assert captured["calls"] == ["llm-rank"]   # not even a ground call
    assert out.candidates and out.candidates[out.best_index].video_id == "v1"


# ════ 7. scope: already-clarifying tail and the analog chain ════

@pytest.mark.asyncio
async def test_rank_already_clarifying_skips_verification(bothub_client):
    """The rank tail already asked to clarify (need_clarify + question):
    nothing would be delivered, so the $0.008 verification is wasted — the
    gate skips and the tail's own clarify question survives."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, "песня из мадагаскар 2",
        _extract_json(source=_MAD2), _weak_pool(),
        rank=_rank_json(confidence="low", need_clarify=True,
                        clarify_question="Из какого фильма трек?"),
        factstop=_FACTSTOP_NO)

    assert captured.get("factstop_calls", 0) == 0
    assert out.need_clarify
    assert out.clarify_question == "Из какого фильма трек?"


@pytest.mark.asyncio
async def test_analog_chain_skips_factstop(bothub_client):
    """The analog chain ranks in SIMILAR mode and delivers tracks
    deliberately NOT from the soundtrack — the membership question would
    always answer "no" there. Even with a source anchor present and a weak
    pool, the gate must not fire (the same plain-flow scope as the
    scene-clip demotion)."""

    class _MetaSvc:
        _lastfm_key = "k"

        async def similar_artists(self, artist, limit=10):
            return ["Neighbor One", "Neighbor Two", "Ref Band"]

    bothub_client._sonar_model = "sonar"
    entries = [
        {"video_id": "r1", "title": "Ref Band - Source Song",
         "channel": "Ref Band - Topic", "duration": 200, "views": 900000},
        {"video_id": "n1", "title": "Neighbor One - First Song",
         "channel": "Neighbor One - Topic", "duration": 210, "views": 500000},
        {"video_id": "n2", "title": "Neighbor Two - Second Song",
         "channel": "Neighbor Two - Topic", "duration": 220, "views": 400000},
    ]
    rank = _rank_json(ranked=[
        {"video_id": "r1", "score": 0.30, "rationale": "Сам описанный трек."},
        {"video_id": "n1", "score": 0.20, "rationale": "Звучит похоже."},
        {"video_id": "n2", "score": 0.15, "rationale": "Тоже рядом."},
    ], confidence="low")

    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault("calls", []).append(gen_name)
        if gen_name == "sonar-ground":
            return {"artist": "Ref Band", "track": "Source Song", "sure": True}
        if gen_name == "sonar-factstop":
            captured["factstop_calls"] = captured.get("factstop_calls", 0) + 1
            return _FACTSTOP_NO
        if gen_name == "llm-rank":
            return rank
        raise AssertionError(f"unexpected LLM call: {gen_name}")

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                return entries
            mock_yt.side_effect = cap
            ctx = PipelineContext(
                description="Аналог песенки из Мадагаскара 2, когда шницели "
                            "падали на Алекса",
                metadata_svc=_MetaSvc(), bothub=bothub_client,
                extract=_extract_json(source=_MAD2),
                t_start=time.perf_counter(), t_extract=0.05)
            out = await FromSourcePipeline().run(ctx)

    assert captured.get("factstop_calls", 0) == 0
    assert "sonar-factstop" not in captured["calls"]
    # the chain delivered its similar-mode result as today
    assert out.candidates and out.candidates[out.best_index].video_id == "n1"
    assert not out.need_clarify


# ════ 8. the latch and the tolerant parse (units) ════

@pytest.mark.asyncio
async def test_verification_latched_once_per_search(bothub_client):
    """The ctx-level latch: the first verdict is cached on the context and
    every later verification reuses it — at most ONE sonar call per search,
    whatever calls the flow makes."""
    bothub_client._sonar_model = "sonar"
    calls = []

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        calls.append(gen_name)
        return _FACTSTOP_NO

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        ctx = PipelineContext(description="d", bothub=bothub_client, extract={})
        first = await _factstop_verify(ctx, "q1")
        second = await _factstop_verify(ctx, "q2")

    assert calls == ["sonar-factstop"]     # the second ask never hit the wire
    assert first == second == ("no", _MAD2_SONGS)


@pytest.mark.asyncio
async def test_verification_failure_never_raises(bothub_client):
    """An exploding bothub call parses as unsure — the helper never raises
    (the _sonar_ground doctrine)."""
    bothub_client._sonar_model = "sonar"

    async def fake_chat(*a, **kw):
        raise RuntimeError("boom")

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        ctx = PipelineContext(description="d", bothub=bothub_client, extract={})
        assert await _factstop_verify(ctx, "q") == ("unsure", [])


@pytest.mark.parametrize("ans,verdict", [
    ({"in_soundtrack": True, "sure": True}, "yes"),
    ({"in_soundtrack": True, "sure": True, "soundtrack_songs": ["a - b"]}, "yes"),
    ({"in_soundtrack": False, "sure": True}, "no"),
    ({"in_soundtrack": False, "sure": False}, "no"),      # a "no" rejects as-is
    ({"in_soundtrack": True, "sure": False}, "unsure"),   # yes-but-not-sure
    ({"in_soundtrack": True}, "unsure"),                  # sure missing
    ({"soundtrack_songs": ["a - b"]}, "unsure"),          # flag missing
    ({}, "unsure"),
    (None, "unsure"),                                     # chat_json garbage
    ("не JSON вообще", "unsure"),
    ({"in_soundtrack": "false", "sure": "true"}, "no"),   # string flags
    ({"in_soundtrack": "true", "sure": "да"}, "yes"),
    ({"in_soundtrack": "может быть", "sure": True}, "unsure"),
])
def test_factstop_verdict_tolerant_parse(ans, verdict):
    assert _factstop_verdict(ans)[0] == verdict


def test_factstop_verdict_songs_cleaned_and_capped():
    """Up to 3 songs, whitespace-stripped, non-strings dropped — the message
    can only name what certainly parsed."""
    verdict, songs = _factstop_verdict({
        "in_soundtrack": False, "sure": True,
        "soundtrack_songs": ["  s1 ", "s2", "s3", "s4", "", 42, None]})
    assert verdict == "no"
    assert songs == ["s1", "s2", "s3"]


# ════ 9. the EN message variant ════

@pytest.mark.asyncio
async def test_reject_message_english_variant(bothub_client):
    """signals.language is the pipeline's only lang notion: a clearly-EN
    query gets the EN wording of the П2 message."""
    bothub_client._sonar_model = "sonar"
    out, _ = await _run(
        bothub_client, "song from madagascar 2 schnitzel scene",
        _extract_json(source=_MAD2, lang="en"), _weak_pool(),
        rank=_rank_json(confidence="medium"), factstop=_FACTSTOP_NO)

    assert out.need_clarify and not out.candidates
    q = out.clarify_question
    assert f'I can\'t confirm this is from "{_MAD2}".' in q
    assert "Here is what is definitely on the soundtrack:" in q
    assert "Tell me which one, or describe the scene again." in q


# ════ 10. the constants pin the audit design ════

def test_factstop_constants():
    """FACTSTOP_MAX_SCORE=0.4 (the gate), FACTSTOP_REJECT_SCORE=0.25 (the
    unsure band floor) — the П2 numbers."""
    assert FACTSTOP_MAX_SCORE == 0.4
    assert FACTSTOP_REJECT_SCORE == 0.25
