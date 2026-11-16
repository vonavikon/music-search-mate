"""Tests for the F1 The Movie incident fixes (prod 13-14.11) in
FromSourcePipeline: the source-anchor out (П1 dead clarify memory) and the
sonar soundtrack seeding of the YouTube query list.

The incident: the user hunted the F1 soundtrack by its scene. (1) Every
from_source SearchOut carried the extract contract's empty {artist, track}
— handler._remember_source_anchor (it reads out.reference["source"] or
["name"]) remembered the movie NEVER, so each fragment answer («это рок
трек, там бред пит пробует болид») re-searched source-less. (2) Discovery
queries were only LLM#1's scene descriptions and the pool filled with
trailers, while sonar knows the OFFICIAL soundtrack (Chris Stapleton "Bad
As I Used To Be" is the first ytsearch hit by exact name).

Contract under test:

  - ANCHOR: at EVERY final SearchOut this pipeline builds, a reference
    missing a real "source" gets it from the tolerant _reference_source
    reader (extract/ctx anchor fields, provenance strings excluded). A real
    existing source is never overwritten; a provenance value ("sonar") is
    scrubbed, never delivered; nothing known anywhere -> the reference
    stays source-less (the handler condition stays off).
  - SEED: on a ground-MISS plain search with a known source, ONE sonar
    question (gen_name="sonar-seed", ctx latch) names up to 3 OFFICIAL
    soundtrack songs; only sure answers seed; the songs LEAD the query
    list before the descriptive phrases, deduped, the shared _MAX_QUERIES
    cap respected. Unsure/garbage/exception -> the list is unchanged and
    the search survives. A ground hit keeps its concrete phrase first and
    pays no seed call; the analog chain never seeds.

Mocking follows tests/test_from_source_factstop.py: chat_json patched on
the bothub_client fixture (dispatched by gen_name), search_metadata patched
in services.llm_search, the pipeline driven through its public run() entry.
"""
import time
from unittest.mock import patch

import pytest

from services.llm_search import _MAX_QUERIES
from services.pipelines.base import PipelineContext
from services.pipelines.from_source import (
    FromSourcePipeline,
    _ensure_reference_source,
    _seed_songs,
    _soundtrack_seed,
)


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Same guard as test_from_source_factstop.py: the enrich step must not
    touch the network (candidates here are fake)."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers (the test_from_source_factstop.py patterns) ════

_F1 = "F1 The Movie"
# The incident fragment: names no source of its own (the anchor carries it).
_F1_FRAGMENT = "это рок трек, там бред пит пробует болид"

# The real official soundtrack of the source (what the seed sonar names):
# an exact "Artist - Song" is the first ytsearch hit — the probe (13.11)
# verified p13Cn0mONP8, channel Chris Stapleton.
_F1_SONGS = [
    "Chris Stapleton - Bad As I Used To Be",
    "Eden - Iron Moments",
    "Mychael Danna - Tengo Que Intentar",
]
_SEED_SURE = {"songs": list(_F1_SONGS), "sure": True}


def _extract_json(queries=None, source=None, lang="ru"):
    """LLM#1 extract — a fake parsed dict (from_source flavor). source lands
    in extract["reference"]["source"] (one of the anchor carry points)."""
    extract = {
        "intro": "Ищем: трек из фильма.",
        "intent": "from_source",
        "signals": {"language": lang, "era": None, "vocal": None,
                    "genre": None, "mood": None},
        "target": {"artist": None, "track": None},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries or ["formula 1 movie crash scene soundtrack"],
    }
    if source is not None:
        extract["reference"]["source"] = source
    return extract


def _rank_json():
    """LLM#2 rank — a strong-pool shape (keeps the fact-stop gate off in the
    seed tests: the source is present there and the gate must not spend a
    verification call on a healthy pool)."""
    return {"ranked": [
        {"video_id": "v1", "score": 0.9, "rationale": "Официальный трек фильма."},
        {"video_id": "v2", "score": 0.8, "rationale": "Тоже из саундтрека."},
        {"video_id": "v3", "score": 0.7, "rationale": "Запасной."},
    ], "best_index": 0, "confidence": "high", "verdict": "Вот он.",
        "need_clarify": False, "clarify_question": ""}


def _pool():
    """Fake yt-dlp results (clean audio candidates, no scene clips)."""
    return [
        {"video_id": "v1", "title": "Chris Stapleton - Bad As I Used To Be",
         "channel": "Chris Stapleton - Topic", "duration": 230, "views": 25_000_000},
        {"video_id": "v2", "title": "Eden - Iron Moments (Official Audio)",
         "channel": "Eden - Topic", "duration": 210, "views": 3_000_000},
        {"video_id": "v3", "title": "F1 The Movie Theme",
         "channel": "OST Channel", "duration": 240, "views": 900000},
    ]


async def _run(bothub_client, description, extract, entries, rank=None,
               ground=None, seed=None, seed_error=None, source_anchor=None,
               pipeline=None):
    """Run the pipeline through its public entry on a PipelineContext.
    captured holds the gen_name of every chat_json call, the sonar-ground /
    sonar-seed questions, the seed call count and the discovery queries.
    `ground`/`seed` are the canned parsed answers per sonar call; seed_error
    makes the seed call raise (the timeout/outage path)."""
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault("calls", []).append(gen_name)
        if gen_name == "sonar-ground":
            captured["ground_q"] = user
            return ground
        if gen_name == "sonar-seed":
            captured["seed_q"] = user
            captured["seed_system"] = system
            captured["seed_calls"] = captured.get("seed_calls", 0) + 1
            if seed_error is not None:
                raise seed_error
            return seed
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
            if source_anchor is not None:
                ctx.source_anchor = source_anchor
            out = await (pipeline or FromSourcePipeline()).run(ctx)
    return out, captured


# ════ 1. the anchor reaches every final SearchOut (П1 fix) ════

@pytest.mark.asyncio
async def test_anchor_lands_in_final_reference(bothub_client):
    """The incident defect: extract's reference is {artist, track} with NO
    source (the from_source extract contract) and the anchor rides
    ctx.source_anchor -> the final SearchOut.reference["source"] IS the
    movie, exactly what handler._remember_source_anchor reads (its own
    condition on the reference must now fire)."""
    assert not bothub_client.sonar_enabled      # the plain no-sonar flow
    out, captured = await _run(
        bothub_client, _F1_FRAGMENT, _extract_json(), _pool(),
        source_anchor=_F1)

    assert out.candidates and out.best_index == 0
    assert out.reference["source"] == _F1
    # the handler's exact read (out.reference["source"] or ["name"]) fires
    ref = out.reference
    handler_source = ((ref.get("source") or ref.get("name") or "")
                      if isinstance(ref, dict) else "").strip()
    assert handler_source == _F1
    assert captured["calls"] == ["llm-rank"]     # no sonar layer was needed


@pytest.mark.asyncio
async def test_own_source_not_overwritten(bothub_client):
    """The query names its own source and the chain (LLM#1 following the
    УНАСЛЕДОВАННЫЙ ИСТОЧНИК rule, or the llm_search anchor inheritance)
    already filled reference.source -> the ctx anchor must NOT clobber it."""
    out, _ = await _run(
        bothub_client, _F1_FRAGMENT, _extract_json(source="Такси 2"), _pool(),
        source_anchor=_F1)

    assert out.reference["source"] == "Такси 2"
    assert _F1 not in str(out.reference)


@pytest.mark.asyncio
async def test_no_source_anywhere_reference_stays_sourceless(bothub_client):
    """Nothing carries a source name (the pre-fix from_source reality): the
    reference keeps the extract's exact shape — no phantom source key, so
    handler._remember_source_anchor's condition never fires."""
    out, _ = await _run(bothub_client, _F1_FRAGMENT, _extract_json(), _pool())

    assert "source" not in out.reference
    assert not (out.reference.get("source") or "").strip()


@pytest.mark.asyncio
async def test_provenance_string_scrubbed_when_anchor_known(bothub_client):
    """extract.reference["source"] == "sonar" is resolution PROVENANCE
    (ref_resolved's contract leaking into the extract), never a film title:
    with a real anchor known it is REPLACED by the movie — "sonar" must not
    poison the handler memory as a remembered film name."""
    out, _ = await _run(
        bothub_client, _F1_FRAGMENT, _extract_json(source="sonar"), _pool(),
        source_anchor=_F1)

    assert out.reference.get("source") == _F1
    assert out.reference.get("source") != "sonar"


@pytest.mark.asyncio
async def test_provenance_string_alone_never_delivered(bothub_client):
    """The same leak with NOTHING else known: the provenance string is
    scrubbed to empty rather than delivered — the handler condition stays
    off exactly as if no source existed."""
    out, _ = await _run(
        bothub_client, _F1_FRAGMENT, _extract_json(source="sonar"), _pool())

    assert not (out.reference.get("source") or "").strip()


# ════ 2. sonar seeding of the query list ════

@pytest.mark.asyncio
async def test_sure_songs_seed_the_query_list(bothub_client):
    """The incident fix: sonar (web-grounded) returns sure + 3 official
    soundtrack songs -> they LEAD the discovery queries in order, the
    descriptive phrases follow, and the whole list respects the shared
    _MAX_QUERIES cap (7 descriptives + 3 seeds -> 8: 5 descriptives
    survive). The seed question names the source."""
    bothub_client._sonar_model = "sonar"
    descriptives = [
        "formula 1 movie crash scene soundtrack",
        "F1 film score first drive scene",
        "F1 The Movie OST scene music",
        "music from F1 movie garage scene",
        "F1 film soundtrack list",
        "formula one movie theme song",
        "F1 movie pit stop scene music",
    ]
    out, captured = await _run(
        bothub_client, _F1_FRAGMENT, _extract_json(queries=descriptives),
        _pool(), seed=_SEED_SURE, source_anchor=_F1)

    assert captured["queries"] == _F1_SONGS + descriptives[:_MAX_QUERIES - 3]
    assert len(captured["queries"]) == _MAX_QUERIES
    # the call order: ground first (a miss), the seed, then the rank
    assert captured["calls"] == ["sonar-ground", "sonar-seed", "llm-rank"]
    # the seed question asks for the OFFICIAL soundtrack of THE source
    assert _F1 in captured["seed_q"]
    assert "ОФИЦИАЛЬНОГО саундтрека" in captured["seed_q"]
    # the pipeline delivered as usual (and the anchor fix rides along)
    assert out.candidates and not out.need_clarify
    assert out.reference["source"] == _F1


@pytest.mark.asyncio
async def test_unsure_answer_leaves_queries_unchanged(bothub_client):
    """sure=false -> NO seeding: the query list is byte-identical to the
    extract's (a hallucinated title as query #1 is worse than none)."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, _F1_FRAGMENT, _extract_json(), _pool(),
        seed={"songs": list(_F1_SONGS), "sure": False}, source_anchor=_F1)

    assert captured["seed_calls"] == 1          # asked once, ignored
    assert captured["queries"] == ["formula 1 movie crash scene soundtrack"]
    assert out.candidates


@pytest.mark.asyncio
async def test_seed_exception_never_breaks_search(bothub_client):
    """A raising seed call (timeout/outage shape) is swallowed: the queries
    are unchanged and the pipeline still delivers — the seeding layer is
    strictly auxiliary (the _sonar_ground doctrine)."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, _F1_FRAGMENT, _extract_json(), _pool(),
        seed_error=RuntimeError("sonar circuit open"), source_anchor=_F1)

    assert captured["seed_calls"] == 1
    assert captured["queries"] == ["formula 1 movie crash scene soundtrack"]
    assert out.candidates and not out.need_clarify


@pytest.mark.asyncio
async def test_five_songs_capped_at_three(bothub_client):
    """The prompt allows up to 3; a 5-song answer seeds exactly the FIRST
    3, in answer order."""
    bothub_client._sonar_model = "sonar"
    five = _F1_SONGS + ["Rosenfeld - Vai Vedrai", "Kenshi Yonezu - Forever Young"]
    out, captured = await _run(
        bothub_client, _F1_FRAGMENT, _extract_json(), _pool(),
        seed={"songs": five, "sure": True}, source_anchor=_F1)

    assert captured["queries"][:3] == _F1_SONGS
    assert captured["queries"][3] == "formula 1 movie crash scene soundtrack"


@pytest.mark.asyncio
async def test_garbage_answer_skips_seeding(bothub_client):
    """A non-dict sonar answer (chat_json garbage) parses to no songs: the
    queries are unchanged, nothing raises."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, _F1_FRAGMENT, _extract_json(), _pool(),
        seed="не JSON вообще", source_anchor=_F1)

    assert captured["seed_calls"] == 1
    assert captured["queries"] == ["formula 1 movie crash scene soundtrack"]
    assert out.candidates


@pytest.mark.asyncio
async def test_dirty_song_lines_cleaned_or_dropped(bothub_client):
    """Hostile payload: newlines/tabs/control characters inside song lines,
    empty/whitespace-only strings, non-string entries -> the seed is
    CLEANED (one-line songs) or dropped — never garbage in the queries,
    never a crash."""
    bothub_client._sonar_model = "sonar"
    dirty = {
        "sure": True,
        "songs": [
            "Chris Stapleton -\nBad As I Used To Be\t",
            "   ",
            "",
            42,
            None,
            "Eden - Iron\x00Moments",
        ],
    }
    out, captured = await _run(
        bothub_client, _F1_FRAGMENT, _extract_json(), _pool(),
        seed=dirty, source_anchor=_F1)

    assert captured["queries"][:2] == [
        "Chris Stapleton - Bad As I Used To Be",
        "Eden - Iron Moments",
    ]
    assert "\n" not in " ".join(captured["queries"])
    assert out.candidates


@pytest.mark.asyncio
async def test_ground_hit_keeps_concrete_first_and_skips_seed(bothub_client):
    """A SUCCESSFUL ground resolution is the strongest signal: its concrete
    phrase stays query #0 and the seed question is never asked (no extra
    sonar call on the healthy path — the fact-stop guards the delivery at
    the other end)."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, "музыка из фильма f1 где бред пит пробует болид",
        _extract_json(), _pool(),
        ground={"artist": "Chris Stapleton",
                "track": "Bad As I Used To Be", "sure": True},
        seed=_SEED_SURE,          # would seed — must never be asked
        source_anchor=_F1)

    assert "sonar-seed" not in captured["calls"]
    assert captured["queries"][0] == "Chris Stapleton - Bad As I Used To Be"
    assert out.candidates


# ════ 3. units: the tolerant parse, the latch, the fill ════

@pytest.mark.parametrize("ans,expected", [
    (None, []),                                    # chat_json garbage
    ("не JSON вообще", []),                        # non-dict
    ({}, []),                                      # sure missing
    ({"songs": ["a - b"]}, []),                    # unsure by default
    ({"sure": False, "songs": ["a - b"]}, []),     # explicit unsure
    ({"sure": True}, []),                          # songs missing
    ({"sure": True, "songs": "не список"}, []),    # songs not a list
    ({"sure": True, "soundtrack_songs": ["a - b"]}, ["a - b"]),  # field name
    ({"sure": "true", "songs": ["a - b"]}, ["a - b"]),           # string flag
    ({"sure": "да", "songs": ["a - b"]}, ["a - b"]),
    ({"sure": True, "songs": ["  s1  ", "s2", "", "   ", 7, None]},
     ["s1", "s2"]),                                # cleaned + non-strings
    ({"sure": True, "songs": ["s1", "s2", "s3", "s4", "s5"]},
     ["s1", "s2", "s3"]),                          # capped at 3
])
def test_seed_songs_tolerant_parse(ans, expected):
    """(2026-11-14, F1 incident) hostile shapes never raise; only sure
    answers with a real list seed, cleaned and capped."""
    assert _seed_songs(ans) == expected


@pytest.mark.asyncio
async def test_seed_latched_once_per_search(bothub_client):
    """The ctx latch (the _factstop_result pattern): the first ask caches
    the songs and every later call in the SAME search reuses them — at most
    one sonar-seed wire call."""
    bothub_client._sonar_model = "sonar"
    calls = []

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        calls.append(gen_name)
        return _SEED_SURE

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        ctx = PipelineContext(
            description="d", bothub=bothub_client,
            extract={"reference": {"source": _F1}})
        first = await _soundtrack_seed(ctx)
        second = await _soundtrack_seed(ctx)

    assert calls == ["sonar-seed"]                # the second ask never hit the wire
    assert first == second == _F1_SONGS


@pytest.mark.asyncio
async def test_no_source_no_seed_call(bothub_client):
    """Without a source name there is no soundtrack to name: [] without a
    single sonar wire call (and the latch still settles the result)."""
    bothub_client._sonar_model = "sonar"
    calls = []

    async def fake_chat(*a, **kw):
        calls.append(kw.get("gen_name"))
        return _SEED_SURE

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        ctx = PipelineContext(description="d", bothub=bothub_client, extract={})
        assert await _soundtrack_seed(ctx) == []
    assert calls == []


def test_ensure_reference_source_reference_none_gets_fresh_dict():
    """A SearchOut with reference=None (a malformed tail shape) still earns
    the anchor: a fresh {"source": s} — the handler reads ref.get("source").
    No source known -> None stays None (never a phantom dict)."""
    from services.llm_search import SearchOut

    ctx = PipelineContext(description="d", extract={})
    ctx.source_anchor = _F1
    out = SearchOut(intent="from_source", signals={}, queries=[], assoc=[],
                    candidates=[], best_index=-1, confidence="low")
    out.reference = None
    assert _ensure_reference_source(ctx, out).reference == {"source": _F1}

    bare = SearchOut(intent="from_source", signals={}, queries=[], assoc=[],
                     candidates=[], best_index=-1, confidence="low")
    bare.reference = None
    assert _ensure_reference_source(
        PipelineContext(description="d", extract={}), bare).reference is None
