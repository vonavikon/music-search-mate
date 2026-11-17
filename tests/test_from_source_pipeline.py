"""Tests for the FromSourcePipeline (Stage 3a of the pipeline refactor).

Contract under test:
  - behavior-preserving move from LegacyPipeline.run: the sonar-ground block
    (gated on bothub.sonar_enabled) asks the same web-grounded question,
    inserts the resolved concrete "{artist} - {track}" as the FIRST search
    phrase with the _ref_key dedup guard, and the flow ranks under
    SYSTEM_RANK_FROM_SOURCE with SearchOut.intent == "from_source";
  - analog chain (the Stage 3 hybrid «похожее на трек из фильма»): an analog
    marker in the description plus a sonar hit switch the search into a
    similar-mode REFERENCE search — getSimilar neighbors lead discovery (the
    reference self-excluded), the reference itself is banned from the best
    even when the LLM ranks it first, the LLM#2 rank call receives
    SYSTEM_RANK_SIMILAR, and SearchOut.intent STILL reports "from_source";
  - marker + sonar miss: the plain from_source flow, unchanged;
  - sonar disabled: no sonar call at all, even with a marker;
  - the analog-marker regex: RU+EN markers match, plain from_source
    phrasings and markers glued into longer Russian words do not;
  - scene-clip demotion (live-test F4): a Film & Animation scene clip or a
    scene-worded title deterministically sinks below the OST track whatever
    the LLM scores say, with the cat=film marker surfaced to the ranker;
    the analog chain (similar mode) and other intents are untouched;
  - stateless shared instance.

ROUTING: services/pipelines/__init__.py maps "from_source" to
FromSourcePipeline (wired by the lead together with deleting the duplicate
sonar block from LegacyPipeline.run); the route() contract lives in
tests/test_pipelines.py. These tests run the pipeline DIRECTLY against a
PipelineContext, the exact object route() hands it in production.

Mocking follows tests/test_similar_pipeline.py: BothubClient.chat_json
patched on the bothub_client fixture (already-parsed dicts, dispatched by
gen_name), search_metadata patched in services.llm_search, no network.
"""
import json
import time

import pytest
from unittest.mock import patch

from services.llm_search import (
    SYSTEM_RANK_FROM_SOURCE,
    SYSTEM_RANK_SIMILAR,
    _is_film_category,
    _is_scene_clip_title,
)
from services.pipelines.base import PipelineContext
from services.pipelines.from_source import (
    _ANALOG_RE,
    _source_fact_key,
    FromSourcePipeline,
)


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Same guard as test_similar_pipeline.py: the enrich step must not touch
    the network (candidates here are fake)."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers (canned fixtures, the similar-test patterns) ════

def _extract_json(intent="from_source", queries=None):
    """LLM#1 extract — a fake parsed dict (from_source flavor)."""
    return {
        "intro": "Ищем: трек из фильма или рекламы.",
        "intent": intent,
        "signals": {"language": "ru", "era": None, "vocal": None,
                    "genre": None, "mood": None},
        "target": {"artist": None, "track": None},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries or ["soundtrack query 1", "soundtrack query 2"],
    }


def _rank_json(ranked=None, best_index=0, confidence="high"):
    """LLM#2 rank — a fake parsed dict; `ranked` overrides the canned scores."""
    if ranked is None:
        ranked = [
            {"video_id": "v1", "score": 0.95, "rationale": "Тот самый трек."},
            {"video_id": "v2", "score": 0.80, "rationale": "Близкий вариант."},
            {"video_id": "v3", "score": 0.70, "rationale": "Запасной."},
        ]
    return {
        "ranked": ranked,
        "best_index": best_index,
        "confidence": confidence,
        "verdict": "Вот эта композиция.",
        "need_clarify": False,
        "clarify_question": "",
    }


def _yt_entries():
    """Fake yt-dlp results (3 candidates)."""
    return [
        {"video_id": "v1", "title": "Cool Soundtrack Theme", "channel": "Film Orchestra",
         "duration": 240, "views": 500000},
        {"video_id": "v2", "title": "Another OST Cue", "channel": "Score House - Topic",
         "duration": 300, "views": 100000},
        {"video_id": "v3", "title": "Soft End Titles", "channel": "Piano Duo",
         "duration": 270, "views": 50000},
    ]


class _MetaSvc:
    """getSimilar for the resolved reference, recording calls. The Last.fm
    enrich layer keys on _lastfm_key but its getinfo/toptags calls raise
    inside enrich_candidates' swallowed try/except (the _MetaSvcQ89 pattern
    from test_similar_pipeline.py)."""

    _lastfm_key = "k"

    def __init__(self, neighbors=None):
        self._neighbors = neighbors if neighbors is not None else []
        self.calls = []

    async def similar_artists(self, artist, limit=10):
        self.calls.append(("getSimilar", artist))
        return self._neighbors


async def _run(bothub_client, description, extract, entries, rank=None,
               sonar=None, metadata_svc=None, pipeline=None, store=None):
    """Run a pipeline DIRECTLY on a PipelineContext (default:
    FromSourcePipeline — the registry is not wired for from_source yet, see
    the routing note). Returns (SearchOut, captured); captured holds
    (gen_name, system) per chat_json call, the sonar question, the discovery
    queries and the llm-rank input message. Any generation other than
    sonar-ground / llm-rank fails the test (the pipeline runs the tail
    only; LLM#1 belongs to the search head). store is the source-fact cache
    injection (None = the pre-cache flow, the surprise-pipeline ctx.store
    contract)."""
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault("calls", []).append((gen_name, system))
        if gen_name == "sonar-ground":
            captured["sonar_q"] = user
            return sonar
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
                description=description,
                metadata_svc=metadata_svc,
                bothub=bothub_client,
                extract=extract,
                t_start=time.perf_counter(),
                t_extract=0.05,
                store=store,
            )
            out = await (pipeline or FromSourcePipeline()).run(ctx)
    return out, captured


_SONAR_HIT = {"artist": "Ref Band", "track": "Source Song", "sure": True}
_CONCRETE = "Ref Band - Source Song"
_PROD_DESCRIPTION = ("Аналог песенки из Магасскао, "
                     "когда на Алекса во сне падали шницели")


def _ref_entries():
    """The reference upload itself plus two neighbor-band tracks."""
    return [
        {"video_id": "r1", "title": "Ref Band - Source Song",
         "channel": "Ref Band - Topic", "duration": 200, "views": 900000},
        {"video_id": "n1", "title": "Neighbor One - First Song",
         "channel": "Neighbor One - Topic", "duration": 210, "views": 500000},
        {"video_id": "n2", "title": "Neighbor Two - Second Song",
         "channel": "Neighbor Two - Topic", "duration": 220, "views": 400000},
    ]


def _ref_tops_rank():
    """LLM#2 ranks the REFERENCE UPLOAD first — only the deterministic ban
    can keep it from winning (the point of the analog-chain test)."""
    return _rank_json(ranked=[
        {"video_id": "r1", "score": 0.98, "rationale": "Сам описанный трек."},
        {"video_id": "n1", "score": 0.75, "rationale": "Звучит похоже."},
        {"video_id": "n2", "score": 0.60, "rationale": "Тоже рядом."},
    ], best_index=0)


# ════ 1. behavior-preserving move (no analog marker) ════

@pytest.mark.asyncio
async def test_e2e_sonar_concrete_first_and_from_source_prompt(bothub_client):
    """Plain from_source (no marker): the sonar hit becomes the FIRST search
    phrase (an equal LLM query deduped by ref_key), the rank call receives
    SYSTEM_RANK_FROM_SOURCE, SearchOut.intent stays "from_source" and the
    output keeps the ranked shape. No getSimilar without the chain."""
    bothub_client._sonar_model = "sonar"  # the layer is enabled
    svc = _MetaSvc()
    concrete = "Bill Medley & Jennifer Warnes - (I've Had) The Time of My Life"
    extract = _extract_json(queries=[
        "carlsberg ad song",
        # ref_key-equal to the sonar phrase (& == and after normalization):
        # the dedup guard must fold it into the inserted concrete query
        "Bill Medley and Jennifer Warnes - (I've Had) The Time of My Life",
    ])

    out, captured = await _run(
        bothub_client, "песня из рекламы Carlsberg", extract, _yt_entries(),
        sonar={"artist": "Bill Medley & Jennifer Warnes",
               "track": "(I've Had) The Time of My Life", "sure": True},
        metadata_svc=svc)

    # the sonar question is unchanged by the move (plus the wave-3c
    # transliteration hint for source names)
    assert captured["sonar_q"] == ("Пользователь ищет музыку: "
                                   "'песня из рекламы Carlsberg'. "
                                   "Какой это трек (артист и название)? "
                                   "Название фильма/игры/сериала может быть русской "
                                   "транслитерацией (Стелларис→Stellaris, "
                                   "Магасскао→Madagascar) — ищи оригинальное название.")
    # concrete first, the equal query deduped, the other LLM query kept
    assert captured["queries"] == [concrete, "carlsberg ad song"]
    # prompt selection: rank under the from_source split, no extract call
    assert [g for g, _ in captured["calls"]] == ["sonar-ground", "llm-rank"]
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_FROM_SOURCE]
    # SearchOut shape and the from_source scenario kept
    assert out.intent == "from_source"
    assert out.confidence == "high"
    assert out.best_index == 0
    assert len(out.candidates) == 3
    assert out.candidates[0].match_score > 0
    assert out.verdict == "Вот эта композиция."
    assert not out.need_clarify
    # the plain flow never touches getSimilar
    assert svc.calls == []


# ════ 2. the analog chain ════

@pytest.mark.asyncio
async def test_analog_chain_similar_mode_reference_banned(bothub_client):
    """The prod case end to end: marker + sonar hit -> the getSimilar
    neighbors lead discovery (the reference self-excluded), the reference
    upload is banned from the best EVEN THOUGH LLM#2 ranked it first, the
    rank call receives SYSTEM_RANK_SIMILAR, and SearchOut.intent stays
    "from_source" (analytics keep the source scenario)."""
    bothub_client._sonar_model = "sonar"
    svc = _MetaSvc(neighbors=["Neighbor One", "Neighbor Two", "Ref Band",
                              "Neighbor Three"])

    out, captured = await _run(
        bothub_client, _PROD_DESCRIPTION, _extract_json(), _ref_entries(),
        rank=_ref_tops_rank(), sonar=_SONAR_HIT, metadata_svc=svc)

    # getSimilar on the sonar-resolved (canonical) artist — no track_search
    assert svc.calls == [("getSimilar", "Ref Band")]
    # neighbors lead discovery, the reference self-excluded, the concrete
    # sonar phrase and the LLM queries follow
    assert captured["queries"][:2] == ["Neighbor One", "Neighbor Two"]
    assert _CONCRETE in captured["queries"]
    assert "Ref Band" not in captured["queries"], \
        "the reference itself must not lead discovery"
    assert "soundtrack query 1" in captured["queries"]
    # the rank call runs under the SIMILAR prompt (sonic match + reference
    # disqualification), while SearchOut.intent keeps the source scenario
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_SIMILAR]
    assert out.intent == "from_source"
    # the reference upload (LLM's #1) is banned: best = first non-reference,
    # the reference sinks to the tail
    assert out.candidates[out.best_index].video_id == "n1"
    assert out.candidates[-1].video_id == "r1"
    # the resolved reference is carried in the output for the ban/analytics
    assert out.reference["artist"] == "Ref Band"
    assert out.reference["track"] == "Source Song"


@pytest.mark.asyncio
async def test_analog_chain_without_metadata_svc(bothub_client):
    """The chain with no metadata service: no neighbors to prepend (the
    concrete sonar phrase leads), but the similar-mode contract still holds —
    SIMILAR prompt and the reference ban from ref_resolved alone."""
    bothub_client._sonar_model = "sonar"

    out, captured = await _run(
        bothub_client, _PROD_DESCRIPTION, _extract_json(), _ref_entries(),
        rank=_ref_tops_rank(), sonar=_SONAR_HIT, metadata_svc=None)

    assert captured["queries"][0] == _CONCRETE
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_SIMILAR]
    assert out.intent == "from_source"
    assert out.candidates[out.best_index].video_id == "n1"
    assert out.candidates[-1].video_id == "r1"


@pytest.mark.asyncio
async def test_analog_marker_sonar_miss_keeps_plain_flow(bothub_client):
    """Marker present but the sonar ground misses (sure=false): the plain
    from_source flow — LLM queries search as-is, FROM_SOURCE prompt, no
    getSimilar, no reference populated."""
    bothub_client._sonar_model = "sonar"
    svc = _MetaSvc()
    extract = _extract_json(queries=["magasskao song", "schnitzel dream scene"])

    out, captured = await _run(
        bothub_client, _PROD_DESCRIPTION, extract, _yt_entries(),
        sonar={"artist": "Кто-то", "track": "", "sure": False}, metadata_svc=svc)

    assert "sonar-ground" in [g for g, _ in captured["calls"]]  # the attempt was made
    assert captured["queries"] == ["magasskao song", "schnitzel dream scene"]
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_FROM_SOURCE]
    assert out.intent == "from_source"
    assert svc.calls == []
    assert not (out.reference.get("artist") or "")
    assert out.candidates[out.best_index].video_id == "v1"


@pytest.mark.asyncio
async def test_sonar_disabled_no_sonar_call_even_with_marker(bothub_client):
    """The layer is off (empty sonar model): not a single sonar call — the
    marker alone never fires the chain (no resolution to chain from)."""
    assert not bothub_client.sonar_enabled
    extract = _extract_json(queries=["magasskao song"])

    out, captured = await _run(
        bothub_client, _PROD_DESCRIPTION, extract, _yt_entries())

    assert [g for g, _ in captured["calls"]] == ["llm-rank"]
    assert captured["queries"] == ["magasskao song"]
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_FROM_SOURCE]
    assert out.intent == "from_source"
    assert len(out.candidates) == 3


# ════ 3. the analog-marker regex ════

@pytest.mark.parametrize("text", [
    _PROD_DESCRIPTION,                       # the prod case itself
    "Хочу похожее на трек из фильма Такси",
    "похож на тему из Пиратов Карибского моря",   # the bare short form
    "что-нибудь напоминающее музыку из сериала Друзья",
    "в духе саундтрека из Интерстеллар",
    "аналогичная песня из рекламы Nike",
    "similar to the song from the Carl's Jr ad",
    "like the track from Fallout",
    "analogous to the music from Mad Men",
])
def test_analog_marker_matches(text):
    """RU+EN analog markers (with natural morphology) fire the detector."""
    assert _ANALOG_RE.search(text)


@pytest.mark.parametrize("text", [
    "песня из рекламы Carlsberg",             # plain from_source
    "трек из фильма Побег из Шоушенка",
    "музыка из игры Doom",
    "неаналог чего-либо",                     # glued into a longer RU word
    "саундтрек похожего духа",                # no «на» after похож\w*
])
def test_analog_marker_no_false_positive(text):
    """Plain from_source phrasings and boundary-violating near-misses do
    not fire the detector."""
    assert not _ANALOG_RE.search(text)


# ════ 4. instance contract ════

def test_from_source_pipeline_is_stateless():
    """One shared instance with no mutable instance attributes: safe for
    concurrent chats (the registry contract)."""
    p = FromSourcePipeline()
    assert p.name == "from_source"
    mutable = {k: v for k, v in vars(p).items()
               if isinstance(v, (list, dict, set))}
    assert not mutable, "the pipeline must hold no mutable instance state"


# ════ 5. scene-clip demotion (live-test F4, 2026-11-05) ════
# «музыка из погони такси 2»: the Film & Animation scene clip "Taxi 2
# (La course)" took llm-rank 0.95 over the OST track at 0.85 — the
# deterministic demotion must beat that LLM gap.

_SCENE_CLIP_ID = "r8h7EIUaIFw"   # the live video id of the scene clip
_OST_ID = "ostvid1"


def _live_entries():
    """The live F4 pool: the scene clip plus the real OST track (and a
    filler so the card has a tail)."""
    return [
        {"video_id": _SCENE_CLIP_ID, "title": "Taxi 2 (La course)",
         "channel": "Taxi Channel", "duration": 180, "views": 2500000},
        {"video_id": _OST_ID,
         "title": "Akhenaton, La Cosca Scratch by DJ Pone - Ultime Poursuite (OST Taxi)",
         "channel": "Taxi OST - Topic", "duration": 260, "views": 800000},
        {"video_id": "ostvid2", "title": "Sur La Route - Taxi OST",
         "channel": "Score House - Topic", "duration": 300, "views": 90000},
    ]


def _live_rank():
    """LLM#2 ranks the scene clip FIRST (the live failure: exact title match
    beat the official OST track)."""
    return _rank_json(ranked=[
        {"video_id": _SCENE_CLIP_ID, "score": 0.95,
         "rationale": "Точное совпадение названия фильма."},
        {"video_id": _OST_ID, "score": 0.85,
         "rationale": "Официальный трек из саундтрека."},
        {"video_id": "ostvid2", "score": 0.40, "rationale": "Запасной."},
    ], best_index=0)


def _film_enrich(url):
    """The enrich pass for the live pool: the scene clip is categorized
    Film & Animation, the OST uploads Music (existing data — the category
    part of the fix adds no new yt-dlp calls)."""
    if _SCENE_CLIP_ID in url:
        return {"categories": ["Film & Animation"]}
    if _OST_ID in url:
        return {"categories": ["Music"]}
    return None


@pytest.mark.asyncio
async def test_live_case_scene_clip_demoted_below_ost(bothub_client):
    """The live case shape: clip 0.95 vs OST 0.85 -> final best = the OST.
    The scene clip sinks to the tail, its match_score takes the scene-clip
    penalty, and the ranker's input carried the cat=film marker (the signal
    the LLM was missing)."""
    assert not bothub_client.sonar_enabled  # plain flow, no sonar call

    with patch("services.search._extract_info_no_dl", side_effect=_film_enrich):
        out, captured = await _run(
            bothub_client, "музыка из погони такси 2", _extract_json(),
            _live_entries(), rank=_live_rank())

    # the deterministic demotion beats the 0.95-vs-0.85 LLM gap
    assert out.candidates[out.best_index].video_id == _OST_ID
    assert out.candidates[-1].video_id == _SCENE_CLIP_ID
    # the penalty is visible in the calibrated score breakdown
    scene = out.candidates[-1]
    assert "scene-clip -0.20" in scene.match_reasons
    # the LLM saw the category signal on the clip line (and only there)
    assert captured["rank_input"].count("cat=film") == 1
    assert f"video_id={_SCENE_CLIP_ID}" in captured["rank_input"]


@pytest.mark.asyncio
async def test_scene_title_words_demoted_without_category(bothub_client):
    """The title-regex path alone (enrich gave no category — the flat search
    output has none): a RU scene word in the title demotes even when the
    category signal is absent. «фрагмент» is deliberately NOT in _NOISE_RE,
    so this isolates the new signal from the [noise?] machinery."""
    entries = [
        {"video_id": "frag1", "title": "Такси 2 фрагмент погони",
         "channel": "Taxi Channel", "duration": 200, "views": 1500000},
        {"video_id": _OST_ID, "title": "Ultime Poursuite (OST Taxi)",
         "channel": "Taxi OST - Topic", "duration": 260, "views": 800000},
    ]
    rank = _rank_json(ranked=[
        {"video_id": "frag1", "score": 0.95, "rationale": "Погоня из фильма."},
        {"video_id": _OST_ID, "score": 0.85, "rationale": "Трек саундтрека."},
    ], best_index=0)

    # the autouse fixture keeps _extract_info_no_dl at None: no categories
    out, _ = await _run(bothub_client, "музыка из погони такси 2",
                        _extract_json(), entries, rank=rank)

    assert out.candidates[out.best_index].video_id == _OST_ID
    assert out.candidates[-1].video_id == "frag1"
    assert "scene-clip -0.20" in out.candidates[-1].match_reasons


@pytest.mark.asyncio
async def test_scene_clip_all_clean_pool_unchanged(bothub_client):
    """A pool with no scene signal (the normal case): order and best are
    exactly the LLM's — the demotion is a no-op."""
    out, _ = await _run(bothub_client, "песня из рекламы Carlsberg",
                        _extract_json(), _yt_entries())

    assert out.candidates[out.best_index].video_id == "v1"
    assert [c.video_id for c in out.candidates] == ["v1", "v2", "v3"]
    assert all("scene-clip" not in (c.match_reasons or "") for c in out.candidates)


@pytest.mark.asyncio
async def test_analog_chain_skips_scene_clip_demotion(bothub_client):
    """The analog chain ranks in SIMILAR mode: a scene-titled neighbor keeps
    the LLM's top spot — the from_source demotion must not leak into the
    chain (F4 scope is the plain flow)."""
    bothub_client._sonar_model = "sonar"
    svc = _MetaSvc(neighbors=["Neighbor One", "Neighbor Two", "Ref Band"])
    entries = [
        {"video_id": "r1", "title": "Ref Band - Source Song",
         "channel": "Ref Band - Topic", "duration": 200, "views": 900000},
        {"video_id": "n1", "title": "Neighbor One - Chase Scene (from the movie)",
         "channel": "Neighbor One - Topic", "duration": 210, "views": 500000},
        {"video_id": "n2", "title": "Neighbor Two - Second Song",
         "channel": "Neighbor Two - Topic", "duration": 220, "views": 400000},
    ]
    rank = _rank_json(ranked=[
        {"video_id": "n1", "score": 0.98, "rationale": "Звучит похоже."},
        {"video_id": "n2", "score": 0.60, "rationale": "Тоже рядом."},
        {"video_id": "r1", "score": 0.30, "rationale": "Сам трек."},
    ], best_index=0)

    out, _ = await _run(bothub_client, _PROD_DESCRIPTION, _extract_json(),
                        entries, rank=rank, sonar=_SONAR_HIT, metadata_svc=svc)

    # no demotion: the scene-titled neighbor stays the best (and unbanned:
    # it is not the reference)
    assert out.candidates[out.best_index].video_id == "n1"
    assert "scene-clip" not in (out.candidates[out.best_index].match_reasons or "")


@pytest.mark.asyncio
async def test_this_track_scene_title_not_demoted(bothub_client):
    """Other intents untouched: under this_track a scene-titled candidate
    keeps today's behavior — no tail-drop, no scene-clip penalty (the
    demotion lives in the from_source tail alone)."""
    from services.pipelines.this_track import ThisTrackPipeline

    extract = {
        "intro": "", "intent": "this_track", "signals": {},
        "target": {"artist": "Wanted Artist", "track": "Wanted Track"},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": ["Wanted Artist Wanted Track"],
    }
    # nothing fuzzy-matches the target: no shortcut, the LLM#2 tail decides
    entries = [
        {"video_id": "s1", "title": "Fan Cut - Chase Scene Excerpt",
         "channel": "Fan Cuts", "duration": 200, "views": 500000},
        {"video_id": "s2", "title": "Other Song",
         "channel": "Other Chan", "duration": 210, "views": 400000},
    ]
    rank = _rank_json(ranked=[
        {"video_id": "s1", "score": 0.90, "rationale": "Похоже на искомое."},
        {"video_id": "s2", "score": 0.50, "rationale": "Запасной."},
    ], best_index=0)

    out, _ = await _run(bothub_client, "wanted artist wanted track", extract,
                        entries, rank=rank, pipeline=ThisTrackPipeline())

    assert out.candidates[out.best_index].video_id == "s1"
    assert out.candidates[0].video_id == "s1"
    assert "scene-clip" not in (out.candidates[out.best_index].match_reasons or "")


# ════ 6. the scene-clip title regex ════
# _SCENE_CLIP_RE (RU+EN) keys the deterministic demotion: it must catch
# scene/excerpt/fragment wordings and stay silent on real track titles
# (the film/scene NAME alone — "La course" — is deliberately not a signal).

@pytest.mark.parametrize("title", [
    "Такси 2 - Сцена погони",
    "сцена из фильма Такси 2",
    "Сцены погонь из фильма",
    "Такси 2 фрагмент погони",
    "нарезка сцен погони",
    "отрывок из фильма",
    "эпизод побега",
    "Taxi 2 chase scene",
    "Escape scene, part 1",
    "excerpt from the movie",
    "Shawshank fragment",
    "clip from the film",
    "moment from the movie",
])
def test_scene_clip_title_matches(title):
    """RU+EN scene-clip wordings fire the detector."""
    assert _is_scene_clip_title(title)


@pytest.mark.parametrize("title", [
    "Ultime Poursuite (OST Taxi)",
    "Akhenaton, La Cosca Scratch by DJ Pone - Ultime Poursuite (OST Taxi)",
    "Brooks Was Here",
    "Taxi 2 (La course)",                        # scene NAME alone: no word signal
    "A Whiter Shade of Pale (2007 Remaster)",
    "Come Clean - Hilary Duff (Official Music Video) 4K",
    "Locomotive Breath",
])
def test_scene_clip_title_no_false_positive(title):
    """Real track titles (OST cues included) do not fire the detector."""
    assert not _is_scene_clip_title(title)


def test_film_category_normalized_match():
    """The category signal matches Film & Animation across spacing/case, and
    only it — Music and the _NON_MUSIC_CATEGORIES set do not count."""
    assert _is_film_category(["Film & Animation"])
    assert _is_film_category(["film  &  animation"])
    assert _is_film_category(["Music", "Film & Animation"])
    assert not _is_film_category(["Music"])
    assert not _is_film_category(["Comedy"])      # non-music, handled elsewhere
    assert not _is_film_category(None)
    assert not _is_film_category([])


# ════ 7. source-fact cache (improvement B4, research card П3) ════
# Sonar grounding has run-to-run variance (the Stellaris-organ case): the
# first successful grounding is persisted as a fact, identical later queries
# reuse it and skip the sonar call. Key = ("src", normalize(description),
# "src") inside the shared facts table; TTL = FACT_TTL (30 days, applied by
# get_fact itself). The cache rides ctx.store (the injection point the
# surprise pipeline established); None = the pre-cache flow.


class _StoreStub:
    """The ctx.store contract the cache needs (the services.store module's
    get_fact/save_fact signatures), recording calls. cached is the raw fact
    string get_fact returns (None = miss); read_error/write_error make the
    corresponding op raise — the cache must degrade, never break search."""

    def __init__(self, cached=None, read_error=None, write_error=None):
        self._cached = cached
        self._read_error = read_error
        self._write_error = write_error
        self.get_calls = []
        self.save_calls = []

    async def get_fact(self, artist, track, lang):
        self.get_calls.append((artist, track, lang))
        if self._read_error:
            raise self._read_error
        return self._cached

    async def save_fact(self, artist, track, lang, fact):
        self.save_calls.append((artist, track, lang, fact))
        if self._write_error:
            raise self._write_error


def _cached_payload(artist="Ref Band", track="Source Song"):
    """A stored source-fact row: JSON {artist, track} in the fact column."""
    return json.dumps({"artist": artist, "track": track}, ensure_ascii=False)


def test_source_fact_key_normalized_and_bounded():
    """The key is the full description through fuzzy.normalize: case,
    punctuation and whitespace fold (the same text always hits), an empty or
    over-cap text is a stop-signal ("") — no cache at all."""
    assert _source_fact_key("Песня из рекламы Carlsberg!") == \
        _source_fact_key("  песня  ИЗ рекламы   carlsberg ")
    assert _source_fact_key("Песня из рекламы Carlsberg!") == \
        "песня из рекламы carlsberg"
    assert _source_fact_key("«Такси 2» — погоня") == "такси 2 погоня"
    assert _source_fact_key("") == ""
    assert _source_fact_key("   ") == ""
    assert _source_fact_key("x" * 301) == ""     # over the cap: no caching
    assert _source_fact_key("x" * 300) == "x" * 300


@pytest.mark.asyncio
async def test_source_fact_cache_hit_skips_sonar(bothub_client):
    """Hit: the stored {artist, track} becomes the concrete FIRST query, the
    sonar call is skipped entirely, nothing is re-saved, and the plain
    from_source flow (FROM_SOURCE prompt, intent kept) runs as after a
    sonar hit."""
    bothub_client._sonar_model = "sonar"
    store = _StoreStub(cached=_cached_payload())
    extract = _extract_json(queries=["carlsberg ad song"])

    out, captured = await _run(
        bothub_client, "песня из рекламы Carlsberg", extract, _yt_entries(),
        sonar=_SONAR_HIT, store=store)

    # one read under the namespace tuple, key = normalized description
    assert store.get_calls == [("src", "песня из рекламы carlsberg", "src")]
    # NO sonar call — the cache answered before it
    assert [g for g, _ in captured["calls"]] == ["llm-rank"]
    # the cached track leads discovery, the LLM queries follow
    assert captured["queries"][0] == _CONCRETE
    assert "carlsberg ad song" in captured["queries"]
    # a hit is not re-saved (the first successful grounding stays)
    assert store.save_calls == []
    # the plain flow is intact
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_FROM_SOURCE]
    assert out.intent == "from_source"
    assert out.candidates[out.best_index].video_id == "v1"


@pytest.mark.asyncio
async def test_source_fact_cache_miss_sonar_hit_saves_fact(bothub_client):
    """Miss + sonar hit: the resolved {artist, track} is saved once under the
    namespace key as a JSON payload (stripped fields — exactly the concrete
    phrase the search used), so the next identical query deterministically
    hits."""
    bothub_client._sonar_model = "sonar"
    store = _StoreStub(cached=None)

    out, captured = await _run(
        bothub_client, "песня из рекламы Carlsberg", _extract_json(),
        _yt_entries(),
        sonar={"artist": " Ref Band ", "track": " Source Song ", "sure": True},
        store=store)

    assert store.get_calls == [("src", "песня из рекламы carlsberg", "src")]
    assert [g for g, _ in captured["calls"]] == ["sonar-ground", "llm-rank"]
    assert captured["queries"][0] == _CONCRETE
    # exactly one save, namespaced, payload = the stripped resolution
    assert store.save_calls == [
        ("src", "песня из рекламы carlsberg", "src",
         '{"artist": "Ref Band", "track": "Source Song"}')]
    assert out.intent == "from_source"


@pytest.mark.asyncio
async def test_source_fact_cache_miss_sonar_miss_nothing_saved(bothub_client):
    """Miss + sonar miss (sure=false): the flow is the plain from_source
    search and NOTHING enters the cache — an unsure answer must not pin a
    wrong track for 30 days."""
    bothub_client._sonar_model = "sonar"
    store = _StoreStub(cached=None)

    out, captured = await _run(
        bothub_client, _PROD_DESCRIPTION, _extract_json(), _yt_entries(),
        sonar={"artist": "Кто-то", "track": "", "sure": False}, store=store)

    assert store.get_calls  # the cache was consulted
    assert store.save_calls == []
    assert [g for g, _ in captured["calls"]] == ["sonar-ground", "llm-rank"]
    assert captured["queries"] == ["soundtrack query 1", "soundtrack query 2"]
    assert out.intent == "from_source"


@pytest.mark.asyncio
async def test_source_fact_cache_errors_degrade_to_sonar(bothub_client):
    """A cache that raises on read AND write: normal flow — sonar grounds,
    the concrete phrase leads, ranking delivers. The cache never breaks the
    search."""
    bothub_client._sonar_model = "sonar"
    store = _StoreStub(cached=None, read_error=RuntimeError("db locked"),
                       write_error=RuntimeError("db locked"))

    out, captured = await _run(
        bothub_client, "песня из рекламы Carlsberg", _extract_json(),
        _yt_entries(), sonar=_SONAR_HIT, store=store)

    assert store.get_calls and store.save_calls  # both attempts were made
    assert [g for g, _ in captured["calls"]] == ["sonar-ground", "llm-rank"]
    assert captured["queries"][0] == _CONCRETE
    assert out.intent == "from_source"
    assert out.candidates[out.best_index].video_id == "v1"


@pytest.mark.asyncio
async def test_source_fact_cache_corrupt_payload_is_a_miss(bothub_client):
    """A corrupt/partial stored row (not JSON, or missing a side) counts as
    a miss: sonar grounds fresh and the successful answer overwrites the
    dead row — the cache self-heals."""
    bothub_client._sonar_model = "sonar"
    store = _StoreStub(cached="не JSON вообще")

    out, captured = await _run(
        bothub_client, "песня из рекламы Carlsberg", _extract_json(),
        _yt_entries(), sonar=_SONAR_HIT, store=store)

    assert [g for g, _ in captured["calls"]] == ["sonar-ground", "llm-rank"]
    assert captured["queries"][0] == _CONCRETE
    assert len(store.save_calls) == 1  # the overwrite happened


@pytest.mark.asyncio
async def test_analog_chain_after_cache_hit(bothub_client):
    """The analog chain fires off a CACHE HIT exactly as off a sonar hit:
    no sonar call, getSimilar on the CACHED artist, neighbors lead discovery,
    the SIMILAR prompt ranks, the reference is banned from the best, and
    SearchOut.intent keeps "from_source"."""
    bothub_client._sonar_model = "sonar"
    store = _StoreStub(cached=_cached_payload())
    svc = _MetaSvc(neighbors=["Neighbor One", "Neighbor Two", "Ref Band"])

    out, captured = await _run(
        bothub_client, _PROD_DESCRIPTION, _extract_json(), _ref_entries(),
        rank=_ref_tops_rank(), metadata_svc=svc, store=store)

    # the cache answered: no sonar call, no re-save
    assert [g for g, _ in captured["calls"]] == ["llm-rank"]
    assert store.save_calls == []
    # the chain ran on the CACHED reference
    assert svc.calls == [("getSimilar", "Ref Band")]
    assert captured["queries"][:2] == ["Neighbor One", "Neighbor Two"]
    assert _CONCRETE in captured["queries"]
    # similar-mode ranking + the ban, intent kept
    assert [s for g, s in captured["calls"] if g == "llm-rank"] == [SYSTEM_RANK_SIMILAR]
    assert out.intent == "from_source"
    assert out.candidates[out.best_index].video_id == "n1"
    assert out.candidates[-1].video_id == "r1"
    assert out.reference["artist"] == "Ref Band"
    assert out.reference["track"] == "Source Song"


# ════ Hard-case escalation (live 06.11 «Iron Man 2»: both dev and prod stuck
# on a rejected score cue — a plain sonar re-ask returned the SAME cue and
# the pool drifted to the famous soundtrack) ═══

from services.pipelines.from_source import _DETAILS_QUESTION


async def _run_escalated(bothub_client, description, extract, entries, rank=None,
                         sonar=None, restart_streak=0, excluded_titles=None):
    """Like _run but wires the restart context: restart_streak and
    excluded_titles are what the handler passes on a miss-streak restart."""
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault("calls", []).append(gen_name)
        if gen_name == "sonar-ground":
            captured["sonar_q"] = user
            return sonar
        if gen_name == "llm-rank":
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
                t_start=time.perf_counter(), t_extract=0.05,
                restart_streak=restart_streak,
                excluded_titles=excluded_titles)
            out = await FromSourcePipeline().run(ctx)
    return out, captured


_IM2_DESC = "дай мне песню из железного человека 2 фильм где он создает ускоритель части"
_IM2_REJECTED = ["Iron Man 2 Soundtrack [John Debney - New Element / Particle Accelerator]"]


@pytest.mark.asyncio
async def test_first_restart_asks_sonar_for_a_different_track(bothub_client):
    """streak=2 (the first miss-streak restart): the sonar QUESTION changes —
    it names the rejected candidates and demands a DIFFERENT track; the
    answer becomes the first discovery query."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run_escalated(
        bothub_client, _IM2_DESC, _extract_json(), _ref_entries(),
        sonar={"artist": "John Debney", "track": "Sledgehammer", "sure": True},
        restart_streak=2, excluded_titles=set(_IM2_REJECTED))
    q = captured["sonar_q"]
    assert "ДРУГОЙ" in q and "New Element" in q   # negatives named, new info demanded
    assert captured["queries"][0] == "John Debney - Sledgehammer"
    assert [g for g in captured["calls"] if g == "sonar-ground"] == ["sonar-ground"]
    assert out.candidates  # the tail ran as usual


@pytest.mark.asyncio
async def test_second_restart_asks_the_user_for_details(bothub_client):
    """streak=4 (the second restart): no sonar, no search, no rank — the bot
    honestly asks the user for details instead of a third guess."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run_escalated(
        bothub_client, _IM2_DESC, _extract_json(), _ref_entries(),
        restart_streak=4, excluded_titles=set(_IM2_REJECTED))
    assert not captured                      # zero LLM calls, zero searches
    assert out.need_clarify and out.clarify_question == _DETAILS_QUESTION
    assert not out.candidates


@pytest.mark.asyncio
async def test_no_restart_keeps_the_standard_cached_question(bothub_client):
    """streak=0: the ladder base is byte-identical — the standard cached
    ground question, no escalated wording."""
    bothub_client._sonar_model = "sonar"
    _, captured = await _run_escalated(
        bothub_client, _IM2_DESC, _extract_json(), _ref_entries(),
        sonar=_SONAR_HIT, restart_streak=0)
    assert "Какой это трек" in captured["sonar_q"]
    assert "ДРУГОЙ" not in captured["sonar_q"]


@pytest.mark.asyncio
async def test_second_restart_detected_by_desc_blocks(bothub_client):
    """The durable restart counter is the «Не то:» block count in the
    description (the handler's streak resets on every restart, so the
    second restart carries streak=2 again — the ladder must still fire)."""
    bothub_client._sonar_model = "sonar"
    desc2 = (_IM2_DESC + "\nНе то: AC/DC - Shoot To Thrill; Iron Man 2 Soundtrack"
             + "\nНе то: John Debney - New Element / Particle Accelerator")
    out, captured = await _run_escalated(
        bothub_client, desc2, _extract_json(), _ref_entries(),
        sonar={"artist": "X", "track": "Y"},   # must never be asked
        restart_streak=2, excluded_titles=set(_IM2_REJECTED))
    assert not captured                          # no LLM calls at all
    assert out.need_clarify and out.clarify_question == _DETAILS_QUESTION


@pytest.mark.asyncio
async def test_first_restart_detected_by_one_desc_block(bothub_client):
    """One «Не то:» block = restart #1 = the escalated question (the
    description channel works without relying on restart_streak)."""
    bothub_client._sonar_model = "sonar"
    desc1 = _IM2_DESC + "\nНе то: Iron Man 2 Soundtrack [John Debney - New Element]"
    _, captured = await _run_escalated(
        bothub_client, desc1, _extract_json(), _ref_entries(),
        sonar={"artist": "John Debney", "track": "Sledgehammer"},
        restart_streak=2, excluded_titles=set(_IM2_REJECTED))
    assert "ДРУГОЙ" in captured["sonar_q"]


# ════ live demotion (audit 11.11 D: owner rule "live never tops the list") ════

@pytest.mark.asyncio
async def test_live_never_tops_when_studio_exists(bothub_client):
    """LLM#2 loved the official live video (1.0 over the studio's 0.85) —
    the final order must still put the studio cut first; the live sinks
    below every clean record."""
    entries = [
        {"video_id": "v1", "title": "Band - Hit Song (Live at Wembley)",
         "channel": "Band Official", "duration": 300, "views": 900_000_000},
        {"video_id": "v2", "title": "Band - Hit Song (Official Video)",
         "channel": "Band - Topic", "duration": 250, "views": 100_000},
    ]
    rank = _rank_json(ranked=[
        {"video_id": "v1", "score": 1.0, "rationale": "gigaview live"},
        {"video_id": "v2", "score": 0.85, "rationale": "studio cut"},
    ])
    out, _ = await _run(bothub_client, "музыка из фильма Разгон", _extract_json(),
                        entries, rank=rank)
    assert out.candidates[out.best_index].video_id == "v2"
    assert out.candidates[-1].video_id == "v1"


@pytest.mark.asyncio
async def test_live_delivered_when_nothing_clean(bothub_client):
    """The owner's escape clause: a live-only pool still delivers the live —
    an honest live beats an empty answer."""
    entries = [
        {"video_id": "v1", "title": "Band - Hit Song (Live at Wembley)",
         "channel": "Band Official", "duration": 300, "views": 900_000_000},
    ]
    rank = _rank_json(ranked=[{"video_id": "v1", "score": 1.0, "rationale": "live only"}])
    out, _ = await _run(bothub_client, "музыка из фильма Разгон", _extract_json(),
                        entries, rank=rank)
    assert out.candidates and out.candidates[out.best_index].video_id == "v1"
