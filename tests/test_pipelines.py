"""Contract tests for the pipeline refactor (services/pipelines/).

Stage 0 contract under test:
  - services/pipelines exposes PipelineContext (dataclass), SearchPipeline
    (ABC: name + async run(ctx) -> SearchOut | None) and route(intent).
  - Stage 1: route() normalizes the raw intent (strip/lower/variant map) and
    maps intent="this_track" (and its spellings) to ThisTrackPipeline.
  - Stage 2a: intent="similar" (and its spellings) maps to SimilarPipeline;
    from_source, None and unknown strings still go to ONE shared
    LegacyPipeline instance with name "legacy".
  - search_candidates keeps its head (LLM#1 extract, block check,
    clarify-early, LLM#1-failure raw fallback) and delegates the rest to
    route(intent).run(ctx).

Mocking follows tests/test_llm_search.py: BothubClient.chat_json patched on
the bothub_client fixture (returns ALREADY parsed dicts), search_metadata
patched in services.llm_search, no network. route() is replaced with a
recording stub so the orchestrator wiring is observable without running the
real legacy tail.
"""
import dataclasses

import pytest
from unittest.mock import AsyncMock, patch

from services.llm_search import SearchOut, search_candidates
from services.pipelines import PipelineContext, SearchPipeline, route
from services.pipelines.similar import SimilarPipeline
from services.pipelines.this_track import ThisTrackPipeline


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Same guard as test_llm_search.py: the enrich step must not touch the
    network (candidates here are fake; the stub returns None -> enrich is
    skipped, the path still executes offline)."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers ════

def _extract_json(intent="similar", reference=None, target=None, queries=None):
    """LLM#1 extract — a fake JSON response (an already parsed dict).
    Mirrors the helper in test_llm_search.py so canned extracts stay
    interchangeable between the two files."""
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


def _yt_entries():
    """Fake yt-dlp results (3 candidates) for the raw-fallback path."""
    return [
        {"video_id": "v1", "title": "Cool Rock Song", "channel": "Rock Band",
         "duration": 240, "views": 500000},
        {"video_id": "v2", "title": "Another Rock Track", "channel": "Other Band",
         "duration": 300, "views": 100000},
        {"video_id": "v3", "title": "Soft Rock Ballad", "channel": "Soft Band",
         "duration": 270, "views": 50000},
    ]


def _blocked_out() -> SearchOut:
    """A sentinel SearchOut the stub pipeline would return; identity-checked
    by the wiring test. Distinct field values make mix-ups visible."""
    return SearchOut(intent="similar", signals={"genre": "rock"}, queries=[],
                     assoc=[], candidates=[], best_index=-1, confidence="low")


class _StubPipeline:
    """Recording stand-in for a SearchPipeline: run is an AsyncMock returning
    a fixed SearchOut, so the orchestrator's delegation is observable."""

    name = "stub"

    def __init__(self, out: SearchOut):
        self.run = AsyncMock(return_value=out)


def _install_route_stub(monkeypatch, out: SearchOut) -> tuple[dict, _StubPipeline]:
    """Replace route() with a recorder that always returns ONE stub pipeline
    yielding `out`. Records the intents route() was called with. Returns
    (calls, stub).

    Both lookup spots are patched: services.pipelines.route (covers
    `pipelines.route(...)` and in-function imports, which resolve the
    attribute at call time) and services.llm_search.route (covers a
    module-level `from services.pipelines import route` binding, if the
    orchestrator uses one).
    """
    import services.llm_search as ls
    import services.pipelines as pl

    calls = {"intents": []}
    stub = _StubPipeline(out)

    def fake_route(intent):
        calls["intents"].append(intent)
        return stub

    monkeypatch.setattr(pl, "route", fake_route)
    monkeypatch.setattr(ls, "route", fake_route, raising=False)
    return calls, stub


def _routed_ctx(stub: _StubPipeline) -> PipelineContext:
    """Assert run() was awaited exactly once; return the PipelineContext
    it received (positional or keyword)."""
    stub.run.assert_awaited_once()
    call = stub.run.await_args
    return call.args[0] if call.args else call.kwargs["ctx"]


class _NullObs:
    """trace.span() result: finish() is a no-op."""

    def finish(self, output=None, usage_details=None, level=None):
        pass


class _TraceSpy:
    """Minimal SearchTrace stand-in: records tag() arguments. span()/finish()
    are no-ops so the trace never touches langfuse."""

    trace_id = "t" * 32  # read by the final log line of a full search

    def __init__(self):
        self.tags: list[str] = []

    def span(self, name, input=None, as_type="span", model=None,
             model_parameters=None) -> _NullObs:
        return _NullObs()

    def tag(self, tags) -> None:
        self.tags.extend(tags or [])

    def finish(self, output=None) -> None:
        pass


# ════ 1. registry contract ════

def test_route_returns_shared_legacy_pipeline_for_all_intents():
    """Stage 1: intent="this_track" routes to ThisTrackPipeline; Stage 2a:
    intent="similar" -> SimilarPipeline; Stage 3: from_source ->
    FromSourcePipeline and surprise -> SurprisePipeline (variants included);
    None and unknown intents still route to the SAME shared LegacyPipeline."""
    from services.pipelines.from_source import FromSourcePipeline
    from services.pipelines.surprise import SurprisePipeline

    this_track = route("this_track")
    assert isinstance(this_track, ThisTrackPipeline)
    assert this_track.name == "this_track"
    assert this_track is route("This Track")   # normalization maps variants
    assert this_track is not route("similar")

    similar = route("similar")
    assert isinstance(similar, SimilarPipeline)
    assert similar.name == "similar"
    assert similar is route("Similar")          # normalization maps variants
    assert similar is route("similar_tracks")
    assert similar is not this_track

    from_source = route("from_source")
    assert isinstance(from_source, FromSourcePipeline)
    assert from_source.name == "from_source"
    assert from_source is route("From Source")  # normalization maps variants

    surprise = route("surprise")
    assert isinstance(surprise, SurprisePipeline)
    assert surprise.name == "surprise"
    assert surprise is route("surprise_me")     # normalization maps variants

    seen = set()
    for intent in (None, "unknown_intent"):
        p = route(intent)
        assert isinstance(p, SearchPipeline)
        assert p.name == "legacy"
        seen.add(id(p))
    assert len(seen) == 1, "route() must return one shared legacy instance, not per-call copies"


def test_search_pipeline_abc_contract():
    """SearchPipeline is abstract: bare instantiation fails; a subclass
    defining name + async run(ctx) -> SearchOut | None instantiates."""
    with pytest.raises(TypeError):
        SearchPipeline()

    class _Ok(SearchPipeline):
        name = "ok"

        async def run(self, ctx) -> SearchOut | None:
            return None

    assert _Ok().name == "ok"


# ════ 2. routing wiring inside search_candidates ════

@pytest.mark.asyncio
async def test_search_candidates_delegates_to_routed_pipeline(bothub_client, monkeypatch):
    """The orchestrator head (LLM#1 extract) delegates the tail to
    route(intent).run(ctx): run awaited exactly once with a PipelineContext
    carrying the canned extract and the user description; the SearchOut
    returned by search_candidates IS the pipeline's object."""
    extract = _extract_json(intent="similar",
                            queries=["Mazzy Star Fade Into You", "Portishead Roads"])
    stub_out = _blocked_out()
    routed, stub = _install_route_stub(monkeypatch, stub_out)

    with patch.object(bothub_client, "chat_json", AsyncMock(return_value=extract)):
        out = await search_candidates("Хочу рок помягче", bothub_client)

    assert routed["intents"] == ["similar"], \
        "route() must be called once with the extract's intent"
    ctx = _routed_ctx(stub)
    assert isinstance(ctx, PipelineContext)
    assert dataclasses.is_dataclass(ctx), "PipelineContext must stay a dataclass"
    assert ctx.extract == extract
    assert ctx.description == "Хочу рок помягче"
    assert out is stub_out, "search_candidates must return the pipeline's SearchOut as-is"


# ════ 3. head behavior preserved in the orchestrator ════

@pytest.mark.asyncio
async def test_block_check_fires_before_routing(bothub_client, monkeypatch):
    """A destructive-query verdict from LLM#1 returns a blocked SearchOut
    WITHOUT delegating to the routed pipeline."""
    extract = _extract_json(intent="similar")
    extract["block"] = "drugs"
    routed, stub = _install_route_stub(monkeypatch, _blocked_out())

    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [extract]
        out = await search_candidates("что-нибудь тонкое", bothub_client)

    assert out.block == "drugs"
    assert out.candidates == [] and out.best_index == -1
    assert mock_chat.await_count == 1, "only LLM#1 extract runs before the block short-circuit"
    assert routed["intents"] == [], "the routed pipeline must not run on a blocked query"
    stub.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_clarify_early_fires_before_routing(bothub_client, monkeypatch):
    """An extract with no queries, no target and no assoc -> need_clarify
    SearchOut straight from the head; the routed pipeline is not called."""
    # queries set AFTER the helper: `queries=[]` would be replaced by the
    # helper's non-empty default (same pattern as test_llm_search.py).
    extract = _extract_json(intent="similar",
                            reference={"artist": None, "track": None},
                            target={"artist": None, "track": None})
    extract["queries"] = []
    routed, stub = _install_route_stub(monkeypatch, _blocked_out())

    with patch.object(bothub_client, "chat_json", AsyncMock(return_value=extract)):
        out = await search_candidates("давай еще подобный вариант", bothub_client)

    assert out.need_clarify is True
    assert out.clarify_question
    assert out.candidates == []
    assert routed["intents"] == [], "clarify-early must short-circuit before routing"
    stub.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm1_failure_raw_fallback_skips_pipeline(bothub_client, monkeypatch):
    """LLM#1 dead (chat_json -> None): the raw-phrase fallback still runs
    (degraded SearchOut with raw YouTube candidates) and the routed pipeline
    is not involved."""
    routed, stub = _install_route_stub(monkeypatch, _blocked_out())

    with patch.object(bothub_client, "chat_json", AsyncMock(return_value=None)):
        with patch("services.llm_search.search_metadata") as mock_yt:
            mock_yt.return_value = _yt_entries()
            out = await search_candidates("какой-то рок", bothub_client)

    assert out.degraded is True
    assert out.intent is None
    assert len(out.candidates) == 3
    assert out.best_index == 0
    # the fallback searched the user's raw phrase
    mock_yt.assert_awaited_once_with(["какой-то рок"], per_query=20, on_progress=None)
    assert routed["intents"] == [], "the LLM-less fallback must not route to a pipeline"
    stub.run.assert_not_awaited()


# ════ 4. SearchOut contract freeze ════

def test_searchout_field_set_frozen():
    """The SearchOut dataclass field set is frozen for later stages: any field
    added/removed/renamed breaks this list and must be a deliberate decision.
    (List read from services/llm_search.py at the time of writing.)"""
    frozen = sorted([
        "intent", "signals", "queries", "assoc", "candidates", "best_index",
        "confidence", "intro", "verdict", "need_clarify", "clarify_question",
        "reference", "target", "block", "degraded", "trace",
    ])
    assert sorted(f.name for f in dataclasses.fields(SearchOut)) == frozen


# ════ 5. trace tags ════

@pytest.mark.asyncio
async def test_intent_trace_tag_applied(bothub_client, monkeypatch):
    """The head still tags the trace with the extract's intent (current code:
    trace.tag([intent]) right after the block check). Verified via a trace
    spy on a stub-routed search."""
    spy = _TraceSpy()
    extract = _extract_json(intent="similar", queries=["test query 1"])
    _install_route_stub(monkeypatch, _blocked_out())

    with patch.object(bothub_client, "chat_json", AsyncMock(return_value=extract)):
        await search_candidates("Хочу рок помягче", bothub_client, trace=spy)

    assert "similar" in spy.tags, "the intent tag must still be applied by the orchestrator head"
