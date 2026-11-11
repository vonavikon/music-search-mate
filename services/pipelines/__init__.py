"""Pipeline registry: intent -> SearchPipeline (pipeline refactor).

search_candidates() resolves the pipeline for the extracted intent here
after its head (LLM#1 extract, fallbacks, destructive-block and
clarify-early checks). Stage 0 mapped every intent to one shared
LegacyPipeline; Stage 1 gives intent="this_track" its own pipeline
(ThisTrackPipeline) and normalizes the raw intent string before the table
lookup — LLM#1 occasionally answers "similar_tracks" or "This Track"
despite the prompt's closed list. Stage 2a gives intent="similar" its own
SimilarPipeline. Stage 2b gives intent="mood" (state/activity descriptions
without a reference) its own MoodPipeline. Stage 3 gives from_source its
own FromSourcePipeline (with the analog chain) and surprise its own
SurprisePipeline. Unknown intents and None still fall back to the legacy
pipeline.
"""
from services.pipelines.base import PipelineContext, SearchPipeline

# Intent -> pipeline table. Populated on the first route() call: the lazy
# pipeline imports keep this package importable from services.llm_search
# without a cycle.
_PIPELINES: dict[str | None, SearchPipeline] = {}

# Spelling variants LLM#1 emits for the canonical intents, mapped to
# the registry keys. Applied after strip+lower; anything not listed passes
# through unchanged and simply misses the table (legacy fallback).
_NORMALIZE: dict[str, str] = {
    "similar_tracks": "similar",
    "similar tracks": "similar",
    "similar_track": "similar",
    "similar track": "similar",
    "similar_artists": "similar",
    "similar_music": "similar",
    "this track": "this_track",
    "thistrack": "this_track",
    "this song": "this_track",
    "find_track": "this_track",
    "track": "this_track",
    "from source": "from_source",
    "fromsource": "from_source",
    "from the source": "from_source",
    "vibe": "mood",
    "mood_music": "mood",
    "mood music": "mood",
    "surprise_me": "surprise",
    "surprise me": "surprise",
}


def _normalize_intent(raw: str | None) -> str | None:
    """Canonical registry key for a raw extract intent: strip, lower, map
    known variants. Unknown strings pass through unchanged (the route()
    table lookup misses and the legacy fallback serves them); None/empty
    stays None."""
    s = (raw or "").strip().lower()
    if not s:
        return None
    return _NORMALIZE.get(s, s)


def _registry() -> dict[str | None, SearchPipeline]:
    """Build the intent table on first use. All pipelines are stateless, so
    one shared instance each serves every search (and concurrent chats)."""
    if not _PIPELINES:
        from services.pipelines.from_source import FromSourcePipeline
        from services.pipelines.legacy import LegacyPipeline
        from services.pipelines.mood import MoodPipeline
        from services.pipelines.similar import SimilarPipeline
        from services.pipelines.surprise import SurprisePipeline
        from services.pipelines.this_track import ThisTrackPipeline
        legacy = LegacyPipeline()
        _PIPELINES.update({
            "this_track": ThisTrackPipeline(),
            "similar": SimilarPipeline(),
            "mood": MoodPipeline(),
            "from_source": FromSourcePipeline(),
            "surprise": SurprisePipeline(),
            None: legacy,
        })
    return _PIPELINES


def route(intent: str | None) -> SearchPipeline:
    """Pipeline for an extracted intent. The raw string is normalized
    (strip/lower/variant map) before the table lookup; unknown intents and
    None fall back to the legacy pipeline."""
    pipelines = _registry()
    key = _normalize_intent(intent)
    return pipelines.get(key) or pipelines[None]
