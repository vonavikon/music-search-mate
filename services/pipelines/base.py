"""Pipeline layer base: the context and the interface for intent pipelines.

Stage 0 of the staged pipeline refactor: search_candidates() in
services/llm_search.py keeps its head (pre-search gate, history context,
LLM#1 extract, LLM-less raw fallback, destructive-block check,
clarify-early check) and delegates the post-extract tail to a pipeline
resolved by intent. Every intent currently routes to LegacyPipeline, which
holds the moved body verbatim; later stages give intents their own
pipelines. Zero behavior change is the Stage 0 invariant.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Circular-import guard: services.llm_search imports this package at
    # module level (route), so SearchOut/BothubClient are typing-only here.
    from collections.abc import Callable

    from services.llm_search import BothubClient, SearchOut
    from services.metadata import MetadataService
    from services.tracing import SearchTrace


@dataclass
class PipelineContext:
    """Everything a pipeline needs from the search_candidates() head.

    Field names mirror the search_candidates() parameters (context is the
    recent conversation history; excluded_ids/excluded_artists are the
    refinement-loop exclusions). extract is the parsed LLM#1 output dict;
    pipelines re-derive intent/signals/queries/etc. from it. t_start is the
    perf_counter checkpoint taken before LLM#1, t_extract the duration of
    the extract step — both feed the per-step latency log lines.
    """

    description: str                       # the user's query text
    context: list[dict] | None = None      # recent queries (recent_context)
    excluded_ids: set[str] | None = None   # video_ids rejected in refinement
    excluded_artists: set[str] | None = None  # _ref_key names of rejected artists
    excluded_titles: set[str] | None = None  # rejected track titles/keys (title-level rejection, live-test F2)
    restart_streak: int = 0                # miss streak len: >=2 rotates the similar neighbor window (F2)
    metadata_svc: "MetadataService | None" = None  # Last.fm enrichment layer
    bothub: "BothubClient | None" = None   # LLM gateway client
    on_progress: "Callable | None" = None  # async streaming-status callback
    trace: "SearchTrace | None" = None     # tracing.SearchTrace of this search
    extract: dict = field(default_factory=dict)  # LLM#1 output
    t_start: float = 0.0                   # perf_counter before LLM#1
    t_extract: float = 0.0                 # LLM#1 extract duration
    user_id: int | None = None             # asking user (personal-taste layers, surprise)
    store: object = None                   # services.store module (or test stub) for personal history


class SearchPipeline(ABC):
    """Base class for intent pipelines.

    run() executes the post-extract tail of a search: query expansion,
    YouTube discovery, filtering, LLM#2 ranking and SearchOut assembly.
    The registry may share one instance across concurrent searches, so
    implementations must be stateless: all per-search state travels in the
    PipelineContext.
    """

    name: str

    @abstractmethod
    async def run(self, ctx: PipelineContext) -> "SearchOut | None":
        """Run the tail for the extract in ctx.

        Returns the assembled SearchOut; None signals that this pipeline
        could not produce a result and the caller must decide on a fallback
        (LegacyPipeline never returns None).
        """
