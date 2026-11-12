"""Similar pipeline: the intent-specific tail for intent="similar"
(Stage 2a of the pipeline refactor).

Behavior-preserving move from LegacyPipeline.run: the similar-specific
orchestration — the LLM#1 reference resolve (_resolve_reference: canonical
artist via Last.fm track.search / yt-dlp channel / sonar, plus the
getSimilar neighbors) and the neighbors injection into the discovery
queries (with the ref_key self-exclusion so the reference itself never
leads discovery) — now lives here. Everything downstream is the shared
legacy stage helpers (_search_youtube, _filter_and_enrich, _rank_and_decide,
_no_results_out): the reference ban and the sonic boost stay inside
_rank_and_decide, shared with every other pipeline. The from_source
sonar-ground and the this_track query expansion remain in legacy.py.

Byte-identical flow for the similar intents (same calls in the same order,
same trace spans, same log lines, same SearchOut fields); the only change
is the LLM#2 system prompt — SYSTEM_RANK_SIMILAR instead of the former
combined SYSTEM_RANK (the Stage 2a prompt split, selected inside
_rank_and_decide via rank_prompt_for).

One later addition: the shared DETERMINISTIC diversity guard
(artist_diversity_guard, the mood pipeline's «не три трека одного
исполнителя» rule, RU for "no three tracks by one artist") runs between
the filter and the rank - 3+ of the top-6 filtered candidates from one
canonical ARTIST keep only the best 2 (acceptance rerun s3: the
«песня как Nights in White Satin» query, RU for "a song like Nights in
White Satin", returned four Jethro Tull tracks in the top-5). It reads
the filter's raw scores/views ordering exactly like mood; the
[neighbor?] signal is rank-time only and never reorders anything
before it.

Miss-streak restart hardening (live-test F2, 2026-11-05): after two «Не то»
presses (the RU "not it" miss button) on one song a restart (a) re-served
the SAME song from other channels -
fixed by the title-level rejection in _filter_and_enrich (excluded_titles
from the context), (b) re-ran the same neighbor window — fixed by rotating
the getSimilar window by ctx.restart_streak (the NEXT _NEIGHBOR_QUERIES
neighbors instead of the same first three), and (c) re-ran query echoes of
the rejected song/artist — fixed by _drop_echo_queries over the final query
list (LLM guesses and neighbors alike). restart_streak=0 and
excluded_titles=None keep today's behavior byte-identical.

Reference-resolve upgrade B1 (research card П4, 2026-11-05): inside
_resolve_reference (services/llm_search.py) the Last.fm track.search query
and the yt-dlp channel probe now run CONCURRENTLY (a Last.fm hit drops the
probe; a miss reuses the already-running probe instead of queueing behind
the Last.fm query), and before a LOW-confidence channel is kept, one sonar
question may rescue the real artist («Ry Cooder - Dior wild man», a DIOR-ad
fan upload unknown to Last.fm). Same contract here: ref_resolved carries
artist/track/source/neighbors; source may now also be "sonar" from the
rescue branch.

Stateless like every SearchPipeline: one shared instance serves concurrent
searches; per-search state travels in the PipelineContext.
"""
import logging
import time

from services.pipelines._diversity import artist_diversity_guard
from services.pipelines.base import PipelineContext, SearchPipeline
from services.pipelines.legacy import (
    _fields_from_extract,
    _filter_and_enrich,
    _no_results_out,
    _rank_and_decide,
    _search_youtube,
)

# Same logger name as the legacy pipeline: the deployed log filters and
# dashboards key on "bot.llm_search".
_log = logging.getLogger("bot.llm_search")


def _descriptor_ref_flag(reference: dict, ref_resolved: dict, queries: list) -> bool:
    """The «артист + слово-дескриптор звучания» case (RU: "artist + a
    sound-descriptor word"), e.g. «как moby blues» ("like moby blues"):
    the user described HOW the artist sounds, never named a song - the
    artist's own tracks in that sound are a legitimate answer and the hard
    artist-wide
    reference ban relaxes to the composite key (in _rank_and_decide). Real
    cue titles («Dior wild man») keep the hard ban. Two independent signals,
    either suffices:
      (1) reference.track is a bare sound word AND no empirical track of
          this artist confirmed it (resolve source != lastfm-track-search);
      (2) LLM#1 itself put the artist's tracks into queries — the extract
          prompt allows this ONLY for descriptor references (wave-3c live
          run: extract gave track=None but queries=[Moby Natural Blues...],
          and the hard ban sank exactly those candidates)."""
    from services.llm_search import _is_sound_descriptor, _ref_key
    artist = (ref_resolved.get("artist") or "").strip()
    if not _ref_key(artist):
        return False
    artist_tracks_in_queries = any(
        (q or "").strip().lower().startswith(artist.lower() + " ")
        for q in queries or [])
    return bool(
        artist_tracks_in_queries
        or (_is_sound_descriptor((reference or {}).get("track") or "")
            and ref_resolved.get("source") != "lastfm-track-search"))


class SimilarPipeline(SearchPipeline):
    """Stage-2a pipeline for intent="similar": reference resolve + neighbor
    injection over the shared legacy stages."""

    name = "similar"

    async def run(self, ctx: PipelineContext) -> "SearchOut | None":
        """Execute the similar tail: reference resolve, getSimilar-neighbors
        query injection, YouTube search, filter/enrich, then the shared
        LLM#2 rank tail (reference ban + sonic boost included)."""
        # Lazy per-call imports: services.llm_search imports this package at
        # module level (cycle), and per-call resolution honors test
        # monkeypatching of the services.llm_search namespace.
        from services.llm_search import (
            _MAX_QUERIES,
            _NEIGHBOR_QUERIES,
            _drop_echo_queries,
            _ref_key,
            _resolve_reference,
        )

        # Local aliases keep the moved body below textually identical to the
        # original orchestrator code (plain local names, not ctx attributes).
        trace = ctx.trace
        t_start = ctx.t_start

        f = _fields_from_extract(ctx.extract)
        intent, reference = f.intent, f.reference
        queries = f.queries

        # --- reference resolve for similar. Before the YouTube call: getSimilar
        # neighbors enter queries (empirics instead of LLM#1 guesses) and the
        # canonical artist strengthens the ban and sonic boost. For other
        # intents it's a no-op dict. The raw extract intent is compared exactly
        # as the legacy body did — the registry already normalized the routing
        # decision, and every downstream stage (ban, boost, scoring) keys on
        # the same raw string. ---
        ref_resolved = {"artist": "", "track": "", "source": "llm", "neighbors": []}
        if intent == "similar" and isinstance(reference, dict) and (
                (reference.get("artist") or "").strip() or (reference.get("track") or "").strip()):
            ref_resolved = await _resolve_reference(reference, ctx.metadata_svc,
                                                    trace=trace, bothub=ctx.bothub)
            # wave-3 s2: descriptor reference (the «как moby blues» query,
            # RU for "like moby blues") - see
            # _descriptor_ref_flag below for the two-signal rationale.
            ref_resolved["descriptor_ref"] = _descriptor_ref_flag(
                reference, ref_resolved, queries)
            if ref_resolved["descriptor_ref"]:
                _log.info("similar: descriptor reference %r -> artist tracks allowed",
                          ref_resolved.get("artist"))

        # The reference's empirical neighbors lead discovery. An empty ref_key
        # cuts off "neighbor = the reference itself" (getSimilar never returns
        # that, but the normalization is free). LLM#1 guesses stay after the
        # neighbors; the tail is trimmed to the cap.
        # Neighbor-window rotation (live-test F2): a miss-streak restart
        # (restart_streak >= 2, the miss count wired by the handler) must not
        # re-run the same first neighbors that produced the rejected pool —
        # the window slides by the streak: [streak : streak + N], so a
        # streak-2 restart takes the NEXT three neighbors (a fresh direction).
        # streak 0 keeps today's window byte-identical; a streak past the
        # neighbor list (limit 6) empties the window and discovery gracefully
        # falls back to the (echo-guarded) LLM queries alone.
        if ref_resolved["neighbors"]:
            ref_k = _ref_key(ref_resolved["artist"])
            offset = max(0, ctx.restart_streak or 0)
            window = ref_resolved["neighbors"][offset:offset + _NEIGHBOR_QUERIES]
            neighbors = [n for n in window
                         if _ref_key(n) != ref_k]
            if neighbors:
                queries = (neighbors + list(queries))[:_MAX_QUERIES]
                _log.info("similar discovery: neighbors %s via getSimilar('%s'), llm-queries kept %d",
                          neighbors, ref_resolved["artist"], len(queries) - len(neighbors))

        # --- echo guard (live-test F2): on a miss-streak restart LLM#1
        # regenerates genre-mate queries whatever the negatives in the
        # description say — the restart re-ran "Procol Harum A Whiter Shade
        # of Pale" verbatim. Applied to the FINAL query list (after the
        # neighbor injection): a neighbor naming a rejected artist is the
        # same echo. No-op without excluded_titles. ---
        queries = _drop_echo_queries(queries, ctx.excluded_titles, ctx.excluded_artists)

        # wave-3c s2: for a descriptor reference the artist's own tracks in
        # the queried sound ARE the answer — but the neighbor phrases sit in
        # front and eat the filter window (live run: filt=8, zero Moby
        # tracks survived while the queries named three). Front-load the
        # artist's phrases for the descriptor case only.
        if ref_resolved.get("descriptor_ref"):
            artist_pref = (ref_resolved.get("artist") or "").strip().lower() + " "
            front = [q for q in queries if q.strip().lower().startswith(artist_pref)]
            if front:
                queries = front + [q for q in queries if q not in front]
                _log.info("similar: descriptor reference -> artist phrases front-loaded (%d)",
                          len(front))

        # step 3: youtube
        raw = await _search_youtube(queries, on_progress=ctx.on_progress, trace=trace)
        t_search = time.perf_counter() - t_start
        if not raw:
            return _no_results_out(f, queries, trace)

        # steps 4-4.5: filter + enrich (excluded_titles = title-level
        # rejection of the same song from other channels/uploads, F2)
        filtered, enriched = await _filter_and_enrich(
            raw, ctx.excluded_ids, ctx.excluded_artists,
            metadata_svc=ctx.metadata_svc, on_progress=ctx.on_progress, trace=trace,
            excluded_titles=ctx.excluded_titles)
        t_filter = time.perf_counter() - t_start

        # --- deterministic diversity guard (before ranking; acceptance s3) ---
        # Same window and key as mood: 3+ of the top-6 filtered candidates
        # from one canonical artist keep only the best 2. Runs on the
        # filter's raw scores/views order — the [neighbor?] marker is a
        # rank-time signal and nothing here reorders the pool.
        filtered, enriched = artist_diversity_guard(filtered, enriched, label="similar")

        # steps 5+: LLM#2 rank (SYSTEM_RANK_SIMILAR via rank_prompt_for),
        # reference ban, sonic boost, scoring, SearchOut assembly.
        return await _rank_and_decide(ctx, f, queries, filtered, enriched, ref_resolved,
                                      len(raw), t_search, t_filter)
