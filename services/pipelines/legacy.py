"""Legacy pipeline: the post-extract body of search_candidates(), moved
verbatim (Stage 0 of the pipeline refactor).

Stages 1-2a carve out per-intent pipelines: this_track routes to
ThisTrackPipeline, similar to SimilarPipeline. LegacyPipeline keeps serving
from_source, None and unknown intents, and hosts the shared stage helpers
below (_fields_from_extract, _search_youtube, _filter_and_enrich,
_rank_and_decide, _no_results_out) that every per-intent pipeline composes
without copying the body. Zero behavior change for the intents served here:
same calls in the same order, same trace spans, same log lines, same
SearchOut fields.
"""
import logging
import re
import time
from typing import NamedTuple, TYPE_CHECKING

from services.pipelines.base import PipelineContext, SearchPipeline
from services.observability import emit_stage

if TYPE_CHECKING:
    from services.llm_search import SearchOut

# The moved log lines keep their historical logger name ("bot.llm_search")
# so deployed log filters and dashboards see no change after the move.
_log = logging.getLogger("bot.llm_search")

# A live/concert recording marker in a video TITLE (EN/RU). Shared by the
# live demotion below and the fact-button record class: facts about a live
# take must not be served on a studio card (and vice versa).
LIVE_RE = re.compile(r"\blive\b|\bвживую\b|\bконцерт", re.I)


def record_class(title: str) -> str:
    """'lv' when the title marks a live/concert recording, else 'st' (studio).
    Threads into the 💡 fact key: the same song pressed on a live card and a
    studio card asks for (and caches) different facts."""
    return "lv" if LIVE_RE.search(title or "") else "st"


class _Fields(NamedTuple):
    """The LLM#1 extract fields a pipeline tail works with. Mirrors the
    lines the search_candidates() head derives for its own early returns."""

    intent: str | None
    signals: dict
    sonic: dict
    assoc: list
    reference: dict
    target: dict
    intro: str
    queries: list
    need_clarify_early: bool
    clarify_q: str


def _fields_from_extract(extract: dict) -> _Fields:
    """Re-derive the tail's working fields from the LLM#1 extract — the same
    lines as the search_candidates() head, which needs them for its own early
    returns. Reaching a pipeline guarantees need_clarify_early is False (the
    head already returned to clarify), so queries here are exactly the
    head's queries."""
    intent = extract.get("intent")
    signals = extract.get("signals") or {}
    sonic = extract.get("sonic") or {}
    assoc = extract.get("association_candidates") or []
    reference = extract.get("reference") or {}
    target = extract.get("target") or {}
    intro = (extract.get("intro") or "").strip()
    queries, need_clarify_early, clarify_q = _resolve_queries(extract)
    return _Fields(intent, signals, sonic, assoc, reference, target, intro,
                   queries, need_clarify_early, clarify_q)


def _resolve_queries(extract: dict) -> tuple[list[str], bool, str]:
    """Lazy per-call binding of the llm_search helper (monkeypatchable)."""
    from services.llm_search import _resolve_queries as _rq
    return _rq(extract)


def _no_results_out(f: _Fields, queries: list, trace) -> "SearchOut":
    """The empty-raw early return: same SearchOut shape as the moved body."""
    from services.llm_search import SearchOut
    return SearchOut(intent=f.intent, signals=f.signals, queries=queries, assoc=f.assoc,
                     reference=f.reference, target=f.target, candidates=[], best_index=-1,
                     confidence="low", intro=f.intro, trace=trace)


async def _search_youtube(queries: list, on_progress=None, trace=None) -> list:
    """step 3: youtube discovery. The search progress ping plus the traced
    search_metadata call; returns the raw candidate list (possibly empty)."""
    # Lazy per-call import: honors test monkeypatching of the
    # services.llm_search namespace (same convention as the pipelines).
    from services.llm_search import search_metadata
    if on_progress:
        await on_progress("search", "Ищу на YouTube…")
    sp_search = trace.span("youtube-search", input={"queries": queries}) if trace else None
    # wave 45-3: per-stage timing - a neutral hook (the subscriber is
    # web/metrics.py; no-op without the web layer). A helper shared by
    # ALL pipelines.
    _t0 = time.perf_counter()
    raw = await search_metadata(queries, per_query=20, on_progress=on_progress)
    emit_stage("yt_search", time.perf_counter() - _t0)
    if sp_search:
        sp_search.finish(output={"raw": len(raw)})
    return raw


async def _filter_and_enrich(raw: list, excluded_ids: set[str] | None = None,
                             excluded_artists: set[str] | None = None,
                             excluded_titles: set[str] | None = None,
                             metadata_svc=None, on_progress=None, trace=None) -> tuple[list, list]:
    """steps 4-4.5: heuristic filter, refinement exclusions, top-5 enrich,
    non-music drop. Returns (filtered, enriched) — the enriched list holds
    copies with canonical fields for the top-5 only. excluded_titles (new,
    live-test F2; None = previous behavior) drops candidates by normalized
    TITLE so the same song cannot return from another channel/upload."""
    from services.llm_search import (
        _filter_candidates,
        _is_non_music,
        _ref_key,
        _title_keys,
        enrich_candidates,
    )

    # wave 45-3: per-stage timing (filter + enrich as one chunk - both
    # live in this helper shared by all pipelines); the hook is neutral,
    # a no-op without a subscriber (web/metrics.py).
    _t0 = time.perf_counter()
    filtered = _filter_candidates(raw, target_n=20)
    # Refinement: deterministically drop rejected video_ids before ranking.
    # LLM text ("not this artist") is hope; a hard video_id exclude is the
    # guarantee that a rejected clip won't return in the refined results.
    if excluded_ids:
        before = len(filtered)
        filtered = [c for c in filtered if c["video_id"] not in excluded_ids]
        if before != len(filtered):
            _log.info("refinement: dropped %d previously-rejected video_id(s)", before - len(filtered))
    # Consecutive misses: also drop everything by the rejected uploaders —
    # a wrong-direction pool keeps offering the same neighborhood even with
    # the video_ids excluded (the "Ry Cooder resolved to a folk singer" case:
    # every next candidate was another folk track).
    if excluded_artists:
        before = len(filtered)
        filtered = [c for c in filtered
                    if _ref_key((c.get("channel") or "").removesuffix(" - Topic"))
                    not in excluded_artists]
        if before != len(filtered):
            _log.info("direction change: dropped %d candidate(s) from rejected artists", before - len(filtered))
    # Title-level rejection (live-test F2): video_id and channel-key excludes
    # both miss the rejected SONG arriving from another channel/upload — the
    # Beat-Club/MrLodurr copies of an already-rejected track. The keys are
    # channel-independent (_title_keys), so "A Whiter Shade of Pale (2007
    # Remaster)" also kills "PROCOL HARUM - A Whiter Shade of Pale".
    if excluded_titles:
        banned = set()
        for t in excluded_titles:
            banned |= _title_keys(t)
        banned.discard("")
        before = len(filtered)
        filtered = [c for c in filtered
                    if not (_title_keys(c.get("title") or "") & banned)]
        if before != len(filtered):
            _log.info("title exclusion: dropped %d same-title candidate(s)", before - len(filtered))
    if on_progress:
        await on_progress("filter", f"Отобрано {len(filtered)} вариантов")

    # step 4.5: enrich the top-5 candidates (second metadata pass).
    # Full extract_info (no download) + Last.fm → canonical
    # artist/track/album/year/genre for ranking. Doesn't mutate filtered —
    # returns copies with enriched fields.
    if on_progress:
        await on_progress("filter", "Уточняю данные треков…")
    sp_filter = trace.span("filter-enrich", input={"filtered": len(filtered)}) if trace else None
    enriched = await enrich_candidates(filtered, metadata_svc=metadata_svc, top_n=5)
    if sp_filter:
        sp_filter.finish(output={"enriched": len(enriched)})

    # Music filter: cut candidates the full extract_info tagged as clearly
    # non-musical (Comedy/Gaming/News...). Only the enriched top-5 have a category.
    non_music_ids = {c["video_id"] for c in enriched[:5] if _is_non_music(c.get("categories"))}
    if non_music_ids:
        filtered = [c for c in filtered if c["video_id"] not in non_music_ids]
        enriched = [c for c in enriched if c["video_id"] not in non_music_ids]
        _log.info("music filter: dropped %d non-music candidate(s): %s",
                  len(non_music_ids), ",".join(sorted(non_music_ids)))
    emit_stage("filter_enrich", time.perf_counter() - _t0)
    return filtered, enriched


async def _rank_and_decide(ctx: PipelineContext, f: _Fields, queries: list,
                           filtered: list, enriched: list, ref_resolved: dict,
                           n_raw: int, t_search: float, t_filter: float,
                           rank_intent: str | None = None) -> "SearchOut":
    """steps 5+: the LLM#2 rank stage and everything after it — score
    mapping, best_index resolution, the similar-only reference ban and sonic
    boost (no-ops for other intents), match scoring, the observability log
    line and the final SearchOut. Shared verbatim by every pipeline.
    n_raw is the pre-filter candidate count (the observability log line).

    rank_intent overrides the intent the RANK MACHINERY runs under: the LLM#2
    system prompt (rank_prompt_for) and the similar-only post-filters below.
    None (the default every legacy caller passes) = f.intent — behavior
    unchanged. The override exists for the from_source analog chain
    (from_source.py), which ranks under "similar" while SearchOut.intent
    stays "from_source" (analytics keep the extracted scenario): everything
    user-visible here (SearchOut.intent, score_candidate, the observability
    log line) keys on f.intent, never on the override."""
    # Lazy per-call imports (monkeypatchable services.llm_search namespace).
    from services.llm_search import (
        Candidate,
        SearchOut,
        _RANK_INPUT_N,
        _SONIC_BOOST,
        _channel_key,
        _format_for_rank,
        _is_reference_candidate,
        _match_video_id,
        _norm_confidence,
        _ref_key,
        pipeline_model,
        rank_prompt_for,
        score_candidate,
    )

    # Local aliases keep the moved body below textually identical to the
    # original orchestrator code (plain local names, not ctx attributes).
    description = ctx.description
    bothub = ctx.bothub
    on_progress = ctx.on_progress
    trace = ctx.trace
    metadata_svc = ctx.metadata_svc
    t_start = ctx.t_start
    t_extract = ctx.t_extract
    intent = f.intent
    # Effective intent of the rank machinery: the analog-chain override or
    # the extracted one. Only the LLM#2 prompt and the similar-only
    # post-filters (reference ban, sonic boost) key on it; everything
    # user-visible keeps f.intent above.
    rank_int = rank_intent or intent
    signals = f.signals
    sonic = f.sonic
    assoc = f.assoc
    reference = f.reference
    target = f.target
    intro = f.intro

    # step 5: LLM#2 rank. The system prompt is selected by the effective rank
    # intent (Stage 2a: STRICT/SIMILAR/FROM_SOURCE; the legacy combined prompt
    # for anything else; rank_intent="similar" from the from_source analog
    # chain ranks under SIMILAR). One call site serves every pipeline — the
    # this_track fall-through funnels into this helper too. The MODEL is
    # selected by the same key (pipeline_model, LLM_MODEL_PIPELINES): a
    # pipeline swap is config-only; None (nothing configured) = the client's
    # own global model.
    if on_progress:
        await on_progress("rank", "Сравниваю кандидатов…")
    # The ranker's input is a top-10 compact instead of all ~20 (latency
    # measurement 2026-11-01: llm-rank burned 3247 tokens vs 1604 for
    # extract, and 10.3s; the top slice almost always has the final say,
    # the rest are already ordered by the filter heuristic). The rank
    # prompts are tight (rationale up to 5 words, rules deduplicated),
    # max_tokens 1200 — rank is the slowest stage (7-20s on identical
    # queries). Candidates outside the top-10 stay in the output with score
    # 0.0 (existing semantics of uncoordinated LLM scores).
    # [neighbor?] markers (doctrine s3): in similar rank mode the reference's
    # empirical getSimilar neighbors are flagged to the LLM — deterministic
    # data (channel == neighbor artist), explained by the SIMILAR prompt
    # legend. Non-similar ranks never pass neighbors.
    rank_neighbors = (ref_resolved.get("neighbors") or None) if rank_int == "similar" else None
    _t0 = time.perf_counter()
    rank = await bothub.chat_json(rank_prompt_for(rank_int),
                                  _format_for_rank(description, signals, enriched[:_RANK_INPUT_N], sonic,
                                                   neighbors=rank_neighbors),
                                  max_tokens=1200, trace=trace, gen_name="llm-rank",
                                  model=pipeline_model(rank_int))
    emit_stage("llm2_rank", time.perf_counter() - _t0)
    t_rank = time.perf_counter() - t_start
    # (2026-11-14, F1 incident B) canonical artists of the enriched top-5
    # ride onto the Candidates below: the official-first guard (llm_search)
    # reads them as official-upload evidence. `filtered` itself never
    # carries canonical fields — only the enriched top-5 copies do.
    _canon_art = {c.get("video_id"): str(c.get("canonical_artist") or "")
                  for c in (enriched or [])[:5] if isinstance(c, dict)}
    if not rank:
        # No rank — return the filtered ones unscored, best=0
        cands = [Candidate(
            c["video_id"], c["title"], c["channel"], c["duration"], c["views"],
            source=c.get("source", "youtube"),
            url=c.get("url") or f"https://www.youtube.com/watch?v={c['video_id']}",
            fallback_url=c.get("fallback_url", ""),
            canonical_artist=_canon_art.get(c["video_id"], ""),
        ) for c in filtered]
        _log.info("search latency extract=%.2fs search=%.2fs filter=%.2fs rank=%.2fs (no rank)",
                  t_extract, t_search, t_filter, t_rank)
        return SearchOut(intent=intent, signals=signals, queries=queries, assoc=assoc,
                         reference=reference, target=target, candidates=cands, best_index=0 if cands else -1,
                         confidence="low", intro=intro, trace=trace)

    scores = {r["video_id"]: r for r in rank.get("ranked", []) if r.get("video_id")}
    ranked = []
    for c in filtered:
        sc = scores.get(c["video_id"])
        sc_val = sc.get("score") if sc else None
        score = float(sc_val) if isinstance(sc_val, (int, float)) else 0.0
        ranked.append(Candidate(
            c["video_id"], c["title"], c["channel"], c["duration"], c["views"],
            score,
            (sc.get("rationale", "") if sc else ""),
            c.get("is_noise", False),
            source=c.get("source", "youtube"),
            url=c.get("url") or f"https://www.youtube.com/watch?v={c['video_id']}",
            fallback_url=c.get("fallback_url", ""),
            canonical_year=str(c.get("canonical_year") or ""),
            canonical_album=str(c.get("canonical_album") or ""),
            canonical_artist=_canon_art.get(c["video_id"], ""),
        ))
    ranked.sort(key=lambda c: c.score, reverse=True)

    # best is the LLM's pick by video_id, not the pre-sort position.
    # sort reorders; a positional best_index from the JSON would point at a
    # different track.
    ranked_llm = rank.get("ranked", []) or []
    llm_best_idx = int(rank.get("best_index", 0))
    best_vid = (ranked_llm[llm_best_idx].get("video_id")
                if ranked_llm and 0 <= llm_best_idx < len(ranked_llm) else None)
    best_vid = _match_video_id(best_vid, filtered) if best_vid else None
    if best_vid:
        best_index = next((i for i, c in enumerate(ranked) if c.video_id == best_vid),
                          0 if ranked else -1)
    else:
        best_index = 0 if ranked else -1

    # --- reference key for similar post-filters (ban and sonic together) ---
    # The resolved artist (Last.fm/yt-dlp) is more precise than the raw LLM
    # field: the ban catches the reference by the performer's canonical name
    # too, not only by what the user named.
    # wave-3 s2 descriptor reference (the «как moby blues» query, RU for
    # "like moby blues"): the user described a SOUND, not a track - the
    # artist's own tracks in that sound are the
    # answer. With a track word the ban narrows to the composite "artist +
    # descriptor" (only the literal phrase sinks); via the queries-signal
    # (extract gave track=None but named the artist's tracks itself) there
    # is no phrase to compose — the artist's sound IS the ask, so no ban
    # key at all: the ranker already judged which of the artist's tracks
    # fits (live run: Natural Blues scored 1.0 and was still sunk by the
    # artist-wide ban).
    ref_key = ""
    if rank_int == "similar" and isinstance(reference, dict):
        track_part = (reference.get("track") or "").strip()
        if ref_resolved.get("descriptor_ref") and track_part:
            ref_key = _ref_key(" ".join(
                (ref_resolved.get("artist") or reference.get("artist") or "",
                 track_part)))
        elif not ref_resolved.get("descriptor_ref"):
            ref_key = _ref_key(ref_resolved.get("artist")
                               or (reference.get("artist") or "").strip())

    # --- hard disqualification of the reference for the similar rank mode ---
    # LLM#1 (queries) and LLM#2 (rank) break the "the reference itself =
    # miss" ban, especially for Russian-language artists (the model fails to
    # recognize Russian artist names as the reference). Guaranteed by a
    # post-filter: a candidate matching the reference (assoc[0].artist) in
    # channel/title goes to the tail (sort by (is_ref, -score)), best shifts
    # to the first non-reference. A violated prompt instruction never
    # reaches the user.
    if rank_int == "similar" and ranked and ref_key:
        banned = sum(1 for c in ranked if _is_reference_candidate(c, ref_key))
        ranked.sort(key=lambda c: (1 if _is_reference_candidate(c, ref_key) else 0, -c.score))
        if 0 <= best_index < len(ranked) and _is_reference_candidate(ranked[best_index], ref_key):
            best_index = next(
                (i for i, c in enumerate(ranked) if not _is_reference_candidate(c, ref_key)),
                best_index)
        if banned:
            _log.info("reference ban: %d candidate(s) matched reference '%s', moved to tail",
                      banned, (reference.get("artist") or "").strip())

    # --- live demotion (audit 11.11, owner rule: "live never tops the
    # list, only when no studio recording exists"). The final order used to
    # be (is_ref, -llm_score): an official live video with gigaviews could
    # win the LLM's 1.0 and top the studio cut. Now LIVE-titled candidates
    # sink below every non-live candidate; when the pool has no non-live
    # candidate the live one still delivers — the owner's exact escape
    # clause. Scoped to live ONLY (the broader is_noise family — covers,
    # karaoke — keeps today's behavior, the this_track scene-title test
    # pins it). An explicitly REQUESTED version carries is_noise=False and
    # is untouched.
    _live_re = LIVE_RE
    live_n = sum(1 for c in ranked
                 if getattr(c, "is_noise", False) and _live_re.search(c.title or ""))
    if live_n and ranked:
        clean = [c for c in ranked
                 if not (getattr(c, "is_noise", False) and _live_re.search(c.title or ""))]
        if clean:
            ranked.sort(key=lambda c: (
                1 if (getattr(c, "is_noise", False) and _live_re.search(c.title or "")) else 0,
                -c.score))
            if (0 <= best_index < len(ranked) and getattr(ranked[best_index], "is_noise", False)
                    and _live_re.search(ranked[best_index].title or "")):
                best_index = next(
                    (i for i, c in enumerate(ranked)
                     if not (getattr(c, "is_noise", False)
                             and _live_re.search(c.title or ""))), best_index)
            _log.info("live demotion: %d live candidate(s) moved below "
                      "%d studio record(s)", live_n, len(clean))
    # --- sonic adjustment: Last.fm getSimilar ---
    # The reference's empirical sonic neighbors (crowd-sourced "sounds
    # like") get the _SONIC_BOOST (+0.20, doctrine s3: sound-alikeness must
    # outrank genre-alikeness — a flat +0.10 was decorative against LLM rank
    # scores and the genre answer kept winning). The LLM ranks by genre
    # label while getSimilar gives an actual similarity signal. The
    # reference is already in the tail (block above) — the re-sort after
    # the boost respects the ban: (is_ref, -score), best = first non-ref
    # (review 2026-09-03: a plain score sort returned the banned reference
    # to #1). Artist and neighbors come from _resolve_reference (canonical
    # artist instead of the raw LLM field; no second getSimilar — neighbors
    # are already fetched during resolve).
    sim_artist = ref_resolved.get("artist") or ""
    sim_artists = ref_resolved.get("neighbors") or []
    if rank_int == "similar" and ranked and sim_artist and not sim_artists and metadata_svc:
        sim_artists = await metadata_svc.similar_artists(sim_artist)
    if rank_int == "similar" and ranked and sim_artists:
        sim_keys = {_ref_key(s) for s in sim_artists}
        boosted = 0
        for c in ranked:
            if _channel_key(c.channel) in sim_keys:
                c.score += _SONIC_BOOST
                boosted += 1
        if boosted:
            if ref_key:
                ranked.sort(key=lambda c: (1 if _is_reference_candidate(c, ref_key) else 0, -c.score))
                best_index = next(
                    (i for i, c in enumerate(ranked) if not _is_reference_candidate(c, ref_key)),
                    best_index)
            else:
                ranked.sort(key=lambda c: c.score, reverse=True)
                best_index = 0
            _log.info("sonic boost: +%.2f to %d candidate(s) near '%s' via getSimilar",
                      _SONIC_BOOST, boosted, sim_artist)

    # Rule-based match-score: a calibrated per-candidate score from features.
    # Replaces the uncalibrated LLM confidence for the auto-download
    # decision in the handler. ref_key feeds the same-artist-as-reference
    # disqualification double-check ("" for non-similar ranks = off).
    for c in ranked:
        s = score_candidate(c, target, intent, c.score, ref_key=ref_key)
        c.match_score = s.score
        c.match_reasons = "; ".join(s.reasons)

    # Observability: a structured log line of pipeline decisions for
    # debugging and for collecting data to tune thresholds/feedback loop.
    # One-liner, readable in docker logs. tid is the Langfuse trace id =
    # correlation_id in the DB.
    best = ranked[best_index] if 0 <= best_index < len(ranked) else None
    _log.info(
        "search done intent=%s conf=%s | queries=%d raw=%d filt=%d | "
        "best=[%.2f %s] %s | top3=%s | target=%s | tid=%s | "
        "latency extract=%.2fs search=%.2fs filter=%.2fs rank=%.2fs",
            intent, _norm_confidence(rank.get("confidence")), len(queries), n_raw, len(ranked),
        best.match_score if best else 0.0,
        (best.match_reasons[:50] if best else "-"),
        (best.title[:42] if best else "none"),
        ", ".join("%.2f" % c.match_score for c in ranked[:3]),
        target or "-",
        (trace.trace_id[:8] if trace else "-"),
        t_extract, t_search, t_filter, t_rank,
    )

    return SearchOut(
        intent=intent, signals=signals, queries=queries, assoc=assoc,
        reference=reference, target=target, candidates=ranked, best_index=best_index,
        confidence=_norm_confidence(rank.get("confidence")),
        intro=intro,
        verdict=(rank.get("verdict") or "").strip(),
        need_clarify=bool(rank.get("need_clarify")),
        clarify_question=(rank.get("clarify_question") or "").strip(),
        trace=trace,
    )


class LegacyPipeline(SearchPipeline):
    """The former orchestrator tail, now serving None and unknown intents
    (this_track, similar, mood, from_source and surprise have their own
    pipelines). run() keeps the original orchestration order over the shared
    stage helpers below; behavior is unchanged (same calls, spans, logs,
    SearchOut fields)."""

    name = "legacy"

    async def run(self, ctx: PipelineContext) -> "SearchOut | None":
        """Execute the search tail: query expansion (a no-op for the intents
        served here, kept as safety), YouTube search, filter, enrich, LLM#2
        rank, scoring, SearchOut assembly."""
        # Lazy per-call imports: services.llm_search imports this package at
        # module level, so a top-level import would be circular; resolving
        # the names on every call also honors test monkeypatching of the
        # services.llm_search namespace.
        from services.llm_search import (
            _expand_queries,
        )

        # Local aliases keep the moved body below textually identical to the
        # original orchestrator code (plain local names, not ctx attributes).
        trace = ctx.trace
        metadata_svc = ctx.metadata_svc
        t_start = ctx.t_start

        f = _fields_from_extract(ctx.extract)
        intent, target = f.intent, f.target
        queries = f.queries

        # --- reference resolve for similar + the getSimilar-neighbors query
        # injection moved to SimilarPipeline (Stage 2a): the registry routes
        # every raw intent that normalizes to "similar" there, so it can no
        # longer reach this body. The from_source sonar-ground block moved
        # to FromSourcePipeline in Stage 3a. For the intents still served
        # here (None, unknown) both were always no-ops. ---
        ref_resolved = {"artist": "", "track": "", "source": "llm", "neighbors": []}

        # Recall expansion for this_track with an exact target — before the YouTube call.
        queries = _expand_queries(queries, intent, target)

        # step 3: youtube
        raw = await _search_youtube(queries, on_progress=ctx.on_progress, trace=trace)
        t_search = time.perf_counter() - t_start
        if not raw:
            return _no_results_out(f, queries, trace)

        # steps 4-4.5: filter + enrich
        filtered, enriched = await _filter_and_enrich(
            raw, ctx.excluded_ids, ctx.excluded_artists,
            metadata_svc=metadata_svc, on_progress=ctx.on_progress, trace=trace,
            excluded_titles=ctx.excluded_titles)
        t_filter = time.perf_counter() - t_start

        # steps 5+: LLM#2 rank and the decision tail
        return await _rank_and_decide(ctx, f, queries, filtered, enriched, ref_resolved,
                                      len(raw), t_search, t_filter)
