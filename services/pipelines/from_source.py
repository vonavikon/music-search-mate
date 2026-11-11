"""From-source pipeline: the intent-specific tail for intent="from_source"
(Stage 3a of the pipeline refactor).

Behavior-preserving move from LegacyPipeline.run: the from_source-specific
orchestration — the sonar-ground block (one web-grounded fact question when
bothub.sonar_enabled; a resolved concrete "{artist} - {track}" becomes the
first search phrase with the _ref_key dedup guard) — now lives here.
Everything downstream composes the shared legacy stage helpers
(_search_youtube, _filter_and_enrich, _rank_and_decide, _no_results_out);
no stage body is copied. LegacyPipeline keeps a duplicate of the block only
until the registry routes "from_source" here (the wiring lands separately).

ANALOG CHAIN (new, the Stage 3 hybrid capability). Prod case: the query
«Аналог песенки из Магасскао, когда на Алекса во сне падали шницели»
(RU: "an analog of the tune from Madagascar, when schnitzels fell on
Alex in his sleep") - the user
identifies a track VIA its source but wants tracks SIMILAR to it. LLM#1
answers intent=from_source (a source scenario), so the plain flow would
deliver the resolved track itself. When an analog marker («аналог ...»,
«похожее на ...», «напоминает ...», «в духе ...» - RU phrasings of
"analog of" / "similar to" / "reminds of" / "in the spirit of" - or
the EN "similar to ...") appears
in the description AND the sonar ground resolved a concrete track, the
search switches into a REFERENCE SEARCH — the same empirical machinery
SimilarPipeline runs, minus the untrusted-reference resolve: the reference
is already canonical from sonar, so _resolve_reference (track_search /
yt-dlp channel) is skipped and only the getSimilar neighbors are fetched
(via MetadataService.similar_artists, exactly _resolve_reference's step 3):
  - up to _NEIGHBOR_QUERIES neighbor queries lead discovery (the reference
    self-excluded by ref_key); the sonar concrete phrase and the LLM#1
    queries follow; the whole list trims to _MAX_QUERIES;
  - the resolved {artist, track} populates the reference fields and the
    ref_resolved shape, so the shared _rank_and_decide applies the similar
    post-filters — the hard reference ban (the reference and its artist
    never win) and the Last.fm sonic boost;
  - the LLM#2 rank prompt is SYSTEM_RANK_SIMILAR (rank_intent="similar"),
    while SearchOut.intent stays "from_source": analytics keep the source
    scenario, the ranking runs in similar mode.
A marker with a sonar miss keeps the plain from_source flow unchanged.

SCENE-CLIP DEMOTION (live-test F4, 2026-11-05). Prod case: the query
«музыка из погони такси 2» (RU: "music from the Taxi 2 chase") - the
scene clip "Taxi 2 (La course)" (YouTube category
Film & Animation) took llm-rank 0.95 over the real OST track at 0.85
despite the FROM_SOURCE prompt rule: the ranker saw no scene signal in the
candidate lines. After the shared rank tail, the PLAIN flow (not the analog
chain, which ranks in similar mode) deterministically demotes scene-clip
candidates — title words (_SCENE_CLIP_RE) or the Film & Animation category
from the enrich pass — to the tail, mirroring the similar-mode reference
ban. The delivery was audio (m4a); the CONTENT was the movie scene.

SOURCE-FACT CACHE (improvement B4, research card П3). Sonar grounding has
run-to-run variance: the same source query (the Stellaris-organ case) can
resolve to different tracks on different days. The first SUCCESSFUL
grounding is now persisted as a fact and every later identical query reuses
it, so repeat asks are deterministic and the sonar call ($0.008) is skipped.
KEY DESIGN: the LLM#1 extract carries nothing source-shaped for
from_source — reference is {artist: null, track: null} by the extract
prompt's own contract, and signals (language/era/vocal/genre/mood) are
descriptors, not the film/game/ad NAME. Mining a "source mention" out of
the free-text description (stripping mood words, top-N tokens) would be
fragile and could merge distinct sources under one key. So the cache keys
on the FULL normalized description (fuzzy.normalize: lowercase,
punctuation->space, whitespace collapsed — deliberately NOT clean_title,
whose junk-token stripping targets YouTube titles and could merge distinct
user texts like "official" ad phrasings). Same query text -> same key ->
the repeat-variance case is fixed; a different phrasing of the same source
simply misses and grounds fresh — the cache can never serve another
source's track. STORAGE: the shared facts table via the untouched
store.get_fact/save_fact (key tuple artist+track+lang): the namespace
occupies artist="src", lang="src" — "src" is never a real language (the
fact button and the web pass "ru"/"en"), so no collision with real track
facts — and the fact column holds a JSON {"artist", "track"} payload. TTL
= store.FACT_TTL (30 days), enforced by get_fact on read; expired rows are
purged by the existing retention job. WIRING: the store rides ctx.store
(the injection point the surprise pipeline already uses); None (no store
injected — legacy tests, callers that never wired store) keeps the
pre-cache behavior. Every cache read/write sits in try/except: a cache
failure degrades to the plain sonar flow, never breaking the search.

Stateless like every SearchPipeline: one shared instance serves concurrent
searches; per-search state travels in the PipelineContext.
"""
import json
import logging
import re
import time
from typing import TYPE_CHECKING

from services.fuzzy import normalize
from services.pipelines.base import PipelineContext, SearchPipeline
from services.pipelines.legacy import (
    _fields_from_extract,
    _filter_and_enrich,
    _no_results_out,
    _rank_and_decide,
    _search_youtube,
)

if TYPE_CHECKING:
    from services.llm_search import SearchOut

# Same logger name as the other pipelines: the deployed log filters and
# dashboards key on "bot.llm_search".
_log = logging.getLogger("bot.llm_search")

# Analog markers (RU+EN, re.I): the description asks for tracks SIMILAR to a
# track identified via its source, not for the track itself. Compiled once.
# Word boundaries are RU-morphology aware (the same care as the superlative
# regex in this_track.py): \w is Unicode, so \b holds between Cyrillic word
# chars and a marker never matches inside a longer Russian word
# («неаналог», RU "non-analog"); the trailing «на» of похож-на requires a
# real word end («похожее наружу», RU "similar-outward", must not match).
# The bare short form «похож на» ("resembles") needs a
# zero-char gap, hence \w* (not \w+) between the stem and « на».
_ANALOG_RE = re.compile(
    r"\bаналог\w*"                                   # аналог/аналоги/аналогичная... (the "analog" family)
    r"|\bпохож\w*\s+на\b"                            # похожее на/похожая на/похож на... ("similar to")
    r"|\bнапомина\w+"                                # напоминает/напоминающий... ("reminds of")
    r"|\bв\s+духе\b"                                 # в духе ("in the spirit of" that soundtrack)
    r"|\bsimilar\s+to\b"
    r"|\blike\s+the\s+(?:song|track|music)\s+from\b"
    r"|\banalogous\s+to\b",
    re.I)

# --- source-fact cache (improvement B4 / П3; see the module docstring) ---
# The namespace tuple inside the shared facts table: ("src", <normalized
# description>, "src"). "src" is not a language the fact button or the web
# ever passes (ru/en), so these rows never collide with real track facts.
_SRC_FACT_ARTIST = "src"
_SRC_FACT_LANG = "src"
# Bound on the cached key: Telegram texts run to 4096 chars; an essay-length
# description skips the cache entirely (read AND write) rather than keying a
# row on a truncated fragment of it.
_SRC_FACT_KEY_MAX = 300


def _source_fact_key(description: str) -> str:
    """The cache key for a from_source description: fuzzy.normalize of the
    full text (lowercase, punctuation->space, collapsed whitespace) —
    deterministic for the same query text. "" (a stop-signal, not a key)
    when the text is empty or over _SRC_FACT_KEY_MAX chars."""
    norm = normalize(description or "")
    return norm if 0 < len(norm) <= _SRC_FACT_KEY_MAX else ""


# --- hard-case escalation (live 06.11, «Iron Man 2», stuck on BOTH dev and
# prod): sonar grounded the accelerator scene to a real score cue, the user
# rejected it (they meant another cue), and the miss-streak restart asked
# sonar the SAME question — got the SAME cue back, the pool drifted to the
# famous soundtrack (AC/DC) and even re-served the same song from another
# upload. Retries must carry NEW information, so the ladder is:
#   fresh query — the standard cached ground (unchanged behavior);
#   restart #1  — the QUESTION changes: name the rejected candidates, ask
#                 for a DIFFERENT track of the same scene (uncached — the
#                 answer is bound to this restart's rejection set);
#   restart #2  — stop spending searches: ask the USER for details.
# RESTART COUNTING: restart_streak alone cannot ladder — the handler counts
# it as len(rejected_ids) of the CURRENT query state, and every restart
# starts a fresh state, so the second restart sees streak=2 again, never 4.
# The durable signal is the description itself: each restart appends a
# «Не то: ...» block (RU "not it:", the handler's reject marker) on top
# of the previous session's desc
# (desc2 = store["desc"] + new rejects), so the number of «Не то:» blocks IS
# the restart count. A direct caller without the handler protocol still
# ladders via restart_streak >= 4 (the unreachable-by-handler safety cap).
_ESCALATE_STREAK = 2
_DETAILS_STREAK = 4
_DETAILS_BLOCKS = 2
_DETAILS_QUESTION = (
    "Не попадаю в точку. Помоги деталями: это песня со словами или "
    "оркестровая музыка? Быстрая или медленная? Помнишь исполнителя "
    "или пару слов из текста?")


async def _sonar_ground_escalated(bothub, description: str,
                                  rejected_titles, trace=None) -> dict:
    """The restart sonar question: a plain re-ask returns the cue the user
    just rejected. The rejected titles are named explicitly (the restart
    description already carries them as «Не то: ...» tail — sonar parses an
    explicit list better) and the question demands a DIFFERENT track.
    rejected_titles arrives as a SET (the search_candidates contract, the
    handler passes set(...)) — a wave2 runner card caught the set-slicing
    TypeError; order-stable listing via sorted()."""
    from services.llm_search import _sonar_ground

    rejected = ", ".join(sorted(rejected_titles or [])[:3])
    neg = f" Уже показаны и отвергнуты: {rejected}." if rejected else ""
    return await _sonar_ground(
        bothub,
        f"Пользователь ищет музыку: {description!r}.{neg} "
        f"Это НЕ они. Какой ДРУГОЙ трек или оркестровый кий звучит в этой "
        f"сцене? Назови один, самый вероятный (артист и название).",
        trace=trace)


async def _sonar_ground_cached(store, bothub, description: str,
                               trace=None) -> dict:
    """_sonar_ground behind the persisted source-fact cache (П3).

    Read path: a fresh JSON {"artist", "track"} payload under the
    ("src", <key>, "src") tuple is returned as a sonar-shaped answer and the
    sonar call is skipped (a corrupt/partial payload counts as a miss — the
    next successful grounding upserts over it). Miss: sonar as today, then a
    successful track resolution is saved for the next identical query. TTL
    is store.FACT_TTL (30 days), applied by get_fact itself. store=None (no
    injection — legacy callers/tests) or a cache error degrades to the plain
    sonar flow: the cache must never break the search."""
    # Lazy per-call import: the monkeypatchable services.llm_search
    # namespace (a patched services.llm_search._sonar_ground is honored).
    from services.llm_search import _sonar_ground

    key = _source_fact_key(description)
    store_used = store is not None and key
    if store_used:
        try:
            raw = await store.get_fact(_SRC_FACT_ARTIST, key, _SRC_FACT_LANG)
            if raw:
                payload = json.loads(raw)
                artist = (payload.get("artist") or "").strip()
                track = (payload.get("track") or "").strip()
                if artist and track:
                    _log.info("source fact cache hit: %r", key[:60])
                    return {"artist": artist, "track": track}
        except Exception as e:
            _log.warning("кеш source-фактов: чтение не удалось, зову sonar: %s",
                         str(e)[:100])

    s_ans = await _sonar_ground(
        bothub, f"Пользователь ищет музыку: {description!r}. "
                f"Какой это трек (артист и название)? "
                f"Название фильма/игры/сериала может быть русской транслитерацией "
                f"(Стелларис→Stellaris, Магасскао→Madagascar) — ищи оригинальное "
                f"название.", trace=trace)
    if store_used and s_ans.get("track") and (s_ans.get("artist") or "").strip():
        # Persist the FIRST successful grounding; identical later queries
        # reuse it deterministically (the П3 variance fix). Only complete
        # {artist, track} answers are stored (a half-answer would be a
        # dead row the read path discards anyway).
        try:
            await store.save_fact(
                _SRC_FACT_ARTIST, key, _SRC_FACT_LANG,
                json.dumps({"artist": s_ans["artist"].strip(),
                            "track": s_ans["track"].strip()},
                           ensure_ascii=False))
        except Exception as e:
            _log.warning("кеш source-фактов: запись не удалась: %s",
                         str(e)[:100])
    return s_ans


def _demote_scene_clips(out, enriched: list) -> None:
    """Deterministic from_source demotion of movie-scene video clips
    (live-test F4), applied to the SearchOut of the shared rank tail.

    Signals, both already-fetched (no new yt-dlp calls):
      - scene words in the title (_is_scene_clip_title, RU+EN);
      - the Film & Animation YouTube category — SearchOut candidates do not
        carry it (the Candidate dataclass has no category field), so the
        video_ids of the enriched dicts (top-5 carry `categories`) are
        matched through.
    Mechanics mirror the similar-mode reference ban in the shared tail:
    flagged clips sink to the tail (stable sort by (is_scene, -llm_score)),
    best shifts to the first clean candidate when the LLM's pick was a clip
    (the LLM's non-clip pick is re-located by video_id), and each flagged
    clip's match_score loses _SCENE_CLIP_PENALTY — the same -0.20 [noise?]
    takes in score_candidate — so it cannot win the ambiguity gap as a
    runner-up either. A violated prompt instruction never reaches the user."""
    # Lazy per-call imports: the monkeypatchable services.llm_search
    # namespace (the same convention as every stage helper).
    from services.llm_search import (
        _SCENE_CLIP_PENALTY,
        _is_film_category,
        _is_scene_clip_title,
    )

    cands = out.candidates
    if not cands:
        return
    film_ids = {c.get("video_id") for c in enriched
                if _is_film_category(c.get("categories"))}

    def _scene(c) -> bool:
        return _is_scene_clip_title(c.title) or c.video_id in film_ids

    flagged = [c for c in cands if _scene(c)]
    if not flagged:
        return
    for c in flagged:
        c.match_score = max(0.0, (c.match_score or 0.0) - _SCENE_CLIP_PENALTY)
        c.match_reasons = ((c.match_reasons + "; ") if c.match_reasons else "") + \
            f"scene-clip -{_SCENE_CLIP_PENALTY:.2f}"
    if len(flagged) == len(cands):
        return  # nothing clean to promote — keep the LLM order, penalty only
    # The LLM's pick by video_id, not the pre-sort position (the list is
    # about to be reordered; _rank_and_decide has the same doctrine).
    best = cands[out.best_index] if 0 <= out.best_index < len(cands) else None
    cands.sort(key=lambda c: (1 if _scene(c) else 0, -c.score))
    if best is None or _scene(best):
        out.best_index = next((i for i, c in enumerate(cands) if not _scene(c)),
                              out.best_index)
    else:
        out.best_index = next((i for i, c in enumerate(cands)
                               if c.video_id == best.video_id), out.best_index)
    _log.info("scene-clip demotion: %d candidate(s) moved to the tail "
              "(from_source title/category signal)", len(flagged))


class FromSourcePipeline(SearchPipeline):
    """Stage-3a pipeline for intent="from_source": the sonar ground plus the
    analog chain (hybrid "similar to a track from a source") over the shared
    legacy stages."""

    name = "from_source"

    async def run(self, ctx: PipelineContext) -> "SearchOut | None":
        """Execute the from_source tail: sonar ground (a concrete track leads
        discovery), the analog chain switch into a similar-mode reference
        search when the description asks for analogs, YouTube search,
        filter/enrich, then the shared LLM#2 rank tail."""
        # Lazy per-call imports: services.llm_search imports this package at
        # module level (cycle), and per-call resolution honors test
        # monkeypatching of the services.llm_search namespace.
        from services.llm_search import (
            _MAX_QUERIES,
            _NEIGHBOR_QUERIES,
            _ref_key,
        )

        # Local aliases keep the moved body textually identical to the
        # original orchestrator code (plain local names, not ctx attributes).
        description = ctx.description
        bothub = ctx.bothub
        trace = ctx.trace
        t_start = ctx.t_start

        f = _fields_from_extract(ctx.extract)
        intent = f.intent
        queries = f.queries

        # Reference inputs for the shared rank tail: the empty dict for the
        # plain flow (identical to the other non-similar pipelines); the
        # analog chain fills both with the sonar-resolved reference.
        ref_resolved = {"artist": "", "track": "", "source": "llm", "neighbors": []}
        rank_intent = None  # None = rank under f.intent (the legacy behavior)

        # --- hard-case escalation ladder (live 06.11 «Iron Man 2»): the
        # SECOND restart stops searching and asks the user for details —
        # two failed grounds plus a drifted pool mean the query text alone
        # cannot name the track, and honest details beat a third guess.
        # Restart counting: «Не то:» blocks in the description (the handler
        # accumulates one per restart, see the ladder note above); the
        # streak cap covers direct callers without the handler protocol. ---
        _restarts = (description or "").count("Не то:")
        if intent == "from_source" and (
                _restarts >= _DETAILS_BLOCKS
                or (ctx.restart_streak or 0) >= _DETAILS_STREAK):
            _log.info("from_source escalation -> asking the user for details "
                      "(restarts=%d streak=%d)", _restarts, ctx.restart_streak)
            from services.llm_search import SearchOut
            return SearchOut(intent=intent, signals=f.signals, queries=[], assoc=f.assoc,
                             reference=f.reference, target=f.target, candidates=[],
                             best_index=-1, confidence="low", intro=f.intro,
                             need_clarify=True, clarify_question=_DETAILS_QUESTION,
                             trace=trace)

        # --- sonar-ground for from_source (moved from LegacyPipeline.run),
        # behind the persisted source-fact cache (B4/П3): a cached
        # "source -> track" fact short-circuits the sonar call; a miss
        # grounds once and saves for the identical next query. Ads/film/TV
        # are fact scenarios where LLM memory lies; one web-grounded attempt
        # yields a concrete track that becomes the first search phrase (then
        # the regular pipeline and rank). The FIRST miss-streak restart
        # switches to the escalated question (new information per retry). ---
        if intent == "from_source" and bothub.sonar_enabled:
            if ((ctx.restart_streak or 0) >= _ESCALATE_STREAK
                    or _restarts >= 1):
                _log.info("from_source escalation: escalated sonar question (restarts=%d)", _restarts)
                s_ans = await _sonar_ground_escalated(bothub, description,
                                                      ctx.excluded_titles, trace=trace)
            else:
                s_ans = await _sonar_ground_cached(ctx.store, bothub,
                                                   description, trace=trace)
            if s_ans.get("track"):
                concrete = f"{s_ans['artist'].strip()} - {s_ans['track'].strip()}"
                queries = [q for q in queries if _ref_key(q) != _ref_key(concrete)]
                queries.insert(0, concrete)
                _log.info("sonar ground (from_source): %r -> %r", description[:60], concrete)

                # --- analog chain (before ranking): the description asks for
                # tracks SIMILAR to the track the source describes. The
                # sonar-resolved {artist, track} is already canonical — the
                # untrusted-reference machinery (_resolve_reference:
                # track_search / yt-dlp channel) is skipped; only the
                # getSimilar neighbors are fetched, like _resolve_reference's
                # step 3. ---
                if _ANALOG_RE.search(description or ""):
                    ref_artist = s_ans["artist"].strip()
                    ref_track = s_ans["track"].strip()
                    # Populate what the shared _rank_and_decide reads:
                    # f.reference feeds SearchOut.reference and the ban's log
                    # line; ref_resolved feeds the ban key (canonical artist)
                    # and the sonic boost.
                    f.reference["artist"] = ref_artist
                    f.reference["track"] = ref_track
                    ref_resolved = {"artist": ref_artist, "track": ref_track,
                                    "source": "sonar", "neighbors": []}
                    if ctx.metadata_svc is not None:
                        ref_resolved["neighbors"] = await ctx.metadata_svc.similar_artists(
                            ref_artist, limit=6)
                    # The reference's empirical neighbors lead discovery (the
                    # reference self-excluded by ref_key — the same guard
                    # SimilarPipeline applies); the concrete sonar phrase and
                    # the LLM#1 queries follow; the tail trims to the cap.
                    if ref_resolved["neighbors"]:
                        ref_k = _ref_key(ref_resolved["artist"])
                        neighbors = [n for n in ref_resolved["neighbors"][:_NEIGHBOR_QUERIES]
                                     if _ref_key(n) != ref_k]
                        if neighbors:
                            queries = (neighbors + list(queries))[:_MAX_QUERIES]
                    # Similar-mode ranking inside the shared tail:
                    # SYSTEM_RANK_SIMILAR + the reference ban + the sonic
                    # boost, while SearchOut.intent stays "from_source".
                    rank_intent = "similar"
                    _log.info("from_source analog chain: resolved reference %r -> similar mode",
                              concrete)

        # step 3: youtube
        raw = await _search_youtube(queries, on_progress=ctx.on_progress, trace=trace)
        t_search = time.perf_counter() - t_start
        if not raw:
            return _no_results_out(f, queries, trace)

        # steps 4-4.5: filter + enrich (excluded_titles = title-level
        # rejection, the same F2 contract similar.py applies: a restart must
        # not serve the rejected cue from another channel/upload — the w9
        # wave2 card caught from_source silently ignoring it)
        filtered, enriched = await _filter_and_enrich(
            raw, ctx.excluded_ids, ctx.excluded_artists,
            metadata_svc=ctx.metadata_svc, on_progress=ctx.on_progress, trace=trace,
            excluded_titles=ctx.excluded_titles)
        t_filter = time.perf_counter() - t_start

        # steps 5+: LLM#2 rank and the decision tail. rank_intent="similar"
        # (analog chain) switches the rank machinery — the SIMILAR prompt,
        # the reference ban, the sonic boost — without touching
        # SearchOut.intent.
        out = await _rank_and_decide(ctx, f, queries, filtered, enriched, ref_resolved,
                                     len(raw), t_search, t_filter, rank_intent=rank_intent)
        # Scene-clip demotion (live-test F4): the PLAIN flow only — the
        # analog chain ranks in similar mode, where the scene-clip policy of
        # from_source does not apply.
        if rank_intent is None:
            _demote_scene_clips(out, enriched)
        return out
