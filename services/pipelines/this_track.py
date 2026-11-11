"""This-track pipeline: the intent-specific tail for intent="this_track"
(Stage 1 of the pipeline refactor).

Composes the shared legacy stages (module-level helpers in legacy.py) and
adds two product features on top:

1. CONFIDENT-MATCH SHORTCUT (skips LLM#2). Prod case: an exact target
   ("Depeche Mode — Enjoy the Silence") makes the rank call redundant when
   fuzzy matching already picks one candidate decisively. After candidate
   filtering, every filtered candidate is fuzzy-scored against the target;
   a best score >= FUZZY_SHORTCUT_MIN with a clear gap to the runner-up (a
   DIFFERENT track_key, the same dedup logic the ambiguity check uses)
   builds the SearchOut directly — no rank call, no rank latency (7-20s).

2. SUPERLATIVE FACT-RESOLVE + FACT-CHECK. Prod case: the query "депеш мод
   самый первый их трек" (RU: "depeche mode their very first track") -
   the artist is named, the track is a discography FACT the LLM
   memory gets wrong. One sonar-ground question resolves the concrete track
   into the target BEFORE query expansion, so discovery and matching target
   it. When LLM#1 already filled target.track, the same question FACT-CHECKS
   that fill (bake-off t4): sonar's confident different track wins, a sonar
   miss keeps the LLM fill.

3. ARTIST RECOVERY FROM THE TRACK STRING (skips LLM#2). Prod case:
   "Deoche mode" — LLM#1 extracts target {artist: None, track: "Deoche
   mode"} (the whole query lands in the track field), the confident-match
   shortcut above requires BOTH fields, never fires, and the flow degrades
   to low-match ranking. When the artist is empty but the track string is
   not, the track string IS the full query: a candidate whose channel
   confidently fuzzy-matches it recovers the artist and fires the same
   shortcut path with the verdict "Channel - Title" (see _artist_recovery).

4. YT-SUGGEST QUERY NORMALIZATION (B3, research П2). YouTube's keyless
   search-suggest endpoint returns canonical spellings for mangled queries
   («deoche mode» -> «depeche mode»). Before query expansion the target's
   artist/track strings are asked for suggestions once; when a suggestion
   near-matches the raw primary query but spells it differently, the top-2
   canonical spellings join the search pool as EXTRA queries right after
   the primary — canonical queries surface the official uploads the LLM's
   mangled target missed, and the artist-recovery/shortcut gates decide on
   the wider pool exactly as before (see _suggest_widen + services/suggest).

Stateless like every SearchPipeline: one shared instance serves concurrent
searches; per-search state travels in the PipelineContext.
"""
import logging
import re
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

from services.match_score import AMBIGUITY_GAP, track_key
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

# Same logger name as the legacy pipeline: the deployed log filters and
# dashboards key on "bot.llm_search".
_log = logging.getLogger("bot.llm_search")

# Minimum best fuzzy pair_match for the confident-match shortcut. Tuned
# against the prod tuning cases (exact artist+title uploads score >= 0.90;
# partial/cover/guest-vocalist versions stay under). Deliberately a module
# constant, not config: promote to config once match-score feedback
# accumulates.
FUZZY_SHORTCUT_MIN = 0.80

# YT-suggest gate (B3): a suggestion must fuzzy-match the raw primary query
# at least this well to count as a canonical RE-SPELLING of it (lower would
# feed unrelated popular queries into the pool), and the top _SUGGEST_EXTRA_N
# differing spellings join the pool as extra search queries.
_SUGGEST_SIM_MIN = 0.60
_SUGGEST_EXTRA_N = 2

# Superlative / discography-fact markers (RU+EN) that turn an artist-only
# this_track query into a fact question for sonar. Compiled once; extended
# with the obvious declension/translation variants of the same idea. The
# three alternatives are the fact KINDS: which one matched decides which
# question sonar is asked (a "last single" phrasing must not resolve the
# debut into target).
_SUPERLATIVE_RE = re.compile(
    r"(?P<debut>"
    r"самый первый"
    r"|дебютн"
    r"|перв(ый|ая|ую|ого|ым) (трек|сингл|альбом|клип|песн|релиз)"
    r"|debut"
    r"|first (track|single|album|song|release)"
    r")"
    r"|(?P<last>"
    r"последн(ий|яя|юю|ее|его|им) (трек|сингл|альбом|клип|песн|релиз)"
    r"|last (track|single|album|song|release)"
    r")"
    r"|(?P<most_famous>"
    r"самая известная|самый известный|самый популярный|самая популярная"
    r"|biggest hit"
    r"|most (famous|popular)"
    r")",
    re.I)

# The sonar question per fact kind. The question must anchor to exactly the
# fact the user's marker names: the answer lands in target and everything
# downstream (queries, matching, verdict) follows it.
_SUPERLATIVE_QUESTIONS = {
    "debut": "Какой первый сингл/дебютный трек у артиста {artist}?",
    "last": "Какой последний сингл/трек у артиста {artist}?",
    "most_famous": "Какая самая известная/популярная песня у артиста {artist}?",
}


def _superlative_kind(description: str) -> str | None:
    """Fact kind of the first superlative marker in the description:
    "debut" | "last" | "most_famous", or None without a marker. Exactly one
    kind group participates in any match (the groups are the pattern's
    top-level alternatives)."""
    m = _SUPERLATIVE_RE.search(description or "")
    if m is None:
        return None
    return next((kind for kind, hit in m.groupdict().items() if hit is not None), None)


def _track_key_of(c: dict) -> str:
    """track_key (match_score) over a filtered raw dict: the function reads
    .title/.channel attributes, filtered candidates are dicts."""
    return track_key(SimpleNamespace(title=c.get("title") or "",
                                     channel=c.get("channel") or ""))


class ThisTrackPipeline(SearchPipeline):
    """Stage-1 pipeline for intent="this_track": the legacy stages plus the
    confident-match shortcut, the artist-recovery shortcut and the
    superlative fact-resolve."""

    name = "this_track"

    async def run(self, ctx: PipelineContext) -> "SearchOut | None":
        """Execute the this_track tail: superlative fact-resolve, query
        expansion, YouTube search, filter/enrich, confident-match shortcut,
        artist recovery from the track string, then the regular LLM#2 rank
        tail when neither shortcut fires."""
        # Lazy per-call imports: services.llm_search imports this package at
        # module level (cycle), and per-call resolution honors test
        # monkeypatching of the services.llm_search namespace.
        from services.llm_search import _expand_queries, _ref_key, _sonar_ground

        f = _fields_from_extract(ctx.extract)
        queries = f.queries
        target = f.target

        # --- superlative fact-resolve (before query expansion). The artist
        # is named but the track is a discography fact ("their first single")
        # — LLM memory lies here, one web-grounded sonar question doesn't.
        # The question asks for exactly the fact the marker names (debut /
        # last / most famous — a mismatched question resolves the WRONG fact
        # into target). The resolved track fills target BEFORE
        # _expand_queries, so the recall phrases and the rank target the
        # concrete track. When LLM#1 already filled target.track the sonar
        # answer doubles as a FACT-CHECK of that fill (bake-off t4: the
        # fuzzy shortcut used to fire on the LLM's own unverified fill):
        # sonar wins on a confident DIFFERENT track, the LLM fill survives a
        # sonar miss, agreement changes nothing. Sonar is respected exactly
        # like the from_source branch (bothub.sonar_enabled gate;
        # _sonar_ground itself also checks and never raises).
        tgt_artist = (target.get("artist") or "").strip()
        tgt_track = (target.get("track") or "").strip()
        fact_kind = None
        if tgt_artist and ctx.bothub.sonar_enabled:
            fact_kind = _superlative_kind(ctx.description or "")
        if fact_kind:
            question = _SUPERLATIVE_QUESTIONS[fact_kind].format(artist=tgt_artist)
            s_ans = await _sonar_ground(
                ctx.bothub,
                f"{question} Пользователь просит: {ctx.description}",
                trace=ctx.trace)
            if s_ans.get("track"):
                resolved_artist = (s_ans.get("artist") or "").strip() or tgt_artist
                resolved_track = s_ans["track"].strip()
                concrete = f"{resolved_artist} - {resolved_track}"
                if tgt_track and _ref_key(f"{tgt_artist} - {tgt_track}") == _ref_key(concrete):
                    # The web-grounded fact-check agrees with the LLM's own
                    # fill: keep it, nothing to resolve or correct.
                    _log.info("superlative: sonar confirmed llm %r",
                              f"{tgt_artist} - {tgt_track}")
                else:
                    if tgt_track:
                        _log.info("superlative: llm %r corrected by sonar %r",
                                  f"{tgt_artist} - {tgt_track}", concrete)
                    else:
                        _log.info("sonar resolve (this_track superlative): %r -> %r",
                                  (ctx.description or "")[:60], concrete)
                    queries = [q for q in queries if _ref_key(q) != _ref_key(concrete)]
                    queries.insert(0, concrete)
                    target["artist"] = resolved_artist
                    target["track"] = resolved_track
            elif tgt_track:
                # Sonar missed on a query whose LLM#1 fill WAS a named track:
                # keep the fill (unlike the artist-only path it is not a
                # guess to throw away) and let the regular flow continue.
                _log.info("superlative: sonar miss, keeping llm fill %r",
                          f"{tgt_artist} - {tgt_track}")

        # --- yt-suggest query normalization (B3, before query expansion):
        # canonical spellings of the mangled target strings join the pool as
        # extra queries so discovery surfaces the official uploads the LLM's
        # target missed. Fail-soft: a suggest outage changes nothing here. ---
        queries = await self._suggest_widen(queries, target)

        # Recall expansion with the (possibly now concrete) target — before the YouTube call.
        queries = _expand_queries(queries, f.intent, target)

        # steps 3-4.5: the shared discovery/filter stages.
        raw = await _search_youtube(queries, on_progress=ctx.on_progress, trace=ctx.trace)
        t_search = time.perf_counter() - ctx.t_start
        if not raw:
            return _no_results_out(f, queries, ctx.trace)
        filtered, enriched = await _filter_and_enrich(
            raw, ctx.excluded_ids, ctx.excluded_artists,
            metadata_svc=ctx.metadata_svc, on_progress=ctx.on_progress, trace=ctx.trace)
        t_filter = time.perf_counter() - ctx.t_start

        # --- confident-match shortcut: right before the LLM#2 rank call ---
        shortcut = self._confident_match(filtered, target)
        if shortcut is not None:
            best_index, best_score, pair_scores = shortcut
            _log.info("shortcut: confident fuzzy match %.2f for target %r",
                      best_score, f"{(target.get('artist') or '').strip()} - "
                                  f"{(target.get('track') or '').strip()}")
            if ctx.trace:
                # Path marker next to the intent tag (tag() accumulates):
                # shortcut traces are filterable in the dashboards.
                ctx.trace.tag(["shortcut"])
            return self._shortcut_out(f, queries, filtered, best_index, pair_scores, ctx.trace)

        # --- artist recovery from the track string (the "Deoche mode" gap):
        # LLM#1 put the whole query into target.track and left target.artist
        # empty, so the confident-match shortcut above (which requires BOTH
        # fields) cannot fire. _artist_recovery treats the track string as
        # the full query and reuses the SAME shortcut path. ---
        recovered = self._artist_recovery(filtered, target)
        if recovered is not None:
            best_index, best_score, pair_scores = recovered
            channel_artist = ((filtered[best_index].get("channel") or "")
                              .removesuffix(" - Topic").strip())
            _log.info("shortcut: artist recovered from track string (%r -> %r)",
                      (target.get("track") or "").strip(), channel_artist)
            if ctx.trace:
                ctx.trace.tag(["shortcut"])
            return self._shortcut_out(
                f, queries, filtered, best_index, pair_scores, ctx.trace,
                verdict=self._recovery_verdict(
                    channel_artist, filtered[best_index].get("title") or ""))

        # Shortcut conditions failed — the regular legacy rank tail, unchanged.
        ref_resolved = {"artist": "", "track": "", "source": "llm", "neighbors": []}
        return await _rank_and_decide(ctx, f, queries, filtered, enriched, ref_resolved,
                                      len(raw), t_search, t_filter)

    async def _suggest_widen(self, queries: list[str], target: dict) -> list[str]:
        """Widen the search pool with canonical spellings from YouTube's
        search-suggest (services/suggest.yt_suggest), BEFORE query expansion.
        Returns the (possibly extended) query list; NEVER raises and never
        reorders — extras land right after the primary query, ahead of the
        recall expansion, so the _MAX_QUERIES cap trims the broad tail first.

        Gate: only when a suggestion fuzzy-matches the raw primary query
        (>= _SUGGEST_SIM_MIN) yet SPELLS it differently (normalized compare)
        — i.e. suggest understood the query and offers a canonical re-spelling
        («deoche mode» -> «depeche mode»). Already-canonical queries get
        nothing (the top suggestion equals the query), unrelated suggestions
        get nothing (similarity under the gate). The fuzzy/shortcut gates
        downstream are untouched: suggestions only widen the POOL."""
        artist = (target.get("artist") or "").strip()
        track = (target.get("track") or "").strip()
        suggest_q = f"{artist} {track}".strip()
        if not suggest_q:
            return queries
        # Lazy imports honor per-call patching of the services namespaces in
        # tests (the module convention; also keeps the import cycle-free).
        from services import fuzzy, suggest
        try:
            suggestions = await suggest.yt_suggest(suggest_q)
        except Exception:  # noqa: BLE001 - fail-soft: suggest must never break search
            _log.debug("yt suggest failed for %r", suggest_q, exc_info=True)
            return queries
        if not suggestions:
            return queries
        raw = queries[0].strip() if queries else suggest_q
        raw_key = fuzzy.normalize(raw)
        canonical = any(
            fuzzy.similarity(s, raw) >= _SUGGEST_SIM_MIN
            and fuzzy.normalize(s) != raw_key
            for s in suggestions)
        if not canonical:
            return queries
        seen = {fuzzy.normalize(q) for q in queries}
        extras = []
        for s in suggestions:
            key = fuzzy.normalize(s)
            if key and key not in seen:
                seen.add(key)
                extras.append(s.strip())
            if len(extras) >= _SUGGEST_EXTRA_N:
                break
        if not extras:
            return queries
        _log.info("yt suggest: %r -> %r", suggest_q, extras)
        return queries[:1] + extras + queries[1:]

    def _confident_match(self, filtered: list, target: dict):
        """Fuzzy shortcut decision. Returns (best_index, best_score,
        pair_scores) when the match is confident enough to skip LLM#2, else
        None (pair_scores: the pair_match of every filtered candidate, in
        filtered order — feeds Candidate.score in the shortcut output).

        Fires when the best fuzzy pair_match >= FUZZY_SHORTCUT_MIN AND the
        best candidate is not noise-flagged (is_noise) AND the gap to the
        best DIFFERENT-track_key candidate (match_score.track_key —
        duplicate uploads of the same song don't count, the same dedup
        logic is_ambiguous uses) >= AMBIGUITY_GAP. Near-equal different
        tracks (two versions of the same title) and noisy bests (tribute
        covers) stay ambiguous -> LLM#2."""
        tgt_artist = (target.get("artist") or "").strip()
        tgt_track = (target.get("track") or "").strip()
        if not tgt_artist or not tgt_track or not filtered:
            return None
        # Lazy import honors per-call patching of services.fuzzy in tests.
        from services import fuzzy
        pair_scores = []
        for c in filtered:
            # Candidate artist: the channel minus the YouTube Music suffix;
            # title: the raw title (fuzzy normalizes internally).
            cand_artist = (c.get("channel") or "").removesuffix(" - Topic")
            cand_title = c.get("title") or ""
            pair_scores.append(fuzzy.pair_match(cand_artist, cand_title, tgt_artist, tgt_track))
        best_index = max(range(len(pair_scores)), key=lambda i: pair_scores[i])
        best_score = pair_scores[best_index]
        if best_score < FUZZY_SHORTCUT_MIN:
            return None
        # Noise guard: a tribute-band cover ("Enjoy the Silence (Cover)" on
        # a tribute channel) can fuzzy-match the target almost perfectly and
        # be the pool's best when the official upload is missing. A noisy
        # best must never be delivered as the original with confidence high
        # — fall through to the LLM#2 rank path, which ranks [noise?]
        # versions down.
        if filtered[best_index].get("is_noise"):
            return None
        best_key = _track_key_of(filtered[best_index])
        runner_up = max((s for s, c in zip(pair_scores, filtered)
                         if _track_key_of(c) != best_key), default=0.0)
        if best_score - runner_up < AMBIGUITY_GAP:
            return None
        return best_index, best_score, pair_scores

    def _artist_recovery(self, filtered: list, target: dict):
        """Fuzzy artist-recovery decision for an artist-less target. Returns
        (best_index, best_score, pair_scores) when the artist can be
        recovered from the track string confidently enough to skip LLM#2,
        else None (pair_scores: pair_match of every filtered candidate, in
        filtered order — feeds Candidate.score in the shortcut output).

        Known LLM#1 gap (prod: «Deoche mode»): the whole query lands in
        target.track and target.artist stays empty, so _confident_match
        (which requires both fields) never fires. Here the track string is
        treated as the full query: every filtered candidate is scored with
        pair_match(channel, title, track_str, track_str) — the artist side
        of the pair IS the track string, so the right candidate's channel
        (the recovered artist) carries the score. Fires when:
          - the best pair's channel fuzzy-matches the track string at the
            usual shortcut confidence: artist_match >= FUZZY_SHORTCUT_MIN
            ("Depeche Mode" vs "Deoche mode" = 0.87; a tribute channel 0.56).
            The channel IS the recovered artist — this, not the blended
            pair, is the recovery's confidence signal;
          - the best pair clears fuzzy.WRONG_ARTIST_CAP (fuzzy's own
            "clearly wrong pairing" bar): the title side is at least
            on-topic, not an interview or a random upload on the right
            channel;
          - the best is not noise-flagged (the same guard as
            _confident_match);
          - the gap to the best DIFFERENT-ARTIST candidate (channel
            artist_match < fuzzy.ARTIST_GATE against the best's channel)
            is >= AMBIGUITY_GAP. Deliberately NOT the usual
            different-track_key gap: every same-artist candidate shares the
            0.4-weighted artist component, so two different tracks of the
            recovered artist always sit within AMBIGUITY_GAP of each other
            and the track_key gap could never fire here. For an artist
            recovery same-artist candidates are agreement, not ambiguity;
            a near-equal OTHER artist is the real ambiguity."""
        tgt_artist = (target.get("artist") or "").strip()
        tgt_track = (target.get("track") or "").strip()
        if tgt_artist or not tgt_track or not filtered:
            return None
        # Lazy import honors per-call patching of services.fuzzy in tests.
        from services import fuzzy
        pair_scores = []
        for c in filtered:
            cand_artist = (c.get("channel") or "").removesuffix(" - Topic")
            cand_title = c.get("title") or ""
            pair_scores.append(fuzzy.pair_match(cand_artist, cand_title, tgt_track, tgt_track))
        best_index = max(range(len(pair_scores)), key=lambda i: pair_scores[i])
        best_score = pair_scores[best_index]
        best_channel = (filtered[best_index].get("channel") or "").removesuffix(" - Topic")
        # The recovery's confidence: the channel matches the queried string.
        if fuzzy.artist_match(best_channel, tgt_track) < FUZZY_SHORTCUT_MIN:
            return None
        # Title-side floor (fuzzy's own wrong-pairing cap).
        if best_score < fuzzy.WRONG_ARTIST_CAP:
            return None
        # Noise guard, mirroring _confident_match: a tribute cover must not
        # be delivered as the original with shortcut confidence.
        if filtered[best_index].get("is_noise"):
            return None
        runner_up = max(
            (s for s, c in zip(pair_scores, filtered)
             if fuzzy.artist_match((c.get("channel") or "").removesuffix(" - Topic"),
                                   best_channel) < fuzzy.ARTIST_GATE),
            default=0.0)
        if best_score - runner_up < AMBIGUITY_GAP:
            return None
        return best_index, best_score, pair_scores

    @staticmethod
    def _recovery_verdict(channel_artist: str, title: str) -> str:
        """Verdict line for the artist-recovery shortcut: "{channel} - {title
        cleaned}". The channel is the recovered artist; the title is cleaned
        of YouTube junk and of a leading artist run, so "Depeche Mode -
        Enjoy the Silence (Official Video)" on channel "Depeche Mode" gives
        "Depeche Mode - Enjoy the Silence". Case is preserved
        (fuzzy.clean_title lowercases — right for scoring, wrong for a
        user-facing line), so this reuses fuzzy's junk regexes directly
        instead of clean_title."""
        from services import fuzzy
        t = fuzzy._BRACKET_GROUP.sub(
            lambda m: " " if fuzzy._bracket_is_junk(m.group(0)[1:-1]) else m.group(0),
            (title or "").strip())
        tokens = [tok for tok in t.split() if tok.lower() not in fuzzy._JUNK_TOKENS]
        # Drop the leading artist run ("Depeche Mode - Enjoy the Silence"):
        # the verdict already names the artist via the channel.
        art_tokens = fuzzy.normalize(channel_artist).split()
        if art_tokens and [tok.lower() for tok in tokens[:len(art_tokens)]] == art_tokens:
            tokens = tokens[len(art_tokens):]
            while tokens and not tokens[0].strip("-–—"):
                tokens.pop(0)
        track_part = " ".join(tokens).strip()
        return f"{channel_artist} - {track_part}" if track_part else channel_artist

    def _shortcut_out(self, f, queries: list, filtered: list,
                      best_index: int, pair_scores: list, trace,
                      verdict: str | None = None) -> "SearchOut":
        """SearchOut for the shortcut: candidates from the filtered list
        (fuzzy score in .score — the score that decided the pick), match
        fields exactly as the rank tail sets them, confidence high, verdict
        one clean line from the matched target. `verdict` overrides the
        default "{artist} - {track}" line (the artist-recovery pass builds
        its own from the recovered channel + cleaned title)."""
        from services.llm_search import Candidate, SearchOut, score_candidate
        tgt_artist = (f.target.get("artist") or "").strip()
        tgt_track = (f.target.get("track") or "").strip()
        cands = []
        for c, fuzzy_score in zip(filtered, pair_scores):
            cand = Candidate(
                c["video_id"], c["title"], c["channel"], c["duration"], c["views"],
                fuzzy_score,
                "",
                c.get("is_noise", False),
                source=c.get("source", "youtube"),
                url=c.get("url") or f"https://www.youtube.com/watch?v={c['video_id']}",
                fallback_url=c.get("fallback_url", ""),
            )
            s = score_candidate(cand, f.target, f.intent, cand.score)
            cand.match_score = s.score
            cand.match_reasons = "; ".join(s.reasons)
            cands.append(cand)
        # Wave2 w3 (judge PARTIAL): the verdict must describe the DELIVERED
        # upload, not the LLM target — the shortcut matched a plain take
        # while the target named a remix, and the card's explanation and the
        # delivered file diverged. Built from the best candidate's own
        # title (same cleaning as the recovery verdict), artist from the
        # canonical target.
        if verdict is not None:
            final_verdict = verdict
        elif 0 <= best_index < len(filtered):
            final_verdict = self._recovery_verdict(
                tgt_artist or (filtered[best_index].get("channel") or "")
                .removesuffix(" - Topic").strip(),
                filtered[best_index].get("title") or "")
        else:
            final_verdict = f"{tgt_artist} - {tgt_track}"
        return SearchOut(
            intent=f.intent, signals=f.signals, queries=queries, assoc=f.assoc,
            reference=f.reference, target=f.target, candidates=cands,
            best_index=best_index, confidence="high", intro=f.intro,
            verdict=final_verdict,
            trace=trace,
        )
