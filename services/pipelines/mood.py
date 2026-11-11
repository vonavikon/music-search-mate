"""Mood pipeline: the intent-specific tail for intent="mood"
(Stage 2b of the pipeline refactor).

Product case: requests like «музыка для бега» (RU: "music for running") or
«грустная музыка для человека которого никто не понимает» (RU: "sad music
for a person nobody understands") have NO reference. Under intent=similar
the reference machinery no-ops and the candidate pool was pure LLM guesses —
the main source of similar misses. Stage 2b gives them an empirical source:
real track pools from Last.fm tag.gettoptracks over the LLM#1 mood_tags.

Orchestration:
  - up to 2 tags (deduped, lowercased; when LLM#1 returns more than 2, a
    curated priority list decides which ones — junk tags lose to real
    Last.fm community tags) -> tag_top_tracks(tag, limit=6);
  - pool resilience (audit m1): a tag that cannot be a Last.fm tag at all
    (Cyrillic junk like «бег», RU for "running") drops BEFORE any fetch; a
    tag whose pool
    comes back empty walks a small synonym chain (max 3 fetches per tag)
    so an intermittent tag.gettoptracks [] does not no-op discovery; and
    when every pool ends up empty the degradation to LLM-only guesses is
    logged as a warning instead of passing silently;
  - the two pools interleave (alternating) so one tag cannot monopolize
    discovery; the combined empirical queries cap at 6; entries that are
    the same song under different spellings/orderings collapse by title
    key (live F3: the same seed echoed from two tag pools);
  - EQUAL QUEUES (B2, live 2026-11-05: «бег» and «музыка для бега» built
    byte-identical pools and the ranker re-delivered the same winner): the
    LLM#1 guess queries and the empirical pool tracks INTERLEAVE — llm
    guess, pool track, llm guess, ... — instead of pools-first, so both
    sources feed the ranker as equals (the Amazon-RecSys selector pattern:
    the LLM wins as a SELECTOR over pooled candidates, not as the pool
    generator); the whole list trims to _MAX_QUERIES;
  - MEMORY OF DELIVERED (B2): after filtering, candidates whose cleaned
    title matches one of the user's recently delivered best_titles
    (store.get_recent_best_titles, lookback 20) drop, and so do candidates
    of recently delivered ARTISTS (the _diversity.artist_key of both
    sides) — a synonymous repeat with a byte-identical pool cannot
    re-deliver the same winner or its artist («Hilary Duff дважды», RU:
    "Hilary Duff twice").
    Honest fallback: when a cut would drop EVERY candidate, the pre-cut
    pool delivers anyway (the surprise pattern). user_id None skips it;
  - POP-POLLUTION GUARD (B2, candidate-side prior): the Last.fm charts for
    workout/running-family tags are pop-polluted (Hilary Duff topped
    «музыка для бега»), while the old pipeline's raw LLM guesses (Daft
    Punk, Mick Gordon) were the better live answers. A POOL-source
    candidate tie-breaks BELOW an LLM-source candidate of the same
    order-of-magnitude views band (a -0.05-shaped penalty on the pre-rank
    pick order — the only order mood owns; LLM#2 scores stay untouched in
    the shared rank tail). search_metadata returns one flat merged list
    with no per-query grouping, so pool provenance is classified on the
    candidate side by record identity with the pool queries;
  - NO SEED BYPASSES THE RANKER: every empirical and LLM query flows
    through the same search -> filter -> diversity -> LLM#2 rank chain;
    nothing in this pipeline promotes a pool seed past the rank;
  - a DETERMINISTIC diversity guard runs before ranking (code, not prompt):
    3+ copies of the same RECORD in the top-6 (the same song under any
    title ordering/channel/re-upload — live F3) keep only the best 1, and
    3+ of the top-6 filtered candidates from the same canonical ARTIST
    (parsed from the "Artist - Track" title, channel minus " - Topic" as
    the fallback) keep only the best 2 - the «не три трека одного
    исполнителя» rule (RU: "no three tracks by one artist"), enforced
    even when the LLM ignores the prompt line;
    keyed by artist, not channel, so one artist cannot fill the slots under
    two upload channels (prod m1: 4/5 slots, one artist);
  - MIX DEMOTE (q275, live 2026-11-06: the best pick for a running query
    was a multi-DJ compilation mix; the «То» (hit button) press went to
    the real track next - the problem was never the genre, it was MIXES
    posing as tracks): AFTER
    the shared rank tail, on the final SearchOut, every candidate whose
    title names itself a mix (compiled markers + the "DJ X, Y, Z - ..."
    3-artists-before-the-dash shape) carries -0.30 on its LLM#2 score and
    the list re-sorts score-descending (stable), best_index re-pointed by
    video_id (the same-record collapse reindex pattern in
    services/llm_search.py). One marker is enough; demoting EVERY candidate
    keeps the list (a uniform -0.30 is order-neutral — the pool never
    empties here); no mixes -> byte-identical passthrough.

Downstream stages are the shared legacy helpers (_search_youtube,
_filter_and_enrich, _rank_and_decide): mood has no target, so scoring flows
through score_candidate's non-target branch exactly like similar; the
reference ban and the sonic boost are similar-only no-ops here. The LLM#2
system prompt is SYSTEM_RANK_MOOD (selected inside _rank_and_decide via
rank_prompt_for).

Stateless like every SearchPipeline: one shared instance serves concurrent
searches; per-search state travels in the PipelineContext.
"""
import logging
import re
import time

from services import store as _default_store  # module-level functions: no instance to inject
from services.pipelines._diversity import (
    artist_diversity_guard,
    artist_key,
    normalized_title_key,
)
from services.pipelines.base import PipelineContext, SearchPipeline
from services.pipelines.legacy import (
    _fields_from_extract,
    _filter_and_enrich,
    _no_results_out,
    _rank_and_decide,
    _search_youtube,
)

# Same logger name as the other pipelines: the deployed log filters and
# dashboards key on "bot.llm_search".
_log = logging.getLogger("bot.llm_search")

# Empirical tag pools: at most two tags feed discovery (a wider blend dilutes
# the state), each pool asks Last.fm for this many tracks, and the combined
# interleaved empirical queries cap here (leaving _MAX_QUERIES slots for the
# LLM guesses after them).
_MAX_MOOD_TAGS = 2
_TAG_POOL_LIMIT = 6
_MAX_EMPIRICAL = 6

# Tag curation: LLM#1 sometimes returns 3+ mood_tags where late entries are
# junk ("songs to run to at 6am"). When more tags arrive than the _MAX_MOOD_TAGS
# slots, curated Last.fm community tags win: priority below (higher = more
# central to the mood cases the bot serves), stable-sorted so unknown tags
# keep the LLM's own order and still fill the slots when nothing curated is
# available. Unknown tags are not banned — they just lose the race for the
# first two slots.
_TAG_PRIORITY = {
    # movement/energy: the classic «музыка для бега»-family queries
    "workout": 3,
    "running": 2, "electronic": 2, "drum and bass": 2, "post-rock": 2,
    "gym": 2, "training": 2, "sport": 2, "sports": 2, "dance": 2,
    "techno": 2, "house": 2, "trance": 2, "dnb": 2, "dubstep": 2,
    "industrial": 2, "synthwave": 2, "shoegaze": 2,
    # states and genres that map to a real Last.fm tag page
    "instrumental": 1, "epic": 1, "sad": 1, "happy": 1, "chill": 1,
    "relax": 1, "relaxation": 1, "sleep": 1, "focus": 1, "party": 1,
    "ambient": 1, "rock": 1, "pop": 1, "jazz": 1, "classical": 1,
    "metal": 1, "punk": 1, "indie": 1, "alternative": 1, "hip hop": 1,
    "rap": 1, "r&b": 1, "soul": 1, "funk": 1, "reggae": 1, "country": 1,
    "folk": 1, "blues": 1, "soundtrack": 1, "dark": 1, "calm": 1,
    "energetic": 1, "motivational": 1, "uplifting": 1, "melancholy": 1,
    "melancholic": 1, "nostalgic": 1, "romantic": 1,
}

# The diversity guard itself lives in services.pipelines._diversity (the
# shared helper, also called by the similar pipeline): 3+ same-ARTIST
# candidates among the top-6 trigger it; each crowded artist keeps only
# the best 2 (the filter's raw ordering — the list is already sorted
# non-noise/views first).

# Synonym fallback chains (audit m1): tag.gettoptracks INTERMITTENTLY returns
# [] for a good tag — rate limit, an empty chart for a tag variant, a network
# blip — and an empty pool used to silently no-op discovery, leaving pure LLM
# guesses (Hilary Duff topping «музыка для бега»). Each chain leads with its
# own key and walks toward broader tags that carry the same energy; the first
# non-empty pool wins. At most _MAX_TAG_FETCHES fetches per tag, sequential.
_TAG_SYNONYMS = {
    # movement/energy: the «музыка для бега» family
    "running": ["running", "workout", "electronic", "drum and bass"],
    "workout": ["workout", "training", "electronic", "dance"],
    "gym": ["gym", "workout", "electronic"],
    "training": ["training", "workout", "electronic"],
    "sport": ["sport", "workout", "electronic"],
    "sports": ["sports", "workout", "electronic"],
    "energetic": ["energetic", "workout", "electronic"],
    "motivational": ["motivational", "uplifting", "energetic"],
    "dance": ["dance", "electronic", "house"],
    "electronic": ["electronic", "techno", "house"],
    "techno": ["techno", "electronic", "house"],
    "house": ["house", "electronic", "dance"],
    "trance": ["trance", "electronic", "dance"],
    "drum and bass": ["drum and bass", "electronic", "techno"],
    "dnb": ["dnb", "drum and bass", "electronic"],
    "dubstep": ["dubstep", "electronic", "drum and bass"],
    "industrial": ["industrial", "electronic", "rock"],
    "synthwave": ["synthwave", "electronic", "ambient"],
    # states and moods
    "sad": ["sad", "melancholy", "acoustic"],
    "melancholy": ["melancholy", "sad", "acoustic"],
    "melancholic": ["melancholic", "melancholy", "acoustic"],
    "happy": ["happy", "uplifting", "pop"],
    "uplifting": ["uplifting", "happy", "dance"],
    "epic": ["epic", "post-rock", "instrumental"],
    "post-rock": ["post-rock", "instrumental", "ambient"],
    "concentration": ["concentration", "ambient", "instrumental"],
    "focus": ["focus", "concentration", "instrumental"],
    "chill": ["chill", "relax", "ambient"],
    "relax": ["relax", "chill", "ambient"],
    "relaxation": ["relaxation", "relax", "ambient"],
    "calm": ["calm", "ambient", "relax"],
    "sleep": ["sleep", "ambient", "calm"],
    "dark": ["dark", "ambient", "rock"],
    "nostalgic": ["nostalgic", "melancholy", "indie"],
    "romantic": ["romantic", "soul", "pop"],
    # genre hubs that anchor many chains
    "instrumental": ["instrumental", "ambient", "soundtrack"],
    "ambient": ["ambient", "instrumental", "chill"],
    "soundtrack": ["soundtrack", "instrumental", "ambient"],
    "rock": ["rock", "alternative", "indie"],
    "pop": ["pop", "dance", "rock"],
}

# One tag costs at most this many Last.fm fetches: the primary plus up to two
# synonym fallbacks, sequential — each call is ~100-500ms, so the worst case
# stays in the same ballpark as the old fixed two-fetch budget per search.
_MAX_TAG_FETCHES = 3

# Junk guard (audit m1): the LLM sometimes emits RU tags that map to no
# Last.fm tag page. A plausible tag is ASCII lowercase word(s) with the usual
# separators ("r&b", "drum and bass", "post-rock", "80s"); a tag containing
# Cyrillic is junk by definition (the extract prompt demands English tags).
_TAG_SHAPE = re.compile(r"[a-z0-9]+(?:[ &'-]+[a-z0-9]+)*")

# The observed junk, kept as an explicit blocklist. The ASCII shape already
# rejects all of them; the list documents the cases that motivated the guard
# and stays defensive if the guard is ever reused on raw (non-lowercased)
# input.
_JUNK_TAGS = frozenset({"бег", "катка", "дота", "грустная"})

# Memory of delivered (B2, live 2026-11-05: «бег» then «музыка для бега»
# re-delivered the same track): look back this many of the user's
# queries.best_title values (store.get_recent_best_titles, the same helper
# and lookback the surprise pipeline uses). A candidate whose cleaned title
# matches one of them never delivers again, and neither does a candidate of
# a recently delivered ARTIST — that second cut is what kills a repeat
# winner across synonymous queries even when both queries build
# byte-identical pools.
_DELIVERED_LOOKBACK = 20

# Pop-pollution guard (B2): the size of the tie-break penalty a POOL-source
# candidate carries in the pre-rank pick order when an LLM-source candidate
# exists in the same views band. The pick order carries no numeric scores
# (the only scores are LLM#2's, assigned downstream in the shared rank tail
# this pipeline must not touch), so the -0.05 materializes as a demotion
# below the same-band LLM candidates — a tie-break, never a cross-band move.
_POOL_SOURCE_PENALTY = 0.05

# Mix demote (q275, live 2026-11-06): the score penalty a mix-titled candidate
# carries on the FINAL SearchOut (after the shared rank tail — unlike the pop
# guard, which runs pre-rank on the pick order). A uniform -0.30 never empties
# the pool: demoting every candidate is order-neutral (stable sort over the
# same relative scores), so the list always survives.
_MIX_DEMOTE = 0.30

# Mix-title markers (compiled locally, case-insensitive; ONE match demotes):
# the word "mix(es)" (a real "Remix" has no \b before "mix" and survives),
# "megamix" (one word — \bmix\b misses it), "non-stop"/"nonstop", "1 hour",
# the one-word "gymmix"/"workoutmix" shapes, and the compilation credit
# "DJ X, Y, Z - ..." (3+ comma-separated artists — 2+ comma groups — before
# the title dash, within the first 80 chars; a plain "A, B - ..." two-artist
# credit is NOT a mix). Deliberately narrow: '\bset\b' is too wide (a real
# track can be called "Set") and is NOT used.
_MIX_MARKERS = (
    re.compile(r"\bmix(?:es)?\b", re.I),
    re.compile(r"megamix", re.I),
    re.compile(r"non.?stop", re.I),
    re.compile(r"\b1 ?hour\b", re.I),
    re.compile(r"\bgym ?mix\b", re.I),
    re.compile(r"workout ?mix", re.I),
    # wave-3b w7: playlist-channel compilations without the word "mix"
    # («THE GAUNTLET 🔥 ... | Heavy Workout Motivation 2026», «Gym Music -
    # Trap Music Video»). Phrase markers only — bare "playlist"/"motivation"
    # would hit real songs (Normani "Motivation", tracks named "Playlist").
    re.compile(r"workout ?motivation", re.I),
    re.compile(r"\bgym ?music\b", re.I),
    re.compile(r"^[^-]{0,80}(?:,[^-]*){2,}- ", re.I),
)


def _plausible_tags(tags: list[str]) -> list[str]:
    """Drop tags that cannot be real Last.fm tags (junk guard): only ASCII
    lowercase word(s) outside the junk blocklist survive — and only the
    survivors are ever fetched."""
    return [t for t in tags
            if _TAG_SHAPE.fullmatch(t) and t not in _JUNK_TAGS]


def _pool_queries(tracks: list[dict]) -> list[str]:
    """"{artist} - {track}" query strings for the tracks that have both
    fields (blank artist/track entries never become queries)."""
    return [f"{m['artist']} - {m['track']}" for m in tracks
            if (m.get("artist") or "").strip() and (m.get("track") or "").strip()]


async def _fetch_tag_pool(metadata_svc, tag: str, seen: set[str]) -> list[str]:
    """The query pool for `tag`, walking its synonym chain while fetches come
    back EMPTY (or raise): an intermittent tag.gettoptracks [] must not no-op
    discovery for a good tag. At most _MAX_TAG_FETCHES fetches; `seen` skips
    candidates the other tag's chain already tried (both chains share hubs
    like "electronic"). Returns [] when the whole chain is empty."""
    for i, candidate in enumerate(dict.fromkeys([tag] + _TAG_SYNONYMS.get(tag, []))):
        if i >= _MAX_TAG_FETCHES:
            break
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            tracks = await metadata_svc.tag_top_tracks(candidate, limit=_TAG_POOL_LIMIT)
        except Exception:
            tracks = []  # rate-limit/network blip: same treatment as an empty pool
        queries = _pool_queries(tracks or [])
        if queries:
            if candidate != tag:
                _log.info("mood tags: pool empty for %r, fallback %r", tag, candidate)
            return queries
    return []


def _mood_tags(extract: dict) -> list[str]:
    """Deduped, lowercased mood_tags from the extract (order preserved)."""
    tags = []
    for t in (extract.get("mood_tags") or []):
        t = str(t).strip().lower()
        if t and t not in tags:
            tags.append(t)
    return tags


def _curate_tags(tags: list[str]) -> list[str]:
    """The discovery tags when LLM#1 returned more than _MAX_MOOD_TAGS:
    stable-sort by _TAG_PRIORITY (descending) and keep the top slots. The
    stable sort preserves the LLM's own order among equals, so all-unknown
    tag lists keep the first two exactly as before the curation existed."""
    if len(tags) <= _MAX_MOOD_TAGS:
        return list(tags)
    ranked = sorted(tags, key=lambda t: -_TAG_PRIORITY.get(t, 0))
    return ranked[:_MAX_MOOD_TAGS]


def _interleave(pools: list[list[str]], cap: int) -> list[str]:
    """Round-robin merge of the tag pools: element 0 of every pool, then
    element 1 of every pool... A short pool runs out without blocking the
    longer one; the merged list caps at `cap`."""
    out = []
    for i in range(max((len(p) for p in pools), default=0)):
        for p in pools:
            if i < len(p):
                out.append(p[i])
                if len(out) >= cap:
                    return out
    return out


def _dedupe_queries(queries: list[str]) -> list[str]:
    """Seed-query title echo (live F3, 2026-11-05): pool entries that are
    the SAME SONG under different spellings or orderings — 'Hilary Duff -
    Come Clean' in one tag pool, a reversed 'Come Clean - Hilary Duff' in
    another (Last.fm tag charts overlap) — collapse to the first
    occurrence by normalized_title_key, so discovery does not echo one
    record as two queries (and the tag-pool seed cannot come back next to
    its own official upload). Runs after the interleave cap: a collapsed
    duplicate simply frees its empirical slot."""
    seen: set[str] = set()
    out = []
    for q in queries:
        key = normalized_title_key(q)
        if not key or key not in seen:
            if key:
                seen.add(key)
            out.append(q)
    return out


def _equal_queries(llm_queries: list[str], pool_queries: list[str],
                   cap: int) -> list[str]:
    """Equal queues (B2): the LLM#1 guesses and the empirical pool tracks
    INTERLEAVE — llm guess, pool track, llm guess, pool track... — instead
    of pools-first, so both sources feed the ranker as equals (the
    Amazon-RecSys selector pattern: the LLM wins as a list-wise SELECTOR
    over pooled candidates, and neither source monopolizes the head slots
    that dominate the search pool). The LLM list leads the alternation —
    the live traces showed the LLM guesses were the better answers for
    pop-polluted workout/running charts. The whole list trims to `cap`;
    same-record entries collapse ACROSS sources (an LLM guess equal to a
    pool track is one query, not two — _dedupe_queries over the merged
    list)."""
    return _dedupe_queries(_interleave([llm_queries, pool_queries], cap))


# --- pop-pollution guard (B2): pool provenance + the views-band tie-break ---

def _pool_record_keys(pool_queries: list[str]) -> list[frozenset]:
    """Record-identity token sets of the empirical pool queries (the same
    cleaned-title sets the diversity guard groups by): the classification
    keys for the pool side of the pop guard."""
    from services import fuzzy
    return [frozenset(fuzzy.clean_title(q).split()) for q in pool_queries]


def _is_pool_record(title: str, pool_keys: list[frozenset]) -> bool:
    """True when `title` is the same RECORD as one of the pool queries:
    equal cleaned token sets, or a >= 2-token subset either way (the same
    relation the diversity guard uses for same-record grouping — 'come
    clean' is 'hilary duff come clean' minus the artist credit, and a
    decorated official upload cleans to the same set). search_metadata
    returns ONE flat merged list with no per-query grouping (the merge in
    services/search.py dedups and interleaves as queries complete), so a
    candidate's provenance is classified on the candidate side, by record
    identity with the pool queries that seeded it."""
    from services import fuzzy
    toks = frozenset(fuzzy.clean_title(title or "").split())
    if not toks:
        return False
    for pk in pool_keys:
        if not pk:
            continue
        if (toks == pk or (len(toks) >= 2 and toks < pk)
                or (len(pk) >= 2 and pk < toks)):
            return True
    return False


def _views_band(views) -> int:
    """Order-of-magnitude views band (0 for unknown/absent): the cheap
    deterministic 'same popularity' scale behind the pop guard's
    tie-break. Monotone in views, so band-descending never contradicts the
    filter's views-descending order."""
    v = views or 0
    return len(str(int(v))) if v > 0 else 0


def _pool_prior_demote(filtered: list, enriched: list,
                       pool_keys: list[frozenset]) -> tuple[list, list]:
    """Pop-pollution guard (B2), candidate side: a POOL-source candidate
    carries the -0.05 prior and sorts BELOW every LLM-source candidate of
    the SAME views band (a tie-break prefer-LLM: same band = a views tie
    for ordering purposes, so the penalty only ever demotes within a
    band). Cross-band order, noise order and the views order inside a band
    are preserved — the sort key inserts only the band and the penalty
    between the filter's existing components, so nothing moves unless a
    same-band LLM-source candidate exists to prefer. The demoted order is
    the pick order every deterministic consumer sees (the LLM#2 rank input
    window, the no-rank fallback, the diversity guard's keep-best);
    LLM#2's own scores stay untouched. `enriched` (index-aligned with
    `filtered`) is reordered by the same permutation."""
    if not pool_keys or not filtered:
        return filtered, enriched
    pool_flags = [_is_pool_record(c.get("title") or "", pool_keys)
                  for c in filtered]
    if not any(pool_flags):
        return filtered, enriched
    llm_bands = {_views_band(c.get("views"))
                 for c, p in zip(filtered, pool_flags) if not p}

    def _key(i: int) -> tuple:
        c = filtered[i]
        penalty = (_POOL_SOURCE_PENALTY
                   if pool_flags[i] and _views_band(c.get("views")) in llm_bands
                   else 0.0)
        # The filter's own order (is_noise, -_vwanted, -views) with the
        # band and the penalty inserted: band-descending is consistent
        # with views-descending, so only the penalty can reorder.
        return (bool(c.get("is_noise")), -int(c.get("_vwanted", 0) or 0),
                -_views_band(c.get("views")), penalty,
                -(int(c.get("views") or 0)))

    order = sorted(range(len(filtered)), key=_key)
    if order != list(range(len(filtered))):
        demoted = sum(1 for i in range(len(filtered)) if _key(i)[3])
        _log.info("mood pop guard: demoted %d pool-source candidate(s) below "
                  "same-views-band llm candidate(s)", demoted)
    out_f = [filtered[i] for i in order]
    out_e = ([enriched[i] for i in order]
             if len(enriched) == len(filtered) else enriched)
    return out_f, out_e


# --- memory of delivered (B2): the two novelty cuts ---

def _novelty_cut(filtered: list, enriched: list, keep,
                 noun: str) -> tuple[list, list]:
    """One novelty cut: keep the candidates passing `keep`, drop the rest
    from `enriched` by video_id too, log the drop count. Honest fallback
    (the surprise pattern): when the cut would drop EVERY candidate, the
    pre-cut pool delivers anyway — an answer beats silence, and the log
    says the pool ran dry."""
    fresh = [c for c in filtered if keep(c)]
    if len(fresh) != len(filtered):
        _log.info("mood novelty: dropped %d recently delivered %s (%d left)",
                  len(filtered) - len(fresh), noun, len(fresh))
    if not fresh:
        _log.info("mood novelty: every candidate is recently delivered %s, "
                  "delivering the best anyway", noun)
        return filtered, enriched
    keep_ids = {c["video_id"] for c in fresh}
    return fresh, [c for c in enriched if c["video_id"] in keep_ids]


def _delivered_novelty(filtered: list, enriched: list,
                       delivered: list[str]) -> tuple[list, list]:
    """Memory of delivered (B2) over the filtered pool, two deterministic
    cuts:

      1. TITLE: a candidate whose fuzzy.clean_title equals the cleaned
         title of a recently delivered best_title drops — the same song
         (any spelling/upload) never delivers twice;
      2. ARTIST: a candidate whose canonical artist key
         (_diversity.artist_key: the validated 'Artist - Track' title
         split, channel minus '- Topic' as the fallback) equals the
         artist key of a delivered title drops — «Hilary Duff дважды»
         dies here even across synonymous queries with byte-identical
         pools, because the follow-up track by the same act cannot
         re-deliver either.

    Each cut logs its own line and falls back honestly when it would empty
    the pool."""
    from services import fuzzy

    title_keys = {k for k in (fuzzy.clean_title(t) for t in delivered) if k}
    if title_keys:
        filtered, enriched = _novelty_cut(
            filtered, enriched,
            lambda c: fuzzy.clean_title(c.get("title") or "") not in title_keys,
            "track(s)")

    artist_keys = {k for k in (artist_key({"title": t, "channel": ""})
                               for t in delivered) if k[1]}
    if artist_keys:
        filtered, enriched = _novelty_cut(
            filtered, enriched,
            lambda c: artist_key(c) not in artist_keys,
            "artist(s)")

    return filtered, enriched


# --- mix handling (q275 + wave2 w7): mixes never beat real tracks, and
# with a rich enough pool they leave the list entirely ---

_MIX_MIN_CLEAN = 5  # real tracks needed before mixes are dropped outright


def _is_mix_title(title: str) -> bool:
    """True when the title names itself a mix/compilation: one marker match
    is enough. Real tracks survive — "Remix" has no word boundary before
    "mix", and no marker touches a plain "Artist - Track" title."""
    return any(rx.search(title or "") for rx in _MIX_MARKERS)


# Query hygiene (wave2 w7): LLM#1 guesses like "high bpm running music mix"
# ASK YouTube for mixes — the junk enters through the query, not only the
# results. Stripped before discovery; a query left without meaning drops
# out (the tag pools and the remaining guesses carry the search).
# wave-3b w7: playlist words join the hygiene — but only as standalone
# playlist/compilation or PAIRED with an activity word before
# motivation/playlist: a bare "motivation" is a real track title (Normani -
# Motivation) and "running music" is a useful discovery phrase — both must
# survive (the compilations they fetch are caught by the TITLE markers).
_MIX_QUERY_RE = re.compile(
    r"\b(mix(es)?|megamix|mixtape|non[\s-]?stop|1[\s-]?hour|"
    r"playlists?|compilations?|"
    r"(?:workout|gym|training|running)\s+(?:motivation|playlist))\b",
    re.IGNORECASE)


def _strip_mix_from_queries(queries: list) -> list:
    """Remove mix-words from every discovery query; never returns an empty
    list (on total wipeout the original queries stand — a search for mixes
    still beats no search at all)."""
    cleaned, dropped = [], 0
    for q in queries or []:
        nq = re.sub(r"\s{2,}", " ", _MIX_QUERY_RE.sub("", q or "").strip(" -,"))
        if nq and nq.lower() not in {"music", "музыка", "song", "songs",
                                     "playlist", "music 2024", "музыка 2024"}:
            cleaned.append(nq)
        else:
            dropped += 1
    if dropped and cleaned:
        _log.info("mood query hygiene: %d query(ies) cleaned/dropped of mix words",
                  dropped)
    return cleaned or list(queries or [])


def _mix_demote(out):
    """Mix handling, the post-rank tail (q275 live + wave2 w7):

    Tier 1 (DROP): with at least _MIX_MIN_CLEAN real tracks in the final
    list, mix-titled candidates are REMOVED — wave2's w7 card showed a
    demoted mix still occupying a visible top-5 slot, and a demoted mix
    remains a dead pick in the miss chain. best re-points by video_id; a
    dropped mix-best hands the delivery slot to the score-top real track.

    Tier 2 (DEMOTE, the scarce-pool fallback): fewer than 5 real tracks —
    every mix-titled candidate carries -0.30 on its LLM#2 score, the list
    re-sorts score-descending (STABLE — equal scores keep the rank tail's
    order), best_index re-points by video_id (the same-record collapse
    reindex pattern in services/llm_search.py). Demoting EVERY candidate is
    order-neutral (a uniform -0.30 preserves relative scores), so the pool
    never empties. No mix titles -> byte-identical passthrough."""
    if not out or not out.candidates:
        return out
    flags = [_is_mix_title(c.title or "") for c in out.candidates]
    if not any(flags):
        return out
    best_idx = out.best_index if 0 <= out.best_index < len(out.candidates) else -1
    best_vid = out.candidates[best_idx].video_id if best_idx >= 0 else ""
    best_was_mix = best_idx >= 0 and flags[best_idx]
    if sum(1 for f in flags if not f) >= _MIX_MIN_CLEAN:
        out.candidates = [c for c, f in zip(out.candidates, flags) if not f]
        if best_was_mix or not best_vid:
            out.best_index = 0
        else:
            out.best_index = next(
                (i for i, c in enumerate(out.candidates) if c.video_id == best_vid), 0)
        _log.info("mood mix drop: removed %d mix candidate(s), %d real track(s) kept",
                  sum(flags), len(out.candidates))
        return out
    for c, is_mix in zip(out.candidates, flags):
        if is_mix:
            c.score -= _MIX_DEMOTE
    out.candidates.sort(key=lambda c: c.score, reverse=True)
    if best_vid:
        new_idx = next((i for i, c in enumerate(out.candidates)
                        if c.video_id == best_vid), -1)
        if best_was_mix and new_idx != best_idx:
            # the demote pushed the mix-best below a real track: the
            # score-top (a real track whenever one exists) wins the slot
            out.best_index = 0
        else:
            out.best_index = new_idx if new_idx >= 0 else out.best_index
    _log.info("mood mix demote: %d candidate(s) -0.30", sum(flags))
    return out


class MoodPipeline(SearchPipeline):
    """Stage-2b pipeline for intent="mood": Last.fm tag pools and LLM guesses
    feed discovery as EQUAL QUEUES (interleaved); memory of delivered and a
    pop guard run before a deterministic diversity guard caps one artist."""

    name = "mood"

    async def run(self, ctx: PipelineContext) -> "SearchOut | None":
        """Execute the mood tail: tag pools and LLM guesses as EQUAL QUEUES
        (interleaved, LLM guess first), YouTube search, filter/enrich,
        memory of delivered (title + artist cuts), pop guard (pool-source
        tie-break demotion), diversity guard, the shared LLM#2 rank tail
        (SYSTEM_RANK_MOOD via rank_prompt_for), then the mix demote on the
        final SearchOut (q275: compilation mixes -0.30, re-sorted,
        best_index re-pointed by video_id)."""
        # Lazy per-call imports: services.llm_search imports this package at
        # module level (cycle), and per-call resolution honors test
        # monkeypatching of the services.llm_search namespace.
        from services.llm_search import (
            _CLARIFY_DEFAULT,
            _CLARIFY_QUESTIONS,
            _MAX_QUERIES,
            SearchOut,
        )

        trace = ctx.trace
        t_start = ctx.t_start

        f = _fields_from_extract(ctx.extract)
        queries = list(f.queries)

        # --- empirical pools from Last.fm community tags ---
        tags = _mood_tags(ctx.extract)
        plausible = _plausible_tags(tags)
        if plausible != tags:
            _log.info("mood tags: dropped junk %s (not plausible Last.fm tags)",
                      [t for t in tags if t not in plausible])
            tags = plausible
        curated = _curate_tags(tags)
        if curated != tags:
            _log.info("mood tags: curated %s -> %s (priority list, junk tags lose)",
                      tags, curated)
        tags = curated
        pools: list[list[str]] = []
        tried_tags = False
        if ctx.metadata_svc is not None and tags:
            tried_tags = True
            seen: set[str] = set()
            for tag in tags[:_MAX_MOOD_TAGS]:
                pools.append(await _fetch_tag_pool(ctx.metadata_svc, tag, seen))
        empirical = _dedupe_queries(_interleave(pools, _MAX_EMPIRICAL))
        if empirical:
            _log.info("mood discovery: %d empirical queries from tags %s, %d llm-queries queued (before cap)",
                      len(empirical), tags[:_MAX_MOOD_TAGS], len(queries))
        elif tried_tags and queries:
            # Empirical discovery degraded to nothing even after the synonym
            # chains: proceed on the LLM guesses as before, but say so — the
            # silent no-op was the m1 audit finding (a).
            _log.warning("mood discovery: all tag pools empty (tags=%r) — LLM guesses only",
                         tags[:_MAX_MOOD_TAGS])

        # Defensive double-check of the head's clarify-early (queries empty
        # here is already caught by _resolve_queries before routing): with
        # BOTH tag pools empty AND no LLM queries there is nothing to search.
        if not empirical and not queries:
            _log.info("mood: no tag pools and no llm queries -> clarify")
            if trace:
                trace.finish(output={"decision": "clarify-early", "intent": f.intent})
            return SearchOut(intent=f.intent, signals=f.signals, queries=[], assoc=f.assoc,
                             reference=f.reference, target=f.target, candidates=[],
                             best_index=-1, confidence="low", intro=f.intro,
                             need_clarify=True,
                             clarify_question=_CLARIFY_QUESTIONS.get(f.intent, _CLARIFY_DEFAULT),
                             trace=trace)

        # Equal queues (B2): llm guess, pool track, llm guess, ... — both
        # sources feed the ranker as equals; the whole list trims to
        # _MAX_QUERIES and same-record entries collapse across sources.
        queries = _equal_queries(queries, empirical, _MAX_QUERIES)

        # Query hygiene (wave2 w7): "high bpm running music mix" ASKS for a
        # mix — strip the mix-words before they ever reach YouTube.
        queries = _strip_mix_from_queries(queries)

        # step 3: youtube
        raw = await _search_youtube(queries, on_progress=ctx.on_progress, trace=trace)
        t_search = time.perf_counter() - t_start
        if not raw:
            return _no_results_out(f, queries, trace)

        # steps 4-4.5: filter + enrich
        filtered, enriched = await _filter_and_enrich(
            raw, ctx.excluded_ids, ctx.excluded_artists,
            metadata_svc=ctx.metadata_svc, on_progress=ctx.on_progress, trace=trace)
        t_filter = time.perf_counter() - t_start

        # --- memory of delivered (B2): the recent best_titles cut both the
        # delivered songs and the delivered ARTISTS from the pool, so a
        # synonymous repeat («бег» -> «музыка для бега», byte-identical
        # pools) cannot re-deliver the previous winner or its artist.
        # user_id None (no personal history) or a store without the helper
        # skips the cuts entirely. ---
        if ctx.user_id:
            store = ctx.store if ctx.store is not None else _default_store
            delivered_fn = getattr(store, "get_recent_best_titles", None)
            if delivered_fn is not None:
                try:
                    delivered = await delivered_fn(
                        ctx.user_id, limit=_DELIVERED_LOOKBACK)
                except Exception:
                    delivered = []
                if delivered:
                    filtered, enriched = _delivered_novelty(
                        filtered, enriched, delivered)

        # --- pop-pollution guard (B2): pool-source candidates tie-break
        # below same-views-band LLM-source candidates (deterministic,
        # pre-rank, cheap). ---
        filtered, enriched = _pool_prior_demote(
            filtered, enriched, _pool_record_keys(empirical))

        # --- deterministic diversity guard (before ranking) ---
        filtered, enriched = self.diversity_guard(filtered, enriched)

        # steps 5+: LLM#2 rank (SYSTEM_RANK_MOOD via rank_prompt_for),
        # scoring, SearchOut assembly. No reference to resolve for mood.
        ref_resolved = {"artist": "", "track": "", "source": "llm", "neighbors": []}
        out = await _rank_and_decide(ctx, f, queries, filtered, enriched, ref_resolved,
                                     len(raw), t_search, t_filter)
        # --- mix demote (q275): compilation mixes tie-break below real
        # tracks on the FINAL SearchOut (post-rank, like every other mood
        # guard this one owns; legacy.py stays untouched). ---
        return _mix_demote(out)

    @staticmethod
    def diversity_guard(filtered: list, enriched: list) -> tuple[list, list]:
        """The «не три трека одного исполнителя» rule (RU: "no three tracks
        by one artist") plus same-record collapse, enforced in code -
        the shared artist_diversity_guard
        (services.pipelines._diversity, also called by the similar
        pipeline) under mood's log label. 3+ copies of the same RECORD in
        the top-6 (the same song under any title ordering/channel — live
        F3) keep only the best 1; 3+ with the same canonical artist (the
        (kind, token-set) from artist_key: validated "Artist - Track"
        title split, channel minus "- Topic" as the fallback, a title-hash
        bucket as the last resort) keep only the best 2 (the filter's raw
        ordering — non-noise by views first) and the rest drop from the
        whole pool, `enriched` included. Returns the (possibly unchanged)
        lists; one log line per pass when a drop happened."""
        return artist_diversity_guard(filtered, enriched, label="mood")
