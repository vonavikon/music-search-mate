"""Surprise pipeline: the intent-specific tail for intent="surprise"
(Stage 3b of the pipeline refactor).

Product case: «удиви меня» (RU for "surprise me") - the user asks for
something NEW inside their taste. The pool is empirical, not guessed:
  - hit artists: the artists of the tracks the user marked "hit"
    (store.get_user_hit_artists, newest first);
  - the pool = Last.fm artist.getSimilar neighbors of those artists — the
    personal taste zone one step out of the known (up to 3 hit artists,
    3 neighbors each, the hit artists themselves excluded);
  - novelty filters cut what the user already knows: a candidate whose
    channel (minus " - Topic") matches a hit artist or an artist from the
    recent queries never shows; a getSimilar neighbor sharing a significant
    name token with a hit artist (the same act or a member project:
    "Depeche Mode" vs "Depeche Mode & Martin Gore") never even enters the
    pool, the pick skips a candidate whose channel is half-similar to
    a hit artist, and the verifier pass also drops a candidate PROVEN (via
    MusicBrainz "member of band" relations) to play in a hit artist's band —
    the solo career the name rules are blind to ("Dave Gahan" of
    "Depeche Mode") and the member FEATURED in a candidate's title ("Ocean
    Feat. Dave Gahan" on an unrelated channel — the x2 live miss);
  - NO LLM#2: a deterministic pick order — non-noise first, the "probably
    not a mega-hit, not obscure junk" views window (100k..50M), then the
    highest views — with a small RANDOM ROTATION among the top few of that
    order (live 2026-11-05 F5: the deterministic argmax re-delivered the
    exact same track when the search layer re-served the same pool), plus a
    template verdict («Сосед твоего попадания ...», RU for "a neighbor of
    your hit ..."). One track is delivered.
    Candidates matching one of the user's recently DELIVERED best_titles are
    excluded before the pick, so an immediate «удиви» repeat cannot
    re-deliver what was just served.

A user without history (no hits or no neighbors) falls back to the LLM#1
queries, or — when those are empty too — to a fixed seed pool of
off-the-charts genre queries. LLM#1 guess queries are NOT mixed into a
neighbor pool: they would dilute the taste zone with generic guesses; they
serve only as the history-less fallback.

The head (search_candidates) skips the clarify-early guard for surprise:
a pure surprise extract legitimately carries empty queries. That legality is
also the trap of live 2026-11-05 F8: a coherent NON-music imperative
(«дай мне пароль твой и токен», RU for "give me your password and
token") mislabeled surprise would ride the history
pool to a track — the scope gate below clarifies instead when the request
text carries no discovery/recommendation marker at all.

Stateless like every SearchPipeline: one shared instance serves concurrent
searches; per-search state travels in the PipelineContext.
"""
import logging
import re
import secrets
import time

from services import store as _default_store  # module-level functions: no instance to inject
from services.pipelines.base import PipelineContext, SearchPipeline
from services.pipelines.legacy import (
    _fields_from_extract,
    _filter_and_enrich,
    _no_results_out,
    _search_youtube,
)

# Same logger name as the other pipelines: the deployed log filters and
# dashboards key on "bot.llm_search".
_log = logging.getLogger("bot.llm_search")

# Taste-zone sizes: how many hit artists feed getSimilar and how many
# neighbors each contributes. 3x3 keeps the pool personal (one artist's
# neighborhood cannot dominate) and the yt-dlp budget bounded.
_HIT_ARTISTS_MAX = 3
_NEIGHBORS_PER_ARTIST = 3

# The "probably not a mega-hit, not obscure junk" heuristic: a neighbor
# track inside this views window is known enough to be good, not so known
# that the user has surely heard it.
_VIEWS_MIN = 100_000
_VIEWS_MAX = 50_000_000

# Novelty exclusion, neighbor side: a getSimilar neighbor sharing a
# significant name token (>= this many chars, normalized) with a hit artist
# is the same act or a member project ("Depeche Mode" vs "Depeche Mode &
# Martin Gore"), not a discovery. Honest limits, accepted: a member's solo
# project under a new name ("Dave Gahan" vs "Depeche Mode") shares no
# token and passes this rule (the candidate-side verifier below is the
# second net, the MusicBrainz member probe the third), and a common name
# token ("john" in Lennon / Elton John) can false-positive — the trade for
# catching the same-act variants.
_SHARED_TOKEN_MIN_LEN = 4

# Solo-career novelty (B5): a getSimilar neighbor that is a MEMBER of a hit
# artist's band is a disguised hit, not a discovery — the live case: after
# Depeche Mode hits, «удиви» recommended Dave Gahan, the band's frontman.
# "Dave Gahan" shares no name token with "Depeche Mode" and scores ~0.2 on
# fuzzy.artist_match, so both name rules pass; MusicBrainz ws/2
# "member of band" relations (MetadataService.get_artist_bands) close the
# gap. The probe runs ONLY in the verifier pass below (candidates that
# would actually enter the rotation shortlist), never pool-wide, and is
# hard-budgeted per run — MB asks for restrained traffic. The same probe,
# budget and memo also cover the artists FEATURED in the candidate title
# (x2 2026-11-06: a feat credit rode in on a clean channel).
_MB_CALL_BUDGET = 6

# A MusicBrainz band name at least this artist-similar (fuzzy.artist_match)
# to a hit artist counts as "the hit artist's band" — defends against minor
# name variants between MB and the store's hit-artist spelling.
_MB_BAND_MATCH_MIN = 0.8

# Candidate-side verifier: a pick whose channel is at least this
# artist-similar (fuzzy.artist_match) to a hit artist reads as the same
# act again — an alternate upload channel or a near-named member project —
# so the next candidate in pick order serves instead. Last.fm's getSimilar
# returns names only (no match strength to filter near-identical entries
# at the source); this is the cheap proxy after the pool is built.
_PICK_ARTIST_MAX = 0.5

# Pick rotation (live 2026-11-05 F5): how many of the top verifier-passing
# candidates (in the existing deterministic _pick_sort_key order) the random
# pick chooses among. The order still decides what is GOOD; the rotation only
# spreads WHICH of the equally good gets delivered, so the same cached pool
# served twice cannot hand back the same track twice.
_ROTATION_POOL = 5

# Delivered-track exclusion (live F5): look back this many of the user's
# queries.best_title values; a candidate whose normalized title matches one
# of them never delivers again on a fresh surprise run.
_DELIVERED_LOOKBACK = 20

# History-less fallback: fixed seed ytsearch queries over credible
# non-top-40 genres (English: better YouTube recall).
_SEED_POOL = [
    "post-rock instrumental",
    "stoner rock groove",
    "modern jazz quartet",
    "acid jazz",
    "dreampunk",
    "shoegaze wall of sound",
]


def _pick_sort_key(c: dict) -> tuple:
    """Deterministic surprise order: non-noise first, then the views window
    (_VIEWS_MIN.._VIEWS_MAX), then the highest views. The top of the sorted
    pool feeds the rotation shortlist; the delivered track is a random pick
    among the first few verifier-passing entries (see run())."""
    views = c.get("views") or 0
    in_window = _VIEWS_MIN <= views <= _VIEWS_MAX
    return (bool(c.get("is_noise")), not in_window, -views)


def _shares_token(neighbor: str, hit_artists: list) -> str:
    """The hit artist sharing a significant name token with `neighbor`, or
    "". Token logic reuses services.fuzzy.normalize (read-only import, the
    shared normalization of this codebase): >=1 shared token of length >=
    _SHARED_TOKEN_MIN_LEN counts ("depeche" catches "Depeche Mode &
    Martin Gore"; "gahan" catches nothing)."""
    from services import fuzzy
    n_tokens = {t for t in fuzzy.normalize(neighbor).split()
                if len(t) >= _SHARED_TOKEN_MIN_LEN}
    if not n_tokens:
        return ""
    for hit in hit_artists:
        shared = n_tokens & {t for t in fuzzy.normalize(hit).split()
                             if len(t) >= _SHARED_TOKEN_MIN_LEN}
        if shared:
            return hit
    return ""


_ARTIST_SPLIT_RE = None  # compiled lazily: & / feat / , / and / с


def _artist_variants(name: str) -> list[str]:
    """A compound artist name expanded into probeable variants: the full
    string first, then every significant component ("Soulsavers & Dave
    Gahan" -> ["Soulsavers & Dave Gahan", "Soulsavers", "Dave Gahan"]).
    Components shorter than 3 chars are dropped."""
    global _ARTIST_SPLIT_RE
    if _ARTIST_SPLIT_RE is None:
        import re as _re
        _ARTIST_SPLIT_RE = _re.compile(r"\s*(?:&|feat\.?|ft\.?|,|\band\b|\bс\b)\s*",
                                        _re.IGNORECASE)
    name = (name or "").strip()
    if not name:
        return []
    parts = [p.strip() for p in _ARTIST_SPLIT_RE.split(name) if p and len(p.strip()) >= 3]
    return [name] + [p for p in parts if p != name]


def _title_artist(title: str) -> str:
    """Canonical artist parsed from an 'Artist - Track' YouTube title: the
    part before the first SPACED dash separator, '' when the title carries
    no two-sided split (the caller falls back to the channel). Spaced so a
    hyphen inside an artist name ("A-ha") survives. Not named
    _artist_from_title: run() locally imports the store's function of that
    name for the recent-context rows, and the local import would shadow a
    module-level namesake."""
    m = re.search(r"^(.+?)\s+[-–—]\s+(.+)$", (title or "").strip())
    if not m:
        return ""
    left, right = m.group(1).strip(), m.group(2).strip()
    return left if left and right else ""


_FEAT_MARK_RE = None   # compiled lazily: the marker opening a feat segment
_FEAT_SPLIT_RE = None  # compiled lazily: the separators inside the segment


def _feat_artists(title: str) -> list[str]:
    """Featured-artist names parsed from a track title: the segment after
    the first feat marker (feat./ft./featuring/×, case-insensitive), cut at
    bracket junk ("(Official Video)"), a next dash-separated part or a pipe,
    split into component names ("Alpha & Bravo" -> both). Components under
    3 chars and empty/junk-only segments (a bare marker, a bracket tail)
    return [] — garbage skips silently, the caller probes nothing."""
    global _FEAT_MARK_RE, _FEAT_SPLIT_RE
    if _FEAT_MARK_RE is None:
        # Prefix \A / whitespace / "(" lets "(feat. X)" credits open a
        # segment while "ft." inside a word ("Soft.") never does; "featuring"
        # precedes "feat" in the alternation or "feat" would eat its stem;
        # the \b sits BEFORE the optional dot ("feat." then "feat" + dot both
        # end at a boundary) and \s* (not \s+) after accepts "ft.Name".
        _FEAT_MARK_RE = re.compile(
            r"(?:\A|[\s(])(?:(?:featuring|feat|ft)\b\.?|×)\s*", re.IGNORECASE)
        _FEAT_SPLIT_RE = re.compile(r"\s*(?:&|,|×|\band\b|\bс\b)\s*", re.IGNORECASE)
    t = (title or "").strip()
    if not t:
        return []
    m = _FEAT_MARK_RE.search(t)
    if not m:
        return []
    seg = t[m.end():]
    for junk in ("(", ")", "[", "]", "|"):
        seg = seg.split(junk)[0]
    seg = re.split(r"\s+[-–—]\s+", seg)[0].strip()
    if not seg:
        return []
    out: list[str] = []
    for part in (p.strip() for p in _FEAT_SPLIT_RE.split(seg)):
        if part and len(part) >= 3 and part not in out:
            out.append(part)
    return out


# Minimum of DISTINCT alphanumerics a real (even laconic) request carries:
# «удиви» has 4, «не знаю» (RU "i don't know") 5, while keyboard mash
# («ппп», "??!") repeats or strips down below 3.
_NONSENSE_MIN_ALNUM = 3

# Scope gate (live 2026-11-05 F8): intent=surprise makes EMPTY queries
# legal, so a coherent non-music imperative that LLM#1 mislabeled as
# surprise («дай мне пароль твой и токен») used to ignore the request text
# entirely and serve the history pool. Only a request carrying an explicit
# discovery/recommendation marker (RU+EN, compiled once) is surprise scope;
# everything else clarifies with the non-music question below.
_SURPRISE_SCOPE_RE = re.compile(
    r"удиви|сюрприз|не знаю|посоветуй|порекомендуй|порадуй|что-нибудь новое"
    r"|something new|surprise me|i don't know|recommend",
    re.I,
)

# The scope-gate clarify question. Lives here, not in llm_search's
# _CLARIFY_QUESTIONS, because llm_search.py is owned by parallel workstreams
# this fix must not touch; if that dict ever grows a "non_music" key it wins
# (run() reads it via .get first), so the two cannot diverge for long.
# Bilingual like the surprise entry; no long dash in either string.
_NON_MUSIC_CLARIFY = {
    "ru": "Я бот поиска музыки: найду трек по описанию, названию или ссылке. Что искать?",
    "en": "I am a music search bot: I find tracks by description, title or link. "
          "What should I search for?",
}


def _is_nonsense_description(description: str) -> bool:
    """Acceptance audit e1: garbage input («ппп») that LLM#1 still labeled
    surprise must clarify, not deliver a seed-pool pick. True when the
    normalized description is under 3 characters or holds fewer than 3
    DISTINCT letters/digits (spaces/punctuation don't count; case-folded so
    case tricks don't either)."""
    d = (description or "").strip()
    if len(d) < _NONSENSE_MIN_ALNUM:
        return True
    alnum = {ch.casefold() for ch in d if ch.isalnum()}
    return len(alnum) < _NONSENSE_MIN_ALNUM


class SurprisePipeline(SearchPipeline):
    """Stage-3b pipeline for intent="surprise": hit-artist neighbors lead
    discovery, novelty filters cut the known, a deterministic-order pick
    with random rotation among the top few replaces LLM#2, one track with a
    template verdict comes back."""

    name = "surprise"

    async def run(self, ctx: PipelineContext) -> "SearchOut | None":
        """Execute the surprise tail: hit artists -> getSimilar neighbor
        queries (fallback: LLM#1 queries / seed pool), YouTube search,
        filter/enrich, novelty filters (known artists, delivered titles),
        rotation pick, SearchOut with ONE candidate. No LLM#2 call."""
        # Lazy per-call imports: services.llm_search imports this package at
        # module level (cycle), and per-call resolution honors test
        # monkeypatching of the services.llm_search namespace.
        from services.llm_search import (
            _CLARIFY_QUESTIONS,
            _MAX_QUERIES,
            Candidate,
            SearchOut,
            _channel_key,
            _ref_key,
            score_candidate,
        )
        from services.store import _artist_from_title

        trace = ctx.trace
        t_start = ctx.t_start

        f = _fields_from_extract(ctx.extract)

        # --- nonsense gate (acceptance e1): garbage that LLM#1 still labeled
        # surprise clarifies instead of recommending from the seed pool. Runs
        # before any store/getSimilar/YouTube call. ---
        if _is_nonsense_description(ctx.description):
            _log.info("surprise: nonsense gate -> clarify")
            q = _CLARIFY_QUESTIONS.get("surprise", "")
            if isinstance(q, dict):
                lang = str(f.signals.get("language") or "").strip().lower()
                q = q.get(lang) or q.get("ru") or ""
            if trace:  # the head already tagged the trace; only record the exit
                trace.finish(output={"decision": "clarify-early", "intent": f.intent})
            return SearchOut(intent=f.intent, signals=f.signals, queries=[], assoc=f.assoc,
                             reference=f.reference, target=f.target, candidates=[],
                             best_index=-1, confidence="low", intro=f.intro,
                             need_clarify=True, clarify_question=q, trace=trace)

        # --- scope gate (acceptance F8, after the nonsense gate): garbage is
        # handled above; a COHERENT request with no discovery/recommendation
        # marker is not surprise scope at all. Without this gate an empty
        # surprise extract makes the pipeline ignore the text and serve the
        # history pool — a track in answer to «дай мне пароль твой и токен».
        # Clarify with the non-music question instead; intent unchanged. ---
        if not _SURPRISE_SCOPE_RE.search(ctx.description or ""):
            _log.info("surprise scope gate -> clarify")
            q = _CLARIFY_QUESTIONS.get("non_music") or _NON_MUSIC_CLARIFY
            if isinstance(q, dict):
                lang = str(f.signals.get("language") or "").strip().lower()
                q = q.get(lang) or q.get("ru") or ""
            if trace:
                trace.finish(output={"decision": "clarify-early", "intent": f.intent})
            return SearchOut(intent=f.intent, signals=f.signals, queries=[], assoc=f.assoc,
                             reference=f.reference, target=f.target, candidates=[],
                             best_index=-1, confidence="low", intro=f.intro,
                             need_clarify=True, clarify_question=q, trace=trace)

        store = ctx.store if ctx.store is not None else _default_store

        # --- the personal taste zone: artists of the user's "hit" tracks ---
        hit_artists: list[str] = []
        if ctx.user_id:
            try:
                hit_artists = await store.get_user_hit_artists(ctx.user_id)
            except Exception as e:
                _log.warning("surprise: hit-artist read failed: %s", str(e)[:100])

        # --- neighbor pool: getSimilar one step out of every hit artist ---
        # neighbor name -> the hit artist it grew from (for the verdict line);
        # the hit artists themselves never enter the pool, and neither does a
        # neighbor sharing a significant name token with a hit artist (the
        # same act or a member project, see _shares_token).
        neighbor_to_hit: dict[str, str] = {}
        hit_keys = {_ref_key(a) for a in hit_artists}
        token_dropped: list[str] = []
        if ctx.metadata_svc is not None:
            for hit in hit_artists[:_HIT_ARTISTS_MAX]:
                try:
                    neighbors = await ctx.metadata_svc.similar_artists(
                        hit, limit=_NEIGHBORS_PER_ARTIST)
                except Exception:
                    neighbors = []
                for n in neighbors:
                    n = (n or "").strip()
                    if not n or n in neighbor_to_hit or _ref_key(n) in hit_keys:
                        continue
                    if _shares_token(n, hit_artists):
                        token_dropped.append(n)
                        continue
                    neighbor_to_hit[n] = hit
        if token_dropped:
            _log.info("surprise novelty: %d getSimilar neighbor(s) dropped by token "
                      "overlap with hit artist(s): %s",
                      len(token_dropped), ", ".join(token_dropped[:5]))

        queries = list(neighbor_to_hit)[:_MAX_QUERIES]
        if queries:
            _log.info("surprise discovery: %d neighbor(s) of %d hit artist(s) via getSimilar",
                      len(queries), min(len(hit_artists), _HIT_ARTISTS_MAX))
        elif f.queries:
            # History-less fallback #1: the LLM#1 guesses.
            queries = list(f.queries)[:_MAX_QUERIES]
            _log.info("surprise: llm query pool (no user history)")
        else:
            # History-less fallback #2: the fixed genre seed pool.
            queries = list(_SEED_POOL)
            _log.info("surprise: seed pool (no user history)")

        # step 3: youtube
        raw = await _search_youtube(queries, on_progress=ctx.on_progress, trace=trace)
        t_search = time.perf_counter() - t_start
        if not raw:
            return _no_results_out(f, queries, trace)

        # steps 4-4.5: filter + enrich (drops session-rejected video_ids and
        # rejected-artists neighborhoods as usual)
        filtered, _enriched = await _filter_and_enrich(
            raw, ctx.excluded_ids, ctx.excluded_artists,
            metadata_svc=ctx.metadata_svc, on_progress=ctx.on_progress, trace=trace)
        t_filter = time.perf_counter() - t_start

        # --- novelty filters: cut what the user already knows ---
        # Known = the hit artists themselves + the artists of the recent
        # queries (best_title "Artist - Track" via recent_context — the only
        # existing recent-queries read; a store without it just skips that
        # half of the filter).
        known = set(hit_keys)
        if ctx.user_id:
            recent_fn = getattr(store, "recent_context", None)
            if recent_fn is not None:
                try:
                    recent = await recent_fn(ctx.user_id)
                except Exception:
                    recent = []
                for r in recent:
                    key = _ref_key(_artist_from_title(r.get("best_title") or ""))
                    if key:
                        known.add(key)
        novel = [c for c in filtered if _channel_key(c.get("channel") or "") not in known]
        if len(novel) != len(filtered):
            _log.info("surprise novelty: dropped %d known-artist candidate(s) (%d left)",
                      len(filtered) - len(novel), len(novel))
        if not novel:
            _log.info("surprise: nothing novel found (raw=%d filt=%d) -> empty",
                      len(raw), len(filtered))
            return _no_results_out(f, queries, trace)

        from services import fuzzy

        # --- delivered-track exclusion (live F5): the search layer can
        # re-serve the SAME pool for a repeat «удиви» (identical queries,
        # cache); a candidate whose normalized title (fuzzy.clean_title, the
        # shared title normalization) matches one of the user's recently
        # delivered best_titles must not deliver twice. A store without the
        # helper (old test stubs) just skips this cut. ---
        if ctx.user_id:
            delivered_fn = getattr(store, "get_recent_best_titles", None)
            delivered_keys: set[str] = set()
            if delivered_fn is not None:
                try:
                    delivered_titles = await delivered_fn(
                        ctx.user_id, limit=_DELIVERED_LOOKBACK)
                except Exception:
                    delivered_titles = []
                delivered_keys = {k for k in (fuzzy.clean_title(t) for t in delivered_titles)
                                  if k}
            if delivered_keys:
                fresh = [c for c in novel
                         if fuzzy.clean_title(c.get("title") or "") not in delivered_keys]
                if len(fresh) != len(novel):
                    _log.info("surprise novelty: dropped %d recently delivered track(s) "
                              "(%d left)", len(novel) - len(fresh), len(fresh))
                if fresh:
                    novel = fresh
                # every novel candidate already delivered: delivering the best
                # anyway beats an empty answer; the delivered-cut log above
                # says the pool ran dry.

        # --- pick (NO LLM#2): deterministic order, random rotation on top ---
        # Order: non-noise, views window, views (the same _pick_sort_key).
        # Rotation (live F5): among the first _ROTATION_POOL candidates that
        # pass the verifier below, pick uniformly at random instead of always
        # argmax — two «удиви» runs over the same cached pool then deliver
        # different tracks. The verifier: a candidate whose channel is
        # half-similar to a hit artist (fuzzy.artist_match >= _PICK_ARTIST_MAX)
        # is the same act again — skip to the next in pick order. Honest
        # limit: the fuzzy rule catches near-named acts and alternate
        # channels, NOT a solo project under a fresh name ("Dave Gahan" vs
        # "Depeche Mode" scores ~0.2); the MusicBrainz member probe below is
        # the net for exactly that, and every skip is logged so misses stay
        # visible.
        ordered = sorted(novel, key=_pick_sort_key)
        shortlist: list[dict] = []
        # B5 member-probe state: only shortlist-bound candidates are probed
        # (verifier-pass integration — a handful of calls, not a pool-wide
        # sweep), at most _MB_CALL_BUDGET MusicBrainz lookups per run, with
        # the per-run verdict memo in mb_seen (repeat channels of the same
        # neighbor — and a feat name already probed as an artist component —
        # cost nothing). No hit artists or a metadata service without the
        # method (old stubs) -> the probe is a no-op.
        mb_bands_fn = getattr(ctx.metadata_svc, "get_artist_bands", None) if hit_artists else None
        mb_budget = _MB_CALL_BUDGET
        mb_seen: dict[str, str] = {}

        async def _member_probe(names: list[str]) -> tuple[str, str]:
            """The first name in `names` PROVEN (MusicBrainz member-of-band
            relations) to play in a hit band, as (name, hit_artist);
            ("", "") when none is. One budget and one verdict memo serve
            BOTH probe sites below (the candidate's artist and the title's
            featured artists): a name is never asked twice in a run, and
            the HTTP calls stay capped however many feat names a title
            carries."""
            nonlocal mb_budget
            for name in names:
                verdict = mb_seen.get(name)
                if verdict is not None:
                    if verdict:
                        return name, verdict
                    continue
                if mb_budget <= 0:
                    _log.debug("surprise novelty: musicbrainz budget used up, %r unchecked",
                               name)
                    break
                mb_budget -= 1
                try:
                    bands = await mb_bands_fn(name)
                except Exception as e:
                    # fail-soft: an MB outage never breaks the pick
                    _log.debug("surprise novelty: musicbrainz lookup failed for %r: %s",
                               name, str(e)[:100])
                    mb_seen[name] = ""
                    continue
                verdict = next(
                    (h for h in hit_artists
                     if any(fuzzy.artist_match(b or "", h) >= _MB_BAND_MATCH_MIN
                            for b in (bands or set()))), "")
                mb_seen[name] = verdict
                if verdict:
                    return name, verdict
            return "", ""

        for c in ordered:
            # Prefer the TITLE-parsed artist ("Dave Gahan - Kingdom"): uploader
            # channels (Medialook Studio & co) are not artists and would defeat
            # both the fuzzy check and the MusicBrainz member probe (x2 case
            # 2026-11-06: Gahan re-delivered under a studio channel).
            channel_artist = _title_artist(c.get("title") or "") or (
                (c.get("channel") or "").removesuffix(" - Topic").strip())
            close_hit = next((h for h in hit_artists
                              if fuzzy.artist_match(channel_artist, h) >= _PICK_ARTIST_MAX), "")
            if close_hit:
                _log.info("surprise novelty: skipped %r (channel %r artist-similar to hit %r)",
                          (c.get("title") or "")[:42], channel_artist, close_hit)
                continue
            if mb_bands_fn is not None:
                # a band of the candidate's artist that fuzzy-matches a hit
                # artist = a disguised hit (the frontman's solo project).
                # Compound names ("Soulsavers & Dave Gahan") defeat a
                # whole-string lookup, so every component is probed too
                # (memo per name; the budget still caps the HTTP calls).
                if channel_artist:
                    _, member_hit = await _member_probe(_artist_variants(channel_artist))
                    if member_hit:
                        _log.info("surprise novelty: %r is a member of hit band %r (artist)",
                                  channel_artist, member_hit)
                        continue
                # x2 (2026-11-06): an artist FEATURED in the track title is
                # probed the same way — the live miss delivered "Goldfrapp -
                # Ocean Feat. Dave Gahan" because the exclusion keyed on the
                # channel (GoldfrappTV) and nothing checked the title's feat
                # credit. First proven feat member drops the candidate by
                # the same rule; empty/garbage extractions probe nothing.
                feat_names = _feat_artists(c.get("title") or "")
                if feat_names:
                    feat_name, member_hit = await _member_probe(feat_names)
                    if member_hit:
                        _log.info("surprise novelty: %r is a member of hit band %r "
                                  "(featured in title)",
                                  feat_name, member_hit)
                        continue
            shortlist.append(c)
            if len(shortlist) >= _ROTATION_POOL:
                break
        if shortlist:
            # secrets.randbelow: the module-level uniform int primitive (no
            # module-level randrange exists in secrets), cryptographically
            # strong and trivially patchable in tests.
            pick_idx = secrets.randbelow(len(shortlist))
            pick = shortlist[pick_idx]
            _log.info("surprise pick: idx=%d of %d novel", pick_idx, len(novel))
        else:
            # Every candidate reads as a hit artist again: delivering the
            # best anyway beats an empty answer; the log says it happened.
            pick = ordered[0]
            _log.info("surprise novelty: all candidates artist-similar to hits, "
                      "delivering the best anyway")

        # --- verdict: a template string, no LLM call ---
        channel_artist = (pick.get("channel") or "").removesuffix(" - Topic").strip()
        # Attribution: name the canonical artist parsed from the TITLE
        # ("Artist - Track" split), not the uploader channel — a label or
        # compilation channel misattributes the recommendation. Channel
        # stays the fallback for titles without an artist split.
        artist_line = _title_artist(pick.get("title") or "") or channel_artist
        # normalized neighbor name -> the hit artist it grew from
        hit_by_neighbor_key = {_ref_key(n): hit for n, hit in neighbor_to_hit.items()}
        hit_of_pick = (hit_by_neighbor_key.get(_channel_key(pick.get("channel") or ""), "")
                       if hit_by_neighbor_key else "")
        if hit_of_pick:
            verdict = (f"Сосед твоего попадания «{artist_line}»: "
                       f"слушатели {hit_of_pick} слушают и это")
        else:
            verdict = f"Из жанровой разведки: {pick.get('title') or ''}"

        cand = Candidate(
            pick["video_id"], pick["title"], pick["channel"], pick["duration"], pick["views"],
            source=pick.get("source", "youtube"),
            url=pick.get("url") or f"https://www.youtube.com/watch?v={pick['video_id']}",
            fallback_url=pick.get("fallback_url", ""),
        )
        # Rule-based match score (no LLM score to calibrate from): keeps the
        # handler's logging/auto-download decision uniform with other intents.
        s = score_candidate(cand, {}, f.intent, 0.0)
        cand.match_score = s.score
        cand.match_reasons = "; ".join(s.reasons)

        _log.info(
            "surprise done conf=medium | queries=%d raw=%d filt=%d novel=%d | "
            "best=[%.2f] %s | hits=%d neighbors=%d | latency search=%.2fs filter=%.2fs",
            len(queries), len(raw), len(filtered), len(novel),
            cand.match_score, cand.title[:42], len(hit_artists), len(neighbor_to_hit),
            t_search, t_filter)

        return SearchOut(intent=f.intent, signals=f.signals, queries=queries, assoc=f.assoc,
                         reference=f.reference, target=f.target,
                         candidates=[cand], best_index=0,
                         confidence="medium", intro=f.intro, verdict=verdict, trace=trace)
