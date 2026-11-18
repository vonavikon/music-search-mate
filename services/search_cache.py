"""Exact-repeat search cache for the bot (П1; port of web/search_cache.py).

A CLEAN first search whose normalized text was answered before is replayed
from SQLite — no LLM, no ytsearch. Only the cacheable shape is ever
consulted/written (bot/handler.py decides: no chat context, no refinement
exclusions, no "Не то:" miss-chain tail, no surprise); refinements and
miss chains always run the live pipeline.

Key = "v{SCHEMA_VERSION}:" + sha256( norm(desc) + "|" + lang ), where norm
is NFKC, lower, ё->е, whitespace collapse, trailing punctuation strip
(.,!?;: and «…»). The version prefix is the bot-side analogue of the web
schema sweep: an algorithm bump (to_payload/from_payload change, guard
change) bumps SCHEMA_VERSION and orphans every old row — they are never
read again and age out via the TTL sweep.

Storage: a SEPARATE file DATA_DIR/bot-search-cache.db (not a table in
music-search.db, and NOT web/search_cache.py's search-cache.db: the web
app defaults to the same DATA_DIR and its engine owns that filename with
an incompatible same-named table — sharing would silently kill both,
since each engine swallows schema errors). A cache reset is one file
deleted, analytics intact. TTL SEARCH_CACHE_TTL_DAYS (default 30; 0 =
everything expired at once), startup sweep + lazy per-key expiry on get.
WAL journal. Values above 64 KB are never stored. Every error is
swallowed and logged — the cache has no right to break a search.

Replay guards (replay_guard, run by the handler on every hit, no LLM):
the stored answer is SHARED across users, but a replay must respect
per-user freshness and the owner's live-demotion doctrine:
- no-repeat: when the stored best was delivered recently to THIS user,
  shift to the first fresh candidate scoring >= _FRESH_MIN_SCORE (the
  search_candidates choke-point logic, mirrored);
- live demotion: noise-flagged live candidates sink below studio ones
  (the legacy pipeline sort, mirrored).

DATA_DIR is read at call time (module global from config), so tests can
point the singleton at a temp directory.
"""
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path

import aiosqlite

from config import DATA_DIR

log = logging.getLogger(__name__)

# Bump when to_payload/from_payload or a guard changes shape/behavior: the
# key prefix changes with it and old rows become unreachable orphans.
# (2026-11-18) RANK_GEN: bumped whenever a wave changes ranking/delivery
# semantics (officials-only, variants, provenance 17.11) - old cached
# answers must die with the same deploy that ships the new behavior.
SCHEMA_VERSION = 2
_KEY_PREFIX = f"v{SCHEMA_VERSION}:"

DEFAULT_TTL_DAYS = 30.0
# A normal payload is a few KB (candidates + verdicts); above 64 KB the
# answer is pathological — storing it is pointless (as in the web engine).
_MAX_VALUE_BYTES = 65536

# Trailing punctuation that does not change a repeat's meaning.
_STRIP_PUNCT = ".,!?;:…"
_WS_RE = re.compile(r"\s+")

# Mirror of the choke-point threshold in services/llm_search.py (the
# no-repeat shift inside search_candidates): a fresh candidate must score
# at least this high to replace a recently delivered best. llm_search
# defines it as a function-local constant, so it cannot be imported
# without touching that module — keep the two values in sync manually.
_FRESH_MIN_SCORE = 0.45

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_cache (
    key        TEXT PRIMARY KEY,
    lang       TEXT NOT NULL DEFAULT '',
    out_json   TEXT NOT NULL,
    created_ts REAL NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_search_cache_created ON search_cache(created_ts);
"""


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number, using default %s", name, raw, default)
        return default


# ---------- keys ----------

def normalize_query(q: str) -> str:
    """NFKC -> lower -> ё->е -> whitespace collapse -> strip trailing
    punctuation. Order matters: collapse spaces BEFORE cutting the tail,
    or «слова .» keeps a trailing space after the dot goes."""
    s = unicodedata.normalize("NFKC", q or "").lower().replace("ё", "е")
    s = _WS_RE.sub(" ", s).strip()
    return s.strip(_STRIP_PUNCT + " \t")


def cache_key(text: str, lang: str) -> str:
    """"v1:" + sha256(norm(text) + "|" + lang): ru/en are separate entries;
    surface variants of one text («Depeche Mode  ENJOY…» == «depeche mode
    enjoy») collapse into one."""
    return _KEY_PREFIX + hashlib.sha256(
        f"{normalize_query(text)}|{lang or ''}".encode("utf-8")).hexdigest()


def is_cacheable(desc: str, intent: str | None) -> bool:
    """The desc may be served from / written to the exact-repeat cache.

    False for: the "Не то:" miss-chain tail (desc2 in the handler's miss
    re-search — a refinement, never a fresh first search); surprise in any
    form — «удиви»/«surprise» in the normalized desc (the ask) or
    intent == "surprise" in the produced answer (the LLM's classification;
    surprise must stay fresh every time, its pool is personal history);
    descs above 200 chars (pasted lyrics / paragraphs — no exact repeats).
    """
    d = desc or ""
    if not d.strip():
        return False
    if "Не то:" in d:
        return False
    if len(d) > 200:
        return False
    n = normalize_query(d)
    if "удиви" in n or "surprise" in n:
        return False
    if (intent or "").strip().lower() == "surprise":
        return False
    return True


# ---------- SearchOut <-> payload (field-by-field, tolerant on read) ----------

# Every Candidate dataclass field (services/llm_search.py): insert_query
# persists c.__dict__ for the top-10 and the card/miss-chain paths read
# url/fallback_url/rationale/match_score, so nothing may be dropped.
_CANDIDATE_FIELDS = (
    "video_id", "title", "channel", "duration", "views",
    "score", "rationale", "is_noise", "match_score", "match_reasons",
    "source", "url", "fallback_url", "canonical_year", "canonical_album",
)
# Every SearchOut field except trace (a live SearchTrace object — never
# serializable; a replayed answer carries none, which _finish_search
# already tolerates) and latency_ms (per-request, set by _do_search).
_OUT_FIELDS = (
    "intent", "signals", "queries", "assoc", "candidates", "best_index",
    "confidence", "intro", "verdict", "need_clarify", "clarify_question",
    "reference", "target", "block", "degraded",
)


def to_payload(out) -> dict:
    """SearchOut -> a JSON-able dict (real dataclass field names)."""
    cands = [{f: getattr(c, f, None) for f in _CANDIDATE_FIELDS}
             for c in (getattr(out, "candidates", None) or [])]
    body = {f: getattr(out, f, None) for f in _OUT_FIELDS}
    body["candidates"] = cands
    return body


def from_payload(payload: dict):
    """payload dict -> SearchOut, or None when unusable (treated as a miss).

    Tolerant to missing keys: every field falls back to its dataclass
    default via getattr on a fresh instance, so an older payload can only
    degrade into a shorter SearchOut, never a crash.
    """
    try:
        from services.llm_search import Candidate, SearchOut
        out = SearchOut(intent=None, signals={}, queries=[], assoc=[],
                        candidates=[], best_index=-1, confidence="low")
        for f in _OUT_FIELDS:
            if f in (payload or {}):
                setattr(out, f, payload[f])
        cands = []
        for row in (out.candidates or []):
            if not isinstance(row, dict):
                continue
            kwargs = {f: row[f] for f in _CANDIDATE_FIELDS if f in row}
            cands.append(Candidate(**kwargs))
        if not cands:
            return None  # nothing deliverable: a miss (live search), not an empty reply
        out.candidates = cands
        if not (0 <= out.best_index < len(cands)):
            out.best_index = 0
        return out
    except Exception:
        log.warning("search cache payload unusable", exc_info=True)
        return None


# ---------- replay guards (no LLM, mirror the live pipeline) ----------

async def replay_guard(out, user_id: int | None):
    """Run every guard on a cache-replayed SearchOut (in the order the live
    pipeline applies them) and return it. Never raises; a failing guard is
    a skipped guard (the stored answer stands)."""
    try:
        await _demote_live(out)
    except Exception:
        log.warning("live demotion guard failed", exc_info=True)
    try:
        await _shift_recent(out, user_id)
    except Exception:
        log.warning("no-repeat guard failed", exc_info=True)
    return out


async def _demote_live(out) -> None:
    """Live demotion, mirroring the legacy rank sort: noise-flagged live
    candidates sink below studio records; best re-points to the first
    non-live when the (post-sort) best slot is live. An explicitly
    requested live version carries is_noise=False and stays put."""
    from services.pipelines.legacy import LIVE_RE

    def _live(c) -> bool:
        return bool(getattr(c, "is_noise", False)) and bool(
            LIVE_RE.search(c.title or ""))

    ranked = out.candidates or []
    live_n = sum(1 for c in ranked if _live(c))
    if not live_n or live_n == len(ranked):
        return  # nothing to sink / nothing to sink BELOW
    ranked.sort(key=lambda c: (1 if _live(c) else 0, -c.score))
    if 0 <= out.best_index < len(ranked) and _live(ranked[out.best_index]):
        out.best_index = next(
            (i for i, c in enumerate(ranked) if not _live(c)), out.best_index)
    log.info("search cache replay: live demotion moved %d live candidate(s) "
             "below %d studio record(s)", live_n, len(ranked) - live_n)


async def _shift_recent(out, user_id: int | None) -> None:
    """No-repeat of a fresh delivery, mirroring the search_candidates choke
    point: when the stored best was delivered recently to THIS user (the
    cache is shared, deliveries are not), shift to the first fresh
    candidate with score >= _FRESH_MIN_SCORE; an honest repeat otherwise.
    this_track is exempt — an explicit track ask is an explicit wish."""
    from services.llm_search import _title_tokens, _same_record
    from services.pipelines import _normalize_intent
    from services.store import get_recent_best_titles

    if (not user_id or not out.candidates
            or not (0 <= out.best_index < len(out.candidates))
            or _normalize_intent(out.intent) not in {"similar", "mood", "from_source"}):
        return

    recent_toks: list[frozenset] = []
    try:
        for t in await get_recent_best_titles(user_id, limit=5):
            recent_toks.append(_title_tokens(t))
    except Exception:
        recent_toks = []

    def _recently_delivered(title: str) -> bool:
        # token-subset match (the choke-point comparator): a delivered SONG
        # matches across any upload/channel decoration.
        toks = _title_tokens(title)
        return any(_same_record(toks, rt) for rt in recent_toks)

    if not recent_toks or not _recently_delivered(out.candidates[out.best_index].title):
        return
    fresh_i = next(
        (i for i, c in enumerate(out.candidates)
         if not _recently_delivered(c.title) and c.score >= _FRESH_MIN_SCORE),
        None)
    if fresh_i is not None:
        log.info("search cache replay: no-repeat shift '%s' -> '%s'",
                 out.candidates[out.best_index].title[:40],
                 out.candidates[fresh_i].title[:40])
        out.best_index = fresh_i
    else:
        log.info("search cache replay: no-repeat — best '%s' delivered "
                 "recently but no fresh candidate scores >= %.2f, keeping "
                 "the honest repeat",
                 out.candidates[out.best_index].title[:40], _FRESH_MIN_SCORE)


# ---------- engine ----------

class _SearchCache:
    """SQLite store of replayed search answers; one instance per process
    (module singleton) holding only config. Every operation opens its own
    connection (the services.store.py pattern, NOT a held aiosqlite
    connection like meta_cache): the bot's one-shot tools and tests must
    never be pinned by the non-daemon aiosqlite worker thread, and WAL
    lives in the DB file, so nothing is lost. Every method swallows
    exceptions — a cache miss is the worst outcome."""

    def __init__(self) -> None:
        self._ttl_sec = max(0.0, _env_float(
            "SEARCH_CACHE_TTL_DAYS", DEFAULT_TTL_DAYS)) * 86400.0
        self._swept = False  # one TTL sweep per process at first touch
        self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def _db(self):
        """A fresh, schema-checked connection per operation (the caller
        context-manages it). The first open per process also sweeps the
        expired rows."""
        base = Path(DATA_DIR)
        base.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(base / "bot-search-cache.db") as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.executescript(_SCHEMA)
            if not self._swept:
                async with self._lock:
                    if not self._swept:
                        await self._sweep_expired(conn)
                        self._swept = True
            yield conn

    async def _sweep_expired(self, db) -> None:
        """Delete everything expired — current and orphaned-prefix rows
        alike (orphans carry created_ts too, so they age out here instead
        of lingering forever); afterwards get() expires lazily per key."""
        await db.execute(
            "DELETE FROM search_cache WHERE created_ts < ?",
            (time.time() - self._ttl_sec,))
        await db.commit()

    async def init(self) -> None:
        """Create the schema + TTL sweep. Idempotent — every call
        re-sweeps (a restart with an expired cache is fixed immediately,
        not by the first get); errors swallowed."""
        try:
            async with self._db() as db:
                await self._sweep_expired(db)
        except Exception:
            log.warning("search cache init failed", exc_info=True)

    async def aclose(self) -> None:
        # per-operation connections: nothing is held to close (the name
        # exists for API parity with services.meta_cache)
        self._swept = False

    async def get(self, key: str) -> dict | None:
        """The stored payload dict, or None (miss / TTL expired / DB error).
        A hit increments hits."""
        try:
            async with self._db() as db:
                async with db.execute(
                        "SELECT out_json, created_ts FROM search_cache "
                        "WHERE key=?", (key,)) as cur:
                    row = await cur.fetchone()
                if row is None:
                    return None
                blob, created = row
                if time.time() - float(created) > self._ttl_sec:
                    await db.execute(
                        "DELETE FROM search_cache WHERE key=?", (key,))
                    await db.commit()
                    return None
                await db.execute(
                    "UPDATE search_cache SET hits = hits + 1 WHERE key=?",
                    (key,))
                await db.commit()
                value = json.loads(blob)
                return value if isinstance(value, dict) else None
        except Exception:
            log.warning("search cache get failed", exc_info=True)
            return None

    async def put(self, key: str, lang: str, payload: dict) -> None:
        """Store a payload (a rewrite = a fresher answer; hits reset).
        Oversized or unserializable payloads are skipped silently."""
        try:
            blob = json.dumps(payload or {}, ensure_ascii=False)
        except (TypeError, ValueError):
            return
        if len(blob.encode("utf-8")) > _MAX_VALUE_BYTES:
            return
        try:
            async with self._db() as db:
                await db.execute(
                    "INSERT INTO search_cache (key, lang, out_json, created_ts, hits) "
                    "VALUES (?,?,?,?,0) "
                    "ON CONFLICT(key) DO UPDATE SET lang=excluded.lang, "
                    "out_json=excluded.out_json, created_ts=excluded.created_ts, "
                    "hits=0",
                    (key, lang or "", blob, time.time()))
                await db.commit()
        except Exception:
            log.warning("search cache put failed", exc_info=True)


_cache: _SearchCache | None = None


def _get() -> _SearchCache:
    global _cache
    if _cache is None:
        _cache = _SearchCache()
    return _cache


async def _reset() -> None:
    """Close and forget the singleton (tests / DATA_DIR change)."""
    global _cache
    cache, _cache = _cache, None
    if cache is not None:
        await cache.aclose()


async def init() -> None:
    """Open the cache: schema + startup sweep. Safe to call repeatedly."""
    await _get().init()


async def aclose() -> None:
    await _reset()


async def get(key: str) -> dict | None:
    return await _get().get(key)


async def put(key: str, lang: str, payload: dict) -> None:
    await _get().put(key, lang, payload)
