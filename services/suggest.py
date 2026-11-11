"""YouTube search-suggest query normalization (research card П2, B3).

YouTube's public search-suggest endpoint
(https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q=...)
is keyless and returns canonical spellings for mangled queries: «deoche mode»
-> «depeche mode». Used as a QUERY NORMALIZER feeding canonical spellings
into the this_track search pool BEFORE the fuzzy gates — the suggestions only
widen the pool, the gates still decide.

Fail-soft by contract: ANY error (network, timeout, malformed payload) logs
at debug and returns [] — a suggest outage must never break a search. One
call per search, sequential with the rest of the pipeline, under a hard
timeout ceiling.

HTTP style follows services/metadata.py: httpx.AsyncClient (no new deps).
"""
import asyncio
import logging
import time
from collections import OrderedDict

import httpx

_log = logging.getLogger("bot.suggest")

_SUGGEST_URL = "https://suggestqueries.google.com/complete/search"
_SUGGEST_N = 4     # max suggestions kept from one response
_CACHE_TTL = 60.0  # seconds a suggest result stays fresh
_CACHE_CAP = 64    # max cached queries (oldest evicted)

# Tiny in-process TTL cache. The pipelines are stateless shared instances, so
# the cache lives at module level; repeated identical queries in a session
# must not re-hit Google. Failures cache too ([] for the same TTL) — a down
# suggest service must not be re-queried on every search. OrderedDict gives
# LRU behavior: hit -> move_to_end, insert -> evict oldest past the cap.
_CACHE: OrderedDict[str, tuple[float, list[str]]] = OrderedDict()


def _cache_get(key: str) -> list[str] | None:
    """Cached suggestions for key, or None on a miss/expired entry."""
    entry = _CACHE.get(key)
    if entry is None:
        return None
    expiry, value = entry
    if expiry <= time.monotonic():
        del _CACHE[key]
        return None
    _CACHE.move_to_end(key)
    return value


def _cache_put(key: str, value: list[str]) -> None:
    """Store suggestions under key; evict the oldest past _CACHE_CAP."""
    _CACHE[key] = (time.monotonic() + _CACHE_TTL, list(value))
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_CAP:
        _CACHE.popitem(last=False)


def _parse_suggestions(data) -> list[str] | None:
    """Parsed suggest payload -> up to _SUGGEST_N non-blank suggestion
    strings, or None when the payload is not the expected client=firefox
    shape (a simple 2-element array: [original_query, [suggestion, ...]];
    non-string and blank entries are dropped). Pure: no I/O, no raise."""
    if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
        return None
    out = [s.strip() for s in data[1] if isinstance(s, str) and s.strip()]
    return out[:_SUGGEST_N]


async def _fetch(query: str, timeout: float) -> list[str] | None:
    """One HTTP GET to the suggest endpoint -> up to _SUGGEST_N suggestion
    strings, or None on ANY error / malformed payload (fail-soft, debug log).
    ds=yt scopes the suggestions to YouTube search."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(_SUGGEST_URL, params={
                "client": "firefox",
                "ds": "yt",
                "q": query,
            })
        if r.status_code != 200:
            _log.debug("yt suggest http %d for %r", r.status_code, query[:60])
            return None
        data = r.json()
    except Exception as e:  # noqa: BLE001 - fail-soft by contract
        _log.debug("yt suggest fetch failed for %r: %s", query[:60], str(e)[:80])
        return None
    out = _parse_suggestions(data)
    if out is None:
        _log.debug("yt suggest unexpected payload for %r", query[:60])
    return out


async def yt_suggest(query: str, timeout: float = 4.0) -> list[str]:
    """Canonical YouTube search suggestions for a mangled query: up to 4
    strings, [] on any error/timeout (fail-soft — NEVER raises). The timeout
    is a hard ceiling: asyncio.wait_for cuts a stuck HTTP call even when the
    per-op httpx timeouts would sum higher. Results (including empty ones)
    are cached for _CACHE_TTL seconds."""
    key = " ".join((query or "").lower().split())
    if not key:
        return []
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        # Hard ceiling on the whole call (fetch + parse), not per HTTP op.
        suggestions = await asyncio.wait_for(_fetch(query.strip(), timeout),
                                             timeout=timeout)
    except TimeoutError:
        _log.debug("yt suggest timed out (%.1fs) for %r", timeout, query[:60])
        suggestions = None
    except Exception as e:  # noqa: BLE001 - fail-soft by contract
        _log.debug("yt suggest failed for %r: %s", query[:60], str(e)[:80])
        suggestions = None
    out = list(suggestions or [])
    _cache_put(key, out)
    return out
