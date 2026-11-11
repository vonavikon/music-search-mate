"""Tests for services/suggest.py (B3: YouTube search-suggest normalization).

Contract under test:
  - parser: the client=firefox 2-element JSON array parses to up to 4
    non-blank suggestion strings; non-string entries drop; anything that is
    not that shape is None (malformed);
  - fetch through mocked HTTP: the real _fetch parses the mocked 200
    response, non-200 and raising transports fail-soft to None;
  - yt_suggest end-to-end: never raises, [] on every error shape (network
    exception, timeout, malformed payload, a fetch that explodes), hard
    timeout ceiling cuts a stuck fetch, empty/blank query short-circuits;
  - TTL cache: an identical second query is served from the cache (no
    second HTTP call, case/whitespace-insensitive key), an expired entry
    re-fetches, the cap of 64 evicts the oldest entry.

The conftest autouse fixture _no_yt_suggest_network replaces
services.suggest._fetch with an offline stub for EVERY test; tests here that
need the real HTTP path restore it from _REAL_FETCH (captured at import
time, before any fixture ran) and mock the transport at the httpx level.
"""
import asyncio
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import services.suggest as suggest_mod
from services.suggest import _CACHE_CAP, _parse_suggestions, yt_suggest

# The REAL fetch, bound at import time (the conftest autouse fixture swaps
# suggest._fetch per test afterwards).
_REAL_FETCH = suggest_mod._fetch


class _FakeResponse:
    """Minimal httpx.Response stand-in: status code + json() (json() may be
    scripted to raise, standing in for a transport decode error)."""

    def __init__(self, payload=None, status_code=200):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _http(payload=None, status_code=200):
    """Class-level patch of the suggest GET with a canned response."""
    return patch.object(
        httpx.AsyncClient, "get",
        AsyncMock(return_value=_FakeResponse(payload, status_code)))


# ════ parser ════

def test_parse_firefox_two_element_array():
    """The client=firefox shape: [original_query, [suggestion, ...]] -> the
    suggestion strings, stripped."""
    assert _parse_suggestions(
        ["deoche mode", ["depeche mode", " depeche mode songs "]]) == \
        ["depeche mode", "depeche mode songs"]


def test_parse_drops_non_strings_and_blanks():
    """Non-string and blank entries drop silently; only real strings count."""
    assert _parse_suggestions(
        ["q", ["depeche mode", 7, None, {}, "  ", ""]]) == ["depeche mode"]


def test_parse_caps_at_four():
    """At most 4 suggestions survive, in response order."""
    assert _parse_suggestions(["q", ["a", "b", "c", "d", "e", "f"]]) == \
        ["a", "b", "c", "d"]


@pytest.mark.parametrize("bad", [
    None, {}, "string", [], ["only-query"], ["q", "not-a-list"],
], ids=["none", "dict", "string", "empty", "no-suggestions", "second-not-list"])
def test_parse_malformed_shapes(bad):
    """Every shape that is not the 2-element array is malformed -> None."""
    assert _parse_suggestions(bad) is None


@pytest.mark.parametrize("ok", [["q", []], ["q", [1, 2, 3]]],
                         ids=["empty-suggestions", "no-strings"])
def test_parse_valid_shape_without_strings(ok):
    """The right shape with zero usable suggestions is a VALID empty result
    ([] — cacheable), not a malformed payload (None)."""
    assert _parse_suggestions(ok) == []


# ════ fetch through mocked HTTP ════

async def test_fetch_parses_mocked_200(monkeypatch):
    """The real _fetch (restored) reads the mocked 200 response and returns
    the parsed suggestions."""
    monkeypatch.setattr(suggest_mod, "_fetch", _REAL_FETCH)
    with _http(["deoche mode", ["depeche mode", "depeche mode playlist"]]):
        out = await suggest_mod._fetch("deoche mode", 4.0)
    assert out == ["depeche mode", "depeche mode playlist"]


async def test_fetch_non_200_fails_soft(monkeypatch):
    monkeypatch.setattr(suggest_mod, "_fetch", _REAL_FETCH)
    with _http(["deoche mode", ["depeche mode"]], status_code=503):
        assert await suggest_mod._fetch("deoche mode", 4.0) is None


async def test_fetch_transport_error_fails_soft(monkeypatch):
    monkeypatch.setattr(suggest_mod, "_fetch", _REAL_FETCH)
    with patch.object(httpx.AsyncClient, "get",
                      AsyncMock(side_effect=httpx.ConnectError("offline"))):
        assert await suggest_mod._fetch("deoche mode", 4.0) is None


# ════ yt_suggest end-to-end (mocked transport / mocked _fetch) ════

async def test_yt_suggest_end_to_end_capped(monkeypatch):
    """Full path through the real fetch with mocked HTTP: parses, caps at 4."""
    monkeypatch.setattr(suggest_mod, "_fetch", _REAL_FETCH)
    with _http(["deoche mode", ["depeche mode", "depeche mode playlist",
                                "depeche mode best of", "depeche mode live",
                                "depeche mode remasters"]]):
        out = await yt_suggest("deoche mode")
    assert out == ["depeche mode", "depeche mode playlist",
                   "depeche mode best of", "depeche mode live"]


async def test_yt_suggest_empty_query_short_circuits(monkeypatch):
    """Blank queries never reach the fetch layer."""
    calls = []

    async def spy(query, timeout):
        calls.append(query)
        return ["x"]

    monkeypatch.setattr(suggest_mod, "_fetch", spy)
    assert await yt_suggest("") == []
    assert await yt_suggest("   ") == []
    assert calls == []


async def test_yt_suggest_malformed_payload_empty(monkeypatch):
    """A 200 response whose body is not the expected array -> [] (not a raise)."""
    monkeypatch.setattr(suggest_mod, "_fetch", _REAL_FETCH)
    with _http({"unexpected": "shape"}):
        assert await yt_suggest("deoche mode") == []


@pytest.mark.parametrize("explode", [
    httpx.ReadTimeout("timed out"),
    RuntimeError("boom"),
], ids=["timeout-exception", "generic-exception"])
async def test_yt_suggest_fetch_raises_fails_soft(monkeypatch, explode):
    """A raising fetch layer (timeout or anything else) -> [], never a raise
    — the fail-soft contract the pipeline relies on."""
    async def broken(query, timeout):
        raise explode

    monkeypatch.setattr(suggest_mod, "_fetch", broken)
    assert await yt_suggest("deoche mode") == []


async def test_yt_suggest_hard_timeout_ceiling(monkeypatch):
    """A stuck fetch is cut by the wait_for ceiling: the call returns []
    within the timeout instead of hanging for the fetch's own sleep."""
    async def stuck(query, timeout):
        await asyncio.sleep(30)

    monkeypatch.setattr(suggest_mod, "_fetch", stuck)
    t0 = time.perf_counter()
    assert await yt_suggest("deoche mode", timeout=0.1) == []
    assert time.perf_counter() - t0 < 5.0


# ════ TTL cache ════

async def test_cache_second_call_no_http(monkeypatch):
    """An identical second query (any case/whitespace form) is served from
    the cache: the fetch layer runs exactly once."""
    calls = []

    async def counting(query, timeout):
        calls.append(query)
        return ["depeche mode"]

    monkeypatch.setattr(suggest_mod, "_fetch", counting)
    first = await yt_suggest("Deoche mode")
    second = await yt_suggest("  deoche MODE ")  # same cache key
    assert first == second == ["depeche mode"]
    assert calls == ["Deoche mode"]


async def test_cache_expiry_refetches(monkeypatch):
    """An expired entry is a miss: the fetch layer runs again."""
    calls = []

    async def counting(query, timeout):
        calls.append(query)
        return ["depeche mode"]

    monkeypatch.setattr(suggest_mod, "_fetch", counting)
    await yt_suggest("deoche mode")
    # Age the cached entry past its expiry without touching the process clock.
    key = "deoche mode"
    expiry, value = suggest_mod._CACHE[key]
    suggest_mod._CACHE[key] = (time.monotonic() - 0.001, value)
    await yt_suggest("deoche mode")
    assert len(calls) == 2


async def test_cache_failures_cached_too(monkeypatch):
    """A failing fetch caches []: a down suggest service is not re-queried
    within the TTL."""
    calls = []

    async def failing(query, timeout):
        calls.append(query)
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(suggest_mod, "_fetch", failing)
    assert await yt_suggest("deoche mode") == []
    assert await yt_suggest("deoche mode") == []
    assert len(calls) == 1


def test_cache_cap_evicts_oldest():
    """Past 64 entries the oldest is evicted; the newest survives."""
    for i in range(_CACHE_CAP):
        suggest_mod._cache_put(f"q{i}", [f"s{i}"])
    suggest_mod._cache_put("newest", ["sn"])
    assert len(suggest_mod._CACHE) == _CACHE_CAP
    assert "q0" not in suggest_mod._CACHE, "the oldest entry must be evicted"
    assert suggest_mod._cache_get("newest") == ["sn"]
    assert suggest_mod._cache_get(f"q{_CACHE_CAP - 1}") == [f"s{_CACHE_CAP - 1}"]
