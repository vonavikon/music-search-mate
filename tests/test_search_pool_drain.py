"""Drain-grace tests for the search pool window (incident 13.11 "F1 The Movie").

Live case (13.11, prod): the user asked for the track from the F1 movie; LLM#1's
FIRST query was the exact "Chris Stapleton - Bad As I Used To Be", but
descriptive queries ("Formula 1 movie first drive crash scene soundtrack")
filled the pool to _POOL_TARGET while the exact-title ytsearch was still in
flight. The early-exit cancel threw its result away: the official video never
reached the pool and the user got a Warner Bros trailer instead of the song.

Fix (2026-11-14): once the pool is full, in-flight requests are DRAINED with a
bounded grace (_DRAIN_GRACE) — done futures merge (even past the target),
still-pending ones are cancelled after the grace. No NEW requests are
dispatched after the target (the early-exit latency win stands), and a full
pool with nothing in flight adds zero waiting.

The fakes replace services.search._extract_one (the sync executor seam) with
time.sleep delays: _extract_one runs on worker threads, so the sleeps delay
the futures WITHOUT blocking the event loop — asyncio.wait's grace timeout
stays wall-clock accurate (an asyncio.sleep inside the sync fake would create
a coroutine and never sleep; the loop-visible delay is what these tests need).
"""
import logging
import re
import time

import pytest

from services import search as S

_EXACT = "Chris Stapleton - Bad As I Used To Be"   # the incident's first query


def _uniq(prefix: str, n: int) -> list[dict]:
    """n unique yt-shaped entries: distinct title+channel per item."""
    return [{"video_id": f"{prefix}-{i}", "title": f"{prefix} title {i}",
             "channel": f"{prefix} channel", "duration": 200, "views": 100}
            for i in range(n)]


def _fake_yt(plan: dict, calls: list | None = None):
    """Fake _extract_one: plan maps query -> (delay_seconds, payload).
    The payload is returned verbatim (a list of entries, or a poison object
    for the hostile case — one_yt passes anything through untouched)."""

    def fake(q, per_query):
        if calls is not None:
            calls.append(q)
        delay, payload = plan[q]
        time.sleep(delay)
        return payload

    return fake


def _no_sc(q, per_query):
    """Hermetic SoundCloud: phase 2 must never fire in these scenarios."""
    raise AssertionError("scsearch must not run: the yt pool is full")


def _patch(monkeypatch, plan, grace: float, calls: list | None = None):
    monkeypatch.setattr(S, "_extract_one", _fake_yt(plan, calls))
    monkeypatch.setattr(S, "_extract_one_sc", _no_sc)
    monkeypatch.setattr(S, "_DRAIN_GRACE", grace)
    return list(plan)


# ════ 1. the exact query finishes inside the grace -> its results land ════

@pytest.mark.asyncio
async def test_drained_exact_query_lands_in_pool(monkeypatch):
    """The incident replay: the exact query (first in the list) is slower than
    the descriptive ones, the pool fills without it — but it completes within
    _DRAIN_GRACE, so the drain merges it instead of cancelling."""
    plan = {
        _EXACT: (0.8, _uniq("exact", 5)),
        "d1": (0.02, _uniq("d1", 25)),
        "d2": (0.02, _uniq("d2", 25)),
        "d3": (0.02, _uniq("d3", 25)),
    }
    queries = _patch(monkeypatch, plan, grace=3.0)
    t0 = time.monotonic()
    out = await S.search_metadata(queries, per_query=20)
    dt = time.monotonic() - t0
    ids = {r["video_id"] for r in out}
    assert {"exact-0", "d1-0", "d2-0"} <= ids     # the drained exact query landed
    assert dt >= 0.7                              # and the drain DID wait for it
    assert dt < S._DRAIN_GRACE + 1.0              # ...within the grace window


# ════ 2. the exact query outlives the grace -> cancelled, bounded wait ════

@pytest.mark.asyncio
async def test_slow_exact_cancelled_after_grace(monkeypatch, caplog):
    """The exact query sleeps well past _DRAIN_GRACE -> cancelled after the
    grace (pool without it), the total wait stays ~grace (not the full sleep),
    and the drain log carries counters only — no query text (PII pattern)."""
    plan = {
        _EXACT: (1.2, _uniq("exact", 5)),
        "d1": (0.02, _uniq("d1", 25)),
        "d2": (0.02, _uniq("d2", 25)),
    }
    queries = _patch(monkeypatch, plan, grace=0.3)
    with caplog.at_level(logging.INFO, logger="services.search"):
        t0 = time.monotonic()
        out = await S.search_metadata(queries, per_query=20)
        dt = time.monotonic() - t0
    ids = {r["video_id"] for r in out}
    assert not any(i.startswith("exact-") for i in ids)   # cancelled -> dropped
    assert len(out) >= S._POOL_TARGET                     # pool is full without it
    assert dt < S._DRAIN_GRACE + 0.5                      # ~grace, NOT the 1.2s sleep
    line = next(ln for ln in caplog.text.splitlines() if "drained=" in ln)
    m = re.search(r"drained=(\d+) cancelled=(\d+)", line)
    assert m and int(m.group(2)) >= 1                     # the slow one was counted
    assert _EXACT not in line and "d1" not in line        # PII: no query text


# ════ 3. dispatch stops at the target: nothing beyond the window starts ════

@pytest.mark.asyncio
async def test_no_dispatch_beyond_window_after_target(monkeypatch):
    """Early exit is preserved: once the pool is full (drain included), no NEW
    query past the sliding window is ever dispatched — the drain only waits
    for what was already started."""
    calls: list[str] = []
    plan = {
        _EXACT: (0.5, _uniq("exact", 5)),
        "d1": (0.02, _uniq("d1", 25)),
        "d2": (0.02, _uniq("d2", 25)),
        "d3": (0.02, _uniq("d3", 25)),
        "d4": (0.02, _uniq("d4", 25)),
        "d5": (0.02, _uniq("d5", 25)),
    }
    queries = _patch(monkeypatch, plan, grace=1.5, calls=calls)
    out = await S.search_metadata(queries, per_query=20)
    assert calls[0] == _EXACT                       # window order: the exact query first
    assert "d4" not in calls and "d5" not in calls  # never dispatched past the target
    assert len(calls) <= S._WINDOW + 1              # window + at most one refill slot
    assert len(out) >= S._POOL_TARGET


# ════ 4. pool filled by the last finisher, nothing in flight -> zero wait ════

@pytest.mark.asyncio
async def test_full_pool_no_inflight_zero_extra_wait(monkeypatch):
    """The pool reaches the target exactly when the LAST future completes and
    nothing is in flight: no drain wait at all — a huge grace must not add a
    single extra second of latency."""
    plan = {
        "qA": (0.02, _uniq("a", 24)),
        "qB": (0.03, _uniq("b", 24)),
    }
    queries = _patch(monkeypatch, plan, grace=5.0)   # huge: proves it is not applied
    t0 = time.monotonic()
    out = await S.search_metadata(queries, per_query=20)
    dt = time.monotonic() - t0
    assert len(out) == 48                            # 24+24: filled by the last finisher
    assert dt < 1.0                                  # ~0s extra: no grace was burned


# ════ 5. hostile: a drained future's payload breaks the merge -> others live ════

@pytest.mark.asyncio
async def test_drained_future_exception_does_not_break_merge(monkeypatch):
    """one_yt swallows extractor exceptions, so a hostile future reaches the
    drain-merge as a POISON RESULT (here: a non-iterable payload — _merge_yt
    raises TypeError on it, the same try/except seam as fut.result()). The
    poison is skipped, the rest of the drain still merges, no crash."""
    poison = object()   # non-iterable: `for r in (poison or [])` raises TypeError
    plan = {
        _EXACT: (0.5, poison),
        "d1": (0.02, _uniq("d1", 25)),
        "d2": (0.02, _uniq("d2", 25)),
        "d3": (0.3, _uniq("d3", 5)),   # also in the drain batch, finishes within grace
    }
    queries = _patch(monkeypatch, plan, grace=1.5)
    out = await S.search_metadata(queries, per_query=20)   # must not raise
    ids = {r["video_id"] for r in out}
    assert "d1-0" in ids and "d2-0" in ids        # the others still merged
    assert len(out) >= 50                         # 25+25 (+d3's 5 if drained)
    assert all(i != poison for i in ids)          # nothing of the poison leaked


# ════ 6. dedup by title+channel keeps working across the drain ════

@pytest.mark.asyncio
async def test_drained_duplicate_not_double_counted(monkeypatch):
    """The drained exact query returns a copy of a track the fast queries
    already took (same title+channel, different video_id): the duplicate is
    deduped exactly as in the harvest loop, while its unique entry lands."""
    dup = {"video_id": "dup-fast", "title": "Bad As I Used To Be",
           "channel": "Chris Stapleton", "duration": 200, "views": 1_000_000}
    dup_slow = dict(dup, video_id="dup-slow")     # same title+channel, other upload
    only_slow = {"video_id": "exact-only", "title": "Exact Only Track",
                 "channel": "Chris Stapleton", "duration": 210, "views": 50}
    plan = {
        _EXACT: (0.5, [dup_slow, only_slow]),
        "d1": (0.02, [dup] + _uniq("d1", 30)),
        "d2": (0.02, _uniq("d2", 25)),
    }
    queries = _patch(monkeypatch, plan, grace=1.5)
    out = await S.search_metadata(queries, per_query=20)
    assert sum(1 for r in out if r["title"] == "Bad As I Used To Be") == 1
    assert any(r["video_id"] == "exact-only" for r in out)   # drained unique landed
