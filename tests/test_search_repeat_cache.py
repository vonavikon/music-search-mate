"""Tests for services/search_cache.py: the exact-repeat search cache (П1).

Engine (normalize/key/TTL/hits/cap), the is_cacheable gate table, the
SearchOut payload roundtrip on the REAL dataclasses, and both replay
guards (no-repeat shift / honest repeat, live demotion). No network:
services.store.get_recent_best_titles is monkeypatched with a fake
recent-deliveries set; DATA_DIR is pointed at tmp_path read-at-call
(the services.search_cache module global), the same isolation pattern
as tests/test_meta_cache_port.py.
"""
import time
from pathlib import Path

import aiosqlite
import pytest

from services import search_cache
from services.llm_search import Candidate, SearchOut

pytestmark = pytest.mark.real_meta_cache  # real engine against temp dirs


@pytest.fixture
async def cache_dir(tmp_path, monkeypatch):
    """Isolated DATA_DIR + a fresh module singleton for every test."""
    await search_cache._reset()
    monkeypatch.setattr(search_cache, "DATA_DIR", tmp_path)
    yield tmp_path
    await search_cache._reset()


def _db_file(cache_dir: Path) -> str:
    return str(cache_dir / "bot-search-cache.db")


async def _fetchone(db_file: str, sql: str, params=()):
    """One-row read through a separate connection (test's eye view)."""
    async with aiosqlite.connect(db_file) as db:
        async with db.execute(sql, params) as cur:
            return await cur.fetchone()


def _out(**kw) -> SearchOut:
    """A real-dataclass SearchOut shaped like a delivered 'ok' answer."""
    defaults = {
        "intent": "similar",
        "signals": {"artist": "Kino", "energy": "high"},
        "queries": ["kino gruppa krovi", "кино группа крови"],
        "assoc": ["Кино", "Цой"],
        "candidates": [
            Candidate("vid1", "Kino — Gruppa Krovi (1988)", "Kino Official",
                      284, 500000, score=0.92, rationale="exact studio hit",
                      match_score=0.88, url="https://yt/vid1"),
            Candidate("vid2", "Gruppa Krovi live concert", "Fan Channel",
                      310, 90000, score=0.55, rationale="live bootleg",
                      is_noise=True, match_score=0.40),
            Candidate("vid3", "Kino — Pachka Cigaret (1989)", "Kino Topic",
                      280, 300000, score=0.70, rationale="another studio hit",
                      match_score=0.66, url="https://yt/vid3",
                      fallback_url="https://sc/vid3"),
        ],
        "best_index": 0,
        "confidence": "high",
        "intro": "Классика русского рока.",
        "verdict": "Тот самый студийный хит.",
        "need_clarify": False,
        "clarify_question": "",
        "reference": {"artist": "Kino", "track": "Gruppa Krovi"},
        "target": {"track": "Gruppa Krovi"},
    }
    defaults.update(kw)
    return SearchOut(**defaults)


# ════ engine: roundtrip + hits ════

async def test_roundtrip_key_normalization_and_hits(cache_dir):
    k1 = search_cache.cache_key("Depeche Mode  ENJOY…", "en")
    k2 = search_cache.cache_key("depeche mode enjoy", "en")
    assert k1 == k2                      # surface variants collapse
    assert search_cache.cache_key("enjoy", "ru") != k1  # lang separates

    assert await search_cache.get(k1) is None  # miss before put
    await search_cache.put(k1, "en", {"intent": "similar", "n": 1})
    assert await search_cache.get(k1) == {"intent": "similar", "n": 1}

    db = _db_file(cache_dir)
    lang, hits1 = await _fetchone(
        db, "SELECT lang, hits FROM search_cache WHERE key=?", (k1,))
    assert lang == "en" and hits1 >= 1   # the get above bumped the counter
    await search_cache.get(k1)
    hits2 = (await _fetchone(
        db, "SELECT hits FROM search_cache WHERE key=?", (k1,)))[0]
    assert hits2 == hits1 + 1            # every get bumps hits


async def test_put_overwrites_and_resets_hits(cache_dir):
    k = search_cache.cache_key("kino", "ru")
    await search_cache.put(k, "ru", {"v": 1})
    await search_cache.get(k)                 # hits: 0 -> 1
    await search_cache.put(k, "ru", {"v": 2})  # a fresher answer wins, hits -> 0
    assert await search_cache.get(k) == {"v": 2}  # and bumps to 1 again
    hits = (await _fetchone(
        _db_file(cache_dir), "SELECT hits FROM search_cache WHERE key=?", (k,)))[0]
    assert hits == 1                     # reset by the rewrite (not 2)


async def test_value_cap_skips_oversized_payloads(cache_dir):
    k = search_cache.cache_key("big", "ru")
    await search_cache.put(k, "ru", {"blob": "x" * 70_000})
    assert await search_cache.get(k) is None
    n = (await _fetchone(_db_file(cache_dir), "SELECT COUNT(*) FROM search_cache"))[0]
    assert n == 0


# ════ engine: TTL ════

async def test_ttl_lazy_expiry_deletes_row_on_get(cache_dir):
    k = search_cache.cache_key("old", "ru")
    await search_cache.put(k, "ru", {"intent": "mood"})
    expired = time.time() - 31 * 86400  # default TTL 30 days
    async with aiosqlite.connect(_db_file(cache_dir)) as raw:
        await raw.execute("UPDATE search_cache SET created_ts=?", (expired,))
        await raw.commit()

    assert await search_cache.get(k) is None
    # expired rows are deleted, not just hidden
    n = (await _fetchone(_db_file(cache_dir), "SELECT COUNT(*) FROM search_cache"))[0]
    assert n == 0


async def test_init_sweeps_expired_rows(cache_dir):
    await search_cache.init()
    k = search_cache.cache_key("fresh", "ru")
    await search_cache.put(k, "ru", {"a": 1})
    expired = time.time() - 40 * 86400
    async with aiosqlite.connect(_db_file(cache_dir)) as raw:
        await raw.execute(
            "INSERT INTO search_cache (key, lang, out_json, created_ts, hits) "
            "VALUES (?,?,?,?,0)",
            (search_cache._KEY_PREFIX + "deadbeef", "ru", "{}", expired))
        await raw.commit()

    await search_cache.init()  # startup sweep

    # the expired row is gone, the live one survives
    got = (await _fetchone(
        _db_file(cache_dir), "SELECT COUNT(*) FROM search_cache"))[0]
    assert got == 1
    assert await search_cache.get(k) == {"a": 1}


# ════ engine: schema version prefix ════

async def test_keys_carry_version_prefix_and_orphans_are_unreachable(cache_dir):
    k = search_cache.cache_key("query", "ru")
    assert k.startswith(f"v{search_cache.SCHEMA_VERSION}:")
    assert k.startswith("v1:") and not k.startswith("v2:")
    assert search_cache.SCHEMA_VERSION == 1

    # a row written under an OLD prefix (a pre-bump database) is orphaned:
    # every current-form key hashes differently, so the v1 lookup can
    # never match it — the row is dead weight until the TTL sweep takes it
    old_k = "v0:" + k[len("v1:"):]
    assert old_k != k
    await search_cache.init()  # create the schema before the raw insert
    async with aiosqlite.connect(_db_file(cache_dir)) as raw:
        await raw.execute(
            "INSERT INTO search_cache (key, lang, out_json, created_ts, hits) "
            "VALUES (?,?,?,?,0)",
            (old_k, "ru", '{"intent": "old"}', time.time()))
        await raw.commit()
    assert await search_cache.get(k) is None  # orphan unreachable


# ════ is_cacheable gate ════

def test_is_cacheable_table():
    ok = search_cache.is_cacheable
    # clean first searches
    assert ok("рок как AC/DC", None)
    assert ok("песня из Такси", "this_track")
    assert ok("Depeche Mode ENJOY…", None)      # punctuation tail is fine
    # the "Не то:" miss-chain tail (handler desc2) is a refinement
    assert not ok("кино\nНе то: Группа Крови; Звезда", None)
    # surprise must stay fresh every time — in the ask...
    assert not ok("удиви меня чем-то новым", None)
    assert not ok("Удиви!", None)
    assert not ok("surprise me with something", None)
    # ...and in the produced intent (put-side check)
    assert not ok("что-нибудь новенькое", "surprise")
    assert ok("что-нибудь новенькое", "similar")
    # oversized descs (pasted lyrics) never repeat exactly
    assert not ok("текст " * 60, None)          # 300 chars > 200
    assert ok("т " * 99 + "x", None)            # 199 chars is fine
    # empty is not a search
    assert not ok("", None)
    assert not ok("   ", None)


# ════ payload roundtrip on the real dataclasses ════

async def test_to_from_payload_roundtrip(cache_dir):
    out = _out()
    payload = search_cache.to_payload(out)
    k = search_cache.cache_key("кино группа крови", "ru")
    await search_cache.put(k, "ru", payload)
    got = search_cache.from_payload(await search_cache.get(k))

    assert got is not None
    assert isinstance(got, SearchOut)
    assert got.intent == out.intent and got.confidence == out.confidence
    assert got.intro == out.intro and got.verdict == out.verdict
    assert got.queries == out.queries and got.assoc == out.assoc
    assert got.signals == out.signals
    assert got.reference == out.reference and got.target == out.target
    assert got.need_clarify is False and got.clarify_question == ""
    assert got.block == "" and got.degraded is False
    assert got.best_index == out.best_index
    assert [c.video_id for c in got.candidates] == ["vid1", "vid2", "vid3"]

    c0, c1, c2 = got.candidates
    assert isinstance(c0, Candidate)
    assert c0.title == out.candidates[0].title
    assert c0.score == 0.92 and c0.match_score == 0.88
    assert c0.url == "https://yt/vid1"
    assert c1.is_noise is True and c1.rationale == "live bootleg"
    assert c2.fallback_url == "https://sc/vid3"
    # trace is never stored (a live object) — the replay carries none
    assert got.trace is None


def test_from_payload_tolerates_missing_keys():
    # an older/partial payload degrades to defaults, never a crash
    got = search_cache.from_payload({
        "intent": "similar",
        "candidates": [{"video_id": "v", "title": "T", "channel": "C",
                        "duration": 100, "views": 1}],
        "best_index": 0,
    })
    assert got is not None
    assert got.confidence == "low"          # dataclass default
    assert got.intro == "" and got.verdict == ""
    assert got.candidates[0].score == 0.0   # candidate default
    assert got.candidates[0].url == ""


def test_from_payload_garbage_degrades_to_miss():
    assert search_cache.from_payload(None) is None
    assert search_cache.from_payload({}) is None
    assert search_cache.from_payload({"candidates": "junk"}) is None
    assert search_cache.from_payload({"candidates": [{"title": "no vid"}]}) is None


# ════ replay guard: no-repeat ════

@pytest.fixture
def recent_deliveries(monkeypatch):
    """Fake the store read: an AsyncMock returning a set of titles."""
    holder = {"titles": []}

    async def _fake(user_id, limit=20):
        return holder["titles"]

    import services.store as store_mod
    monkeypatch.setattr(store_mod, "get_recent_best_titles", _fake)
    return holder


async def test_guard_shifts_recently_delivered_best(recent_deliveries):
    recent_deliveries["titles"] = ["Kino — Gruppa Krovi (1988)"]
    out = _out()  # best = vid1 (that very record); vid3 is a fresh studio copy
    got = await search_cache.replay_guard(out, 42)
    # live demotion ran first (vid2 sinks to the tail), then the no-repeat
    # shift picked the first FRESH candidate: the studio copy vid3
    assert [c.video_id for c in got.candidates] == ["vid1", "vid3", "vid2"]
    assert got.candidates[got.best_index].video_id == "vid3"


async def test_guard_honest_repeat_when_no_fresh_candidate(recent_deliveries):
    # EVERY candidate was delivered recently -> the answer stands
    recent_deliveries["titles"] = [
        "Kino — Gruppa Krovi (1988)", "Gruppa Krovi live concert",
        "Kino — Pachka Cigaret (1989)"]
    out = _out()
    got = await search_cache.replay_guard(out, 42)
    assert got.best_index == 0               # honest repeat


async def test_guard_honest_repeat_when_fresh_too_weak(recent_deliveries):
    # the only non-recent candidate scores below the 0.45 trust bar
    recent_deliveries["titles"] = [
        "Kino — Gruppa Krovi (1988)", "Kino — Gruppa Krovi (remastered)"]
    out = _out(candidates=[
        Candidate("vid1", "Kino — Gruppa Krovi (1988)", "Kino Official",
                  284, 1, score=0.92),
        Candidate("weak", "Other Song live bootleg", "Fan", 300, 1,
                  score=0.15),  # fresh but weak
    ], best_index=0)
    got = await search_cache.replay_guard(out, 42)
    assert got.best_index == 0


async def test_guard_skipped_for_this_track_and_no_user(recent_deliveries):
    recent_deliveries["titles"] = ["Kino — Gruppa Krovi (1988)"]
    # this_track: an explicit track ask is an explicit wish to repeat
    got = await search_cache.replay_guard(_out(intent="this_track"), 42)
    assert got.best_index == 0
    # no user id (degraded call shape): nothing to compare against
    got = await search_cache.replay_guard(_out(), None)
    assert got.best_index == 0


async def test_guard_matches_record_across_upload_decoration(recent_deliveries):
    # the delivered title is decorated ("Fight Club • ..."-style), the
    # token-subset comparator must still recognize the same record
    recent_deliveries["titles"] = ["Gruppa Krovi • Kino • 1988 full album"]
    out = _out()
    got = await search_cache.replay_guard(out, 42)
    # vid1 is recognized as the delivered record despite the decoration
    # ("full album"/"(1988)" are junk) -> the shift goes to the fresh vid3
    assert got.candidates[got.best_index].video_id == "vid3"


# ════ replay guard: live demotion ════

async def test_guard_live_sinks_below_studio(recent_deliveries):
    # live noise scores above the studio copy: the replay re-sorts
    out = _out(candidates=[
        Candidate("lv1", "Gruppa Krovi LIVE вживую концерт", "Fan",
                  300, 999999, score=0.90, is_noise=True),
        Candidate("st1", "Kino — Gruppa Krovi (studio)", "Kino Official",
                  284, 500000, score=0.60),
        Candidate("st2", "Kino — Gruppa Krovi (1988 remaster)", "Kino Topic",
                  280, 1, score=0.50),
    ], best_index=0)
    got = await search_cache.replay_guard(out, 42)

    ids = [c.video_id for c in got.candidates]
    assert ids == ["st1", "st2", "lv1"]      # studio first, live at the tail
    assert got.candidates[got.best_index].video_id == "st1"
    assert not got.candidates[got.best_index].is_noise


async def test_guard_requested_live_stays(recent_deliveries):
    # is_noise=False = an explicitly requested live version: not demoted
    out = _out(candidates=[
        Candidate("lv1", "Kino live MTV unplugged", "Kino",
                  300, 1, score=0.90, is_noise=False),
        Candidate("st1", "Kino — Gruppa Krovi (studio)", "Kino Official",
                  284, 1, score=0.60),
    ], best_index=0)
    got = await search_cache.replay_guard(out, 42)
    assert got.candidates[got.best_index].video_id == "lv1"
    assert [c.video_id for c in got.candidates] == ["lv1", "st1"]


async def test_guard_all_live_keeps_order(recent_deliveries):
    # nothing to sink BELOW: the answer stands as cached
    out = _out(candidates=[
        Candidate("lv1", "Gruppa Krovi live", "Fan", 300, 1,
                  score=0.90, is_noise=True),
    ], best_index=0)
    got = await search_cache.replay_guard(out, 42)
    assert got.best_index == 0
    assert got.candidates[0].video_id == "lv1"
