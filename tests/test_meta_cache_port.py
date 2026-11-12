"""Tests for services/meta_cache.py: the ported web meta-cache engine.

The module binds DATA_DIR at call time (a module global imported from
config), so each test points services.meta_cache.DATA_DIR at tmp_path
and resets the singleton - every test gets its own meta-cache.db and
artstore/. No network: urllib.request.urlopen is monkeypatched where a
download is exercised.
"""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import aiosqlite
import pytest

from services import meta_cache

pytestmark = pytest.mark.real_meta_cache


@pytest.fixture
async def cache_dir(tmp_path, monkeypatch):
    """Isolated DATA_DIR + a fresh module singleton for every test."""
    await meta_cache._reset()
    monkeypatch.setattr(meta_cache, "DATA_DIR", tmp_path)
    yield tmp_path
    await meta_cache._reset()


def _db_file(cache_dir: Path) -> str:
    return str(cache_dir / "meta-cache.db")


async def _fetchone(db_file: str, sql: str, params=()):
    """One-row read through a separate connection (test's eye view)."""
    async with aiosqlite.connect(db_file) as db:
        async with db.execute(sql, params) as cur:
            return await cur.fetchone()


# ════ meta_track ════

async def test_track_roundtrip_last_seen_refresh_and_hits(cache_dir):
    assert await meta_cache.get_track("v1") is None  # miss before put

    await meta_cache.put_track("v1", {"title": "Gruppa Krovi", "artist": "Kino"})
    assert await meta_cache.get_track("v1") == {"title": "Gruppa Krovi",
                                                "artist": "Kino"}

    db = _db_file(cache_dir)
    ls1, hits1 = await _fetchone(
        db, "SELECT last_seen, hits FROM meta_track WHERE video_id='v1'")
    assert hits1 >= 1  # the get above already bumped the counter

    await meta_cache.get_track("v1")
    ls2, hits2 = await _fetchone(
        db, "SELECT last_seen, hits FROM meta_track WHERE video_id='v1'")
    assert hits2 == hits1 + 1  # every get bumps hits
    assert ls2 >= ls1         # ...and refreshes last_seen (sliding TTL)


async def test_put_track_strips_volatile_keys_and_skips_empty(cache_dir):
    await meta_cache.put_track("v2", {
        "title": "T", "video_id": "v2", "took_ms": 12,
        "remaining": 3, "cache": {"a": 1},
    })
    assert await meta_cache.get_track("v2") == {"title": "T"}

    # an empty dict is never stored
    await meta_cache.put_track("v3", {})
    assert await meta_cache.get_track("v3") is None


# ════ meta_artist ════

async def test_bio_upsert_per_language(cache_dir):
    await meta_cache.put_bio("Queen", "en", "English rock band")
    assert await meta_cache.get_bio("Queen", "en") == "English rock band"
    assert await meta_cache.get_bio("Queen", "ru") is None  # not bought yet

    # the second language is a SECOND row, not an overwrite
    await meta_cache.put_bio("  QUEEN ", "RU", "Британская рок-группа")
    assert await meta_cache.get_bio("queen", "en") == "English rock band"
    assert await meta_cache.get_bio("Queen", "ru") == "Британская рок-группа"

    n = (await _fetchone(_db_file(cache_dir),
                         "SELECT COUNT(*) FROM meta_artist"))[0]
    assert n == 2


# ════ meta_fact ════

async def test_fact_roundtrip_keyed_by_artist_track_lang(cache_dir):
    assert await meta_cache.get_fact("Kino", "Gruppa Krovi", "ru") is None

    await meta_cache.put_fact("Kino", "Gruppa Krovi", "ru",
                              "Записан в 1988 году")
    # the key is normalized: case/whitespace do not create duplicates
    assert await meta_cache.get_fact("  kino ", "GRUPPA KROVI", "ru") == \
        "Записан в 1988 году"
    # another language is another fact
    assert await meta_cache.get_fact("Kino", "Gruppa Krovi", "en") is None


# ════ schema sweep ════

async def test_schema_sweep_wipes_all_cache_tables(cache_dir):
    await meta_cache.init()
    await meta_cache.put_track("v1", {"title": "T"})
    await meta_cache.put_bio("A", "en", "bio")
    await meta_cache.put_fact("A", "T", "en", "fact")

    db = _db_file(cache_dir)
    async with aiosqlite.connect(db) as raw:
        await raw.execute(
            "UPDATE meta_kv SET value='0' WHERE key='schema_version'")
        await raw.commit()

    await meta_cache.init()  # re-init sees version 0 -> self-sweep

    for table in ("meta_track", "meta_artist", "meta_fact", "art_index"):
        assert (await _fetchone(db, f"SELECT COUNT(*) FROM {table}"))[0] == 0
    assert (await _fetchone(
        db, "SELECT value FROM meta_kv WHERE key='schema_version'"))[0] == \
        str(meta_cache.META_SCHEMA_VERSION)
    assert await meta_cache.get_track("v1") is None


# ════ artstore manifest + artwork_path ════

async def test_artwork_manifest_indexed_and_looked_up(cache_dir):
    art = cache_dir / "artstore"
    art.mkdir(parents=True)
    (art / "000.jpg").write_bytes(b"\xff\xd8\xff" + b"a" * 100)
    (art / "001.jpg").write_bytes(b"\xff\xd8\xff" + b"b" * 100)
    manifest = [
        {"file": "000.jpg", "artist": "Nick Cave", "album": "Let Love In"},
        {"file": "001.jpg", "artist": "Nick Cave",
         "album": "The Boatman's Call"},
        {"file": "gone.jpg", "artist": "Missing", "album": "X"},  # no file
    ]
    (art / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    await meta_cache.init()

    # hit by artist+album (keys normalized)
    p = await meta_cache.artwork_path("nick  cave", "let love in")
    assert p is not None and p.name == "000.jpg" and p.is_file()

    # unknown album -> miss
    assert await meta_cache.artwork_path("Nick Cave", "No Such Album") is None

    # album=None -> any album of the artist
    any_p = await meta_cache.artwork_path("NICK CAVE", None)
    assert any_p is not None and any_p.parent == art

    # an entry whose file is missing on disk is never indexed
    assert await meta_cache.artwork_path("Missing", "X") is None
    assert await meta_cache.artwork_path("Nobody", None) is None

    # a hit refreshes last_seen
    db = _db_file(cache_dir)
    ls1 = (await _fetchone(
        db, "SELECT last_seen FROM art_index WHERE album_key='let love in'"))[0]
    await meta_cache.artwork_path("Nick Cave", "Let Love In")
    ls2 = (await _fetchone(
        db, "SELECT last_seen FROM art_index WHERE album_key='let love in'"))[0]
    assert ls2 >= ls1

    # a re-open re-imports the manifest without duplicating rows
    await meta_cache._reset()
    await meta_cache.init()
    n = (await _fetchone(db, "SELECT COUNT(*) FROM art_index"))[0]
    assert n == 2


# ════ store_artwork ════

class _FakeResp:
    """Context-manager stand-in for urllib's addinfourl-ish response."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, size: int = -1) -> bytes:
        return self._payload[:size] if size and size > 0 else self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


async def test_store_artwork_downloads_indexes_and_reuses(
        cache_dir, monkeypatch):
    calls = []
    payload = b"\xff\xd8\xff"  # bare JPEG magic, 3 bytes

    def fake_urlopen(url, timeout=None):
        calls.append((url, timeout))
        return _FakeResp(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    url = "https://art.example/cover.jpg"
    p1 = await meta_cache.store_artwork(
        "Aphex Twin", "Selected Ambient Works", url)
    assert p1 is not None and p1.is_file()
    assert p1.parent == cache_dir / "artstore"
    assert p1.name == hashlib.sha1(url.encode("utf-8")).hexdigest()[:16] + ".jpg"
    assert p1.read_bytes() == payload
    assert len(calls) == 1 and calls[0][0] == url
    assert calls[0][1] is not None  # a timeout was passed

    # the index row landed (keys normalized)
    n = (await _fetchone(_db_file(cache_dir),
                         "SELECT COUNT(*) FROM art_index "
                         "WHERE artist_key='aphex twin'"))[0]
    assert n == 1

    # second call: cache hit, returns the same file WITHOUT a new download
    p2 = await meta_cache.store_artwork(
        "aphex twin", "selected ambient works", url)
    assert p2 == p1
    assert len(calls) == 1


async def test_store_artwork_rejects_non_jpeg(cache_dir, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResp(b"GIF89a" + b"x" * 10))
    assert await meta_cache.store_artwork(
        "X", None, "https://art.example/bad.gif") is None
    n = (await _fetchone(_db_file(cache_dir),
                         "SELECT COUNT(*) FROM art_index"))[0]
    assert n == 0


# ════ TTL ════

async def test_ttl_expiry_by_old_last_seen(cache_dir):
    await meta_cache.put_track("vt", {"title": "Old"})
    await meta_cache.put_bio("A", "en", "old bio")
    await meta_cache.put_fact("A", "T", "en", "old fact")

    db = _db_file(cache_dir)
    expired = time.time() - 361 * 86400  # default META_CACHE_TTL_DAYS = 360
    async with aiosqlite.connect(db) as raw:
        await raw.execute("UPDATE meta_track SET last_seen=?", (expired,))
        await raw.execute("UPDATE meta_artist SET last_seen=?", (expired,))
        await raw.execute("UPDATE meta_fact SET last_seen=?", (expired,))
        await raw.commit()

    assert await meta_cache.get_track("vt") is None
    assert await meta_cache.get_bio("A", "en") is None
    assert await meta_cache.get_fact("A", "T", "en") is None

    # expired rows are deleted, not just hidden
    for table in ("meta_track", "meta_artist", "meta_fact"):
        assert (await _fetchone(db, f"SELECT COUNT(*) FROM {table}"))[0] == 0


# ════ value cap ════

async def test_value_cap_skips_oversized_entries(cache_dir):
    await meta_cache.put_track("big", {"title": "x" * 70_000})
    assert await meta_cache.get_track("big") is None

    await meta_cache.put_bio("A", "en", "y" * 70_000)
    assert await meta_cache.get_bio("A", "en") is None

    await meta_cache.put_fact("A", "T", "en", "z" * 70_000)
    assert await meta_cache.get_fact("A", "T", "en") is None


# ════ stats ════

async def test_stats_counts_bytes_and_version(cache_dir):
    await meta_cache.init()
    await meta_cache.put_track("v1", {"title": "T"})
    await meta_cache.put_bio("A", "en", "bio")

    s = meta_cache.stats()
    assert s["schema_version"] == meta_cache.META_SCHEMA_VERSION
    assert s["entries"]["meta_track"] == 1
    assert s["entries"]["meta_artist"] == 1
    assert s["entries"]["meta_fact"] == 0
    assert s["db_bytes"] > 0
    assert s["artstore_bytes"] >= 0  # artstore/ exists but is empty
