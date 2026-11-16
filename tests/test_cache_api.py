"""Tests for services/cache_api.py (the msm-cache HTTP facade).

Every test builds the app through create_app() against a tmp DB + tmp
artstore and drives it with the httpx ASGI transport (no network, no
uvicorn). The service opens the SAME file a "bot" process would write,
so row assertions go through a SECOND aiosqlite connection - exactly
the two-process view the deployment relies on (2026-11-14).

Not covered here (server acceptance): the real cross-process write
race; test_put_survives_locked_db_busy_timeout is its in-process
stand-in (a second connection inside BEGIN IMMEDIATE).
"""
import asyncio
import json
import time
from pathlib import Path
from urllib.parse import quote

import aiosqlite
import httpx
import pytest

from services import cache_api

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def db_file(tmp_path, monkeypatch) -> Path:
    # no ambient token may leak into create_app() calls below
    monkeypatch.delenv("MSM_CACHE_TOKEN", raising=False)
    return tmp_path / "meta-cache.db"


@pytest.fixture
def art_dir(tmp_path) -> Path:
    d = tmp_path / "artstore"
    d.mkdir()
    return d


@pytest.fixture
async def client(db_file, art_dir):
    app = cache_api.create_app(token=TOKEN, db_path=db_file, art_dir=art_dir)
    await app.state.store.open()  # ASGI transport runs no lifespan
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://msm-cache") as c:
        yield c
    await app.state.store.close()


async def _one(db_file: Path, sql: str, params=()):
    """One-row read through a SEPARATE connection (the bot's eye view)."""
    async with aiosqlite.connect(db_file) as db:
        async with db.execute(sql, params) as cur:
            return await cur.fetchone()


# ════ auth / lifecycle ════

async def test_auth_rejects_missing_and_wrong_token(client):
    r = await client.get("/track/abc")
    assert r.status_code == 401
    r = await client.get("/track/abc",
                         headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    r = await client.get("/track/abc", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"base": None}  # a miss is base=null, not 404


def test_empty_token_refuses_to_start(db_file, art_dir):
    # fail-fast: an empty token means an OPEN cache - a refused start
    with pytest.raises(RuntimeError):
        cache_api.create_app(token="", db_path=db_file, art_dir=art_dir)
    with pytest.raises(RuntimeError):  # empty env token, same outcome
        cache_api.create_app(db_path=db_file, art_dir=art_dir)


async def test_healthz_needs_no_auth(client, db_file):
    r = await client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["db"] == str(db_file)


# ════ meta_track ════

async def test_track_roundtrip_hits_and_last_seen(client, db_file):
    base = {"title": "Gruppa Krovi", "artist": "Kino",
            "video_id": "v1", "took_ms": 5}  # volatile keys, as a live card
    r = await client.put("/track/v1", json={"base": base}, headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    r = await client.get("/track/v1", headers=AUTH)
    # volatile keys stripped exactly as the bot's put_track does
    assert r.json() == {"base": {"title": "Gruppa Krovi", "artist": "Kino"}}

    hits1, ls1 = await _one(
        db_file, "SELECT hits, last_seen FROM meta_track WHERE video_id='v1'")
    await client.get("/track/v1", headers=AUTH)
    hits2, ls2 = await _one(
        db_file, "SELECT hits, last_seen FROM meta_track WHERE video_id='v1'")
    assert hits1 >= 1           # a hit bumps hits (copied get semantics)
    assert hits2 == hits1 + 1
    assert ls2 >= ls1           # ...and refreshes last_seen (sliding TTL)


# ════ meta_artist ════

async def test_bio_roundtrip_hits_and_second_lang_row(client, db_file):
    r = await client.put("/bio", params={"artist": "Kino"}, json={"lang": "EN", "bio": "Rock band"},
                         headers=AUTH)
    assert r.status_code == 200
    r = await client.get("/bio", params={"artist": "kino", "lang": "en"}, headers=AUTH)
    assert r.json() == {"bio": "Rock band"}  # lang + key normalized
    assert (await client.get("/bio", params={"artist": "Kino", "lang": "ru"},
                             headers=AUTH)).json() == {"bio": None}

    hits1, _ = await _one(
        db_file, "SELECT hits, last_seen FROM meta_artist WHERE lang='en'")
    await client.get("/bio", params={"artist": "KINO", "lang": "en"}, headers=AUTH)
    hits2, _ = await _one(
        db_file, "SELECT hits, last_seen FROM meta_artist WHERE lang='en'")
    assert hits1 >= 1
    assert hits2 == hits1 + 1

    # a second language is a SECOND row, never an overwrite (put_bio copy)
    await client.put("/bio", params={"artist": "Kino"}, json={"lang": "ru", "bio": "Рок-группа"},
                     headers=AUTH)
    n = (await _one(db_file, "SELECT COUNT(*) FROM meta_artist"))[0]
    assert n == 2


async def test_bio_key_normalization(client, db_file):
    # «Depeche Mode» vs «depeche mode » - one normalize_key, one row
    await client.put("/bio", params={"artist": "Depeche Mode"},
                     json={"lang": "en", "bio": "Basildon band"}, headers=AUTH)
    r = await client.get("/bio", params={"artist": "depeche mode ", "lang": "en"},
                         headers=AUTH)
    assert r.json() == {"bio": "Basildon band"}
    n = (await _one(db_file, "SELECT COUNT(*) FROM meta_artist"))[0]
    assert n == 1


# ════ meta_fact ════

async def test_fact_roundtrip_refreshes_last_seen(client, db_file):
    put = await client.put(
        "/fact", params={"artist": "Kino", "track": "Gruppa Krovi"},
        json={"lang": "RU", "fact": "Recorded in 1988"}, headers=AUTH)
    assert put.status_code == 200
    r = await client.get(
        "/fact", params={"artist": "kino", "track": "gruppa krovi",
                         "lang": "ru"}, headers=AUTH)
    assert r.json() == {"fact": "Recorded in 1988"}

    # meta_fact has no hits column - only last_seen moves on a hit
    ls1 = (await _one(db_file, "SELECT last_seen FROM meta_fact"))[0]
    await client.get("/fact", params={"artist": "Kino",
                                      "track": "Gruppa Krovi", "lang": "ru"},
                     headers=AUTH)
    ls2 = (await _one(db_file, "SELECT last_seen FROM meta_fact"))[0]
    assert ls2 >= ls1


# ════ meta_lyrics (new table) ════

async def test_lyrics_roundtrip_refreshes_last_seen(client, db_file):
    data = {"plain": "la la la", "synced": "[00:01] la", "provider": "test"}
    put = await client.put("/lyrics",
                           params={"artist": "Kino", "track": "Gruppa Krovi"},
                           json={"data": data}, headers=AUTH)
    assert put.status_code == 200
    r = await client.get("/lyrics",
                         params={"artist": "KINO", "track": "gruppa krovi"},
                         headers=AUTH)
    assert r.json() == {"lyrics": data}

    ls1 = (await _one(db_file, "SELECT last_seen FROM meta_lyrics"))[0]
    await client.get("/lyrics",
                     params={"artist": "Kino", "track": "Gruppa Krovi"},
                     headers=AUTH)
    ls2 = (await _one(db_file, "SELECT last_seen FROM meta_lyrics"))[0]
    assert ls2 >= ls1  # a hit refreshes last_seen (no hits column)


# ════ art_index / artstore ════

async def test_art_hit_bytes_and_404(client, db_file, art_dir):
    jpeg = b"\xff\xd8\xff" + b"cover-bytes" * 20  # JPEG magic + payload
    (art_dir / "kino-cover.jpg").write_bytes(jpeg)
    async with aiosqlite.connect(db_file) as db:
        await db.execute(
            "INSERT INTO art_index (artist_key, album_key, file, bytes, "
            "last_seen) VALUES ('kino', '-', 'kino-cover.jpg', ?, ?)",
            (len(jpeg), time.time()))
        await db.commit()

    # album omitted: any album of the artist, newest first
    r = await client.get("/art", params={"artist": "KINO"}, headers=AUTH)
    assert r.status_code == 200
    assert r.content == jpeg
    assert r.headers["content-type"].startswith("image/jpeg")

    # album="" collapses to the '-' key, exactly as _album_key does
    r = await client.get("/art", params={"artist": "Kino", "album": ""}, headers=AUTH)
    assert r.status_code == 200
    assert r.content == jpeg

    assert (await client.get("/art", params={"artist": "Unknown"},
                             headers=AUTH)).status_code == 404

    # a row whose file is gone from disk reads as a miss (artwork_path copy)
    async with aiosqlite.connect(db_file) as db:
        await db.execute(
            "INSERT INTO art_index (artist_key, album_key, file, bytes, "
            "last_seen) VALUES ('ghost', '-', 'missing.jpg', 1, ?)",
            (time.time(),))
        await db.commit()
    assert (await client.get("/art", params={"artist": "ghost"}, headers=AUTH)).status_code == 404


# ════ invalidate ════

async def test_invalidate_deletes_only_its_kind(client, db_file, art_dir):
    await client.put("/track/v1", json={"base": {"title": "T"}}, headers=AUTH)
    await client.put("/bio", params={"artist": "Kino"}, json={"lang": "en", "bio": "b1"},
                     headers=AUTH)
    await client.put("/bio", params={"artist": "Kino"}, json={"lang": "ru", "bio": "b2"},
                     headers=AUTH)
    await client.put("/fact", params={"artist": "Kino", "track": "Song"},
                     json={"lang": "ru", "fact": "f"}, headers=AUTH)
    await client.put("/lyrics", params={"artist": "Kino", "track": "Song"},
                     json={"data": {"plain": "x"}}, headers=AUTH)
    jpeg = b"\xff\xd8\xffcover"
    art_file = art_dir / "art1.jpg"
    art_file.write_bytes(jpeg)
    async with aiosqlite.connect(db_file) as db:
        await db.execute(
            "INSERT INTO art_index (artist_key, album_key, file, bytes, "
            "last_seen) VALUES ('kino', 'album', 'art1.jpg', ?, ?)",
            (len(jpeg), time.time()))
        await db.commit()

    # track: only the track row goes, the rest stays untouched
    r = await client.post("/invalidate",
                          json={"kind": "track", "video_id": "v1"},
                          headers=AUTH)
    assert r.json() == {"deleted": 1}
    assert (await client.get("/track/v1",
                             headers=AUTH)).json() == {"base": None}
    assert (await client.get("/bio", params={"artist": "Kino", "lang": "en"},
                             headers=AUTH)).json() == {"bio": "b1"}

    # bio without lang: EVERY lang row of the artist
    r = await client.post("/invalidate", json={"kind": "bio", "artist": "Kino"},
                          headers=AUTH)
    assert r.json() == {"deleted": 2}
    assert (await client.get("/bio", params={"artist": "Kino", "lang": "en"},
                             headers=AUTH)).json() == {"bio": None}

    r = await client.post("/invalidate",
                          json={"kind": "fact", "artist": "Kino",
                                "track": "song", "lang": "ru"}, headers=AUTH)
    assert r.json() == {"deleted": 1}

    r = await client.post("/invalidate",
                          json={"kind": "lyrics", "artist": "Kino",
                                "track": "Song"}, headers=AUTH)
    assert r.json() == {"deleted": 1}

    # art: the row AND the file behind it (album key normalized)
    r = await client.post("/invalidate",
                          json={"kind": "art", "artist": "Kino",
                                "album": "Album"}, headers=AUTH)
    assert r.json() == {"deleted": 1}
    assert not art_file.exists()
    assert (await client.get("/art", params={"artist": "Kino"}, headers=AUTH)).status_code == 404

    # unknown kind is a 400, not a silent zero
    r = await client.post("/invalidate", json={"kind": "nonsense"},
                          headers=AUTH)
    assert r.status_code == 400


# ════ two-writer discipline ════

async def test_put_survives_locked_db_busy_timeout(client, db_file):
    """While a second connection holds BEGIN IMMEDIATE (the bot
    mid-write), the service put must WAIT (busy_timeout=5000) and land,
    not fail - the core of the two-writer discipline (2026-11-14)."""
    other = await aiosqlite.connect(db_file, isolation_level=None)
    try:
        async def hold_write_lock():
            await other.execute("BEGIN IMMEDIATE")
            await other.execute(
                "INSERT INTO meta_track (video_id, base_json, created_ts, "
                "last_seen, hits) VALUES ('holder','{}',0,0,0)")
            await asyncio.sleep(0.4)  # simulate the bot's transaction
            await other.execute("COMMIT")

        task = asyncio.create_task(hold_write_lock())
        await asyncio.sleep(0.1)  # let the holder actually take the lock
        r = await client.put("/track/victim",
                             json={"base": {"title": "Survivor"}},
                             headers=AUTH)
        await task
        assert r.status_code == 200
        row = await _one(db_file,
                         "SELECT base_json FROM meta_track "
                         "WHERE video_id='victim'")
        assert row is not None
        assert json.loads(row[0]) == {"title": "Survivor"}
    finally:
        await other.close()


# ════ validation ════

async def test_validation_limits(client):
    # bio over the 20k char cap
    r = await client.put("/bio", params={"artist": "Kino"},
                         json={"lang": "en", "bio": "x" * 20001},
                         headers=AUTH)
    assert r.status_code == 400
    # empty lang / empty bio
    assert (await client.put("/bio", params={"artist": "Kino"},
                             json={"lang": "", "bio": "ok"},
                             headers=AUTH)).status_code == 400
    assert (await client.put("/bio", params={"artist": "Kino"},
                             json={"lang": "en", "bio": ""},
                             headers=AUTH)).status_code == 400
    # empty track base
    assert (await client.put("/track/v", json={"base": {}},
                             headers=AUTH)).status_code == 400
    # wrong types are pydantic 422
    assert (await client.put("/bio", params={"artist": "Kino"},
                             json={"lang": "en", "bio": 42},
                             headers=AUTH)).status_code == 422
    # an artist that normalizes to nothing
    assert (await client.get("/bio", params={"artist": "   ", "lang": "en"},
                             headers=AUTH)).status_code == 400
    # lyrics payload over the 64 KiB blob cap
    assert (await client.put(
        "/lyrics", params={"artist": "A", "track": "B"},
        json={"data": {"plain": "y" * 70000}},
        headers=AUTH)).status_code == 400


# ════ (2026-11-14) стык-фикс: слэш-имена в query, не в пути ════

async def test_bio_slash_artist_roundtrip(client, db_file):
    """«AC/DC»: normalize_key хранит слэш, path-сегмент его не переживает —
    ключи едут query-параметрами и декодируются до нормализации."""
    put = await client.put("/bio", params={"artist": "AC/DC"},
                           json={"lang": "en", "bio": "hard rock"}, headers=AUTH)
    assert put.status_code == 200
    r = await client.get("/bio", params={"artist": "ac/dc", "lang": "en"},
                         headers=AUTH)
    assert r.json() == {"bio": "hard rock"}
