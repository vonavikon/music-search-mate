"""msm-cache: HTTP read/write facade over the SHARED bot metadata cache.

Architecture (2026-11-14): the bot keeps writing data/meta-cache.db
directly through its services/meta_cache.py singleton; msm-cache opens
THE SAME file as a SECOND process so the website on the neighbor server
(RTT ~0.6 ms) reads and writes the same entities. services.meta_cache is
deliberately NOT imported (its singleton binds the bot DATA_DIR): this
module owns its own aiosqlite connection and re-creates the same tables
with CREATE TABLE IF NOT EXISTS - the DDL below is copied verbatim from
services/meta_cache.py so a DB first opened by this service is exactly
what the bot expects (and vice versa).

Two-writer discipline with the bot: WAL journal mode + busy_timeout on
both sides, and EVERY service write (put, hit-refresh, invalidate, TTL
delete) runs inside BEGIN IMMEDIATE. Hit semantics are copied from
meta_cache gets: a hit bumps hits (where the table has the column) and
refreshes last_seen (sliding TTL); a put never resets created_ts/hits.

Error discipline differs from the bot module ON PURPOSE: meta_cache
swallows every DB error (the cache must never break a card), but an HTTP
facade has no silent-miss path - sqlite errors surface as 503 (no
traceback in the response body), bad input as 400/422.

meta_lyrics is NEW in the shared DB, modeled on the web version's table
(web/meta_cache.py: artist_key+track_key PK, created_ts/last_seen, a
last_seen index) but storing the payload as ONE JSON blob because the
HTTP contract here is {"data": {...}}.

Env: MSM_CACHE_TOKEN (required non-empty; an empty token refuses startup
- fail-fast, there is deliberately NO open mode), MSM_CACHE_DB (default
/app/data/meta-cache.db), MSM_CACHE_ART (default /app/data-mate/artstore),
MSM_CACHE_PORT (default 8090), MSM_CACHE_HOST (default 127.0.0.1 - the
service is loopback-only; the neighbor web server reaches it through the
host-side tunnel), META_CACHE_TTL_DAYS (default 360, the same knob the
bot reads).
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import hmac
import json
import logging
import os
import re
import sqlite3
import time
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel

log = logging.getLogger(__name__)

# ---------- shared-cache contract (copied from services/meta_cache.py) ----------

# Keep in sync with services.meta_cache.META_SCHEMA_VERSION: a DB first
# opened HERE gets the version row so the bot's init does not read it as
# "fresh" and sweep every cache table (its sweep drops and recreates).
_CACHE_SCHEMA_VERSION = 1

DEFAULT_TTL_DAYS = 360.0
# Two-writer discipline: the bot's transactions must be waited out, not failed.
_BUSY_TIMEOUT_MS = 5000
# Per-value byte cap as in meta_cache (a card is ~1-2 KB, bio/fact a few KB).
_MAX_VALUE_BYTES = 65536
# API-level validation: stored strings are at most 20k chars.
_MAX_TEXT_CHARS = 20000
# put_track never persists these volatile keys of a live pipeline result.
_TRACK_STRIP_KEYS = ("video_id", "took_ms", "remaining", "cache")

_WS_RE = re.compile(r"\s+")


def normalize_key(s: str) -> str:
    """Artist/track key: NFKC -> casefold -> collapse whitespace -> strip.

    Copied verbatim from services/meta_cache.py (same file, same keys):
    NFKC unifies unicode forms (fullwidth "Ｍｅｔａｌｌｉｃａ", diacritic
    compositions), casefold is more aggressive than lower (German ss),
    whitespace collapse glues superficial spelling variants.
    """
    norm = unicodedata.normalize("NFKC", s or "").casefold()
    return _WS_RE.sub(" ", norm).strip()


def _album_key(album: str | None) -> str:
    """Album component of an art_index key; None/empty collapses to '-'.

    Copied from services/meta_cache.py - art keys must match or artworks
    written by the bot never resolve through this service.
    """
    return normalize_key(album) or "-"


# DDL copied verbatim from services/meta_cache.py (so the tables are
# byte-compatible with the bot's sweep/recreate); only meta_lyrics is new.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta_kv (
    "key"   TEXT PRIMARY KEY,
    value   TEXT
);
CREATE TABLE IF NOT EXISTS meta_track (
    video_id   TEXT PRIMARY KEY,
    base_json  TEXT NOT NULL,
    created_ts REAL NOT NULL,
    last_seen  REAL NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta_artist (
    artist_key TEXT NOT NULL,
    lang       TEXT NOT NULL,
    bio        TEXT NOT NULL,
    created_ts REAL NOT NULL,
    last_seen  REAL NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (artist_key, lang)
);
CREATE TABLE IF NOT EXISTS meta_fact (
    artist_key TEXT NOT NULL,
    track_key  TEXT NOT NULL,
    lang       TEXT NOT NULL,
    fact       TEXT NOT NULL,
    created_ts REAL NOT NULL,
    last_seen  REAL NOT NULL,
    PRIMARY KEY (artist_key, track_key, lang)
);
CREATE TABLE IF NOT EXISTS art_index (
    artist_key TEXT NOT NULL,
    album_key  TEXT NOT NULL,
    file       TEXT NOT NULL,
    bytes      INTEGER NOT NULL DEFAULT 0,
    last_seen  REAL NOT NULL,
    PRIMARY KEY (artist_key, album_key)
);
CREATE TABLE IF NOT EXISTS meta_lyrics (
    artist_key  TEXT NOT NULL,
    track_key   TEXT NOT NULL,
    data_json   TEXT NOT NULL,
    created_ts  REAL NOT NULL,
    last_seen   REAL NOT NULL,
    PRIMARY KEY (artist_key, track_key)
);
CREATE INDEX IF NOT EXISTS idx_meta_track_seen ON meta_track(last_seen);
CREATE INDEX IF NOT EXISTS idx_meta_artist_seen ON meta_artist(last_seen);
CREATE INDEX IF NOT EXISTS idx_meta_fact_seen ON meta_fact(last_seen);
CREATE INDEX IF NOT EXISTS idx_art_index_seen ON art_index(last_seen);
CREATE INDEX IF NOT EXISTS idx_meta_lyrics_seen ON meta_lyrics(last_seen);
"""


def _env_float(name: str, default: float) -> float:
    """Env parameter as float; empty/garbage means default (never breaks start)."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number, using default %s", name, raw, default)
        return default


# ---------- op counters ----------
# Counters only - never a payload, key or token in the logs (repo PII pattern).
_OPS: collections.Counter = collections.Counter()


def _count(op: str, outcome: str) -> None:
    _OPS[f"{op}:{outcome}"] += 1
    total = sum(_OPS.values())
    log.info("msm-cache op=%s outcome=%s", op, outcome)
    if total % 100 == 0:
        log.info("msm-cache counters total=%d detail=%s", total, dict(_OPS))


class CacheStore:
    """Own aiosqlite connection to the SHARED meta-cache.db.

    All statements run under one asyncio lock (a single connection is
    shared by every request); every write opens BEGIN IMMEDIATE so the
    bot's writer and this one queue on the SQLite write lock instead of
    failing each other (busy_timeout waits the lock out, 5000 ms).
    """

    def __init__(self, db_path: str | Path, art_dir: str | Path,
                 ttl_days: float | None = None) -> None:
        self.db_path = Path(db_path)
        self.art_dir = Path(art_dir)
        ttl = _env_float("META_CACHE_TTL_DAYS", DEFAULT_TTL_DAYS) \
            if ttl_days is None else float(ttl_days)
        self._ttl_sec = max(0.0, ttl) * 86400.0
        self._conn: aiosqlite.Connection | None = None
        self._conn_lock = asyncio.Lock()
        self._op_lock = asyncio.Lock()

    # ---------- lifecycle ----------

    async def open(self) -> None:
        """Open the connection and make sure the schema (incl. the new
        meta_lyrics) exists. Raises on failure - a service that cannot
        reach its DB must not come up healthy (restart policy retries)."""
        async with self._conn_lock:
            if self._conn is not None:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.art_dir.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(self.db_path,
                                           isolation_level=None)
            try:
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
                await conn.executescript(_SCHEMA)
                # A DB first opened HERE would have no meta_kv version row;
                # the bot's init would read that as "fresh" and sweep every
                # table. OR IGNORE leaves an existing bot-written value.
                await conn.execute(
                    'INSERT OR IGNORE INTO meta_kv ("key", value) '
                    "VALUES ('schema_version', ?)",
                    (str(_CACHE_SCHEMA_VERSION),))
            except Exception:
                with contextlib.suppress(Exception):
                    await conn.close()
                raise
            self._conn = conn

    async def close(self) -> None:
        async with self._conn_lock:
            conn, self._conn = self._conn, None
            if conn is not None:
                with contextlib.suppress(Exception):
                    await conn.close()

    async def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            await self.open()
        return self._conn  # type: ignore[return-value]

    def _expired(self, last_seen: float) -> bool:
        return time.time() - float(last_seen) > self._ttl_sec

    @asynccontextmanager
    async def _txn(self, db: aiosqlite.Connection):
        """One write transaction: BEGIN IMMEDIATE .. COMMIT/ROLLBACK.

        The two-writer discipline with the bot: taking the RESERVED lock
        up front (instead of at first write) makes lock conflicts wait
        inside busy_timeout instead of failing at COMMIT time.
        """
        await db.execute("BEGIN IMMEDIATE")
        try:
            yield
            await db.execute("COMMIT")
        except Exception:
            with contextlib.suppress(sqlite3.Error):
                await db.execute("ROLLBACK")
            raise

    # ---------- meta_track ----------

    async def get_track(self, video_id: str) -> dict | None:
        """Card by video_id or None; semantics copied from
        meta_cache.get_track: expired row -> delete + miss, hit ->
        hits+1 and last_seen=now (popular rows survive LRU eviction)."""
        if not video_id:
            return None
        async with self._op_lock:
            db = await self._db()
            async with db.execute(
                    "SELECT base_json, last_seen FROM meta_track "
                    "WHERE video_id=?", (video_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                return None
            base_blob, last_seen = row
            if self._expired(last_seen):
                async with self._txn(db):
                    await db.execute("DELETE FROM meta_track WHERE video_id=?",
                                     (video_id,))
                return None
            try:
                base = json.loads(base_blob)
            except ValueError:
                return None  # corrupt row reads as a miss, never a 500
            if not isinstance(base, dict):
                return None
            async with self._txn(db):
                await db.execute(
                    "UPDATE meta_track SET hits=hits+1, last_seen=? "
                    "WHERE video_id=?", (time.time(), video_id))
            return base

    async def put_track(self, video_id: str, base: dict) -> None:
        """Upsert a card; copied from meta_cache.put_track - volatile keys
        stripped, created_ts/hits of an existing row kept, last_seen=now.
        Raises ValueError on an (after-strip) empty or oversized payload."""
        clean = {k: v for k, v in base.items() if k not in _TRACK_STRIP_KEYS}
        if not video_id or not clean:
            raise ValueError("base is empty (after stripping volatile keys)")
        blob = json.dumps(clean, ensure_ascii=False)
        if len(blob.encode("utf-8")) > _MAX_VALUE_BYTES:
            raise ValueError("base exceeds the 64 KiB value cap")
        now = time.time()
        async with self._op_lock:
            db = await self._db()
            async with self._txn(db):
                await db.execute(
                    "INSERT INTO meta_track (video_id, base_json, created_ts, "
                    "last_seen, hits) VALUES (?,?,?,?,0) "
                    "ON CONFLICT(video_id) DO UPDATE SET "
                    "base_json=excluded.base_json, last_seen=excluded.last_seen",
                    (video_id, blob, now, now))

    # ---------- meta_artist ----------

    async def get_bio(self, artist: str, lang: str) -> str | None:
        """Bio for ONE language or None; copied from meta_cache.get_bio:
        expired row -> delete + miss, hit -> hits+1 and last_seen=now."""
        key = normalize_key(artist)
        lang = (lang or "").strip().lower()
        if not key or not lang:
            return None
        async with self._op_lock:
            db = await self._db()
            async with db.execute(
                    "SELECT bio, last_seen FROM meta_artist "
                    "WHERE artist_key=? AND lang=?", (key, lang)) as cur:
                row = await cur.fetchone()
            if row is None:
                return None
            bio, last_seen = row
            if self._expired(last_seen):
                async with self._txn(db):
                    await db.execute(
                        "DELETE FROM meta_artist WHERE artist_key=? AND lang=?",
                        (key, lang))
                return None
            if not bio:
                return None
            async with self._txn(db):
                await db.execute(
                    "UPDATE meta_artist SET hits=hits+1, last_seen=? "
                    "WHERE artist_key=? AND lang=?", (time.time(), key, lang))
            return bio

    async def put_bio(self, artist: str, lang: str, bio: str) -> None:
        """Upsert the bio of ONE language; copied from meta_cache.put_bio -
        a second language is a second row, created_ts/hits are kept."""
        key = normalize_key(artist)
        lang = (lang or "").strip().lower()
        if not key or not lang or not bio:
            raise ValueError("artist key, lang and bio must be non-empty")
        if len(bio.encode("utf-8")) > _MAX_VALUE_BYTES:
            raise ValueError("bio exceeds the 64 KiB value cap")
        now = time.time()
        async with self._op_lock:
            db = await self._db()
            async with self._txn(db):
                await db.execute(
                    "INSERT INTO meta_artist (artist_key, lang, bio, created_ts, "
                    "last_seen, hits) VALUES (?,?,?,?,?,0) "
                    "ON CONFLICT(artist_key, lang) DO UPDATE SET "
                    "bio=excluded.bio, last_seen=excluded.last_seen",
                    (key, lang, bio, now, now))

    # ---------- meta_fact ----------

    async def get_fact(self, artist: str, track: str, lang: str) -> str | None:
        """Fact by (artist, track, lang) or None; copied from
        meta_cache.get_fact: facts have no hits column - a hit only
        refreshes last_seen."""
        key, tkey = normalize_key(artist), normalize_key(track)
        lang = (lang or "").strip().lower()
        if not key or not tkey or not lang:
            return None
        async with self._op_lock:
            db = await self._db()
            async with db.execute(
                    "SELECT fact, last_seen FROM meta_fact "
                    "WHERE artist_key=? AND track_key=? AND lang=?",
                    (key, tkey, lang)) as cur:
                row = await cur.fetchone()
            if row is None:
                return None
            fact, last_seen = row
            if self._expired(last_seen):
                async with self._txn(db):
                    await db.execute(
                        "DELETE FROM meta_fact WHERE artist_key=? AND "
                        "track_key=? AND lang=?", (key, tkey, lang))
                return None
            async with self._txn(db):
                await db.execute(
                    "UPDATE meta_fact SET last_seen=? WHERE artist_key=? AND "
                    "track_key=? AND lang=?", (time.time(), key, tkey, lang))
            return fact

    async def put_fact(self, artist: str, track: str, lang: str,
                       fact: str) -> None:
        """Upsert a fact (lang stored lowercased); created_ts kept on
        conflict, last_seen=now - as meta_cache.put_fact."""
        key, tkey = normalize_key(artist), normalize_key(track)
        lang = (lang or "").strip().lower()
        if not key or not tkey or not lang:
            raise ValueError("artist/track keys and lang must be non-empty")
        if not fact:
            raise ValueError("fact must be non-empty")
        if len(fact.encode("utf-8")) > _MAX_VALUE_BYTES:
            raise ValueError("fact exceeds the 64 KiB value cap")
        now = time.time()
        async with self._op_lock:
            db = await self._db()
            async with self._txn(db):
                await db.execute(
                    "INSERT INTO meta_fact (artist_key, track_key, lang, fact, "
                    "created_ts, last_seen) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(artist_key, track_key, lang) DO UPDATE SET "
                    "fact=excluded.fact, last_seen=excluded.last_seen",
                    (key, tkey, lang, fact, now, now))

    # ---------- meta_lyrics (new in the shared DB) ----------

    async def get_lyrics(self, artist: str, track: str) -> dict | None:
        """Lyrics payload dict or None; like the web version's meta_lyrics
        there is no hits column - a hit only refreshes last_seen."""
        key, tkey = normalize_key(artist), normalize_key(track)
        if not key or not tkey:
            return None
        async with self._op_lock:
            db = await self._db()
            async with db.execute(
                    "SELECT data_json, last_seen FROM meta_lyrics "
                    "WHERE artist_key=? AND track_key=?", (key, tkey)) as cur:
                row = await cur.fetchone()
            if row is None:
                return None
            blob, last_seen = row
            if self._expired(last_seen):
                async with self._txn(db):
                    await db.execute(
                        "DELETE FROM meta_lyrics WHERE artist_key=? AND "
                        "track_key=?", (key, tkey))
                return None
            try:
                data = json.loads(blob)
            except ValueError:
                return None
            if not isinstance(data, dict):
                return None
            async with self._txn(db):
                await db.execute(
                    "UPDATE meta_lyrics SET last_seen=? WHERE artist_key=? "
                    "AND track_key=?", (time.time(), key, tkey))
            return data

    async def put_lyrics(self, artist: str, track: str, data: dict) -> None:
        """Upsert the lyrics payload as one JSON blob (created_ts kept on
        conflict, last_seen=now); ValueError on an oversized payload."""
        key, tkey = normalize_key(artist), normalize_key(track)
        if not key or not tkey:
            raise ValueError("artist/track keys must be non-empty")
        blob = json.dumps(data, ensure_ascii=False)
        if len(blob.encode("utf-8")) > _MAX_VALUE_BYTES:
            raise ValueError("data exceeds the 64 KiB value cap")
        now = time.time()
        async with self._op_lock:
            db = await self._db()
            async with self._txn(db):
                await db.execute(
                    "INSERT INTO meta_lyrics (artist_key, track_key, data_json, "
                    "created_ts, last_seen) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(artist_key, track_key) DO UPDATE SET "
                    "data_json=excluded.data_json, last_seen=excluded.last_seen",
                    (key, tkey, blob, now, now))

    # ---------- art_index / artstore ----------

    async def artwork_read(self, artist: str,
                           album: str | None) -> tuple[bytes, str] | None:
        """(bytes, filename) of the stored artwork or None - the matching
        logic of meta_cache.artwork_path copied 1:1 (normalize_key +
        _album_key, album=None means any album newest-first, expired rows
        are deleted together with their file, a hit refreshes last_seen,
        a row whose file is gone is skipped)."""
        key = normalize_key(artist)
        if not key:
            return None
        async with self._op_lock:
            db = await self._db()
            if album is None:
                query = ("SELECT album_key, file, last_seen FROM art_index "
                         "WHERE artist_key=? ORDER BY last_seen DESC")
                params: tuple = (key,)
            else:
                query = ("SELECT album_key, file, last_seen FROM art_index "
                         "WHERE artist_key=? AND album_key=?")
                params = (key, _album_key(album))
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()
            for album_key, fname, last_seen in rows:
                if self._expired(last_seen):
                    async with self._txn(db):
                        await self._delete_art_row(db, key, album_key)
                    continue
                fpath = self.art_dir / fname
                try:
                    if not fpath.is_file():
                        continue
                except OSError:
                    continue
                async with self._txn(db):
                    await db.execute(
                        "UPDATE art_index SET last_seen=? WHERE artist_key=? "
                        "AND album_key=?", (time.time(), key, album_key))
                try:
                    data = await asyncio.to_thread(fpath.read_bytes)
                except FileNotFoundError:
                    continue  # vanished between check and read - a miss
                return data, fname
            return None

    async def _delete_art_row(self, db: aiosqlite.Connection,
                              artist_key: str, album_key: str) -> None:
        """Delete an art_index row together with its file on disk (as
        meta_cache._delete_art_row); the caller owns the transaction."""
        async with db.execute(
                "SELECT file FROM art_index WHERE artist_key=? AND album_key=?",
                (artist_key, album_key)) as cur:
            row = await cur.fetchone()
        await db.execute(
            "DELETE FROM art_index WHERE artist_key=? AND album_key=?",
            (artist_key, album_key))
        if row is not None:
            with contextlib.suppress(OSError):
                fpath = self.art_dir / row[0]
                if fpath.is_file():
                    fpath.unlink()

    # ---------- invalidate / health ----------

    async def invalidate(self, kind: str, video_id: str | None,
                         artist: str | None, track: str | None,
                         lang: str | None, album: str | None) -> int:
        """DELETE rows by key - the reset path NEITHER product had: a
        poisoned row was visible to both the bot and the website with no
        way out (2026-11-14). Returns the number of deleted rows; raises
        ValueError on a bad kind/keys. For bio/fact an omitted lang
        means EVERY lang of the key; for art an omitted album means
        every album (files on disk removed with their rows)."""
        kind = (kind or "").strip().lower()
        async with self._op_lock:
            db = await self._db()
            if kind == "track":
                if not video_id:
                    raise ValueError("kind=track requires video_id")
                async with self._txn(db):
                    cur = await db.execute(
                        "DELETE FROM meta_track WHERE video_id=?", (video_id,))
                    return max(0, cur.rowcount or 0)
            if kind == "bio":
                key = normalize_key(artist or "")
                if not key:
                    raise ValueError("kind=bio requires artist")
                lang_v = (lang or "").strip().lower()
                async with self._txn(db):
                    if lang_v:
                        cur = await db.execute(
                            "DELETE FROM meta_artist WHERE artist_key=? "
                            "AND lang=?", (key, lang_v))
                    else:
                        cur = await db.execute(
                            "DELETE FROM meta_artist WHERE artist_key=?",
                            (key,))
                    return max(0, cur.rowcount or 0)
            if kind == "fact":
                key, tkey = normalize_key(artist or ""), normalize_key(track or "")
                if not key or not tkey:
                    raise ValueError("kind=fact requires artist and track")
                lang_v = (lang or "").strip().lower()
                async with self._txn(db):
                    if lang_v:
                        cur = await db.execute(
                            "DELETE FROM meta_fact WHERE artist_key=? AND "
                            "track_key=? AND lang=?", (key, tkey, lang_v))
                    else:
                        cur = await db.execute(
                            "DELETE FROM meta_fact WHERE artist_key=? AND "
                            "track_key=?", (key, tkey))
                    return max(0, cur.rowcount or 0)
            if kind == "lyrics":
                key, tkey = normalize_key(artist or ""), normalize_key(track or "")
                if not key or not tkey:
                    raise ValueError("kind=lyrics requires artist and track")
                async with self._txn(db):
                    cur = await db.execute(
                        "DELETE FROM meta_lyrics WHERE artist_key=? AND "
                        "track_key=?", (key, tkey))
                    return max(0, cur.rowcount or 0)
            if kind == "art":
                key = normalize_key(artist or "")
                if not key:
                    raise ValueError("kind=art requires artist")
                album_k = _album_key(album) if album is not None else None
                async with self._txn(db):
                    if album_k is not None:
                        cond, params = "artist_key=? AND album_key=?", (key, album_k)
                    else:
                        cond, params = "artist_key=?", (key,)
                    async with db.execute(
                            f"SELECT file FROM art_index WHERE {cond}",
                            params) as cur:
                        files = [r[0] for r in await cur.fetchall()]
                    await db.execute(f"DELETE FROM art_index WHERE {cond}",
                                     params)
                    for fname in files:
                        with contextlib.suppress(OSError):
                            fpath = self.art_dir / fname
                            if fpath.is_file():
                                fpath.unlink()
                    return len(files)
        raise ValueError(f"unknown invalidate kind: {kind!r}")

    async def health(self) -> bool:
        """Cheapest possible probe: one SELECT through the shared conn."""
        async with self._op_lock:
            db = await self._db()
            async with db.execute("SELECT 1") as cur:
                row = await cur.fetchone()
            return row is not None


# ---------- request bodies ----------

class TrackPutIn(BaseModel):
    base: dict


class BioPutIn(BaseModel):
    lang: str
    bio: str


class FactPutIn(BaseModel):
    lang: str
    fact: str


class LyricsPutIn(BaseModel):
    data: dict


class InvalidateIn(BaseModel):
    kind: str
    video_id: str | None = None
    artist: str | None = None
    track: str | None = None
    lang: str | None = None
    album: str | None = None


# ---------- validation helpers ----------

def _text(value: str, field: str) -> str:
    """Non-empty string of at most _MAX_TEXT_CHARS chars (API contract)."""
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=400,
                            detail=f"{field} must be a non-empty string")
    if len(value) > _MAX_TEXT_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"{field} exceeds {_MAX_TEXT_CHARS} chars")
    return value


def _norm_lang(value: str | None) -> str:
    lang = (value or "").strip().lower()
    if not lang:
        raise HTTPException(status_code=400,
                            detail="lang must be a non-empty string")
    if len(lang) > 8:
        raise HTTPException(status_code=400, detail="lang is too long")
    return lang


def _require_key(value: str, field: str) -> None:
    if not normalize_key(value):
        raise HTTPException(
            status_code=400, detail=f"{field} is empty after key normalization")


def _db_error(op: str) -> HTTPException:
    """503 without a traceback in the response; the server-side log
    carries the details (counters only, never a payload)."""
    log.warning("msm-cache op=%s outcome=db_error", op, exc_info=True)
    return HTTPException(status_code=503,
                         detail="cache database unavailable")


# ---------- app factory ----------

def create_app(token: str | None = None, db_path: str | Path | None = None,
               art_dir: str | Path | None = None,
               ttl_days: float | None = None) -> FastAPI:
    """Build the msm-cache app. Fails fast (RuntimeError) on an empty
    token: an unauthenticated cache open to the loopback of a shared
    host is worse than a refused start (2026-11-14)."""
    tok = os.getenv("MSM_CACHE_TOKEN", "") if token is None else str(token)
    if not tok.strip():
        raise RuntimeError(
            "MSM_CACHE_TOKEN is empty: msm-cache refuses to start "
            "(fail-fast; there is no open mode)")
    store = CacheStore(
        db_path if db_path is not None
        else os.getenv("MSM_CACHE_DB", "/app/data/meta-cache.db"),
        art_dir if art_dir is not None
        else os.getenv("MSM_CACHE_ART", "/app/data-mate/artstore"),
        ttl_days=ttl_days)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        await store.open()  # unopenable DB -> failed startup -> restart
        yield
        await store.close()

    app = FastAPI(title="msm-cache", lifespan=_lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store

    def _auth(authorization: str = Header(default="")) -> None:
        scheme, _, value = authorization.strip().partition(" ")
        ok = scheme.lower() == "bearer" and hmac.compare_digest(
            tok.encode("utf-8"), value.strip().encode("utf-8"))
        if not ok:
            raise HTTPException(
                status_code=401, detail="unauthorized",
                headers={"WWW-Authenticate": "Bearer"})

    api = APIRouter(dependencies=[Depends(_auth)])

    @api.get("/track/{video_id}")
    async def track_get(video_id: str) -> dict:
        try:
            base = await store.get_track(video_id)
        except sqlite3.Error:
            raise _db_error("track.get")
        _count("track.get", "hit" if base is not None else "miss")
        return {"base": base}

    @api.put("/track/{video_id}")
    async def track_put(video_id: str, body: TrackPutIn) -> dict:
        if not body.base:
            raise HTTPException(status_code=400,
                                detail="base must be a non-empty object")
        try:
            await store.put_track(video_id, body.base)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except sqlite3.Error:
            raise _db_error("track.put")
        _count("track.put", "ok")
        return {"ok": True}

    @api.get("/bio")
    async def bio_get(artist: str = Query(...),
                      lang: str = Query(...)) -> dict:
        _require_key(artist, "artist")
        lang_v = _norm_lang(lang)
        try:
            bio = await store.get_bio(artist, lang_v)
        except sqlite3.Error:
            raise _db_error("bio.get")
        _count("bio.get", "hit" if bio is not None else "miss")
        return {"bio": bio}

    @api.put("/bio")
    async def bio_put(artist: str = Query(...), body: BioPutIn = None) -> dict:
        _require_key(artist, "artist")
        lang_v = _norm_lang(body.lang)
        bio = _text(body.bio, "bio")
        try:
            await store.put_bio(artist, lang_v, bio)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except sqlite3.Error:
            raise _db_error("bio.put")
        _count("bio.put", "ok")
        return {"ok": True}

    @api.get("/fact")
    async def fact_get(artist: str = Query(...), track: str = Query(...),
                       lang: str = Query(...)) -> dict:
        _require_key(artist, "artist")
        _require_key(track, "track")
        lang_v = _norm_lang(lang)
        try:
            fact = await store.get_fact(artist, track, lang_v)
        except sqlite3.Error:
            raise _db_error("fact.get")
        _count("fact.get", "hit" if fact is not None else "miss")
        return {"fact": fact}

    @api.put("/fact")
    async def fact_put(artist: str = Query(...), track: str = Query(...),
                       body: FactPutIn = None) -> dict:
        _require_key(artist, "artist")
        _require_key(track, "track")
        lang_v = _norm_lang(body.lang)
        fact = _text(body.fact, "fact")
        try:
            await store.put_fact(artist, track, lang_v, fact)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except sqlite3.Error:
            raise _db_error("fact.put")
        _count("fact.put", "ok")
        return {"ok": True}

    @api.get("/lyrics")
    async def lyrics_get(artist: str = Query(...),
                         track: str = Query(...)) -> dict:
        _require_key(artist, "artist")
        _require_key(track, "track")
        try:
            data = await store.get_lyrics(artist, track)
        except sqlite3.Error:
            raise _db_error("lyrics.get")
        _count("lyrics.get", "hit" if data is not None else "miss")
        return {"lyrics": data}

    @api.put("/lyrics")
    async def lyrics_put(artist: str = Query(...), track: str = Query(...),
                         body: LyricsPutIn = None) -> dict:
        _require_key(artist, "artist")
        _require_key(track, "track")
        if not isinstance(body.data, dict):
            raise HTTPException(status_code=400,
                                detail="data must be an object")
        try:
            await store.put_lyrics(artist, track, body.data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except sqlite3.Error:
            raise _db_error("lyrics.put")
        _count("lyrics.put", "ok")
        return {"ok": True}

    @api.get("/art")
    async def art_get(artist: str = Query(...),
                      album: str | None = Query(default=None)) -> Response:
        _require_key(artist, "artist")
        try:
            found = await store.artwork_read(artist, album)
        except (sqlite3.Error, OSError):
            raise _db_error("art.get")
        if found is None:
            _count("art.get", "miss")
            raise HTTPException(status_code=404, detail="no artwork")
        data, _fname = found
        _count("art.get", "hit")
        return Response(content=data, media_type="image/jpeg")

    @api.post("/invalidate")
    async def invalidate(body: InvalidateIn) -> dict:
        try:
            deleted = await store.invalidate(
                body.kind, body.video_id, body.artist, body.track,
                body.lang, body.album)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except sqlite3.Error:
            raise _db_error("invalidate." + (body.kind or "?"))
        _count("invalidate." + (body.kind or "?").lower(), f"deleted={deleted}")
        return {"deleted": deleted}

    app.include_router(api)

    @app.get("/healthz")
    async def healthz() -> dict:
        """The only endpoint WITHOUT auth: the compose healthcheck curls
        it (2026-11-14); 503 when the shared DB is unreachable."""
        try:
            await store.health()
        except Exception:
            log.warning("msm-cache healthz: db unreachable", exc_info=True)
            raise HTTPException(status_code=503,
                                detail="cache database unavailable")
        return {"ok": True, "db": str(store.db_path)}

    return app


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not a number, using default %s", name, raw, default)
        return default


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    application = create_app()  # RuntimeError on an empty token: fail fast
    host = os.getenv("MSM_CACHE_HOST", "127.0.0.1")  # loopback by default
    port = _env_int("MSM_CACHE_PORT", 8090)
    log.info("msm-cache starting: host=%s port=%d db=%s art=%s",
             host, port, application.state.store.db_path,
             application.state.store.art_dir)
    uvicorn.run(application, host=host, port=port, log_level="info")
