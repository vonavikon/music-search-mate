"""Long-lived metadata cache for the bot: port of the web meta_cache engine.

First external calls are expensive (yt-dlp extract, Last.fm bio, LLM
translation, facts, artwork downloads). This module is the bot-side
LOCAL store of everything external sources already delivered: data
survives restarts, TTL 360 days (env META_CACHE_TTL_DAYS), combined
ceiling 10 GiB (env META_CACHE_MAX_BYTES) covering meta-cache.db plus
the artstore/ directory, with LRU eviction by last_seen.

Engine ported faithfully from the web version (web/meta_cache.py):
- sliding TTL: every get refreshes last_seen (popular rows survive);
- lazy TTL cleanup at writes, throttled to once per 10 min (plus init);
- LRU byte ceiling: above 90% of the cap evict batches of 1000
  oldest-by-last_seen rows (rows younger than 7 days are protected),
  stop at 80%, VACUUM after eviction rounds;
- 64 KB value cap per stored text; WAL journal mode;
- every DB error is swallowed and logged: a cache miss means the live
  path, the cache never breaks the caller.

Bot additions over the web version:
- schema sweep: meta_kv.schema_version != META_SCHEMA_VERSION drops and
  recreates every cache table (an algorithm change bumps the version
  and the whole cache self-sweeps);
- artstore/: album artwork files indexed in art_index, imported from
  manifest.json at init, downloaded by store_artwork (stdlib urllib,
  JPEG magic + 250 KB size verified) and evicted together with their
  rows under the combined db+artstore ceiling.

DATA_DIR is read at call time (module global from config), so tests can
point the singleton at a temp directory.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import unicodedata
import urllib.request
from pathlib import Path

import aiosqlite

from config import DATA_DIR

log = logging.getLogger(__name__)

META_SCHEMA_VERSION = 1
DEFAULT_TTL_DAYS = 360.0
DEFAULT_MAX_BYTES = 10 * 1024 ** 3  # 10 GiB combined: meta-cache.db + artstore/

# Lazy TTL cleanup at writes - no more than once per 10 minutes.
CLEANUP_INTERVAL_SEC = 600.0
# Ceiling: above 90% of MAX_BYTES start LRU eviction, below 80% is enough.
_CEILING_HIGH = 0.9
_CEILING_LOW = 0.8
# LRU eviction never touches rows younger than 7 days.
_CEILING_PROTECT_SEC = 7.0 * 86400.0
# Rows deleted per LRU eviction round.
_LRU_BATCH = 1000
# Values above 64 KB are never stored (a card is ~1-2 KB, bio/fact a few KB).
_MAX_VALUE_BYTES = 65536
# Artwork limits: JPEG magic + at most 250 KB per file (as in web artcache).
_MAX_ARTWORK_BYTES = 250_000
_ARTWORK_TIMEOUT_SEC = 15
_JPEG_MAGIC = b"\xff\xd8\xff"

_META_TABLES = ("meta_track", "meta_artist", "meta_fact")
_ALL_TABLES = _META_TABLES + ("art_index",)

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
CREATE INDEX IF NOT EXISTS idx_meta_track_seen ON meta_track(last_seen);
CREATE INDEX IF NOT EXISTS idx_meta_artist_seen ON meta_artist(last_seen);
CREATE INDEX IF NOT EXISTS idx_meta_fact_seen ON meta_fact(last_seen);
CREATE INDEX IF NOT EXISTS idx_art_index_seen ON art_index(last_seen);
"""

# put_track never persists these volatile keys of a live pipeline result.
_TRACK_STRIP_KEYS = ("video_id", "took_ms", "remaining", "cache")

_WS_RE = re.compile(r"\s+")


def normalize_key(s: str) -> str:
    """Artist/track key: NFKC -> casefold -> collapse whitespace -> strip.

    NFKC unifies unicode forms (fullwidth "Ｍｅｔａｌｌｉｃａ", diacritic
    compositions), casefold is more aggressive than lower (German ss),
    whitespace collapse glues superficial spelling variants.
    """
    norm = unicodedata.normalize("NFKC", s or "").casefold()
    return _WS_RE.sub(" ", norm).strip()


def _album_key(album: str | None) -> str:
    """Album component of an art_index key; None/empty collapses to '-'."""
    return normalize_key(album) or "-"


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


def _db_path() -> Path:
    return Path(DATA_DIR) / "meta-cache.db"


def _artstore_dir() -> Path:
    return Path(DATA_DIR) / "artstore"


def _artstore_bytes(art_dir: Path) -> int:
    """Sum of file sizes under artstore/ (walked only at eviction/stats time)."""
    total = 0
    try:
        for root, _dirs, files in os.walk(art_dir):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _download_artwork(url: str) -> bytes | None:
    """Blocking artwork download (runs in a worker thread via to_thread)."""
    try:
        with urllib.request.urlopen(url, timeout=_ARTWORK_TIMEOUT_SEC) as resp:
            return resp.read(_MAX_ARTWORK_BYTES + 1)
    except Exception:
        log.warning("artwork download failed: %s", url, exc_info=True)
        return None


class _MetaCache:
    """SQLite metadata store; one instance per process (module singleton).

    init() is idempotent and every method also works without it: the
    connection opens lazily on first use. All public methods swallow
    exceptions - the cache has no right to break a card/bio/fact.
    """

    def __init__(self) -> None:
        ttl = _env_float("META_CACHE_TTL_DAYS", DEFAULT_TTL_DAYS)
        cap = _env_float("META_CACHE_MAX_BYTES", DEFAULT_MAX_BYTES)
        self._ttl_sec = max(0.0, ttl) * 86400.0
        self._max_bytes = int(cap)
        self._high_water = self._max_bytes * _CEILING_HIGH
        self._low_water = self._max_bytes * _CEILING_LOW
        self._conn: aiosqlite.Connection | None = None
        self._conn_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._last_cleanup = 0.0

    # ---------- lifecycle ----------

    async def init(self) -> None:
        """Open the DB: schema sweep check + artstore manifest + startup
        cleanup (TTL + ceiling). A restart with an expired/overgrown cache
        is fixed immediately, not by the first get. Errors are swallowed
        (the lazy path retries later)."""
        try:
            db = await self._db()
            await self._apply_schema(db)  # cheap re-check; sweeps on mismatch
            async with self._cleanup_lock:
                await self._cleanup_expired(db)
                await self._enforce_ceiling(db)
                self._last_cleanup = time.time()
        except Exception:
            log.warning("meta cache init failed", exc_info=True)

    async def close(self) -> None:
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass
            self._conn = None

    async def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            async with self._conn_lock:
                if self._conn is None:
                    base = Path(DATA_DIR)
                    base.mkdir(parents=True, exist_ok=True)
                    _artstore_dir().mkdir(parents=True, exist_ok=True)
                    conn = await aiosqlite.connect(_db_path())
                    await conn.execute("PRAGMA journal_mode=WAL")
                    self._conn = conn  # set before maintenance that uses it
                    await self._apply_schema(conn)
                    await self._index_manifest(conn)
                    try:
                        async with self._cleanup_lock:
                            await self._cleanup_expired(conn)
                            await self._enforce_ceiling(conn)
                            self._last_cleanup = time.time()
                    except Exception:
                        log.warning("meta cache startup cleanup failed",
                                    exc_info=True)
        return self._conn

    # ---------- schema ----------

    async def _apply_schema(self, db: aiosqlite.Connection) -> None:
        """Create tables; if meta_kv.schema_version != META_SCHEMA_VERSION,
        DROP every cache table and recreate (the self-sweep)."""
        version: str | None = None
        try:
            async with db.execute(
                    'SELECT value FROM meta_kv WHERE "key"=\'schema_version\'') as cur:
                row = await cur.fetchone()
            version = str(row[0]) if row is not None else None
        except sqlite3.Error:
            version = None  # fresh DB file: no meta_kv yet
        if version == str(META_SCHEMA_VERSION):
            await db.executescript(_SCHEMA)  # no-op, keeps missing indexes fresh
            return
        log.warning("meta cache schema_version %s != %s - sweeping all cache "
                    "tables", version, META_SCHEMA_VERSION)
        for table in ("meta_kv", *_ALL_TABLES):
            await db.execute(f"DROP TABLE IF EXISTS {table}")
        await db.commit()
        await db.executescript(_SCHEMA)
        await db.execute(
            'INSERT OR REPLACE INTO meta_kv ("key", value) '
            "VALUES ('schema_version', ?)", (str(META_SCHEMA_VERSION),))
        await db.commit()

    async def _index_manifest(self, db: aiosqlite.Connection) -> None:
        """Import artstore/manifest.json entries into art_index.

        Entries whose file is missing on disk are skipped. Idempotent:
        when art_index already holds >= len(manifest) rows, do nothing
        (a re-index after new files lands is INSERT OR REPLACE anyway).
        """
        manifest_path = _artstore_dir() / "manifest.json"
        if not manifest_path.is_file():
            return
        try:
            try:
                entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("artstore manifest unreadable", exc_info=True)
                return
            if not isinstance(entries, list):
                return
            async with db.execute("SELECT COUNT(*) FROM art_index") as cur:
                have = int((await cur.fetchone())[0])
            if have >= len(entries):
                return
            art_dir = _artstore_dir()
            now = time.time()
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                fname = entry.get("file")
                artist = entry.get("artist")
                if not isinstance(fname, str) or not fname:
                    continue
                if not isinstance(artist, str) or not normalize_key(artist):
                    continue
                fpath = art_dir / fname
                try:
                    if not fpath.is_file():
                        continue  # file gone from disk - skip the entry
                    fbytes = fpath.stat().st_size
                except OSError:
                    continue
                await db.execute(
                    "INSERT OR REPLACE INTO art_index (artist_key, album_key, "
                    "file, bytes, last_seen) VALUES (?,?,?,?,?)",
                    (normalize_key(artist), _album_key(entry.get("album")),
                     fname, fbytes, now))
            await db.commit()
        except Exception:
            log.warning("artstore manifest indexing failed", exc_info=True)

    # ---------- meta_track ----------

    async def get_track(self, video_id: str) -> dict | None:
        """Card by video_id or None (miss / TTL expired / DB failure).

        A hit refreshes last_seen and bumps hits (popular rows survive
        LRU eviction)."""
        if not video_id:
            return None
        try:
            db = await self._db()
            async with db.execute(
                    "SELECT base_json, last_seen FROM meta_track "
                    "WHERE video_id=?", (video_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                return None
            base_blob, last_seen = row
            if self._expired(last_seen):
                await db.execute(
                    "DELETE FROM meta_track WHERE video_id=?", (video_id,))
                await db.commit()
                return None
            base = json.loads(base_blob)
            if not isinstance(base, dict):
                return None
            await db.execute(
                "UPDATE meta_track SET hits=hits+1, last_seen=? "
                "WHERE video_id=?", (time.time(), video_id))
            await db.commit()
            return base
        except Exception:
            log.warning("meta cache get_track failed", exc_info=True)
            return None

    async def put_track(self, video_id: str, base: dict) -> None:
        """Upsert a card. Skips empty dicts and strips volatile keys
        (video_id/took_ms/remaining/cache). created_ts/hits of an existing
        row are kept; errors are swallowed."""
        if not video_id or not isinstance(base, dict) or not base:
            return
        clean = {k: v for k, v in base.items() if k not in _TRACK_STRIP_KEYS}
        if not clean:
            return
        try:
            base_blob = json.dumps(clean, ensure_ascii=False)
        except (TypeError, ValueError):
            return
        if len(base_blob.encode("utf-8")) > _MAX_VALUE_BYTES:
            return
        try:
            db = await self._db()
            now = time.time()
            await db.execute(
                "INSERT INTO meta_track (video_id, base_json, created_ts, "
                "last_seen, hits) VALUES (?,?,?,?,0) "
                "ON CONFLICT(video_id) DO UPDATE SET "
                "base_json=excluded.base_json, last_seen=excluded.last_seen",
                (video_id, base_blob, now, now))
            await db.commit()
            await self._maintain_after_write()
        except Exception:
            log.warning("meta cache put_track failed", exc_info=True)

    # ---------- meta_artist ----------

    async def get_bio(self, artist: str, lang: str) -> str | None:
        """Bio of the artist for ONE language, or None. A hit refreshes
        last_seen and bumps hits."""
        key = normalize_key(artist)
        lang = (lang or "").strip().lower()
        if not key or not lang:
            return None
        try:
            db = await self._db()
            async with db.execute(
                    "SELECT bio, last_seen FROM meta_artist "
                    "WHERE artist_key=? AND lang=?", (key, lang)) as cur:
                row = await cur.fetchone()
            if row is None:
                return None
            bio, last_seen = row
            if self._expired(last_seen):
                await db.execute(
                    "DELETE FROM meta_artist WHERE artist_key=? AND lang=?",
                    (key, lang))
                await db.commit()
                return None
            if not bio:
                return None
            await db.execute(
                "UPDATE meta_artist SET hits=hits+1, last_seen=? "
                "WHERE artist_key=? AND lang=?", (time.time(), key, lang))
            await db.commit()
            return bio
        except Exception:
            log.warning("meta cache get_bio failed", exc_info=True)
            return None

    async def put_bio(self, artist: str, lang: str, bio: str) -> None:
        """Upsert the bio of ONE language ('en'/'ru'/...): a second language
        becomes a second row and never overwrites the first."""
        key = normalize_key(artist)
        lang = (lang or "").strip().lower()
        if not key or not lang or not isinstance(bio, str) or not bio:
            return
        if len(bio.encode("utf-8")) > _MAX_VALUE_BYTES:
            return
        try:
            db = await self._db()
            now = time.time()
            await db.execute(
                "INSERT INTO meta_artist (artist_key, lang, bio, created_ts, "
                "last_seen, hits) VALUES (?,?,?,?,?,0) "
                "ON CONFLICT(artist_key, lang) DO UPDATE SET "
                "bio=excluded.bio, last_seen=excluded.last_seen",
                (key, lang, bio, now, now))
            await db.commit()
            await self._maintain_after_write()
        except Exception:
            log.warning("meta cache put_bio failed", exc_info=True)

    # ---------- meta_fact ----------

    async def get_fact(self, artist: str, track: str, lang: str) -> str | None:
        """Fact by (artist, track, lang) or None; a get refreshes last_seen
        (facts have no hits column - eviction is by last_seen only)."""
        key, tkey = normalize_key(artist), normalize_key(track)
        lang = (lang or "").strip().lower()
        if not key or not tkey or not lang:
            return None
        try:
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
                await db.execute(
                    "DELETE FROM meta_fact WHERE artist_key=? AND "
                    "track_key=? AND lang=?", (key, tkey, lang))
                await db.commit()
                return None
            await db.execute(
                "UPDATE meta_fact SET last_seen=? WHERE artist_key=? AND "
                "track_key=? AND lang=?", (time.time(), key, tkey, lang))
            await db.commit()
            return fact
        except Exception:
            log.warning("meta cache get_fact failed", exc_info=True)
            return None

    async def put_fact(self, artist: str, track: str, lang: str,
                       fact: str) -> None:
        """Upsert a fact (lang stored lowercased, as passed)."""
        key, tkey = normalize_key(artist), normalize_key(track)
        lang = (lang or "").strip().lower()
        if not key or not tkey or not lang:
            return
        if not isinstance(fact, str) or not fact:
            return
        if len(fact.encode("utf-8")) > _MAX_VALUE_BYTES:
            return
        try:
            db = await self._db()
            now = time.time()
            await db.execute(
                "INSERT INTO meta_fact (artist_key, track_key, lang, fact, "
                "created_ts, last_seen) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(artist_key, track_key, lang) DO UPDATE SET "
                "fact=excluded.fact, last_seen=excluded.last_seen",
                (key, tkey, lang, fact, now, now))
            await db.commit()
            await self._maintain_after_write()
        except Exception:
            log.warning("meta cache put_fact failed", exc_info=True)

    # ---------- art_index / artstore ----------

    async def artwork_path(self, artist: str,
                           album: str | None) -> Path | None:
        """Path of the stored artwork for (artist, album), or None.

        album=None means any album of the artist (most recent last_seen
        first). The Path is returned only when the file exists on disk;
        a hit refreshes the row's last_seen."""
        key = normalize_key(artist)
        if not key:
            return None
        try:
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
            art_dir = _artstore_dir()
            for album_key, fname, last_seen in rows:
                if self._expired(last_seen):
                    await self._delete_art_row(db, key, album_key)
                    continue
                fpath = art_dir / fname
                try:
                    if not fpath.is_file():
                        continue
                except OSError:
                    continue
                await db.execute(
                    "UPDATE art_index SET last_seen=? WHERE artist_key=? "
                    "AND album_key=?", (time.time(), key, album_key))
                await db.commit()
                return fpath
            return None
        except Exception:
            log.warning("meta cache artwork_path failed", exc_info=True)
            return None

    async def store_artwork(self, artist: str, album: str | None,
                            url: str) -> Path | None:
        """Download url as artwork for (artist, album) into artstore/.

        A artwork_path hit returns immediately (no download). The payload
        must be a JPEG (ff d8 ff) of at most 250 KB; the file lands at
        artstore/{sha1(url)[:16]}.jpg and is indexed in art_index. After
        the write the combined db+artstore ceiling is enforced. Any
        failure -> None."""
        key = normalize_key(artist)
        if not key or not isinstance(url, str) or not url:
            return None
        try:
            existing = await self.artwork_path(artist, album)
            if existing is not None:
                return existing
            data = await asyncio.to_thread(_download_artwork, url)
            if (not data or len(data) > _MAX_ARTWORK_BYTES
                    or not data.startswith(_JPEG_MAGIC)):
                return None
            art_dir = _artstore_dir()
            art_dir.mkdir(parents=True, exist_ok=True)
            fname = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16] + ".jpg"
            fpath = art_dir / fname
            fpath.write_bytes(data)
            now = time.time()
            db = await self._db()
            await db.execute(
                "INSERT OR REPLACE INTO art_index (artist_key, album_key, "
                "file, bytes, last_seen) VALUES (?,?,?,?,?)",
                (key, _album_key(album), fname, len(data), now))
            await db.commit()
            await self._enforce_ceiling(db)
            return fpath
        except Exception:
            log.warning("meta cache store_artwork failed", exc_info=True)
            return None

    async def _delete_art_row(self, db: aiosqlite.Connection,
                              artist_key: str, album_key: str) -> None:
        """Delete an art_index row together with its file on disk."""
        async with db.execute(
                "SELECT file FROM art_index WHERE artist_key=? AND album_key=?",
                (artist_key, album_key)) as cur:
            row = await cur.fetchone()
        await db.execute(
            "DELETE FROM art_index WHERE artist_key=? AND album_key=?",
            (artist_key, album_key))
        await db.commit()
        if row is not None:
            try:
                fpath = _artstore_dir() / row[0]
                if fpath.is_file():
                    fpath.unlink()
            except OSError:
                pass

    # ---------- maintenance ----------

    def _expired(self, last_seen: float) -> bool:
        return time.time() - float(last_seen) > self._ttl_sec

    async def _maintain_after_write(self) -> None:
        """Lazy maintenance after a write.

        TTL cleanup - at most once per CLEANUP_INTERVAL_SEC (and right
        away on the first write if init() was never called). The ceiling
        is checked on every write with a cheap stat first (db file +
        -wal): the heavy combined walk runs only when it overflows.
        artstore growth is enforced from store_artwork directly."""
        try:
            if time.time() - self._last_cleanup >= CLEANUP_INTERVAL_SEC:
                db = await self._db()
                async with self._cleanup_lock:
                    if time.time() - self._last_cleanup >= CLEANUP_INTERVAL_SEC:
                        await self._cleanup_expired(db)
                        await self._enforce_ceiling(db)
                        self._last_cleanup = time.time()
                        return
            if self._quick_bytes() > self._high_water:
                db = await self._db()
                async with self._cleanup_lock:
                    await self._enforce_ceiling(db)
        except Exception:
            log.warning("meta cache post-write maintenance failed",
                        exc_info=True)

    async def _cleanup_expired(self, db: aiosqlite.Connection) -> None:
        """DELETE everything whose last_seen is older than the TTL.

        last_seen is refreshed by every get, so only unclaimed data goes.
        art_index rows expire together with their files."""
        old = time.time() - self._ttl_sec
        for table in _META_TABLES:
            await db.execute(
                f"DELETE FROM {table} WHERE last_seen < ?", (old,))
        async with db.execute(
                "SELECT rowid, file FROM art_index WHERE last_seen < ?",
                (old,)) as cur:
            rows = await cur.fetchall()
        if rows:
            await db.execute(
                f"DELETE FROM art_index WHERE rowid IN "
                f"({','.join('?' * len(rows))})", [r[0] for r in rows])
            art_dir = _artstore_dir()
            for _rowid, fname in rows:
                try:
                    fpath = art_dir / fname
                    if fpath.is_file():
                        fpath.unlink()
                except OSError:
                    pass
        await db.commit()

    async def _enforce_ceiling(self, db: aiosqlite.Connection) -> None:
        """LRU ceiling over the COMBINED size (db bytes + artstore bytes):
        above 90% of the cap delete the oldest-by-last_seen rows in
        batches of _LRU_BATCH (rows younger than 7 days are untouchable),
        art rows together with their files first, then meta tables while
        still over; VACUUM after each round; stop below 80%. When nothing
        old is left (everything is young) give up: a slight overshoot is
        better than losing fresh rows."""
        size = await self._combined_bytes()
        if size <= self._high_water:
            return
        log.warning("meta cache %.1fMB exceeds 90%% of %dMB combined cap - "
                    "LRU cleanup started", size / 1048576.0,
                    self._max_bytes // 1048576)
        cutoff = time.time() - _CEILING_PROTECT_SEC
        removed_total = 0
        for _ in range(1000):  # fuse against an endless loop
            size = await self._combined_bytes()
            if size <= self._low_water:
                break
            removed = await self._evict_art_batch(db, cutoff)
            if await self._combined_bytes() > self._low_water:
                for table in _META_TABLES:
                    cur = await db.execute(
                        f"DELETE FROM {table} WHERE rowid IN (SELECT rowid "
                        f"FROM {table} WHERE last_seen < ? ORDER BY last_seen "
                        f"LIMIT ?)", (cutoff, _LRU_BATCH))
                    removed += max(0, cur.rowcount or 0)
            await db.commit()
            if removed == 0:
                break  # nothing old left: only young rows, do not touch
            removed_total += removed
            await db.execute("VACUUM")
        if removed_total:
            log.warning("meta cache LRU cleanup removed %d row(s), "
                        "now %.1fMB", removed_total,
                        (await self._combined_bytes()) / 1048576.0)

    async def _evict_art_batch(self, db: aiosqlite.Connection,
                               cutoff: float) -> int:
        """Delete the _LRU_BATCH oldest-by-last_seen art_index rows (older
        than cutoff) together with their files; returns rows removed."""
        async with db.execute(
                "SELECT rowid, file FROM art_index WHERE last_seen < ? "
                "ORDER BY last_seen LIMIT ?", (cutoff, _LRU_BATCH)) as cur:
            rows = await cur.fetchall()
        if not rows:
            return 0
        await db.execute(
            f"DELETE FROM art_index WHERE rowid IN "
            f"({','.join('?' * len(rows))})", [r[0] for r in rows])
        art_dir = _artstore_dir()
        for _rowid, fname in rows:
            try:
                fpath = art_dir / fname
                if fpath.is_file():
                    fpath.unlink()
            except OSError:
                pass
        return len(rows)

    def _quick_bytes(self) -> int:
        """Cheap db size (main file + -wal) without a checkpoint - the
        per-write threshold check: two stats are cheaper than any SQL."""
        db_path = _db_path()
        total = 0
        for suffix in ("", "-wal"):
            try:
                total += db_path.with_name(db_path.name + suffix) \
                    .stat().st_size
            except OSError:
                pass
        return total

    async def _db_bytes(self) -> int:
        """Honest main-file size: wal_checkpoint(TRUNCATE) first, otherwise
        fresh writes would still sit in -wal and understate the size."""
        if self._conn is not None:
            try:
                await self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
        try:
            return _db_path().stat().st_size
        except OSError:
            return 0

    async def _combined_bytes(self) -> int:
        """The ceiling quantity: meta-cache.db bytes + artstore/ bytes."""
        return await self._db_bytes() + _artstore_bytes(_artstore_dir())


# ---------- module-level API (lazy singleton) ----------

_cache: _MetaCache | None = None


def _get() -> _MetaCache:
    """The process-wide singleton; created synchronously (no IO), the
    connection and init happen lazily inside its methods."""
    global _cache
    if _cache is None:
        _cache = _MetaCache()
    return _cache


async def _reset() -> None:
    """Close and forget the singleton (tests / DATA_DIR change)."""
    global _cache
    cache, _cache = _cache, None
    if cache is not None:
        await cache.close()


async def init() -> None:
    """Open the cache: schema, manifest import, startup cleanup. Safe to
    call repeatedly (idempotent; re-checks the schema version)."""
    await _get().init()


async def aclose() -> None:
    """Close the connection. The bot never calls this (it runs forever);
    ONE-SHOT tools (scripts/warm_cache.py, diagnostics) MUST call it on
    exit: the aiosqlite worker thread is non-daemon, so an unclosed
    connection hangs the interpreter at shutdown."""
    await _reset()


async def get_track(video_id: str) -> dict | None:
    return await _get().get_track(video_id)


async def put_track(video_id: str, base: dict) -> None:
    await _get().put_track(video_id, base)


async def get_bio(artist: str, lang: str) -> str | None:
    return await _get().get_bio(artist, lang)


async def put_bio(artist: str, lang: str, bio: str) -> None:
    await _get().put_bio(artist, lang, bio)


async def get_fact(artist: str, track: str, lang: str) -> str | None:
    return await _get().get_fact(artist, track, lang)


async def put_fact(artist: str, track: str, lang: str, fact: str) -> None:
    await _get().put_fact(artist, track, lang, fact)


async def artwork_path(artist: str, album: str | None) -> Path | None:
    return await _get().artwork_path(artist, album)


async def store_artwork(artist: str, album: str | None,
                        url: str) -> Path | None:
    return await _get().store_artwork(artist, album, url)


def stats() -> dict:
    """Sync best-effort snapshot: row counts per table, db bytes, artstore
    bytes, schema_version. Reads via a throwaway stdlib sqlite3 connection
    (wal_checkpoint(TRUNCATE) needs no async cooperation); missing pieces
    stay zero/None instead of failing."""
    db_file, art_dir = _db_path(), _artstore_dir()
    out: dict = {
        "entries": {table: 0 for table in _ALL_TABLES},
        "db_bytes": 0,
        "artstore_bytes": 0,
        "schema_version": None,
    }
    try:
        if db_file.exists():
            con = sqlite3.connect(str(db_file))
            try:
                try:
                    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
                for table in _ALL_TABLES:
                    try:
                        out["entries"][table] = int(
                            con.execute(f"SELECT COUNT(*) FROM {table}")
                            .fetchone()[0])
                    except sqlite3.Error:
                        pass
                try:
                    row = con.execute(
                        'SELECT value FROM meta_kv WHERE "key"='
                        "'schema_version'").fetchone()
                    if row is not None:
                        try:
                            out["schema_version"] = int(row[0])
                        except (TypeError, ValueError):
                            out["schema_version"] = row[0]
                except sqlite3.Error:
                    pass
            finally:
                con.close()
    except Exception:
        log.warning("meta cache stats failed", exc_info=True)
    try:
        out["db_bytes"] = os.path.getsize(db_file)
    except OSError:
        pass
    out["artstore_bytes"] = _artstore_bytes(art_dir)
    return out
