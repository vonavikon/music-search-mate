#!/usr/bin/env python3
"""Pre-fill the bot's persistent metadata cache (services/meta_cache.py).

Runs inside the bot container (env and /app/data/music-search.db present):

    docker exec music-search-mate python -m scripts.warm_cache --seed
    docker exec music-search-mate python -m scripts.warm_cache --covers
    docker exec music-search-mate python -m scripts.warm_cache --bio --limit 50
    docker exec music-search-mate python -m scripts.warm_cache --bio-ru --dry
    docker exec music-search-mate python -m scripts.warm_cache --bio-ru --limit 20
    docker exec music-search-mate python -m scripts.warm_cache --stats

or on a dev machine against a local DB copy:

    python -m scripts.warm_cache --seed --db ~/copies/music-search.db

Modes (composable flags, run in the order below):
  --seed    read-only pass over the bot SQLite DB (SELECTs only):
            file_cache rows -> mc.put_track(video_id, base) with non-null
            fields only (file_cache.track maps to the contract's `title`),
            facts rows -> mc.put_fact(artist, track, lang, fact).
            Prints counts per table. No network.
  --covers  distinct (artist, album) pairs from file_cache (album NULL ->
            None) where mc.artwork_path is None and artwork_url is known:
            mc.store_artwork(artist, album, url). 0.3s pause between
            downloads, per-file failures tolerated. Prints hit/miss/skip.
  --bio     distinct file_cache artists without a cached EN bio:
            Last.fm artist.getInfo via services.metadata.MetadataService
            (built like bot/main.py; _artist_getinfo has a 24h in-memory
            cache), put_bio on non-empty results. 0.35s pause. --limit N.
  --bio-ru  artists with a cached EN bio and no RU one: translate EN->RU via
            services.llm_search.BothubClient.chat_text (constructed like
            bot/main.py). This mode costs LLM cents - it runs only when
            explicitly flagged. 0.5s pause. --limit N; --dry lists the
            targets without calling the LLM.
  --stats   print mc.stats().

Default DB path: --db > MSM_DB env > config.DATA_DIR/'music-search.db'.

The row->cache mapping lives in pure functions (track_base_from_row,
covers_plan, bio_pool) exercised by tests/test_warm_cache.py; the async
runners take the cache and the network callables as parameters, so tests
inject a FakeMC and offline fakes. services.meta_cache is imported lazily
inside run(): the module must be importable in tests without the cache
implementation present.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR  # noqa: E402

DB_ENV = "MSM_DB"
COVERS_PAUSE_SEC = 0.3
BIO_PAUSE_SEC = 0.35
BIO_RU_PAUSE_SEC = 0.5
TRANSLATE_SYSTEM_PROMPT = (
    "Translate the artist biography to Russian, keep it concise, no commentary."
)
TRANSLATE_MAX_TOKENS = 600

# file_cache columns -> put_track base-dict keys (contract field names; the
# DB column is `track`, the meta_cache contract calls it `title`).
_TRACK_FIELDS: tuple[tuple[str, str], ...] = (
    ("artist", "artist"),
    ("track", "title"),
    ("album", "album"),
    ("artwork_url", "artwork_url"),
    ("year", "year"),
    ("genre", "genre"),
    ("bio", "bio"),
    ("file_id", "file_id"),
)
# Columns never copied into the base dict: video_id is the cache key (passed
# separately), updated_at is row bookkeeping, not metadata.


# --------------------------------------------------------------------------
# Row helpers (rows are sqlite3.Row or dict: both have .keys() and [])
# --------------------------------------------------------------------------

def _row_get(row: Mapping, column: str) -> Any:
    """Value of `column` when the row has it, else None."""
    return row[column] if column in row.keys() else None


def _txt(value: Any) -> str:
    """TEXT column -> stripped str ('' for NULL/absent)."""
    return str(value).strip() if value is not None else ""


def _cap(seq: Sequence, limit: int | None) -> list:
    return list(seq)[: max(0, limit)] if limit is not None else list(seq)


# --------------------------------------------------------------------------
# Pure mapping logic (tested directly)
# --------------------------------------------------------------------------

def track_base_from_row(row: Mapping) -> dict | None:
    """file_cache row -> put_track base dict (contract field names, non-null
    fields only). None when both artist and title are empty: nothing
    identifies the track, so the row is skipped."""
    base: dict = {}
    for column, field in _TRACK_FIELDS:
        value = _txt(_row_get(row, column))
        if value:
            base[field] = value
    if not base.get("artist") and not base.get("title"):
        return None
    return base


def covers_plan(rows: Iterable[Mapping]) -> dict:
    """Distinct (artist, album) pairs from file_cache rows -> download plan.

    targets: unique (artist, album, artwork_url) with a non-empty artist and
             a known URL, first-seen order (the first non-empty URL wins for
             a pair seen first without one);
    missing_url: unique (artist, album) pairs with no known URL in any row
             (nothing to download -> reported as skips);
    no_artist: rows with an empty artist (no cache key).
    album NULL/empty normalizes to None - the key for album-less pairs.
    """
    targets: list[tuple[str, str | None, str]] = []
    seen: set[tuple[str, str | None]] = set()
    missing_url: list[tuple[str, str | None]] = []
    missing_seen: set[tuple[str, str | None]] = set()
    no_artist = 0
    for row in rows:
        artist = _txt(_row_get(row, "artist"))
        if not artist:
            no_artist += 1
            continue
        album = _txt(_row_get(row, "album")) or None
        url = _txt(_row_get(row, "artwork_url"))
        pair = (artist, album)
        if not url:
            if pair not in seen and pair not in missing_seen:
                missing_seen.add(pair)
                missing_url.append(pair)
            continue
        if pair in missing_seen:  # an earlier row had no URL, this one does
            missing_seen.discard(pair)
            missing_url.remove(pair)
        if pair not in seen:
            seen.add(pair)
            targets.append((artist, album, url))
    return {"targets": targets, "missing_url": missing_url,
            "no_artist": no_artist}


def bio_pool(rows: Iterable[Mapping], limit: int | None = None) -> list[str]:
    """Distinct non-empty artists from file_cache rows, first-seen order,
    capped at `limit` (the shared pool for --bio and --bio-ru)."""
    pool: list[str] = []
    seen: set[str] = set()
    for row in rows:
        artist = _txt(_row_get(row, "artist"))
        if artist and artist not in seen:
            seen.add(artist)
            pool.append(artist)
    return _cap(pool, limit)


# --------------------------------------------------------------------------
# Seed passes (pure logic + awaited cache calls; `mc` is services.meta_cache)
# --------------------------------------------------------------------------

async def seed_file_cache(rows: Iterable[Mapping], mc: Any) -> dict:
    """file_cache rows -> mc.put_track(video_id, base). Skips rows where
    artist and title are both empty (or video_id is missing)."""
    counts = {"rows": 0, "seeded": 0, "skipped": 0}
    for row in rows:
        counts["rows"] += 1
        video_id = _txt(_row_get(row, "video_id"))
        base = track_base_from_row(row)
        if not video_id or base is None:
            counts["skipped"] += 1
            continue
        await mc.put_track(video_id, base)
        counts["seeded"] += 1
    return counts


async def seed_facts(rows: Iterable[Mapping], mc: Any) -> dict:
    """facts rows -> mc.put_fact(artist, track, lang, fact); lang passes
    through verbatim. Skips rows with an empty artist, track or fact."""
    counts = {"rows": 0, "seeded": 0, "skipped": 0}
    for row in rows:
        counts["rows"] += 1
        artist = _txt(_row_get(row, "artist"))
        track = _txt(_row_get(row, "track"))
        fact = _txt(_row_get(row, "fact"))
        if not (artist and track and fact):
            counts["skipped"] += 1
            continue
        await mc.put_fact(artist, track, _row_get(row, "lang"), fact)
        counts["seeded"] += 1
    return counts


# --------------------------------------------------------------------------
# Network-mode runners (callables injected; tests pass offline fakes)
# --------------------------------------------------------------------------

async def run_covers(plan: dict, mc: Any, pause: float = COVERS_PAUSE_SEC,
                     sleep: Callable[[float], Any] = asyncio.sleep) -> dict:
    """Artwork pass over covers_plan output. hit = artwork already cached
    (artwork_path returned a path), miss = stored, skip = no URL known or no
    artist, failed = store_artwork raised (tolerated; the next run retries).
    Pause only between actual download attempts."""
    hits = misses = failed = 0
    for artist, album, url in plan["targets"]:
        try:
            path = await mc.artwork_path(artist, album)
        except Exception:
            path = None
        if path:
            hits += 1
            continue
        try:
            await mc.store_artwork(artist, album, url)
            misses += 1
        except Exception:
            failed += 1
        await sleep(pause)
    skipped = len(plan["missing_url"]) + plan["no_artist"]
    return {"hit": hits, "miss": misses, "failed": failed, "skip": skipped}


async def run_bio(pool: Sequence[str], mc: Any,
                  fetch_bio: Callable[[str], Any],
                  limit: int | None = None, pause: float = BIO_PAUSE_SEC,
                  sleep: Callable[[float], Any] = asyncio.sleep) -> dict:
    """EN-bio pass: fetch only for artists without a cached EN bio
    (get_bio(artist, 'en') is None); put_bio on a non-empty result.
    fetch_bio is an async (artist) -> str (Last.fm artist.getInfo summary,
    MetadataService._artist_getinfo). warm artists cost no network and no
    pause; the pause lands between actual fetch attempts."""
    artists = _cap(pool, limit)
    warm = saved = empty = failed = 0
    for artist in artists:
        try:
            cached = await mc.get_bio(artist, "en")
        except Exception:
            cached = None
        if cached:
            warm += 1
            continue
        try:
            bio = _txt(await fetch_bio(artist))
        except Exception:
            failed += 1
            await sleep(pause)
            continue
        if bio:
            from services.name_clean import strip_bio_disambiguation
            bio = strip_bio_disambiguation(bio) or bio
            await mc.put_bio(artist, "en", bio)
            saved += 1
        else:
            empty += 1
        await sleep(pause)
    return {"pool": len(artists), "warm": warm, "saved": saved,
            "empty": empty, "failed": failed}


async def select_ru_targets(pool: Sequence[str],
                            mc: Any) -> list[tuple[str, str]]:
    """(artist, en_bio) for artists with a cached EN bio and no RU one."""
    targets: list[tuple[str, str]] = []
    for artist in pool:
        from services.name_clean import strip_bio_disambiguation
        en = strip_bio_disambiguation(await mc.get_bio(artist, "en"))
        if not en:
            continue
        if await mc.get_bio(artist, "ru"):
            continue
        targets.append((artist, en))
    return targets


async def run_bio_ru(pool: Sequence[str], mc: Any,
                     translate: Callable[[str], Any] | None = None,
                     limit: int | None = None, pause: float = BIO_RU_PAUSE_SEC,
                     sleep: Callable[[float], Any] = asyncio.sleep,
                     dry: bool = False,
                     log: Callable[[str], None] = print) -> dict:
    """EN->RU pass. translate is an async (en_bio) -> str | None
    (BothubClient.chat_text with TRANSLATE_SYSTEM_PROMPT). dry lists the
    targets via `log` and never calls translate; empty/None results are
    skipped without a put."""
    artists = _cap(pool, limit)
    targets = await select_ru_targets(artists, mc)
    if dry:
        for artist, en in targets:
            log(f"would translate: {artist} (EN bio {len(en)} chars)")
        return {"pool": len(artists), "dry_listed": len(targets),
                "translated": 0, "skipped_empty": 0, "failed": 0}
    if translate is None:
        raise ValueError("run_bio_ru needs a translate callable without --dry")
    translated = skipped = failed = 0
    for artist, en in targets:
        try:
            ru = _txt(await translate(en))
        except Exception:
            failed += 1
            await sleep(pause)
            continue
        if not ru:
            skipped += 1
        else:
            await mc.put_bio(artist, "ru", ru)
            translated += 1
        await sleep(pause)
    return {"pool": len(artists), "dry_listed": 0, "translated": translated,
            "skipped_empty": skipped, "failed": failed}


# --------------------------------------------------------------------------
# DB access (read-only: SELECTs only)
# --------------------------------------------------------------------------

def open_db(path: str) -> sqlite3.Connection:
    """Open the bot SQLite DB read-only (URI mode=ro; falls back to a normal
    connection for WAL setups whose -shm/-wal need write access)."""
    try:
        conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.OperationalError:
        conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def file_cache_rows(db_path: str) -> list[sqlite3.Row]:
    with contextlib.closing(open_db(db_path)) as db:
        return db.execute("SELECT * FROM file_cache").fetchall()


def facts_rows(db_path: str) -> list[sqlite3.Row]:
    with contextlib.closing(open_db(db_path)) as db:
        try:
            return db.execute("SELECT * FROM facts").fetchall()
        except sqlite3.OperationalError:
            print("--seed facts: table missing in this DB, skipped",
                  file=sys.stderr)
            return []


def default_db_path(cli_value: str | None) -> str:
    """--db > MSM_DB env > config.DATA_DIR/'music-search.db'."""
    return cli_value or os.getenv(DB_ENV) or str(DATA_DIR / "music-search.db")


# --------------------------------------------------------------------------
# Thin async main around the pure parts
# --------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    from services import meta_cache as mc  # lazy: contract module

    db_path = default_db_path(args.db)
    if not args.stats and not Path(db_path).exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 2

    await mc.init()
    svc = bothub = None
    rc = 0
    try:
        if args.seed:
            tracks = await seed_file_cache(file_cache_rows(db_path), mc)
            facts = await seed_facts(facts_rows(db_path), mc)
            print(f"--seed file_cache: {tracks['rows']} rows, "
                  f"{tracks['seeded']} seeded, {tracks['skipped']} skipped "
                  f"(empty artist+title)")
            print(f"--seed facts: {facts['rows']} rows, "
                  f"{facts['seeded']} seeded, {facts['skipped']} skipped "
                  f"(empty fields)")

        if args.covers:
            plan = covers_plan(file_cache_rows(db_path))
            counts = await run_covers(plan, mc)
            print(f"--covers: {len(plan['targets'])} pair(s) with URL, "
                  f"hit {counts['hit']}, miss {counts['miss']}, "
                  f"skip {counts['skip']}, failed {counts['failed']}")

        if args.bio:
            from config import load_config
            from services.metadata import MetadataService
            cfg = load_config()
            if not cfg.lastfm_api_key:
                print("LASTFM_API_KEY is empty - --bio needs it "
                      "(config.load_config)", file=sys.stderr)
                rc = 2
            else:
                svc = MetadataService(cfg.lastfm_api_key)  # like bot/main.py

                async def _fetch_bio(artist: str) -> str:
                    info = await svc._artist_getinfo(artist)  # 24h TTL cache
                    return _txt((info or {}).get("bio"))

                pool = bio_pool(file_cache_rows(db_path))
                counts = await run_bio(pool, mc, _fetch_bio,
                                       limit=args.limit)
                print(f"--bio: pool {counts['pool']}, warm {counts['warm']}, "
                      f"saved {counts['saved']}, no-bio {counts['empty']}, "
                      f"failed {counts['failed']}")

        if args.bio_ru:
            pool = bio_pool(file_cache_rows(db_path), limit=args.limit)
            if args.dry:
                counts = await run_bio_ru(pool, mc, dry=True)
                print(f"--bio-ru --dry: {counts['dry_listed']} artist(s) "
                      f"would be translated (pool {counts['pool']})")
            else:
                from config import load_config
                from services.llm_search import BothubClient
                cfg = load_config()
                bothub = BothubClient(cfg.bothub_base, cfg.bothub_api_key,
                                       cfg.llm_model,
                                       sonar_model=cfg.llm_model_sonar)

                async def _translate(text: str) -> str | None:
                    return await bothub.chat_text(
                        TRANSLATE_SYSTEM_PROMPT, text,
                        max_tokens=TRANSLATE_MAX_TOKENS)

                counts = await run_bio_ru(pool, mc, translate=_translate,
                                          limit=args.limit)
                print(f"--bio-ru: translated {counts['translated']}, "
                      f"empty {counts['skipped_empty']}, "
                      f"failed {counts['failed']} (pool {counts['pool']})")

        if args.stats:
            st = mc.stats()
            if hasattr(st, "__await__"):  # tolerate an async stats() impl
                st = await st
            if isinstance(st, Mapping):
                st = json.dumps(st, ensure_ascii=False, default=str)
            print(f"--stats: {st}")
    finally:
        if svc is not None:
            await svc.aclose()
        if bothub is not None:
            await bothub.aclose()
        # The aiosqlite worker thread is non-daemon: without this the
        # one-shot process hangs at interpreter shutdown.
        if hasattr(mc, "aclose"):
            await mc.aclose()
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.warm_cache",
        description="Pre-fill the persistent metadata cache "
                    "(services.meta_cache). Modes compose; with no mode flag "
                    "this prints help.")
    parser.add_argument("--seed", action="store_true",
                        help="file_cache -> put_track, facts -> put_fact "
                             "(read-only DB pass, no network)")
    parser.add_argument("--covers", action="store_true",
                        help="download missing (artist, album) artwork via "
                             "store_artwork (network, 0.3s pause)")
    parser.add_argument("--bio", action="store_true",
                        help="fetch EN bios via Last.fm for artists without "
                             "one (network, 0.35s pause)")
    parser.add_argument("--bio-ru", action="store_true",
                        help="translate cached EN bios to RU via LLM "
                             "(costs cents; runs only when flagged)")
    parser.add_argument("--stats", action="store_true",
                        help="print meta_cache stats()")
    parser.add_argument("--db", metavar="PATH", default=None,
                        help=f"bot SQLite DB path (default: ${DB_ENV} env or "
                             f"{DATA_DIR / 'music-search.db'})")
    parser.add_argument("--limit", type=int, metavar="N", default=None,
                        help="cap the artist pool for --bio/--bio-ru")
    parser.add_argument("--dry", action="store_true",
                        help="with --bio-ru: list targets without calling "
                             "the LLM")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.seed or args.covers or args.bio or args.bio_ru
            or args.stats):
        build_parser().print_help()
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
