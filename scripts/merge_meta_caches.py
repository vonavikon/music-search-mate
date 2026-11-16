#!/usr/bin/env python3
"""One-shot merge of the web site meta-cache into the bot meta-cache (2026-11-14).

The bot and the web site each kept their own metadata cache. From now on
the bot host owns the single shared cache. This script is run
once on the server: it takes the path to the main (bot) database and the
path to a copied site database file:

    python3 scripts/merge_meta_caches.py --from /tmp/web-meta-cache.db \
                                        --to /srv/bot/data/meta-cache.db
    python3 scripts/merge_meta_caches.py --from /tmp/web-meta-cache.db \
                                        --to /srv/bot/data/meta-cache.db --apply

Default mode is dry-run: nothing is written, the receiver is opened
read-only (its bytes, mtime and side files are untouched) and the script
prints what --apply would do. With --apply the receiver file is first
copied to {receiver}.bak-{YYYYmmdd-HHMMSS} (plus its -wal sibling when
present), then all writes run inside one transaction; any failure rolls
back and leaves the receiver untouched. The source is opened read-only
(mode=ro) and is never modified.

Merge rules, identical for every table (a conflict is an equal primary
key in the receiver):
  - last_seen = max(receiver, source); created_ts = min(receiver, source);
    hits = receiver + source;
  - the payload (base JSON / bio / fact / lyrics data) is taken from the
    row with the GREATER last_seen; on equal last_seen the receiver (bot)
    wins;
  - source rows with last_seen older than 360 days (the sliding TTL the
    bot runtime uses) are not transferred at all;
  - a source row that merges into exactly the receiver row it already
    produced writes nothing (idempotent re-runs report 0 changes).

Table mapping (site -> bot):
  meta_track  -> meta_track   payload column `base`/`base_json`; the site
                              `plays` column is a site product metric and
                              is NEVER transferred (no plays column is
                              created in the receiver); a site-only
                              `tracks_json` column has no receiver column
                              either and is likewise dropped;
  meta_artist -> meta_artist  the site stores ONE row per artist with
                              bio_en + bio_ru; the bot stores one row per
                              (artist_key, lang); one site row expands to
                              at most two receiver rows, empty bios
                              produce no row;
  meta_fact   -> meta_fact    keys are passed through normalize_key;
  meta_lyrics -> meta_lyrics  the bot has no lyrics table: it is created
                              here (artist_key, track_key, data, created_ts,
                              last_seen, hits) and the site rows are moved
                              with normalize_key keys; a site `data` JSON
                              column is copied verbatim, the web schema
                              (synced_lrc/plain/provider/duration_sec) is
                              packed into a JSON object;
  art_index   -> NOT merged   site cover files live in its own static/;
                              the report only counts and lists site
                              artist+album keys missing from the receiver
                              art_index so a human can rsync the files.

Column names are discovered with PRAGMA table_info, so both the documented
schemas and the schemas emitted by web/meta_cache.py and
services/meta_cache.py are accepted (base/base_json, key/artist_key,
artist/artist_key, track/track_key, optional hits/plays columns).

Report discipline (PII): only counters and artist/album keys are printed;
bio, fact, base JSON and lyrics contents are never logged.

Idempotency caveat: re-running --apply re-adds source hits on conflicts
(the a+b rule is stateless); rows that arrived with source hits=0 (the
web schema has no hits column for meta_fact/meta_lyrics) rewrite nothing.

Exit codes: 0 ok; 2 setup error (missing file, same file twice,
destination is not a bot meta-cache); 3 corrupt/unreadable database or a
failed write (rolled back).

Standalone by design: stdlib only, no bot imports. normalize_key below is
a verbatim copy of services/meta_cache.py:normalize_key so the script
runs on a bare server checkout without the bot runtime installed.
"""

import argparse
import dataclasses
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

# Sliding TTL of the bot runtime (META_CACHE_TTL_DAYS default): source rows
# older than this are not transferred (2026-11-14).
TTL_DAYS = 360.0
TTL_SEC = TTL_DAYS * 86400.0

# The lyrics table the script creates in the receiver; mirrors the shape of
# the other bot meta tables (PK + bookkeeping + a last_seen index).
LYRICS_DDL = (
    """
    CREATE TABLE IF NOT EXISTS meta_lyrics (
        artist_key TEXT NOT NULL,
        track_key   TEXT NOT NULL,
        data_json   TEXT NOT NULL,
        created_ts  REAL NOT NULL,
        last_seen   REAL NOT NULL,
        PRIMARY KEY (artist_key, track_key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_meta_lyrics_seen ON meta_lyrics(last_seen)",
)

# Web-schema lyrics columns packed into the receiver `data_json` payload
# (the column name follows the LIVE msm-cache service schema, 2026-11-14).
_LYRICS_WEB_COLS = ("synced_lrc", "plain", "provider", "duration_sec")

_WS_RE = re.compile(r"\s+")


def normalize_key(s: str) -> str:
    """Artist/track key: NFKC -> casefold -> collapse whitespace -> strip.

    Verbatim copy of services/meta_cache.py:normalize_key (2026-11-14):
    the merge must key site rows exactly like the bot runtime does, but
    the script stays runnable on a server without the bot package, so the
    function is duplicated here instead of imported. NFKC unifies unicode
    forms (fullwidth "Ｍｅｔａｌｌｉｃａ", diacritic compositions), casefold is
    more aggressive than lower (German ss), whitespace collapse glues
    superficial spelling variants.
    """
    norm = unicodedata.normalize("NFKC", s or "").casefold()
    return _WS_RE.sub(" ", norm).strip()


# ---------- small sqlite helpers ----------

def table_columns(conn: sqlite3.Connection, table: str) -> set:
    """Column names of `table`; empty set when the table does not exist."""
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def first_col(cols: set, names) -> str | None:
    """First of `names` present in `cols` (schemas name columns differently)."""
    for name in names:
        if name in cols:
            return name
    return None


def _min(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def _max(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if a >= b else b


def _expired(last_seen, now: float) -> bool:
    """Source row older than the sliding TTL (NULL last_seen counts as expired)."""
    return last_seen is None or (now - last_seen) > TTL_SEC


@dataclasses.dataclass
class TableReport:
    """Counters for one merged table; `line()` is the printable report row."""
    name: str
    src_rows: int = 0
    expired: int = 0
    taken: int = 0        # new primary keys inserted from the source
    conflicts: int = 0    # primary keys already present in the receiver
    bot_wins: int = 0     # conflicts where the receiver content stays
    src_wins: int = 0     # conflicts where the source content replaces it
    updated: int = 0      # conflicts whose merged row differs (written in --apply)
    skipped_empty: int = 0
    bad_keys: int = 0
    note: str = ""

    def line(self) -> str:
        core = (f"{self.name}: source_rows={self.src_rows} "
                f"expired={self.expired} taken={self.taken} "
                f"conflicts={self.conflicts} bot_wins={self.bot_wins} "
                f"src_wins={self.src_wins} updated={self.updated}")
        extras = []
        if self.skipped_empty:
            extras.append(f"empty bios skipped: {self.skipped_empty}")
        if self.bad_keys:
            extras.append(f"unusable keys skipped: {self.bad_keys}")
        if self.note:
            extras.append(self.note)
        if extras:
            core += "  [" + "; ".join(extras) + "]"
        return core


class ReceiverTable:
    """Receiver side of one mergeable table.

    SELECT/INSERT/UPDATE are built from the columns that actually exist in
    the receiver (payload column and the optional hits column differ
    between the documented and the runtime schemas). `merge` applies the
    shared conflict rules to one source row; writes are executed only when
    `write` is set (apply mode) AND the merged row differs from the stored
    one, which makes repeated runs report 0 changes.
    """

    def __init__(self, dst, table, payload_col, with_hits, exists=True):
        self.dst = dst
        self.table = table
        self.payload_col = payload_col
        self.with_hits = with_hits
        self.exists = exists  # False: dry-run before meta_lyrics creation

    def _select(self, pk_cols, pk):
        if not self.exists:
            return None
        sql = (f"SELECT {self.payload_col}, created_ts, last_seen"
               + (", hits" if self.with_hits else "")
               + f" FROM {self.table} WHERE "
               + " AND ".join(f"{c} = ?" for c in pk_cols))
        row = self.dst.execute(sql, pk).fetchone()
        if row is None:
            return None
        current = {"payload": row[0], "created_ts": row[1], "last_seen": row[2]}
        if self.with_hits:
            current["hits"] = row[3]
        return current

    def merge(self, pk_cols, pk, payload, s_created, s_last, s_hits,
              rep: TableReport, write: bool) -> None:
        current = self._select(pk_cols, pk)
        if current is None:
            # New primary key: the source row is taken as is.
            rep.taken += 1
            if write:
                cols = list(pk_cols) + [self.payload_col, "created_ts", "last_seen"]
                vals = list(pk) + [payload,
                                   s_created if s_created is not None else s_last,
                                   s_last]
                if self.with_hits:
                    cols.append("hits")
                    vals.append(s_hits or 0)
                self.dst.execute(
                    f"INSERT INTO {self.table} ({', '.join(cols)}) "
                    f"VALUES ({', '.join(['?'] * len(cols))})", vals)
            return
        # Conflict: bookkeeping is merged, the payload is taken from the
        # fresher row (equal last_seen -> the receiver, the bot, wins).
        rep.conflicts += 1
        src_fresher = (s_last or 0.0) > (current["last_seen"] or 0.0)
        if src_fresher:
            rep.src_wins += 1
        else:
            rep.bot_wins += 1
        merged = {
            "payload": payload if src_fresher else current["payload"],
            "created_ts": _min(current["created_ts"], s_created),
            "last_seen": _max(current["last_seen"], s_last),
        }
        if self.with_hits:
            merged["hits"] = (current.get("hits") or 0) + (s_hits or 0)
        if all(merged[k] == current[k] for k in merged):
            return  # already merged before: idempotent no-op
        rep.updated += 1
        if write:
            sets = [f"{self.payload_col} = ?", "created_ts = ?", "last_seen = ?"]
            vals = [merged["payload"], merged["created_ts"], merged["last_seen"]]
            if self.with_hits:
                sets.append("hits = ?")
                vals.append(merged["hits"])
            vals.extend(pk)
            self.dst.execute(
                f"UPDATE {self.table} SET {', '.join(sets)} WHERE "
                + " AND ".join(f"{c} = ?" for c in pk_cols), vals)


# ---------- per-table source readers ----------

def merge_tracks(src, dst, write: bool, now: float):
    """meta_track: PK video_id, JSON payload, site plays column dropped."""
    rep = TableReport("meta_track")
    scols = table_columns(src, "meta_track")
    payload = first_col(scols, ("base", "base_json"))
    if payload is None or not {"video_id", "created_ts", "last_seen"} <= scols:
        rep.note = "table absent or unusable in source, 0 rows"
        return rep, 0
    plays_col = first_col(scols, ("plays",))
    dcols = table_columns(dst, "meta_track")
    rt = ReceiverTable(dst, "meta_track", first_col(dcols, ("base", "base_json")),
                       "hits" in dcols)
    parts = ["video_id", payload, "created_ts", "last_seen",
             "hits" if "hits" in scols else "NULL",
             plays_col or "NULL"]
    plays_dropped = 0
    for video_id, blob, created, last, hits, plays in src.execute(
            f"SELECT {', '.join(parts)} FROM meta_track"):
        rep.src_rows += 1
        if plays is not None:
            plays_dropped += 1  # site product metric, counted and dropped
        if _expired(last, now):
            rep.expired += 1
            continue
        if not video_id:
            rep.bad_keys += 1
            continue
        rt.merge(("video_id",), (video_id,), blob, created, last, hits or 0,
                 rep, write)
    return rep, plays_dropped


def merge_artists(src, dst, write: bool, now: float):
    """meta_artist: site row (bio_en, bio_ru) -> two bot rows (artist_key, lang)."""
    rep = TableReport("meta_artist")
    scols = table_columns(src, "meta_artist")
    key_col = first_col(scols, ("key", "artist_key"))
    if key_col is None or not {"bio_en", "bio_ru", "created_ts", "last_seen"} <= scols:
        rep.note = "table absent or unusable in source, 0 rows"
        return rep
    dcols = table_columns(dst, "meta_artist")
    rt = ReceiverTable(dst, "meta_artist", "bio", "hits" in dcols)
    parts = [key_col, "bio_en", "bio_ru", "created_ts", "last_seen",
             "hits" if "hits" in scols else "NULL"]
    for key, bio_en, bio_ru, created, last, hits in src.execute(
            f"SELECT {', '.join(parts)} FROM meta_artist"):
        rep.src_rows += 1
        if _expired(last, now):
            rep.expired += 1
            continue
        artist_key = normalize_key(key or "")
        if not artist_key:
            rep.bad_keys += 1
            continue
        for lang, bio in (("en", bio_en), ("ru", bio_ru)):
            if not bio:  # empty bios never produce a receiver row
                rep.skipped_empty += 1
                continue
            rt.merge(("artist_key", "lang"), (artist_key, lang), bio,
                     created, last, hits or 0, rep, write)
    return rep


def merge_facts(src, dst, write: bool, now: float):
    """meta_fact: PK (artist_key, track_key, lang), keys via normalize_key."""
    rep = TableReport("meta_fact")
    scols = table_columns(src, "meta_fact")
    acol = first_col(scols, ("artist", "artist_key"))
    tcol = first_col(scols, ("track", "track_key"))
    if acol is None or tcol is None or not {"lang", "fact",
                                            "created_ts", "last_seen"} <= scols:
        rep.note = "table absent or unusable in source, 0 rows"
        return rep
    dcols = table_columns(dst, "meta_fact")
    rt = ReceiverTable(dst, "meta_fact", "fact", "hits" in dcols)
    parts = [acol, tcol, "lang", "fact", "created_ts", "last_seen",
             "hits" if "hits" in scols else "NULL"]
    for artist, track, lang, fact, created, last, hits in src.execute(
            f"SELECT {', '.join(parts)} FROM meta_fact"):
        rep.src_rows += 1
        if _expired(last, now):
            rep.expired += 1
            continue
        artist_key, track_key = normalize_key(artist or ""), normalize_key(track or "")
        if not artist_key or not track_key or not lang:
            rep.bad_keys += 1
            continue
        rt.merge(("artist_key", "track_key", "lang"), (artist_key, track_key, lang),
                 fact, created, last, hits or 0, rep, write)
    return rep


def merge_lyrics(src, dst, write: bool, now: float):
    """meta_lyrics: receiver table is created here, site rows moved with keys.

    A site `data` JSON column is copied verbatim; the web runtime schema
    (synced_lrc/plain/provider/duration_sec) is packed into a JSON object.
    """
    rep = TableReport("meta_lyrics")
    scols = table_columns(src, "meta_lyrics")
    acol = first_col(scols, ("artist", "artist_key"))
    tcol = first_col(scols, ("track", "track_key"))
    web_cols = [c for c in _LYRICS_WEB_COLS if c in scols]
    has_data = "data" in scols or "data_json" in scols
    usable = (acol is not None and tcol is not None
              and {"created_ts", "last_seen"} <= scols
              and (has_data or web_cols))
    if not usable:
        rep.note = "table absent or unusable in source, 0 rows"
        return rep
    dst_has = bool(table_columns(dst, "meta_lyrics"))
    if not dst_has:
        if write:
            for stmt in LYRICS_DDL:  # DDL is transactional in sqlite
                dst.execute(stmt)
            rep.note = "receiver table created"
        else:
            rep.note = "receiver table will be created"
    rx_cols = table_columns(dst, "meta_lyrics") if dst_has else []
    rt = ReceiverTable(dst, "meta_lyrics", "data_json",
                       "hits" in rx_cols, exists=dst_has or write)
    data_col = "data" if "data" in scols else "data_json"
    parts = [acol, tcol] + ([data_col] if has_data else web_cols) \
        + ["created_ts", "last_seen", "hits" if "hits" in scols else "NULL"]
    n_payload = 1 if has_data else len(web_cols)
    for row in src.execute(f"SELECT {', '.join(parts)} FROM meta_lyrics"):
        values = list(row)
        artist, track = values[0], values[1]
        rep.src_rows += 1
        created, last, hits = values[2 + n_payload], values[3 + n_payload], \
            values[4 + n_payload]
        if _expired(last, now):
            rep.expired += 1
            continue
        artist_key, track_key = normalize_key(artist or ""), normalize_key(track or "")
        if not artist_key or not track_key:
            rep.bad_keys += 1
            continue
        if has_data:
            payload = values[2]
        else:
            payload = json.dumps(
                dict(zip(web_cols, values[2:2 + n_payload])), ensure_ascii=False)
        rt.merge(("artist_key", "track_key"), (artist_key, track_key), payload,
                 created, last, hits or 0, rep, write)
    return rep


def art_diff(src, dst):
    """Site art_index keys missing in the receiver: (total, missing, note).

    Informational only: the site art_index is never merged, cover files
    sit in the site static/ and a human decides what to rsync.
    """
    scols = table_columns(src, "art_index")
    dcols = table_columns(dst, "art_index")
    sac = first_col(scols, ("artist", "artist_key"))
    sal = first_col(scols, ("album", "album_key"))
    dac = first_col(dcols, ("artist", "artist_key"))
    dal = first_col(dcols, ("album", "album_key"))
    if sac is None or sal is None:
        return 0, [], "art_index absent in source, nothing to compare"
    if dac is None or dal is None:
        return 0, [], "art_index absent in receiver, cannot compare"
    recv_keys = {(normalize_key(a or ""), normalize_key(b or ""))
                 for a, b in dst.execute(f"SELECT {dac}, {dal} FROM art_index")}
    total, missing = 0, []
    for a, b in src.execute(f"SELECT {sac}, {sal} FROM art_index"):
        total += 1
        key = (normalize_key(a or ""), normalize_key(b or ""))
        if key not in recv_keys:
            missing.append(key)
    return total, sorted(missing), ""


# ---------- validation, backup, report ----------

def open_ro(path: Path) -> sqlite3.Connection:
    """Read-only connection (mode=ro): the file cannot be modified through it."""
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def quick_check(conn: sqlite3.Connection, label: str) -> None:
    rows = conn.execute("PRAGMA quick_check").fetchall()
    bad = [str(r[0]) for r in rows if str(r[0]) != "ok"]
    if bad:
        raise sqlite3.DatabaseError("; ".join(bad[:3]))


def validate_receiver(dst: sqlite3.Connection) -> list:
    """The destination must look like the bot meta-cache (fail fast, exit 2)."""
    problems = []
    for table, payload_names in (("meta_track", ("base", "base_json")),
                                 ("meta_artist", ("bio",)),
                                 ("meta_fact", ("fact",))):
        cols = table_columns(dst, table)
        if not cols:
            problems.append(f"missing table {table}")
        elif first_col(cols, payload_names) is None:
            problems.append(f"table {table} has no payload column "
                            f"{'/'.join(payload_names)}")
    return problems


def backup_receiver(dst_file: Path):
    """Copy the receiver (and its -wal sibling) to {db}.bak-{timestamp}."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bak = dst_file.with_name(f"{dst_file.name}.bak-{stamp}")
    n = 0
    while bak.exists():
        n += 1
        bak = dst_file.with_name(f"{dst_file.name}.bak-{stamp}-{n}")
    shutil.copy2(dst_file, bak)
    wal = dst_file.with_name(dst_file.name + "-wal")
    wal_bak = None
    if wal.exists():
        wal_bak = Path(str(bak) + "-wal")
        shutil.copy2(wal, wal_bak)
    return bak, wal_bak


def _totals(reports) -> TableReport:
    total = TableReport("summary")
    for rep in reports:
        for field in ("src_rows", "expired", "taken", "conflicts",
                      "bot_wins", "src_wins", "updated"):
            setattr(total, field, getattr(total, field) + getattr(rep, field))
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="merge_meta_caches.py",
        description="One-shot merge of the web site meta-cache into the bot "
                    "meta-cache (2026-11-14). Dry-run by default.")
    parser.add_argument("--from", dest="src_path", required=True, metavar="SITE_DB",
                        help="copied web meta-cache.db (source; opened read-only, "
                             "never modified)")
    parser.add_argument("--to", dest="dst_path", required=True, metavar="BOT_DB",
                        help="bot meta-cache.db (receiver; becomes the shared cache)")
    parser.add_argument("--apply", action="store_true",
                        help="write the merge: back up the receiver first, then "
                             "write inside one transaction")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the merge plan without writing (default)")
    args = parser.parse_args(argv)
    if args.dry_run and args.apply:
        parser.error("--dry-run and --apply are mutually exclusive")

    src_file, dst_file = Path(args.src_path), Path(args.dst_path)
    if not src_file.is_file():
        print(f"error: source database not found: {src_file}", file=sys.stderr)
        return 2
    if not dst_file.is_file():
        print(f"error: destination database not found: {dst_file} "
              "(the bot cache must already exist; nothing is created "
              "from scratch)", file=sys.stderr)
        return 2
    if os.path.samefile(src_file, dst_file):
        print("error: --from and --to point to the same file", file=sys.stderr)
        return 2

    # --- open and validate both databases without writing anything ---
    try:
        src = open_ro(src_file)
        quick_check(src, "source")
    except sqlite3.Error as exc:
        print(f"error: source database is corrupt or not a database: "
              f"{src_file} ({exc})", file=sys.stderr)
        return 3
    try:
        dst = open_ro(dst_file)
        quick_check(dst, "destination")
    except sqlite3.OperationalError:
        # A WAL database without its -shm file cannot be opened read-only;
        # retry read-write with query_only (still zero writes).
        try:
            dst = sqlite3.connect(dst_file)
            dst.execute("PRAGMA query_only = 1")
            quick_check(dst, "destination")
        except sqlite3.Error as exc:
            print(f"error: destination database is corrupt: {dst_file} "
                  f"({exc})", file=sys.stderr)
            return 3
    except sqlite3.Error as exc:
        print(f"error: destination database is corrupt: {dst_file} ({exc})",
              file=sys.stderr)
        return 3

    problems = validate_receiver(dst)
    if problems:
        print(f"error: destination is not a bot meta-cache: "
              + "; ".join(problems), file=sys.stderr)
        return 2

    now = time.time()
    print("== merge meta caches (2026-11-14) ==")
    print(f"mode: {'apply (backup first, one transaction)'
                if args.apply else 'dry-run (default; no writes, receiver opened read-only)'}")
    print(f"source (site copy, read-only): {src_file}")
    print(f"receiver (bot cache): {dst_file}")

    if not args.apply:
        # --- dry-run: compute the whole merge against the read-only receiver ---
        track_rep, plays = merge_tracks(src, dst, False, now)
        reports = [track_rep,
                   merge_artists(src, dst, False, now),
                   merge_facts(src, dst, False, now),
                   merge_lyrics(src, dst, False, now)]
        _print_report(reports, plays, src, dst)
        print("dry-run complete: nothing has been written; "
              "re-run with --apply to merge")
        src.close()
        dst.close()
        return 0

    # --- apply: backup, then one transaction; any failure rolls back ---
    dst.close()
    try:
        backup_path, backup_wal = backup_receiver(dst_file)
    except OSError as exc:
        print(f"error: cannot back up the receiver: {exc}", file=sys.stderr)
        src.close()
        return 2
    dst_rw = sqlite3.connect(dst_file)
    dst_rw.isolation_level = None  # explicit transaction control
    try:
        dst_rw.execute("BEGIN IMMEDIATE")
        track_rep, plays = merge_tracks(src, dst_rw, True, now)
        reports = [track_rep,
                   merge_artists(src, dst_rw, True, now),
                   merge_facts(src, dst_rw, True, now),
                   merge_lyrics(src, dst_rw, True, now)]
        dst_rw.execute("COMMIT")
    except sqlite3.Error as exc:
        try:
            dst_rw.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        print(f"error: merge failed and was rolled back "
              f"(receiver untouched, backup kept): {exc}", file=sys.stderr)
        src.close()
        dst_rw.close()
        return 3
    _print_report(reports, plays, src, dst_rw)
    print(f"backup: {backup_path}" + (f" (+ {backup_wal})" if backup_wal else ""))
    print("apply complete: the receiver now owns the merged cache; "
          "the source was not modified")
    src.close()
    dst_rw.close()
    return 0


def _print_report(reports, plays_dropped, src, dst) -> None:
    """Counters and artist/album keys only - never row contents (PII)."""
    for rep in reports:
        print(rep.line())
    total, missing, note = art_diff(src, dst)
    print("covers (informational only, site art_index is never merged):")
    if note:
        print(f"  {note}")
    else:
        print(f"  site art keys: {total} total, {len(missing)} "
              f"missing in receiver art_index")
        for artist_key, album_key in missing:
            print(f"  missing cover key: artist='{artist_key}' album='{album_key}'")
    print(f"skipped product metrics: source plays values dropped for "
          f"{plays_dropped} rows (site product metric, never transferred; "
          f"no plays column is created in the receiver)")
    summary = _totals(reports)
    print(f"summary: source_rows={summary.src_rows} expired={summary.expired} "
          f"taken={summary.taken} conflicts={summary.conflicts} "
          f"bot_wins={summary.bot_wins} src_wins={summary.src_wins} "
          f"updated={summary.updated} written={summary.taken + summary.updated}")


if __name__ == "__main__":
    sys.exit(main())
