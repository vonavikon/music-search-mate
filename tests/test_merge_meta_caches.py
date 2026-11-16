"""Tests for scripts/merge_meta_caches.py (2026-11-14).

One-shot merge of the web site meta-cache into the bot meta-cache. Two
temp databases are built by hand with the documented schemas (bot is the
receiver, site is the source) and the script is exercised through the
real CLI (subprocess): dry-run must not touch the receiver file, --apply
must back it up first and merge by last_seen freshness, the site `plays`
product metric must never travel, and a source without meta_lyrics must
be handled honestly ("0 rows", no crash).
"""

import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_meta_caches.py"
DAY = 86400.0

# Bot (receiver) cache schema, as documented for the merge (2026-11-14).
BOT_SCHEMA = """
CREATE TABLE meta_track (
    video_id TEXT PRIMARY KEY,
    base TEXT,
    created_ts REAL,
    last_seen REAL,
    hits INTEGER
);
CREATE TABLE meta_artist (
    artist_key TEXT,
    lang TEXT,
    bio TEXT,
    created_ts REAL,
    last_seen REAL,
    hits INTEGER,
    PRIMARY KEY (artist_key, lang)
);
CREATE TABLE meta_fact (
    artist_key TEXT,
    track_key TEXT,
    lang TEXT,
    fact TEXT,
    created_ts REAL,
    last_seen REAL,
    hits INTEGER,
    PRIMARY KEY (artist_key, track_key, lang)
);
CREATE TABLE art_index (
    artist TEXT,
    album TEXT,
    file TEXT,
    last_seen REAL,
    PRIMARY KEY (artist, album)
);
"""

# Site (source) cache schema: one bio row per artist, lyrics included,
# `plays` is a site product metric.
SITE_SCHEMA = """
CREATE TABLE meta_track (
    video_id TEXT PRIMARY KEY,
    base TEXT,
    created_ts REAL,
    last_seen REAL,
    hits INTEGER,
    plays INTEGER
);
CREATE TABLE meta_artist (
    key TEXT PRIMARY KEY,
    bio_en TEXT,
    bio_ru TEXT,
    created_ts REAL,
    last_seen REAL,
    hits INTEGER
);
CREATE TABLE meta_fact (
    artist TEXT,
    track TEXT,
    lang TEXT,
    fact TEXT,
    created_ts REAL,
    last_seen REAL,
    hits INTEGER,
    PRIMARY KEY (artist, track, lang)
);
CREATE TABLE meta_lyrics (
    artist TEXT,
    track TEXT,
    data TEXT,
    created_ts REAL,
    last_seen REAL,
    hits INTEGER,
    PRIMARY KEY (artist, track)
);
CREATE TABLE art_index (
    artist TEXT,
    album TEXT,
    file TEXT,
    last_seen REAL,
    PRIMARY KEY (artist, album)
);
"""


def _load_script_module():
    """Import the standalone script by path (it must not import the bot)."""
    spec = importlib.util.spec_from_file_location("merge_meta_caches", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MMC = _load_script_module()


def make_db(path, schema):
    conn = sqlite3.connect(path)
    conn.executescript(schema)
    conn.commit()
    return conn


def make_pair(tmp_path):
    """(src_path, dst_path): an empty site cache and an empty bot cache."""
    src = tmp_path / "web-meta-cache.db"
    dst = tmp_path / "bot-meta-cache.db"
    make_db(src, SITE_SCHEMA).close()
    make_db(dst, BOT_SCHEMA).close()
    return src, dst


def run_merge(src, dst, apply=False):
    cmd = [sys.executable, str(SCRIPT), "--from", str(src), "--to", str(dst)]
    if apply:
        cmd.append("--apply")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", env=env)


def report_line(stdout, table):
    """The 'table: key=value ...' report line for one table."""
    for line in stdout.splitlines():
        if line.startswith(table + ":"):
            return line
    raise AssertionError(f"no {table} report line in output:\n{stdout}")


def tokens(line):
    parsed = {}
    for part in line.split():
        if "=" in part:
            key, value = part.split("=", 1)
            parsed[key] = value
    return parsed


def rows(path, sql, params=()):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------- normalize_key is a faithful standalone copy ----------

def test_normalize_key_matches_bot_runtime():
    """The script's copy must behave exactly like services.meta_cache."""
    from services.meta_cache import normalize_key as bot_normalize_key
    samples = ["Metallica", "Ｍｅｔａｌｌｉｃａ", "Ёлка", "елка",
               "  Foo   Bar ", "Björk", "P!nk", "Кино", "", None]
    for sample in samples:
        assert MMC.normalize_key(sample) == bot_normalize_key(sample)


# ---------- bio conversion ----------

def test_bio_converted_to_two_rows_empty_bio_skipped(tmp_path):
    src, dst = make_pair(tmp_path)
    now = time.time()
    conn = sqlite3.connect(src)
    # Artist one: only bio_en (bio_ru NULL) -> single 'en' row.
    conn.execute("INSERT INTO meta_artist VALUES (?,?,?,?,?,?)",
                 ("Radiohead", "english band bio", None, now - 50 * DAY,
                  now - 5 * DAY, 1))
    # Artist two: empty-string bio_en, real bio_ru -> single 'ru' row.
    conn.execute("INSERT INTO meta_artist VALUES (?,?,?,?,?,?)",
                 ("Кино", "", "русское био", now - 40 * DAY, now - 4 * DAY, 2))
    conn.commit()
    conn.close()

    proc = run_merge(src, dst, apply=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    got = rows(dst, "SELECT artist_key, lang, bio, hits FROM meta_artist "
                    "ORDER BY artist_key, lang")
    assert got == [("radiohead", "en", "english band bio", 1),
                   ("кино", "ru", "русское био", 2)]

    parsed = tokens(report_line(proc.stdout, "meta_artist"))
    assert parsed["source_rows"] == "2"
    assert parsed["taken"] == "2"
    assert "empty bios skipped: 2" in report_line(proc.stdout, "meta_artist")


def test_bio_conflict_fresher_last_seen_wins_both_directions(tmp_path):
    src, dst = make_pair(tmp_path)
    now = time.time()
    conn = sqlite3.connect(src)
    # Fresher site row: its bio must win over the stale bot bio.
    conn.execute("INSERT INTO meta_artist VALUES (?,?,?,?,?,?)",
                 ("Artist A", "site bio A", "сайт био A", now - 100 * DAY,
                  now - 1 * DAY, 3))
    # Staler site row: the bot bio must stay.
    conn.execute("INSERT INTO meta_artist VALUES (?,?,?,?,?,?)",
                 ("Artist B", "site bio B", "сайт био B", now - 100 * DAY,
                  now - 30 * DAY, 3))
    conn.commit()
    conn.close()
    conn = sqlite3.connect(dst)
    conn.execute("INSERT INTO meta_artist VALUES (?,?,?,?,?,?)",
                 ("artist a", "en", "bot bio A", now - 300 * DAY,
                  now - 10 * DAY, 2))
    conn.execute("INSERT INTO meta_artist VALUES (?,?,?,?,?,?)",
                 ("artist b", "ru", "bot bio B", now - 300 * DAY,
                  now - 2 * DAY, 2))
    conn.commit()
    conn.close()

    proc = run_merge(src, dst, apply=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    got = rows(dst, "SELECT artist_key, lang, bio, created_ts, last_seen, hits "
                    "FROM meta_artist ORDER BY artist_key, lang")
    # (2026-11-14) conflict rules: content from the fresher row,
    # created_ts=min, last_seen=max, hits=a+b.
    assert got == [
        ("artist a", "en", "site bio A", now - 300 * DAY, now - 1 * DAY, 5),
        ("artist a", "ru", "сайт био A", now - 100 * DAY, now - 1 * DAY, 3),
        ("artist b", "en", "site bio B", now - 100 * DAY, now - 30 * DAY, 3),
        ("artist b", "ru", "bot bio B", now - 300 * DAY, now - 2 * DAY, 5),
    ]

    parsed = tokens(report_line(proc.stdout, "meta_artist"))
    assert parsed["conflicts"] == "2"
    assert parsed["bot_wins"] == "1"
    assert parsed["src_wins"] == "1"
    assert parsed["updated"] == "2"


# ---------- tracks ----------

def test_track_conflict_fresher_json_wins_plays_never_copied(tmp_path):
    src, dst = make_pair(tmp_path)
    now = time.time()
    conn = sqlite3.connect(src)
    # v1: site is fresher -> site JSON wins. v2: bot is fresher -> bot JSON stays.
    conn.execute("INSERT INTO meta_track VALUES (?,?,?,?,?,?)",
                 ("v1", json.dumps({"title": "site one"}), now - 90 * DAY,
                  now - 1 * DAY, 5, 77))
    conn.execute("INSERT INTO meta_track VALUES (?,?,?,?,?,?)",
                 ("v2", json.dumps({"title": "site two"}), now - 90 * DAY,
                  now - 30 * DAY, 1, 9))
    conn.commit()
    conn.close()
    conn = sqlite3.connect(dst)
    conn.execute("INSERT INTO meta_track VALUES (?,?,?,?,?)",
                 ("v1", json.dumps({"title": "bot one"}), now - 200 * DAY,
                  now - 10 * DAY, 2))
    conn.execute("INSERT INTO meta_track VALUES (?,?,?,?,?)",
                 ("v2", json.dumps({"title": "bot two"}), now - 200 * DAY,
                  now - 2 * DAY, 4))
    conn.commit()
    conn.close()

    proc = run_merge(src, dst, apply=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    got = rows(dst, "SELECT video_id, base, last_seen, hits FROM meta_track "
                    "ORDER BY video_id")
    assert got == [
        ("v1", json.dumps({"title": "site one"}), now - 1 * DAY, 7),
        ("v2", json.dumps({"title": "bot two"}), now - 2 * DAY, 5),
    ]

    # The plays product metric never travels: no plays column appears.
    cols = {r[1] for r in rows(dst, "PRAGMA table_info(meta_track)")}
    assert "plays" not in cols
    assert "skipped product metrics" in proc.stdout
    assert "77" not in proc.stdout  # values are never printed (PII)

    parsed = tokens(report_line(proc.stdout, "meta_track"))
    assert parsed["conflicts"] == "2"
    assert parsed["bot_wins"] == "1"
    assert parsed["src_wins"] == "1"


# ---------- facts ----------

def test_fact_conflict_hits_summed_last_seen_max(tmp_path):
    src, dst = make_pair(tmp_path)
    now = time.time()
    conn = sqlite3.connect(src)
    conn.execute("INSERT INTO meta_fact VALUES (?,?,?,?,?,?,?)",
                 ("Artist X", "Track X", "en", "site fact", now - 200 * DAY,
                  now - 1 * DAY, 4))
    conn.commit()
    conn.close()
    conn = sqlite3.connect(dst)
    conn.execute("INSERT INTO meta_fact VALUES (?,?,?,?,?,?,?)",
                 ("artist x", "track x", "en", "bot fact", now - 100 * DAY,
                  now - 10 * DAY, 3))
    conn.commit()
    conn.close()

    proc = run_merge(src, dst, apply=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    got = rows(dst, "SELECT fact, created_ts, last_seen, hits FROM meta_fact")
    assert got == [("site fact", now - 200 * DAY, now - 1 * DAY, 7)]

    parsed = tokens(report_line(proc.stdout, "meta_fact"))
    assert parsed["conflicts"] == "1"
    assert parsed["src_wins"] == "1"
    assert parsed["taken"] == "0"


# ---------- lyrics ----------

def test_lyrics_table_created_rows_moved_idempotent(tmp_path):
    src, dst = make_pair(tmp_path)
    now = time.time()
    conn = sqlite3.connect(src)
    conn.execute("INSERT INTO meta_lyrics VALUES (?,?,?,?,?,?)",
                 ("Artist L", "Track L", json.dumps({"plain": "la la"}),
                  now - 50 * DAY, now - 5 * DAY, 0))
    conn.execute("INSERT INTO meta_lyrics VALUES (?,?,?,?,?,?)",
                 ("Artist L", "Track M", json.dumps({"plain": "na na"}),
                  now - 50 * DAY, now - 5 * DAY, 0))
    conn.commit()
    conn.close()
    src_hash_before = file_hash(src)

    proc = run_merge(src, dst, apply=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "receiver table created" in report_line(proc.stdout, "meta_lyrics")

    # Table exists in the receiver, rows moved with normalized keys.
    tables = {r[0] for r in rows(dst, "SELECT name FROM sqlite_master "
                                      "WHERE type='table'")}
    assert "meta_lyrics" in tables
    got = rows(dst, "SELECT artist_key, track_key, data_json FROM meta_lyrics "
                    "ORDER BY track_key")
    assert got == [("artist l", "track l", json.dumps({"plain": "la la"})),
                   ("artist l", "track m", json.dumps({"plain": "na na"}))]

    def dump(path):
        return sorted(rows(path, "SELECT name, sql FROM sqlite_master")) + \
            rows(path, "SELECT * FROM meta_lyrics")

    snapshot = dump(dst)

    # Second apply of the same source: zero changes, identical database.
    proc2 = run_merge(src, dst, apply=True)
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    parsed = tokens(report_line(proc2.stdout, "meta_lyrics"))
    assert parsed["taken"] == "0"
    assert parsed["updated"] == "0"
    assert dump(dst) == snapshot

    # The source file is never modified.
    assert file_hash(src) == src_hash_before


# ---------- TTL ----------

def test_rows_expired_361_days_not_moved(tmp_path):
    src, dst = make_pair(tmp_path)
    now = time.time()
    conn = sqlite3.connect(src)
    conn.execute("INSERT INTO meta_track VALUES (?,?,?,?,?,?)",
                 ("old", "{}", now - 400 * DAY, now - 361 * DAY, 1, 5))
    conn.execute("INSERT INTO meta_track VALUES (?,?,?,?,?,?)",
                 ("fresh", "{}", now - 10 * DAY, now - 359 * DAY, 1, 5))
    conn.execute("INSERT INTO meta_artist VALUES (?,?,?,?,?,?)",
                 ("Old Artist", "bio", "био", now - 400 * DAY,
                  now - 361 * DAY, 1))
    conn.execute("INSERT INTO meta_fact VALUES (?,?,?,?,?,?,?)",
                 ("Old Artist", "Old Track", "en", "fact", now - 400 * DAY,
                  now - 361 * DAY, 1))
    conn.execute("INSERT INTO meta_lyrics VALUES (?,?,?,?,?,?)",
                 ("Old Artist", "Old Track", "{}", now - 400 * DAY,
                  now - 361 * DAY, 0))
    conn.commit()
    conn.close()

    proc = run_merge(src, dst, apply=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Only the fresh track travelled.
    assert rows(dst, "SELECT video_id FROM meta_track") == [("fresh",)]
    assert rows(dst, "SELECT artist_key FROM meta_artist") == []
    assert rows(dst, "SELECT artist_key FROM meta_fact") == []
    assert rows(dst, "SELECT artist_key FROM meta_lyrics") == []

    for table in ("meta_track", "meta_artist", "meta_fact", "meta_lyrics"):
        parsed = tokens(report_line(proc.stdout, table))
        assert parsed["expired"] == "1", table
    assert tokens(report_line(proc.stdout, "meta_track"))["taken"] == "1"


# ---------- backup ----------

def test_backup_created_only_in_apply(tmp_path):
    src, dst = make_pair(tmp_path)
    now = time.time()
    conn = sqlite3.connect(src)
    conn.execute("INSERT INTO meta_track VALUES (?,?,?,?,?,?)",
                 ("v1", "{}", now - 10 * DAY, now - 1 * DAY, 1, 0))
    conn.commit()
    conn.close()
    conn = sqlite3.connect(dst)
    conn.execute("INSERT INTO meta_artist VALUES (?,?,?,?,?,?)",
                 ("artist a", "en", "bot bio", now - 10 * DAY, now - 1 * DAY, 1))
    conn.commit()
    conn.close()
    receiver_before = dst.read_bytes()

    def backups():
        return sorted(dst.parent.glob(dst.name + ".bak-*"))

    dry = run_merge(src, dst, apply=False)
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert backups() == []

    applied = run_merge(src, dst, apply=True)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    made = backups()
    assert len(made) == 1
    # The backup is taken before any write: it holds the pre-merge bytes.
    assert made[0].read_bytes() == receiver_before
    # The live receiver did change (the site row arrived).
    assert dst.read_bytes() != receiver_before


# ---------- dry-run purity ----------

def test_dry_run_changes_nothing(tmp_path):
    src, dst = make_pair(tmp_path)
    now = time.time()
    conn = sqlite3.connect(src)
    conn.execute("INSERT INTO meta_track VALUES (?,?,?,?,?,?)",
                 ("v1", "{}", now - 10 * DAY, now - 1 * DAY, 1, 3))
    conn.commit()
    conn.close()

    hash_before = file_hash(dst)
    stat_before = dst.stat()

    dry = run_merge(src, dst, apply=False)
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert tokens(report_line(dry.stdout, "meta_track"))["taken"] == "1"

    assert file_hash(dst) == hash_before
    assert dst.stat().st_mtime_ns == stat_before.st_mtime_ns
    # No journal side files, no backups.
    assert list(dst.parent.glob(dst.name + "-*")) == []
    assert list(dst.parent.glob(dst.name + ".bak-*")) == []


# ---------- hostile source ----------

def test_source_without_meta_lyrics_table_reports_zero_rows(tmp_path):
    src, dst = make_pair(tmp_path)
    conn = sqlite3.connect(src)
    conn.execute("DROP TABLE meta_lyrics")
    conn.execute("INSERT INTO meta_track VALUES (?,?,?,?,?,?)",
                 ("v1", "{}", time.time() - DAY, time.time() - DAY, 0, 1))
    conn.commit()
    conn.close()

    proc = run_merge(src, dst, apply=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = report_line(proc.stdout, "meta_lyrics")
    assert "0 rows" in line

    tables = {r[0] for r in rows(dst, "SELECT name FROM sqlite_master "
                                      "WHERE type='table'")}
    assert "meta_lyrics" not in tables  # nothing to merge -> no creation
    assert rows(dst, "SELECT video_id FROM meta_track") == [("v1",)]


def test_missing_source_nonzero_exit(tmp_path):
    _src, dst = make_pair(tmp_path)
    proc = run_merge(tmp_path / "nope.db", dst)
    assert proc.returncode == 2
    assert "not found" in proc.stderr


def test_corrupt_source_nonzero_exit(tmp_path):
    src, dst = make_pair(tmp_path)
    src.write_bytes(b"this is definitely not a sqlite database")
    proc = run_merge(src, dst)
    assert proc.returncode == 3
    assert "corrupt" in proc.stderr


# ---------- covers report ----------

def test_dry_run_lists_site_art_keys_missing_in_receiver(tmp_path):
    src, dst = make_pair(tmp_path)
    now = time.time()
    conn = sqlite3.connect(src)
    conn.execute("INSERT INTO art_index VALUES (?,?,?,?)",
                 ("Artist One", "Album X", "x.jpg", now))
    conn.execute("INSERT INTO art_index VALUES (?,?,?,?)",
                 ("Artist Two", "Album Y", "y.jpg", now))
    conn.commit()
    conn.close()
    conn = sqlite3.connect(dst)
    conn.execute("INSERT INTO art_index VALUES (?,?,?,?)",
                 ("artist one", "album x", "x.jpg", now))
    conn.commit()
    conn.close()

    proc = run_merge(src, dst, apply=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "site art keys: 2 total, 1 missing in receiver art_index" in proc.stdout
    assert "artist='artist two' album='album y'" in proc.stdout
    # The site art_index is never merged into the receiver.
    assert rows(dst, "SELECT artist, album FROM art_index") == \
        [("artist one", "album x")]
