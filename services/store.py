"""SQLite logging of queries/candidates/feedback via aiosqlite.

Both a journal (observability) and a labeling dataset for rank improvement.
Every search = a row in queries; every "hit/miss" feedback = a row in feedback.
"""
import json
import time
import aiosqlite
from config import DB_PATH, DATA_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    user_id INTEGER NOT NULL,
    correlation_id TEXT,
    description TEXT NOT NULL,
    intent TEXT,
    signals_json TEXT,
    queries_json TEXT,
    candidates_json TEXT,
    best_video_id TEXT,
    best_title TEXT,
    best_score REAL,
    confidence TEXT,
    match_score REAL,
    auto_download INTEGER,
    ambiguous INTEGER,
    outcome TEXT,
    latency_ms INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    user_id INTEGER NOT NULL,
    query_id INTEGER,
    kind TEXT NOT NULL,
    video_id TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    query_id INTEGER REFERENCES queries(id),
    user_id INTEGER NOT NULL,
    video_id TEXT,
    title TEXT,
    feedback TEXT NOT NULL,
    shown_rank INTEGER
);
CREATE TABLE IF NOT EXISTS user_prefs (
    user_id INTEGER PRIMARY KEY,
    lang TEXT NOT NULL DEFAULT 'ru',
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    user_id INTEGER NOT NULL,
    day TEXT NOT NULL,             -- YYYY-MM-DD
    searches INTEGER DEFAULT 0,    -- LLM-поиски по описанию
    urls INTEGER DEFAULT 0,        -- скачивания по ссылке (без LLM)
    facts INTEGER DEFAULT 0,       -- sonar-факты по кнопке 💡 (заход 27)
    videos INTEGER DEFAULT 0,      -- скачанные видео (заход 27)
    bytes_sent INTEGER DEFAULT 0,  -- МБ отправленных файлов
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, day)
);
CREATE TABLE IF NOT EXISTS wtp_survey (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    user_id INTEGER NOT NULL,
    trigger_kind TEXT NOT NULL,    -- 'day' | 'global' | 'fact' | 'video' | 'degraded'
    feature TEXT NOT NULL,         -- 'search' | 'url'
    usage_24h INTEGER,
    response_kind TEXT,            -- 'button' | 'free' | 'optout' | 'ignored'
    stars_value REAL,
    free_text TEXT,
    shown_count INTEGER DEFAULT 1  -- сколько раз показан (анти-спам)
);
CREATE INDEX IF NOT EXISTS idx_wtp_user ON wtp_survey(user_id, created_at);
CREATE TABLE IF NOT EXISTS sessions (
    user_id INTEGER PRIMARY KEY,
    artist TEXT,
    track TEXT,
    video_id TEXT,
    last_query TEXT,
    shown_ids TEXT,
    last_active REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS query_state (
    query_id INTEGER PRIMARY KEY REFERENCES queries(id),
    state_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS file_cache (
    video_id TEXT PRIMARY KEY,
    file_id TEXT NOT NULL,
    artist TEXT,
    track TEXT,
    album TEXT,
    year TEXT,
    genre TEXT,
    bio TEXT,
    artwork_url TEXT,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    artist TEXT NOT NULL,
    track TEXT NOT NULL,
    lang TEXT NOT NULL,
    fact TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (artist, track, lang)
);
CREATE INDEX IF NOT EXISTS idx_queries_user ON queries(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_file_cache_updated ON file_cache(updated_at);
"""

# Live-session TTL: how long to keep the "current track" for "more like this".
SESSION_TTL = 20 * 60  # seconds (spec: 10-30 min, 20 chosen)

# Per-session cache of shown video_id (refinement excludes them for "no repeats").
MAX_SHOWN = 50  # cap so the cache doesn't grow unbounded in a long session


async def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        # WAL: the journal lives in a separate file — readers don't block the writer,
        # writes are batched. synchronous=NORMAL is safe with WAL (at most the last
        # commit is lost on power loss, not DB integrity). For a single-process bot
        # this removes "database is locked" stalls on concurrent read/write from different callbacks.
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.executescript(SCHEMA)
        await _migrate(db)
        await db.commit()


async def _migrate(db: aiosqlite.Connection) -> None:
    """Adds columns to existing tables.

    CREATE TABLE IF NOT EXISTS doesn't touch already-created tables, so newer
    columns (match_score/auto_download/ambiguous; shown_ids) are added separately
    on old DBs. A fresh DB gets them from SCHEMA — a no-op here."""
    async def _add_col(table: str, col: str, decl: str) -> None:
        cur = await db.execute(f"PRAGMA table_info({table})")
        names = {row[1] for row in await cur.fetchall()}
        if col not in names:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    await _add_col("queries", "match_score", "REAL")
    await _add_col("queries", "auto_download", "INTEGER")
    await _add_col("queries", "ambiguous", "INTEGER")
    await _add_col("sessions", "shown_ids", "TEXT")
    # file_cache.artwork_url: the column was once added by hand in prod only,
    # SCHEMA/migration/read-write lagged behind (cache hits without artwork).
    await _add_col("file_cache", "artwork_url", "TEXT")
    # Fact (the 💡 button) and video counters (per-item gates).
    # DEFAULT 0 is mandatory — without it ALTER yields NULL in existing rows,
    # incr NULL+1=NULL (the counter never grows), .get("facts",0) returns None.
    await _add_col("usage", "facts", "INTEGER DEFAULT 0")
    await _add_col("usage", "videos", "INTEGER DEFAULT 0")
    await _add_col("usage", "sends", "INTEGER DEFAULT 0")
    # Funnel: the outcome of every request (ok/clarify/empty/blocked/
    # limit/error/ssrf) + latency; historical rows = ok (found and logged)
    await _add_col("queries", "outcome", "TEXT")
    await _add_col("queries", "latency_ms", "INTEGER")
    await db.execute("UPDATE queries SET outcome='ok' WHERE outcome IS NULL")
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id, created_at)")
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_queries_day ON queries(created_at)")
    await db.execute(
        "UPDATE usage SET facts=0 WHERE facts IS NULL OR videos IS NULL")
    await db.execute(
        "UPDATE usage SET videos=0 WHERE videos IS NULL")


async def insert_query(
    *,
    user_id: int,
    correlation_id: str,
    description: str,
    intent: str | None,
    signals: dict | None,
    queries: list[str] | None,
    candidates: list[dict] | None,
    best_video_id: str | None,
    best_title: str | None,
    best_score: float | None,
    confidence: str | None,
    match_score: float | None = None,
    auto_download: bool | None = None,
    ambiguous: bool | None = None,
    outcome: str = "ok",
    latency_ms: int | None = None,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """INSERT INTO queries
               (created_at, user_id, correlation_id, description, intent,
                signals_json, queries_json, candidates_json,
                best_video_id, best_title, best_score, confidence,
                match_score, auto_download, ambiguous, outcome, latency_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), user_id, correlation_id, description, intent,
                json.dumps(signals, ensure_ascii=False) if signals else None,
                json.dumps(queries, ensure_ascii=False) if queries else None,
                json.dumps(candidates, ensure_ascii=False) if candidates else None,
                best_video_id, best_title, best_score, confidence,
                match_score,
                None if auto_download is None else int(auto_download),
                None if ambiguous is None else int(ambiguous),
                outcome, latency_ms,
            ),
        )
        await db.commit()
        return cur.lastrowid


async def insert_feedback(
    *,
    query_id: int,
    user_id: int,
    video_id: str | None,
    title: str | None,
    feedback: str,  # "hit" | "miss"
    shown_rank: int | None,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO feedback
               (created_at, query_id, user_id, video_id, title, feedback, shown_rank)
               VALUES (?,?,?,?,?,?,?)""",
            (time.time(), query_id, user_id, video_id, title, feedback, shown_rank),
        )
        await db.commit()


async def insert_event(
    *, user_id: int, kind: str,
    query_id: int | None = None, video_id: str | None = None,
) -> None:
    """A funnel event tied to a query.
    kind: dl_audio | cache_hit | dl_video | fact | similar | refine."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO events (created_at, user_id, query_id, kind, video_id)
                   VALUES (?,?,?,?,?)""",
                (time.time(), user_id, query_id, kind, video_id),
            )
            await db.commit()
    except Exception:
        pass  # telemetry must not break delivery


async def funnel_today() -> dict:
    """Today's funnel (the container's UTC day): query outcomes + events."""
    day0 = _day_start()
    out: dict = {"outcomes": {}, "events": {}}
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT outcome, COUNT(*) FROM queries WHERE created_at>=? GROUP BY outcome",
                (day0,),
            ) as cur:
                out["outcomes"] = {k or "ok": n for k, n in await cur.fetchall()}
            async with db.execute(
                "SELECT kind, COUNT(*) FROM events WHERE created_at>=? GROUP BY kind",
                (day0,),
            ) as cur:
                out["events"] = {k: n for k, n in await cur.fetchall()}
            async with db.execute(
                "SELECT feedback, COUNT(*) FROM feedback WHERE created_at>=? GROUP BY feedback",
                (day0,),
            ) as cur:
                out["feedback"] = {k: n for k, n in await cur.fetchall()}
    except Exception:
        pass
    return out


def _day_start() -> float:
    import datetime as _dt
    return _dt.datetime.combine(_dt.date.today(), _dt.time.min).timestamp()


async def count_recent(user_id: int, hours: int = 1) -> int:
    """How many searches the user made in the last N hours (rate limit)."""
    cutoff = time.time() - hours * 3600
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM queries WHERE user_id=? AND created_at>=?",
            (user_id, cutoff),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0


async def get_user_lang(user_id: int) -> str | None:
    """The user's interface language (asked on /start). None = not chosen."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT lang FROM user_prefs WHERE user_id=?", (user_id,)) as cur:
            row = await cur.fetchone()
            return row[0] if row else None


async def set_user_lang(user_id: int, lang: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_prefs (user_id, lang, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET lang=excluded.lang, updated_at=excluded.updated_at",
            (user_id, lang, time.time()))
        await db.commit()


async def get_usage(user_id: int, day: str | None = None) -> dict:
    """The user's usage counters for a day. day defaults to today."""
    import datetime as _dt
    day = day or _dt.date.today().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT searches, urls, facts, videos, sends, bytes_sent
               FROM usage WHERE user_id=? AND day=?""",
            (user_id, day)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else {
                "searches": 0, "urls": 0, "facts": 0, "videos": 0,
                "sends": 0, "bytes_sent": 0}


async def incr_usage(user_id: int, field: str, amount: int = 1) -> None:
    """Increment a usage counter (searches/urls/facts/videos/bytes_sent) with an UPSERT.
    The INSERT writes ALL counters (zeros explicit): facts/videos were added on prod
    via ALTER without DEFAULT, and a new-day row with a single field left NULLs in the
    rest (NULL+1=NULL, caps silently never fired). COALESCE in UPDATE additionally
    heals pre-existing NULL rows between restarts."""
    assert field in ("searches", "urls", "facts", "videos", "sends", "bytes_sent")
    import datetime as _dt
    day = _dt.date.today().isoformat()
    vals = {"searches": 0, "urls": 0, "facts": 0, "videos": 0,
            "sends": 0, "bytes_sent": 0}
    vals[field] = amount
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"""INSERT INTO usage (user_id, day, searches, urls, facts, videos,
                                  sends, bytes_sent, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, day) DO UPDATE SET
                {field} = COALESCE({field}, 0) + ?, updated_at = ?""",
            (user_id, day, vals["searches"], vals["urls"], vals["facts"],
             vals["videos"], vals["sends"], vals["bytes_sent"], time.time(),
             amount, time.time()))
        await db.commit()


async def get_global_usage(day: str | None = None) -> dict:
    """Total usage of ALL users for a day (global budget)."""
    import datetime as _dt
    day = day or _dt.date.today().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COALESCE(SUM(searches),0) as s, COALESCE(SUM(urls),0) as u, "
            "COALESCE(SUM(facts),0) as f, COALESCE(SUM(videos),0) as v, "
            "COALESCE(SUM(sends),0) as sd, "
            "COALESCE(SUM(bytes_sent),0) as b FROM usage WHERE day=?",
            (day,)) as cur:
            row = await cur.fetchone()
        return {"searches": row[0], "urls": row[1], "facts": row[2],
                "videos": row[3], "sends": row[4], "bytes_sent": row[5]}


# ══════ Sonar fact cache ══════
# A track fact doesn't change: the TTL is long, repeat requests (by the same or
# other users) are free. One sonar call = $0.008 — the cache is the main cost lever.

FACT_TTL = 30 * 24 * 3600  # 30 days


async def get_fact(artist: str, track: str, lang: str) -> str | None:
    """A cached fact or None (missing/expired). The key is normalized as on write."""
    key = (artist.strip().lower(), track.strip().lower(), lang)
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT fact, updated_at FROM facts WHERE artist=? AND track=? AND lang=?",
            key) as cur:
            row = await cur.fetchone()
    if not row or time.time() - row[1] > FACT_TTL:
        return None
    return row[0]


async def save_fact(artist: str, track: str, lang: str, fact: str) -> None:
    """UPSERT a fact. Key normalization = same as on read (case/whitespace)."""
    key = (artist.strip().lower(), track.strip().lower(), lang)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO facts (artist, track, lang, fact, updated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(artist, track, lang) DO UPDATE SET
                 fact=excluded.fact, updated_at=excluded.updated_at""",
            (*key, fact, time.time()))
        await db.commit()


# ══════ WTP survey ══════

async def should_show_survey(user_id: int) -> bool:
    """WTP survey anti-spam: at most once per 7 days, lifetime cap of 3 shows, optout forever."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # opted out forever
        async with db.execute(
            "SELECT COUNT(*) FROM wtp_survey WHERE user_id=? AND response_kind='optout'",
            (user_id,)) as cur:
            if (await cur.fetchone())[0] > 0:
                return False
        # lifetime cap of 3 shows
        async with db.execute(
            "SELECT COALESCE(SUM(shown_count),0) FROM wtp_survey WHERE user_id=?",
            (user_id,)) as cur:
            if (await cur.fetchone())[0] >= 3:
                return False
        # at most once per 7 days
        cutoff = time.time() - 7 * 86400
        async with db.execute(
            "SELECT COUNT(*) FROM wtp_survey WHERE user_id=? AND created_at>=?",
            (user_id, cutoff)) as cur:
            return (await cur.fetchone())[0] == 0


async def save_survey_shown(user_id: int, trigger_kind: str, feature: str,
                            usage_24h: int) -> None:
    """Record a survey shown (no response — it arrives separately via button/text)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO wtp_survey (created_at, user_id, trigger_kind, feature, usage_24h) "
            "VALUES (?,?,?,?,?)",
            (time.time(), user_id, trigger_kind, feature, usage_24h))
        await db.commit()


async def save_survey_response(user_id: int, response_kind: str,
                               stars_value: float | None = None,
                               free_text: str = "") -> None:
    """Record a WTP survey response (updates the latest shown record)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE wtp_survey SET response_kind=?, stars_value=?, free_text=?
               WHERE id = (SELECT id FROM wtp_survey WHERE user_id=? AND response_kind IS NULL
                          ORDER BY id DESC LIMIT 1)""",
            (response_kind, stars_value, free_text, user_id))
        await db.commit()


async def get_query_correlation(query_id: int) -> str | None:
    """The query's correlation_id = the Langfuse trace_id (v1.10). None for old/foreign ids."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT correlation_id FROM queries WHERE id=?", (query_id,),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else None


async def recent_context(user_id: int, limit: int = 3, window_min: int = 30) -> list[dict]:
    """The user's recent requests with outcomes, for conversation context.

    Returns [{description, intent, best_title, outcome}], chronological.
    outcome: 'hit' (a "hit" was given) | 'miss' (only "miss") | 'none' (no feedback).
    LLM#1 gets this as "previous context" to tie clarifications together
    ("the same but faster", "no, a different beer").
    """
    cutoff = time.time() - window_min * 60
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT q.description, q.intent, q.best_title, q.confidence, q.created_at AS ts,
                      (SELECT GROUP_CONCAT(f.feedback) FROM feedback f
                       WHERE f.query_id = q.id) AS fb
               FROM queries q
               WHERE q.user_id = ? AND q.created_at >= ?
               ORDER BY q.created_at DESC LIMIT ?""",
            (user_id, cutoff, limit),
        ) as cur:
            rows = await cur.fetchall()
    out = []
    for r in reversed(rows):  # chronological: oldest → newest
        fb = r["fb"] or ""
        outcome = "hit" if "hit" in fb else ("miss" if "miss" in fb else "none")
        out.append({
            "description": r["description"],
            "intent": r["intent"],
            "best_title": r["best_title"],
            "ts": r["ts"],
            "outcome": outcome,
        })
    return out


async def upsert_session(
    *,
    user_id: int,
    artist: str,
    track: str,
    video_id: str,
    last_query: str = "",
) -> None:
    """Remember the user's current track + accumulate its video_id into the shown cache.

    One row per user, overwritten on every track delivery. shown_ids accumulates
    the video_id of every delivered track of the session — refinement excludes them,
    so "another option" never returns what was already shown. Survives a process
    restart (unlike chat_data)."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT shown_ids FROM sessions WHERE user_id=?", (user_id,)
        )
        row = await cur.fetchone()
        shown = json.loads(row[0]) if row and row[0] else []
        if video_id not in shown:
            shown.append(video_id)
        shown = shown[-MAX_SHOWN:]
        await db.execute(
            """INSERT INTO sessions (user_id, artist, track, video_id, last_query, shown_ids, last_active)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET
                 artist=excluded.artist, track=excluded.track, video_id=excluded.video_id,
                 last_query=excluded.last_query, shown_ids=excluded.shown_ids,
                 last_active=excluded.last_active""",
            (user_id, artist, track, video_id, last_query, json.dumps(shown), time.time()),
        )
        await db.commit()


async def get_session(user_id: int, ttl: int = SESSION_TTL) -> dict | None:
    """The user's current track if the session is alive (last_active within ttl). None otherwise."""
    cutoff = time.time() - ttl
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT artist, track, video_id, last_query, shown_ids FROM sessions
               WHERE user_id=? AND last_active>=?""",
            (user_id, cutoff),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    return {
        "artist": row["artist"],
        "track": row["track"],
        "video_id": row["video_id"],
        "last_query": row["last_query"],
        "shown_ids": json.loads(row["shown_ids"]) if row["shown_ids"] else [],
    }


# ---------- persistent callback state + tg_file_id cache ----------

async def save_query_state(query_id: int, state_json: str) -> None:
    """Callback-button state (candidates + verdict/intro/desc/rejected_ids) — into the DB.

    chat_data currently keeps it in memory, so after a process restart the
    hit/miss/refine/similar/pick callback falls into "session expired". Mirror it
    into SQLite: chat_data stays the first (fast) cache, the DB is the source for
    restart recovery."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO query_state (query_id, state_json, updated_at)
               VALUES (?,?,?)
               ON CONFLICT(query_id) DO UPDATE SET
                 state_json=excluded.state_json, updated_at=excluded.updated_at""",
            (query_id, state_json, time.time()),
        )
        await db.commit()


async def load_query_state(query_id: int) -> str | None:
    """state_json for query_id, or None if the search wasn't saved or was deleted."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT state_json FROM query_state WHERE query_id=?", (query_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else None


async def delete_query_state(query_id: int) -> None:
    """Evict the state (track found / options exhausted) — the buttons are no longer needed."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM query_state WHERE query_id=?", (query_id,))
        await db.commit()


async def cache_file(
    *,
    video_id: str,
    file_id: str,
    artist: str = "",
    track: str = "",
    album: str = "",
    year: str = "",
    genre: str = "",
    bio: str = "",
    artwork_url: str = "",
) -> None:
    """tg_file_id by video_id: after the first upload TG returns a file_id, and the
    same file can be sent again without downloading (yt-dlp/ffmpeg) or re-uploading.
    Canonical metadata is stored alongside the file_id (for captions and the
    session) — it's stable across sends."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO file_cache (video_id, file_id, artist, track, album, year, genre, bio,
                                       artwork_url, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(video_id) DO UPDATE SET
                 file_id=excluded.file_id, artist=excluded.artist, track=excluded.track,
                 album=excluded.album, year=excluded.year, genre=excluded.genre,
                 bio=excluded.bio, artwork_url=excluded.artwork_url,
                 updated_at=excluded.updated_at""",
            (video_id, file_id, artist, track, album, year, genre, bio,
             artwork_url, time.time()),
        )
        await db.commit()


async def get_cached_file(video_id: str) -> dict | None:
    """The cached record (file_id + canonical metadata) or None."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT file_id, artist, track, album, year, genre, bio, artwork_url
               FROM file_cache WHERE video_id=?""",
            (video_id,),
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def update_cached_artwork(video_id: str, artwork_url: str) -> None:
    """Backfill artwork_url on old cache rows (created before the column was added
    by hand in prod only). A narrow UPDATE without re-upserting the row."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE file_cache SET artwork_url=? WHERE video_id=?",
            (artwork_url, video_id),
        )
        await db.commit()


# ══════ Personal data retention (audit) ══════
# queries.description and feedback store user texts indefinitely (and backups
# multiply them). Retention cuts by days; functional tables (file_cache, usage,
# sessions, user_prefs) are untouched. Expired facts are purged along the way.

async def purge_old_data(retention_days: int) -> dict:
    """Deletes queries/feedback older than retention_days, facts older than FACT_TTL.
    Returns counters for the log."""
    cutoff = time.time() - retention_days * 86400
    facts_cutoff = time.time() - FACT_TTL
    out = {}
    async with aiosqlite.connect(DB_PATH) as db:
        for table in ("queries", "feedback"):
            cur = await db.execute(
                f"DELETE FROM {table} WHERE created_at < ?", (cutoff,))
            out[table] = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        cur = await db.execute("DELETE FROM facts WHERE updated_at < ?", (facts_cutoff,))
        out["facts"] = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        await db.commit()
    return out

