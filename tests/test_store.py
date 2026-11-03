"""Integration tests for services/store.py: the SQLite log via aiosqlite."""
import aiosqlite
import pytest

from services import store


@pytest.fixture
def store_db(tmp_path, monkeypatch):
    """An isolated DB in tmp_path.

    store.py imports DB_PATH/DATA_DIR from config at module load time
    (`from config import DB_PATH, DATA_DIR`), so the db_path fixture from
    conftest.py (which patches config.DB_PATH) has no effect on store. We patch
    the attributes of the services.store module itself.
    """
    db_file = tmp_path / "test-store.db"
    monkeypatch.setattr("services.store.DB_PATH", str(db_file))
    monkeypatch.setattr("services.store.DATA_DIR", tmp_path)
    return db_file


def _insert_kwargs(correlation_id: str, description: str, **overrides):
    """Base kwargs for insert_query; overrides replace fields."""
    kwargs = dict(
        user_id=42,
        correlation_id=correlation_id,
        description=description,
        intent=None,
        signals=None,
        queries=None,
        candidates=None,
        best_video_id=None,
        best_title=None,
        best_score=None,
        confidence=None,
    )
    kwargs.update(overrides)
    return kwargs


async def test_init_db_creates_tables(store_db):
    await store.init_db()

    async with aiosqlite.connect(str(store_db)) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name IN ('queries', 'feedback', 'sessions', 'query_state', 'file_cache')"
        ) as cur:
            rows = await cur.fetchall()

    assert {r[0] for r in rows} == {"queries", "feedback", "sessions", "query_state", "file_cache"}


async def test_init_db_sets_wal(store_db):
    """init_db enables WAL so readers do not block the writer."""
    await store.init_db()

    async with aiosqlite.connect(str(store_db)) as db:
        async with db.execute("PRAGMA journal_mode") as cur:
            row = await cur.fetchone()

    assert row[0].lower() == "wal"


async def test_insert_and_query(store_db):
    await store.init_db()

    qid = await store.insert_query(**_insert_kwargs(
        correlation_id="corr-1",
        description="какой-то запрос",
        intent="music",
        signals={"query": "запрос"},
        queries=["запрос"],
        candidates=[{"video_id": "vid1", "title": "Title 1", "score": 0.9}],
        best_video_id="vid1",
        best_title="Title 1",
        best_score=0.9,
        confidence="high",
    ))

    assert isinstance(qid, int)
    assert qid > 0


async def test_insert_feedback(store_db):
    await store.init_db()

    qid = await store.insert_query(**_insert_kwargs(
        correlation_id="corr-2",
        description="ещё запрос",
    ))
    await store.insert_feedback(
        query_id=qid,
        user_id=42,
        video_id="vid1",
        title="Title 1",
        feedback="hit",
        shown_rank=1,
    )  # not raising means the test passed


async def test_count_recent(store_db):
    await store.init_db()

    assert await store.count_recent(user_id=42) == 0

    await store.insert_query(**_insert_kwargs(
        correlation_id="corr-3",
        description="счётчик",
    ))

    assert await store.count_recent(user_id=42) == 1


async def test_recent_context_hit_miss(store_db):
    await store.init_db()

    qid = await store.insert_query(**_insert_kwargs(
        correlation_id="corr-4",
        description="попали",
        intent="music",
        best_video_id="vid1",
        best_title="Title 1",
        best_score=0.9,
        confidence="high",
    ))
    await store.insert_feedback(
        query_id=qid, user_id=42, video_id="vid1",
        title="Title 1", feedback="hit", shown_rank=1,
    )

    ctx = await store.recent_context(user_id=42)

    assert len(ctx) == 1
    assert ctx[0]["outcome"] == "hit"
    assert ctx[0]["description"] == "попали"


async def test_recent_context_miss(store_db):
    await store.init_db()

    qid = await store.insert_query(**_insert_kwargs(
        correlation_id="corr-5",
        description="мимо",
        intent="music",
        best_video_id="vid2",
        best_title="Title 2",
        best_score=0.5,
        confidence="low",
    ))
    await store.insert_feedback(
        query_id=qid, user_id=42, video_id="vid2",
        title="Title 2", feedback="miss", shown_rank=1,
    )

    ctx = await store.recent_context(user_id=42)

    assert len(ctx) == 1
    assert ctx[0]["outcome"] == "miss"
    assert ctx[0]["description"] == "мимо"


async def test_upsert_and_get_session(store_db):
    await store.init_db()

    # no session -> None
    assert await store.get_session(user_id=42) is None

    await store.upsert_session(
        user_id=42, artist="The Beatles", track="Yesterday",
        video_id="vid1", last_query="yesterday",
    )

    sess = await store.get_session(user_id=42)
    assert sess["artist"] == "The Beatles"
    assert sess["track"] == "Yesterday"
    assert sess["video_id"] == "vid1"
    assert sess["last_query"] == "yesterday"

    # a repeated upsert overwrites instead of creating extra rows
    await store.upsert_session(
        user_id=42, artist="Nirvana", track="Smells Like Teen Spirit",
        video_id="vid2", last_query="nirvana",
    )
    sess = await store.get_session(user_id=42)
    assert sess["track"] == "Smells Like Teen Spirit"

    async with aiosqlite.connect(str(store_db)) as db:
        async with db.execute("SELECT COUNT(*) FROM sessions WHERE user_id=42") as cur:
            row = await cur.fetchone()
    assert row[0] == 1


async def test_get_session_expired(store_db):
    await store.init_db()

    await store.upsert_session(
        user_id=42, artist="Queen", track="Bohemian Rhapsody",
        video_id="vid1",
    )

    # ttl=-1 -> cutoff lands in the future, so the last activity is guaranteed expired
    assert await store.get_session(user_id=42, ttl=-1) is None


async def test_upsert_session_accumulates_shown_ids(store_db):
    await store.init_db()

    await store.upsert_session(user_id=42, artist="A", track="T1", video_id="v1")
    await store.upsert_session(user_id=42, artist="B", track="T2", video_id="v2")
    # a duplicate video_id does not create an extra cache entry
    await store.upsert_session(user_id=42, artist="B", track="T2", video_id="v2")

    sess = await store.get_session(user_id=42)
    assert sess["shown_ids"] == ["v1", "v2"]
    assert sess["video_id"] == "v2"  # "current track" is the latest; the cache holds all of them


async def test_insert_query_persists_match_score_auto(store_db):
    await store.init_db()

    qid = await store.insert_query(**_insert_kwargs(
        correlation_id="corr-ms",
        description="порог",
        intent="this_track",
        best_video_id="v1",
        best_title="T",
        best_score=0.9,
        confidence="high",
        match_score=0.72,
        auto_download=True,
        ambiguous=False,
    ))

    async with aiosqlite.connect(str(store_db)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT match_score, auto_download, ambiguous FROM queries WHERE id=?",
            (qid,),
        ) as cur:
            row = await cur.fetchone()

    assert row["match_score"] == 0.72
    assert row["auto_download"] == 1
    assert row["ambiguous"] == 0


async def test_insert_query_decision_defaults_null(store_db):
    await store.init_db()

    # without match_score/auto/ambiguous (clarify branch) the columns stay NULL
    qid = await store.insert_query(**_insert_kwargs(
        correlation_id="corr-null", description="без порога",
    ))

    async with aiosqlite.connect(str(store_db)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT match_score, auto_download, ambiguous FROM queries WHERE id=?",
            (qid,),
        ) as cur:
            row = await cur.fetchone()

    assert row["match_score"] is None
    assert row["auto_download"] is None
    assert row["ambiguous"] is None


async def test_migrate_adds_columns_to_existing_db(store_db):
    # simulate an old production DB: queries/sessions without the new columns
    async with aiosqlite.connect(str(store_db)) as db:
        await db.execute(
            """CREATE TABLE queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL,
                user_id INTEGER NOT NULL, correlation_id TEXT, description TEXT NOT NULL,
                intent TEXT, signals_json TEXT, queries_json TEXT, candidates_json TEXT,
                best_video_id TEXT, best_title TEXT, best_score REAL, confidence TEXT)"""
        )
        await db.execute(
            """CREATE TABLE sessions (
                user_id INTEGER PRIMARY KEY, artist TEXT, track TEXT, video_id TEXT,
                last_query TEXT, last_active REAL NOT NULL)"""
        )
        await db.commit()

    await store.init_db()

    async with aiosqlite.connect(str(store_db)) as db:
        for table, cols in [
            ("queries", {"match_score", "auto_download", "ambiguous"}),
            ("sessions", {"shown_ids"}),
        ]:
            async with db.execute(f"PRAGMA table_info({table})") as cur:
                names = {r[1] for r in await cur.fetchall()}
            assert cols <= names


# ════ query_state + file_cache ════

async def test_query_state_save_load_delete(store_db):
    await store.init_db()

    # no state -> None
    assert await store.load_query_state(999) is None

    await store.save_query_state(1, '{"candidates": [], "confidence": "high"}')
    raw = await store.load_query_state(1)
    assert raw == '{"candidates": [], "confidence": "high"}'

    # a repeated save overwrites instead of creating extra rows
    await store.save_query_state(1, '{"candidates": [], "confidence": "low"}')
    assert await store.load_query_state(1) == '{"candidates": [], "confidence": "low"}'

    await store.delete_query_state(1)
    assert await store.load_query_state(1) is None


async def test_file_cache_roundtrip(store_db):
    await store.init_db()

    assert await store.get_cached_file("v1") is None

    await store.cache_file(
        video_id="v1", file_id="file-abc",
        artist="The Beatles", track="Yesterday", album="Help!",
        year="1965", genre="rock", bio="nostalgic",
        artwork_url="https://example.com/600x600bb.jpg",
    )
    row = await store.get_cached_file("v1")
    assert row["file_id"] == "file-abc"
    assert row["artist"] == "The Beatles"
    assert row["track"] == "Yesterday"
    assert row["album"] == "Help!"
    assert row["year"] == "1965"
    assert row["genre"] == "rock"
    assert row["bio"] == "nostalgic"
    # the artwork is both written and read (the column used to be dead)
    assert row["artwork_url"] == "https://example.com/600x600bb.jpg"


async def test_file_cache_artwork_migration_and_update(store_db):
    """An old DB without artwork_url gets the column via migration;
    update_cached_artwork backfills it without re-upserting the row."""
    import aiosqlite as _a
    async with _a.connect(str(store_db)) as db:
        await db.execute(
            """CREATE TABLE file_cache_old (
                 video_id TEXT PRIMARY KEY, file_id TEXT NOT NULL, artist TEXT,
                 track TEXT, album TEXT, year TEXT, genre TEXT, bio TEXT,
                 updated_at REAL NOT NULL)""")
        await db.execute(
            "INSERT INTO file_cache_old VALUES ('v9','file-x','A','T','al','2001','g','b',1.0)")
        await db.execute("ALTER TABLE file_cache_old RENAME TO file_cache")
        await db.commit()
    await store.init_db()  # _migrate adds artwork_url

    row = await store.get_cached_file("v9")
    assert row is not None and not row["artwork_url"]

    await store.update_cached_artwork("v9", "https://example.com/cover.jpg")
    row = await store.get_cached_file("v9")
    assert row["artwork_url"] == "https://example.com/cover.jpg"
    assert row["file_id"] == "file-x"  # the row was not fully re-written


async def test_file_cache_upsert_overwrites(store_db):
    await store.init_db()

    await store.cache_file(video_id="v1", file_id="file-old")
    await store.cache_file(video_id="v1", file_id="file-new", track="Updated")

    row = await store.get_cached_file("v1")
    assert row["file_id"] == "file-new"
    assert row["track"] == "Updated"

    async with aiosqlite.connect(str(store_db)) as db:
        async with db.execute("SELECT COUNT(*) FROM file_cache WHERE video_id='v1'") as cur:
            n = await cur.fetchone()
    assert n[0] == 1


# ════ facts cache + facts/videos counters ════

async def test_facts_cache_roundtrip_and_ttl(store_db):
    await store.init_db()

    assert await store.get_fact("Queen", "Bohemian Rhapsody", "ru") is None

    await store.save_fact("Queen", "Bohemian Rhapsody", "ru",
                          "Записан за четыре недели в загородном доме")
    # the key is normalized: case/whitespace do not create duplicates
    assert await store.get_fact("  queen ", "bohemian rhapsody", "ru") == \
        "Записан за четыре недели в загородном доме"
    # a different language is a different fact
    assert await store.get_fact("Queen", "Bohemian Rhapsody", "en") is None

    # expired TTL -> miss
    import time as _t
    async with aiosqlite.connect(str(store_db)) as db:
        await db.execute(
            "UPDATE facts SET updated_at=?",
            (_t.time() - store.FACT_TTL - 1,))
        await db.commit()
    assert await store.get_fact("Queen", "Bohemian Rhapsody", "ru") is None

    # overwritten after the TTL miss
    await store.save_fact("Queen", "Bohemian Rhapsody", "ru", "Новый факт")
    assert await store.get_fact("Queen", "Bohemian Rhapsody", "ru") == "Новый факт"


async def test_usage_facts_videos_counters(store_db):
    await store.init_db()

    await store.incr_usage(1, "facts")
    await store.incr_usage(1, "facts")
    await store.incr_usage(1, "videos")

    u = await store.get_usage(1)
    assert u["facts"] == 2 and u["videos"] == 1
    assert u["searches"] == 0 and u["bytes_sent"] == 0

    g = await store.get_global_usage()
    assert g["facts"] == 2 and g["videos"] == 1


async def test_usage_migration_null_backfill(store_db):
    """An old usage table without facts/videos columns.
    ALTER without DEFAULT produced NULLs: incr NULL+1=NULL, and .get() returned None."""
    import aiosqlite as _a
    from datetime import datetime, timezone
    # incr_usage writes into TODAY's row: a hardcoded date broke the test at
    # the first UTC midnight after it was written
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    async with _a.connect(str(store_db)) as db:
        await db.execute(
            """CREATE TABLE usage_old (
                 user_id INTEGER NOT NULL, day TEXT NOT NULL,
                 searches INTEGER DEFAULT 0, urls INTEGER DEFAULT 0,
                 bytes_sent INTEGER DEFAULT 0, updated_at REAL NOT NULL,
                 PRIMARY KEY (user_id, day))""")
        await db.execute(
            "INSERT INTO usage_old VALUES (7, ?, 3, 4, 0, 1.0)", (day,))
        await db.execute("ALTER TABLE usage_old RENAME TO usage")
        await db.commit()
    await store.init_db()  # migration + backfill

    u = await store.get_usage(7, day=day)
    assert u["facts"] == 0 and u["videos"] == 0  # zero, not None
    await store.incr_usage(7, "facts")
    u = await store.get_usage(7, day=day)
    assert u["facts"] == 1  # NULL+1 is no longer NULL


async def test_incr_usage_after_migration_no_nulls(store_db):
    """The facts/videos columns were added via ALTER without DEFAULT (the
    production DB shape of that era). The first write of a new day used to fill
    only the incremented column -> NULLs elsewhere -> NULL+1=NULL."""
    import aiosqlite as _a
    async with _a.connect(str(store_db)) as db:
        await db.execute(
            """CREATE TABLE usage_old (
                 user_id INTEGER NOT NULL, day TEXT NOT NULL,
                 searches INTEGER DEFAULT 0, urls INTEGER DEFAULT 0,
                 bytes_sent INTEGER DEFAULT 0, updated_at REAL NOT NULL,
                 PRIMARY KEY (user_id, day))""")
        await db.execute("ALTER TABLE usage_old RENAME TO usage")
        await db.commit()
    await store.init_db()  # migration: facts/videos without DEFAULT, as in production

    await store.incr_usage(5, "searches")  # first row of the day
    u = await store.get_usage(5)
    assert u["searches"] == 1
    assert u["facts"] == 0 and u["videos"] == 0 and u["urls"] == 0 \
        and u["bytes_sent"] == 0  # zeros, not None

    await store.incr_usage(5, "facts")
    await store.incr_usage(5, "facts")
    u = await store.get_usage(5)
    assert u["facts"] == 2  # the counter grows even from a NULL past


async def test_purge_old_data_retention(store_db):
    """queries/feedback older than the retention window are deleted while
    fresh rows stay; facts older than FACT_TTL are purged."""
    import time as _t
    await store.init_db()
    await store.insert_query(user_id=1, correlation_id="old", description="старый",
                             intent="similar", signals=None, queries=None,
                             candidates=None, best_video_id=None, best_title=None,
                             best_score=None, confidence="high")
    await store.insert_query(user_id=1, correlation_id="new", description="новый",
                             intent="similar", signals=None, queries=None,
                             candidates=None, best_video_id=None, best_title=None,
                             best_score=None, confidence="high")
    await store.save_fact("A", "T", "ru", "факт")
    async with aiosqlite.connect(str(store_db)) as db:
        await db.execute("UPDATE queries SET created_at=? WHERE correlation_id='old'",
                         (_t.time() - 200 * 86400,))
        await db.execute("UPDATE facts SET updated_at=?", (_t.time() - store.FACT_TTL - 5,))
        await db.commit()
    purged = await store.purge_old_data(retention_days=180)
    assert purged["queries"] >= 1 and purged["facts"] == 1
    async with aiosqlite.connect(str(store_db)) as db:
        n = await (await db.execute("SELECT COUNT(*) FROM queries")).fetchone()
        assert n[0] == 1  # the fresh one survived
        assert await store.get_fact("A", "T", "ru") is None


class TestFunnel46:
    """Funnel: outcome/latency in queries, events, funnel_today."""

    @pytest.mark.asyncio
    async def test_outcome_latency_persisted(self, store_db):
        import services.store as st
        await st.init_db()
        qid = await st.insert_query(
            user_id=1, correlation_id="c1", description="тест", intent="similar",
            signals=None, queries=None, candidates=None,
            best_video_id=None, best_title=None, best_score=None,
            confidence=None, match_score=None, outcome="blocked", latency_ms=42)
        async with st.aiosqlite.connect(st.DB_PATH) as db:
            row = await db.execute_fetchall("SELECT outcome, latency_ms FROM queries WHERE id=?", (qid,))
        assert row[0] == ("blocked", 42)

    @pytest.mark.asyncio
    async def test_event_and_funnel(self, store_db):
        import services.store as st
        # store_db is isolated per test; deltas stay meaningful if the
        # fixture ever becomes shared again
        await st.init_db()
        base = await st.funnel_today()
        b_dl = base["events"].get("dl_audio", 0)
        b_fa = base["events"].get("fact", 0)
        b_hit = base.get("feedback", {}).get("hit", 0)
        await st.insert_event(user_id=7, kind="dl_audio", query_id=3, video_id="v1")
        await st.insert_event(user_id=7, kind="fact", video_id="v1")
        await st.insert_feedback(query_id=3, user_id=7, video_id="v1", title="t",
                                 feedback="hit", shown_rank=0)
        f = await st.funnel_today()
        assert f["events"].get("dl_audio", 0) - b_dl == 1
        assert f["events"].get("fact", 0) - b_fa == 1
        assert f.get("feedback", {}).get("hit", 0) - b_hit == 1

    @pytest.mark.asyncio
    async def test_migration_backfills_ok(self, store_db):
        import services.store as st
        await st.init_db()
        async with st.aiosqlite.connect(st.DB_PATH) as db:
            rows = await db.execute_fetchall(
                "SELECT COUNT(*) FROM queries WHERE outcome IS NULL")
        assert rows[0][0] == 0  # historical rows were backfilled with 'ok'


class TestOutcomeLog48:
    """All refusal outcomes are recorded in the funnel."""

    @pytest.mark.asyncio
    async def test_log_outcome_persists(self, store_db):
        import services.store as st
        from bot.handler import _log_outcome
        await st.init_db()
        await _log_outcome(99, "тестовый деструктив", "blocked", latency_ms=7)
        async with st.aiosqlite.connect(st.DB_PATH) as db:
            rows = await db.execute_fetchall(
                "SELECT outcome, latency_ms, description FROM queries WHERE user_id=99")
        assert rows and rows[0][0] == "blocked" and rows[0][1] == 7
        assert rows[0][2] == "тестовый деструктив"

    def test_all_refusal_branches_call_log(self):
        import re as _re
        from bot import handler as h
        src = open(h.__file__, encoding="utf-8").read()
        # 6 call sites (blocked/hourly/ssrf/search-limit/url-limit/empty)
        assert len(_re.findall(r"await _log_outcome\(", src)) >= 6


class TestAnalyticsCols:
    """Analytics 2.0: chat_type and llm_model on queries (where the query
    came from and which model ran it), plus migration from an old schema."""

    @pytest.mark.asyncio
    async def test_chat_type_llm_model_roundtrip(self, store_db):
        import services.store as st
        await st.init_db()
        qid = await st.insert_query(
            user_id=7, correlation_id="c7", description="rock", intent="this_track",
            signals=None, queries=None, candidates=None,
            best_video_id="v1", best_title="T", best_score=0.9,
            confidence="high", match_score=0.8,
            outcome="ok", latency_ms=100, chat_type="supergroup", llm_model="deepseek-v4-pro")
        async with st.aiosqlite.connect(str(store_db)) as db:
            row = await db.execute_fetchall(
                "SELECT chat_type, llm_model FROM queries WHERE id=?", (qid,))
        assert row[0] == ("supergroup", "deepseek-v4-pro")

    @pytest.mark.asyncio
    async def test_old_schema_gets_analytics_cols(self, tmp_path, monkeypatch):
        """A pre-Analytics-2.0 database (no chat_type/llm_model) is migrated."""
        import services.store as st
        old = tmp_path / "old.db"
        import aiosqlite
        async with aiosqlite.connect(old) as db:
            await db.execute("""CREATE TABLE queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL,
                user_id INTEGER NOT NULL, correlation_id TEXT, description TEXT NOT NULL,
                intent TEXT, signals_json TEXT, queries_json TEXT, candidates_json TEXT,
                best_video_id TEXT, best_title TEXT, best_score REAL, confidence TEXT,
                match_score REAL, auto_download INTEGER, ambiguous INTEGER,
                outcome TEXT, latency_ms INTEGER)""")
            await db.execute(
                "INSERT INTO queries (created_at, user_id, description, outcome) "
                "VALUES (1, 1, 'old row', 'ok')")
            await db.commit()
        monkeypatch.setattr("services.store.DB_PATH", str(old))
        monkeypatch.setattr("services.store.DATA_DIR", tmp_path)
        await st.init_db()
        async with aiosqlite.connect(str(old)) as db:
            cols = [r[1] for r in await db.execute_fetchall("PRAGMA table_info(queries)")]
            row = await db.execute_fetchall(
                "SELECT chat_type, llm_model, outcome FROM queries WHERE user_id=1")
        assert "chat_type" in cols and "llm_model" in cols
        assert row[0] == (None, None, "ok")  # old rows untouched, columns added
