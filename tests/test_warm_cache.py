"""Tests for scripts/warm_cache: row->cache mapping logic against a FakeMC.

The pure parts (track_base_from_row, covers_plan, bio_pool) and the async
runners (seed_file_cache, seed_facts, run_covers, run_bio, run_bio_ru) are
exercised with a contract double of services.meta_cache; network callables
are injected fakes. MetadataService/BothhubClient are never imported, so the
suite is hermetic (no network, no API keys, no services.meta_cache on disk).
"""
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # pyproject sets pythonpath=["."] too; explicit
    sys.path.insert(0, str(REPO))

from scripts import warm_cache  # noqa: E402

# Minimal real schema, copied from services/store.py @ HEAD (column lists
# verbatim; the DDL "IF NOT EXISTS" tail dropped for brevity).
FILE_CACHE_DDL = """
CREATE TABLE file_cache (
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
)
"""
FACTS_DDL = """
CREATE TABLE facts (
    artist TEXT NOT NULL,
    track TEXT NOT NULL,
    lang TEXT NOT NULL,
    fact TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (artist, track, lang)
)
"""


class FakeMC:
    """services.meta_cache contract double: plain dicts, async methods."""

    def __init__(self):
        self.tracks = {}      # video_id -> base dict
        self.bios = {}        # (artist, lang) -> bio
        self.facts = {}       # (artist, track, lang) -> fact
        self.artworks = {}    # (artist, album|None) -> cached path
        self.store_calls = []  # (artist, album, url)

    async def init(self):
        pass

    async def get_track(self, video_id):
        return self.tracks.get(video_id)

    async def put_track(self, video_id, base):
        self.tracks[video_id] = base

    async def get_bio(self, artist, lang):
        return self.bios.get((artist, lang))

    async def put_bio(self, artist, lang, bio):
        self.bios[(artist, lang)] = bio

    async def get_fact(self, artist, track, lang):
        return self.facts.get((artist, track, lang))

    async def put_fact(self, artist, track, lang, fact):
        self.facts[(artist, track, lang)] = fact

    async def artwork_path(self, artist, album):
        return self.artworks.get((artist, album))

    async def store_artwork(self, artist, album, url):
        self.store_calls.append((artist, album, url))
        self.artworks[(artist, album)] = f"/cache/{artist}/{album or '_'}.jpg"

    async def stats(self):
        return {"tracks": len(self.tracks), "bios": len(self.bios),
                "facts": len(self.facts), "artworks": len(self.artworks)}


def fc_row(video_id, artist=None, track=None, album=None, year=None,
           genre=None, bio=None, artwork_url=None, file_id="fid"):
    return {"video_id": video_id, "file_id": file_id, "artist": artist,
            "track": track, "album": album, "year": year, "genre": genre,
            "bio": bio, "artwork_url": artwork_url, "updated_at": 1.0}


def fact_row(artist, track, lang, fact):
    return {"artist": artist, "track": track, "lang": lang, "fact": fact,
            "updated_at": 1.0}


def make_db(tmp_path, file_cache=(), facts=()):
    db = sqlite3.connect(tmp_path / "bot.db")
    db.execute(FILE_CACHE_DDL)
    db.execute(FACTS_DDL)
    db.executemany(
        "INSERT INTO file_cache (video_id, file_id, artist, track, album,"
        " year, genre, bio, artwork_url, updated_at)"
        " VALUES (:video_id, :file_id, :artist, :track, :album, :year,"
        " :genre, :bio, :artwork_url, :updated_at)", list(file_cache))
    db.executemany(
        "INSERT INTO facts (artist, track, lang, fact, updated_at)"
        " VALUES (:artist, :track, :lang, :fact, :updated_at)", list(facts))
    db.commit()
    db.row_factory = sqlite3.Row
    return db


def fetch_rows(db, table):
    return db.execute(f"SELECT * FROM {table}").fetchall()


class SleepSpy:
    """Offline asyncio.sleep double that records the pauses."""

    def __init__(self):
        self.pauses = []

    async def __call__(self, seconds):
        self.pauses.append(seconds)


# -- seed: file_cache -> put_track -----------------------------------------

async def test_seed_file_cache_counts_and_field_mapping(tmp_path):
    db = make_db(tmp_path, file_cache=[
        fc_row("v1", artist="A", track="T1", album="Album", year="1999",
               genre="rock", artwork_url="https://a/1.jpg"),
        fc_row("v2", artist="B", track="T2"),  # most fields NULL
        fc_row("v3"),  # artist+title both empty -> skipped
    ])
    mc = FakeMC()
    counts = await warm_cache.seed_file_cache(fetch_rows(db, "file_cache"), mc)

    assert counts == {"rows": 3, "seeded": 2, "skipped": 1}
    assert set(mc.tracks) == {"v1", "v2"}
    # file_cache.track maps to the contract's `title`; NULL fields dropped
    assert mc.tracks["v1"] == {"artist": "A", "title": "T1", "album": "Album",
                               "artwork_url": "https://a/1.jpg",
                               "year": "1999", "genre": "rock", "file_id": "fid"}
    assert mc.tracks["v2"] == {"artist": "B", "title": "T2", "file_id": "fid"}


async def test_seed_file_cache_skips_only_when_artist_and_title_both_empty():
    mc = FakeMC()
    counts = await warm_cache.seed_file_cache([
        fc_row("v1", track="Title only"),   # no artist -> still seeded
        fc_row("v2", artist="Artist only"),  # no title -> still seeded
        fc_row("v3"),                        # both empty -> skipped
        fc_row("v4", artist="  ", track=""),  # whitespace == empty -> skipped
    ], mc)

    assert counts == {"rows": 4, "seeded": 2, "skipped": 2}
    assert set(mc.tracks) == {"v1", "v2"}


# -- seed: facts -> put_fact ------------------------------------------------

async def test_seed_facts_lang_passthrough(tmp_path):
    db = make_db(tmp_path, facts=[
        fact_row("A", "T", "en", "English fact"),
        fact_row("A", "T", "ru", "Русский факт"),
        fact_row("A", "T2", "en", ""),  # empty fact -> skipped
    ])
    mc = FakeMC()
    counts = await warm_cache.seed_facts(fetch_rows(db, "facts"), mc)

    assert counts == {"rows": 3, "seeded": 2, "skipped": 1}
    assert mc.facts == {("A", "T", "en"): "English fact",
                        ("A", "T", "ru"): "Русский факт"}  # lang verbatim


# -- covers -----------------------------------------------------------------

def test_covers_plan_dedupes_pairs_and_normalizes_album():
    plan = warm_cache.covers_plan([
        fc_row("v1", artist="A", album="X", artwork_url="https://a/x1.jpg"),
        fc_row("v2", artist="A", album="X", artwork_url="https://a/x2.jpg"),
        fc_row("v3", artist="A", artwork_url="https://a/na.jpg"),  # album NULL
        fc_row("v4", artist="B"),  # no URL anywhere -> missing_url
        fc_row("v5", track="T"),   # no artist -> dropped
    ])

    assert plan["targets"] == [("A", "X", "https://a/x1.jpg"),  # first URL wins
                               ("A", None, "https://a/na.jpg")]  # NULL album -> None
    assert plan["missing_url"] == [("B", None)]
    assert plan["no_artist"] == 1


async def test_run_covers_respects_artwork_path_hits():
    mc = FakeMC()
    await mc.store_artwork("A", "X", "https://a/x1.jpg")  # already cached
    mc.store_calls.clear()  # only calls made by run_covers matter below
    plan = warm_cache.covers_plan([
        fc_row("v1", artist="A", album="X", artwork_url="https://a/x1.jpg"),
        fc_row("v2", artist="B", album="Y", artwork_url="https://b/y.jpg"),
    ])
    spy = SleepSpy()

    counts = await warm_cache.run_covers(plan, mc, sleep=spy)

    assert counts == {"hit": 1, "miss": 1, "failed": 0, "skip": 0}
    # hit pair never re-stored; pause only after the actual download
    assert mc.store_calls == [("B", "Y", "https://b/y.jpg")]
    assert spy.pauses == [warm_cache.COVERS_PAUSE_SEC]


async def test_run_covers_tolerates_store_failures():
    mc = FakeMC()

    async def boom(artist, album, url):
        raise OSError("disk full")

    mc.store_artwork = boom
    plan = warm_cache.covers_plan([
        fc_row("v1", artist="A", album="X", artwork_url="https://a/x.jpg"),
        fc_row("v2", artist="B", album="Y", artwork_url="https://b/y.jpg"),
    ])

    counts = await warm_cache.run_covers(plan, mc, sleep=SleepSpy())

    assert counts == {"hit": 0, "miss": 0, "failed": 2, "skip": 0}


# -- bio pool / --bio -------------------------------------------------------

def test_bio_pool_distinct_and_limit():
    rows = [fc_row("v1", artist="B"), fc_row("v2", artist="A"),
            fc_row("v3", artist="B"), fc_row("v4", track="T")]
    assert warm_cache.bio_pool(rows) == ["B", "A"]
    assert warm_cache.bio_pool(rows, limit=1) == ["B"]


async def test_run_bio_fetches_only_missing():
    mc = FakeMC()
    await mc.put_bio("Warm", "en", "cached bio")

    async def fetch_bio(artist):
        if artist == "Boom":
            raise RuntimeError("lastfm down")
        return {"New": "fresh bio", "Empty": "  "}[artist]

    spy = SleepSpy()
    counts = await warm_cache.run_bio(
        ["Warm", "New", "Empty", "Boom"], mc, fetch_bio, sleep=spy)

    assert counts == {"pool": 4, "warm": 1, "saved": 1, "empty": 1,
                      "failed": 1}
    assert mc.bios == {("Warm", "en"): "cached bio", ("New", "en"): "fresh bio"}
    assert spy.pauses == [warm_cache.BIO_PAUSE_SEC] * 3  # warm = no pause


# --bio-ru ------------------------------------------------------------------

async def test_bio_ru_dry_lists_without_llm():
    mc = FakeMC()
    await mc.put_bio("A", "en", "A bio in english")
    await mc.put_bio("B", "en", "B bio in english")
    await mc.put_bio("B", "ru", "Био B")  # RU already there
    # "C" has no EN bio at all

    async def translate(text):
        raise AssertionError("LLM must not be called in --dry")

    listed = []
    counts = await warm_cache.run_bio_ru(
        ["A", "B", "C"], mc, translate=translate, dry=True,
        log=listed.append)

    assert counts["dry_listed"] == 1 and counts["translated"] == 0
    assert len(listed) == 1 and "A" in listed[0] and "B" not in listed[0]
    assert ("A", "ru") not in mc.bios  # dry writes nothing


async def test_bio_ru_translates_and_skips_empty_results():
    mc = FakeMC()
    await mc.put_bio("A", "en", "A bio")
    await mc.put_bio("B", "en", "B bio")

    async def translate(text):
        return {"A bio": "   ", "B bio": "Био B"}[text]  # A -> empty result

    spy = SleepSpy()
    counts = await warm_cache.run_bio_ru(["A", "B"], mc, translate=translate,
                                         sleep=spy)

    assert counts == {"pool": 2, "dry_listed": 0, "translated": 1,
                      "skipped_empty": 1, "failed": 0}
    assert mc.bios == {("A", "en"): "A bio", ("B", "en"): "B bio",
                       ("B", "ru"): "Био B"}
    assert spy.pauses == [warm_cache.BIO_RU_PAUSE_SEC] * 2


# -- misc -------------------------------------------------------------------

def test_default_db_path_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("MSM_DB", str(tmp_path / "env.db"))
    assert warm_cache.default_db_path(None) == str(tmp_path / "env.db")
    assert warm_cache.default_db_path("/cli/path.db") == "/cli/path.db"
    monkeypatch.delenv("MSM_DB")
    assert warm_cache.default_db_path(None).endswith("music-search.db")
