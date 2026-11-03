import asyncio
import pytest
from unittest.mock import patch
from services.search import _normalize, _normalize_sc, search_metadata, enrich_candidates


def test_normalize_sc_valid():
    entry = {
        "id": "sc123",
        "title": "Great Track",
        "uploader": "Cool Artist",
        "duration": 240,
        "webpage_url": "https://soundcloud.com/coolartist/great-track",
    }
    result = _normalize_sc(entry)
    assert result is not None
    assert result["video_id"] == "sc123"
    assert result["source"] == "soundcloud"
    assert result["url"] == "https://soundcloud.com/coolartist/great-track"
    assert result["channel"] == "Cool Artist"
    assert result["duration"] == 240


def test_normalize_sc_no_webpage_url():
    """Without webpage_url the track cannot be downloaded, so it is dropped."""
    entry = {"id": "sc123", "title": "Track", "uploader": "Artist"}
    assert _normalize_sc(entry) is None


def test_normalize_sc_no_title():
    entry = {"id": "sc123", "webpage_url": "https://..."}
    assert _normalize_sc(entry) is None


def test_normalize_sc_no_id():
    entry = {"title": "Track", "webpage_url": "https://sc.com/track"}
    assert _normalize_sc(entry) is None


@pytest.mark.asyncio
async def test_search_metadata_merges_sources():
    """YouTube + SoundCloud merge: YouTube is primary, SC becomes fallback_url."""
    yt_results = [{
        "video_id": "abc", "title": "Song Name", "uploader": "Artist",
        "duration": 200, "view_count": 1000,
    }]
    sc_results = [{
        "id": "sc1", "title": "Song Name", "uploader": "Artist",
        "duration": 200, "webpage_url": "https://sc.com/a/song",
    }]

    with patch("services.search._extract_one", return_value=yt_results):
        with patch("services.search._extract_one_sc", return_value=sc_results):
            results = await search_metadata(["test query"], per_query=5)

    assert len(results) == 1
    assert results[0]["source"] == "youtube"
    assert results[0]["fallback_url"] == "https://sc.com/a/song"


@pytest.mark.asyncio
async def test_search_metadata_sc_only():
    """A SoundCloud-only result (no YouTube duplicate) makes it into the output."""
    yt_results: list = []
    sc_results = [{
        "id": "sc2", "title": "Unique Track", "uploader": "Artist",
        "duration": 180, "webpage_url": "https://sc.com/a/unique",
    }]

    with patch("services.search._extract_one", return_value=yt_results):
        with patch("services.search._extract_one_sc", return_value=sc_results):
            results = await search_metadata(["unique query"], per_query=5)

    assert len(results) == 1
    assert results[0]["source"] == "soundcloud"
    assert results[0]["url"] == "https://sc.com/a/unique"
    assert results[0].get("fallback_url", "") == ""


@pytest.mark.asyncio
async def test_search_metadata_sc_failure_graceful():
    """SoundCloud fails — only YouTube results, no crash."""
    yt_results = [{"video_id": "abc", "title": "Song", "uploader": "Artist", "duration": 200, "view_count": 1000}]

    with patch("services.search._extract_one", return_value=yt_results):
        with patch("services.search._extract_one_sc", side_effect=Exception("SC down")):
            results = await search_metadata(["test"], per_query=5)

    assert len(results) == 1
    assert results[0]["source"] == "youtube"
    assert results[0].get("fallback_url", "") == ""


@pytest.mark.asyncio
async def test_search_metadata_timeout_graceful():
    """A hung ytsearch (TimeoutError from run_in_executor) yields an empty result instead of crashing."""
    with patch("services.search._extract_one", side_effect=asyncio.TimeoutError()):
        with patch("services.search._extract_one_sc", side_effect=asyncio.TimeoutError()):
            results = await search_metadata(["hang"], per_query=5)
    assert results == []


@pytest.mark.asyncio
async def test_enrich_candidates_timeout_graceful():
    """A hung extract_info (TimeoutError from run_in_executor) leaves the candidate without enriched fields."""
    candidates = [{"video_id": "abc", "title": "Song", "channel": "Artist", "duration": 200, "source": "youtube", "url": "https://..."}]

    with patch("services.search._extract_info_no_dl", side_effect=asyncio.TimeoutError()):
        result = await enrich_candidates(candidates, metadata_svc=None, top_n=1)

    assert "canonical_artist" not in result[0]


@pytest.mark.asyncio
async def test_enrich_candidates_ytdlp_extract():
    """enrich_candidates calls extract_info for the top candidates."""
    candidates = [{
        "video_id": "abc", "title": "Song (Official)", "channel": "Artist - Topic",
        "duration": 240, "views": 1000, "source": "youtube",
        "url": "https://www.youtube.com/watch?v=abc",
    }]

    fake_info = {
        "artist": "Artist Name",
        "track": "Song Name",
        "album": "Album Name",
        "release_date": "20230101",
    }

    with patch("services.search._extract_info_no_dl", return_value=fake_info):
        result = await enrich_candidates(candidates, metadata_svc=None, top_n=1)

    assert result[0]["canonical_artist"] == "Artist Name"
    assert result[0]["canonical_track"] == "Song Name"
    assert result[0]["canonical_album"] == "Album Name"
    assert result[0]["canonical_year"] == "2023"


@pytest.mark.asyncio
async def test_enrich_candidates_extract_fails_graceful():
    """On an extract_info error the candidate is returned without enriched fields."""
    candidates = [{"video_id": "abc", "title": "Song", "channel": "Artist", "duration": 200, "source": "youtube", "url": "https://..."}]

    with patch("services.search._extract_info_no_dl", side_effect=Exception("fail")):
        result = await enrich_candidates(candidates, metadata_svc=None, top_n=1)

    assert "canonical_artist" not in result[0]


@pytest.mark.asyncio
async def test_enrich_does_not_mutate_original():
    """The original candidate list is not mutated."""
    candidates = [{"video_id": "abc", "title": "Song", "channel": "Artist"}]
    original = dict(candidates[0])

    with patch("services.search._extract_info_no_dl", return_value={"artist": "X"}):
        await enrich_candidates(candidates, metadata_svc=None, top_n=1)

    assert candidates[0] == original


@pytest.mark.asyncio
async def test_enrich_respects_top_n():
    """Only the top-N candidates are enriched; the rest are left unchanged."""
    candidates = [
        {"video_id": f"id{i}", "title": f"Track {i}", "channel": "Artist", "source": "youtube",
         "url": f"https://www.youtube.com/watch?v=id{i}"}
        for i in range(10)
    ]

    fake_info = {"artist": "Artist", "track": "Track"}
    with patch("services.search._extract_info_no_dl", return_value=fake_info):
        result = await enrich_candidates(candidates, metadata_svc=None, top_n=3)

    # the first 3 are enriched
    for i in range(3):
        assert "canonical_artist" in result[i]
    # the remaining 7 are not
    for i in range(3, 10):
        assert "canonical_artist" not in result[i]


@pytest.mark.asyncio
async def test_enrich_candidates_stores_categories():
    """A full extract_info returning categories -> they are stored for the music filter."""
    candidates = [{
        "video_id": "abc", "title": "Song", "channel": "Artist - Topic",
        "duration": 240, "views": 1000, "source": "youtube",
        "url": "https://www.youtube.com/watch?v=abc",
    }]
    fake_info = {"artist": "Artist Name", "track": "Song", "categories": ["Music"]}

    with patch("services.search._extract_info_no_dl", return_value=fake_info):
        result = await enrich_candidates(candidates, metadata_svc=None, top_n=1)

    assert result[0]["categories"] == ["Music"]


# ════ yt-first with early exit + SC fallback ════

@pytest.mark.asyncio
async def test_early_exit_pool_target(monkeypatch):
    """Once the pool reaches _POOL_TARGET, remaining yt queries are cancelled and SC is not called."""
    from services import search as S
    calls = {"yt": 0, "sc": 0}

    def fake_yt(q, per_query):
        calls["yt"] += 1
        return [{"video_id": f"{q}-{i}", "title": f"T {q} {i}", "channel": f"Ch {q}",
                 "duration": 200, "views": 100} for i in range(30)]

    def fake_sc(q, per_query):
        calls["sc"] += 1
        return []

    monkeypatch.setattr(S, "_extract_one", fake_yt)
    monkeypatch.setattr(S, "_extract_one_sc", fake_sc)
    out = await S.search_metadata(["q1", "q2", "q3", "q4"], per_query=20)
    assert calls["sc"] == 0                     # SC skipped: pool already full from yt
    assert S._POOL_TARGET <= len(out) <= S._POOL_TARGET + S._WINDOW * 30
    assert calls["yt"] == S._WINDOW             # window: no extra queries fired, pool is full


@pytest.mark.asyncio
async def test_sc_fallback_when_yt_thin(monkeypatch):
    """YT fell short of _POOL_FLOOR — SoundCloud kicks in, solo candidates included in the output."""
    from services import search as S

    def fake_yt(q, per_query):
        return [{"video_id": f"yt-{q}", "title": f"Only One {q}", "channel": f"C {q}",
                 "duration": 200, "views": 100}]

    def fake_sc(q, per_query):
        return [{"title": f"SC Track {q}", "uploader": f"SC Artist {q}",
                 "duration": 210, "views": 5,
                 "webpage_url": f"https://soundcloud.com/x/{q}"}]

    monkeypatch.setattr(S, "_extract_one", fake_yt)
    monkeypatch.setattr(S, "_extract_one_sc", fake_sc)
    out = await S.search_metadata(["q1", "q2", "q3"], per_query=5)
    yt = [r for r in out if r.get("source") == "youtube"]
    sc = [r for r in out if r.get("source") == "soundcloud"]
    assert len(yt) == 3                          # dedup by title+channel: 1 per query
    assert len(sc) == 3                          # SC-only entries without a YouTube duplicate
    assert all(r.get("url", "").startswith("https://soundcloud.com/") for r in sc)
