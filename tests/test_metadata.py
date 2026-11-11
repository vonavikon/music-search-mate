"""Unit tests for pure functions in metadata.py: the gate against same-named
tracks, the pseudo-artist stop-list, canonical artist protection and album scoring."""
import pytest
from services.metadata import (
    _gate_artist,
    _gate_track,
    _is_pseudo_artist,
    _normalize_album,
    _pick_album,
    _tag_key,
)


def test_gate_track_exact_match():
    assert _gate_track("Yesterday", "Yesterday") == "Yesterday"


def test_gate_track_normalizes_punct_case():
    """Case/punctuation are normalized: the Last.fm name matches the requested one by key."""
    assert _gate_track("Bohemian Rhapsody", "bohemian  rhapsody!") == "Bohemian Rhapsody"


def test_gate_track_keeps_lfm_suffix():
    """Last.fm added a suffix (Album Version); the stem matched at length >= 8 — take the canonical name."""
    assert _gate_track("Under Pressure (Album Version)", "Under Pressure") == "Under Pressure (Album Version)"


def test_gate_track_diverged_keeps_requested():
    """The name diverged (someone else's same-named track) — keep the requested title, no substitution."""
    assert _gate_track("Creep", "Different Song") == "Different Song"


def test_gate_track_short_name_no_contains():
    """Short name (len < 8): contains check not applied, divergence -> the requested title."""
    assert _gate_track("Song for You", "Song") == "Song"


def test_gate_track_empty_requested_trusts_lfm():
    """No requested title — nothing to compare against, trust Last.fm."""
    assert _gate_track("Track", "") == "Track"


# --- canonical artist protection ---

def test_tag_key_ampersand_becomes_and():
    """& is normalized to "and" BEFORE punctuation: "A & B" == "A and B"."""
    assert _tag_key("A & B") == _tag_key("A and B")


def test_gate_artist_exact_match():
    assert _gate_artist("Radiohead", "Radiohead") == "Radiohead"


def test_gate_artist_strips_feat_suffix():
    """Content ID produced a feat-string while Last.fm returned the clean artist — take the clean one."""
    assert _gate_artist("Akon", "Akon feat. Colby O'Donis") == "Akon"


def test_gate_artist_short_name_keeps_requested():
    """Short name (len < 8) diverged — no substitution (same-name risk)."""
    assert _gate_artist("Prince", "Akon") == "Akon"


def test_gate_artist_diverged_keeps_requested():
    """The name diverged while the requested one is long — keep the requested one."""
    assert _gate_artist("Procol Harum", "FLY Records") == "FLY Records"


def test_gate_artist_empty_requested_trusts_new():
    """No requested artist — nothing to compare against, trust the source."""
    assert _gate_artist("Radiohead", "") == "Radiohead"


def test_is_pseudo_artist_various():
    assert _is_pseudo_artist("Various Artists")
    assert _is_pseudo_artist("Soundtrack")


def test_is_pseudo_artist_feat_aggregate():
    assert _is_pseudo_artist("Akon feat. Colby O'Donis")


def test_is_pseudo_artist_real_band_not_pseudo():
    assert not _is_pseudo_artist("Radiohead")
    assert not _is_pseudo_artist("Simon & Garfunkel")


# --- album candidate scoring ---

def test_normalize_album_strips_deluxe():
    assert _normalize_album("The Wall (Deluxe Edition)") == "The Wall"


def test_normalize_album_strips_remaster_year_after():
    assert _normalize_album("Abbey Road (Remastered 2011)") == "Abbey Road"


def test_normalize_album_strips_anniversary():
    assert _normalize_album("Thriller (25th Anniversary Edition)") == "Thriller"


def test_normalize_album_keeps_non_edition():
    assert _normalize_album("Pet Sounds (Stereo / Mono)") == "Pet Sounds (Stereo / Mono)"


def test_normalize_album_never_blanks():
    """An album consisting entirely of an edition marker word is not blanked out."""
    assert _normalize_album("Deluxe") == "Deluxe"


def test_pick_album_prefers_short_canonical():
    assert _pick_album(["The Wall (Deluxe Edition)", "The Wall"]) == "The Wall"


def test_pick_album_single_normalizes():
    assert _pick_album(["Rumours (Expanded Edition)"]) == "Rumours"


def test_pick_album_empty():
    assert _pick_album([]) == ""
    assert _pick_album(["", "  "]) == ""


# ════ Last.fm TTL cache ════

class TestTTLCache:
    def test_hit_miss_expiry_evict(self):
        from services.metadata import _TTLCache
        c = _TTLCache(ttl=10, maxsize=2)
        assert c.get("a") == (None, False)
        c.put("a", [1, 2])
        assert c.get("a") == ([1, 2], True)
        c.put("b", "x")
        c.put("c", "y")          # maxsize=2 exceeded -> full eviction
        assert c.get("a") == (None, False)
        assert c.get("c") == ("y", True)


@pytest.mark.asyncio
async def test_lastfm_cache_single_network_call(monkeypatch):
    """Calling similar_artists twice for the same artist -> a single network call."""
    from services.metadata import MetadataService
    svc = MetadataService("k")
    calls = {"n": 0}

    async def fake_uncached(artist, limit):
        calls["n"] += 1
        return ["Rick Nelson"]

    monkeypatch.setattr(svc, "_similar_artists_uncached", fake_uncached)
    a = await svc.similar_artists("Bobby Vee")
    b = await svc.similar_artists("bobby vee ")  # cache key is normalized
    assert a == b == ["Rick Nelson"]
    assert calls["n"] == 1
    await svc.aclose()


# ════ tag.gettoptracks (the mood pipeline's empirical pools) ════

class _FakeResp:
    """A minimal httpx.Response stand-in: status_code + json payload."""

    def __init__(self, status_code=200, payload=None, raise_exc=None):
        self.status_code = status_code
        self._payload = payload or {}
        self._raise = raise_exc

    def json(self):
        if self._raise:
            raise self._raise
        return self._payload


def _lfm_tag_payload(*tracks):
    """A Last.fm tag.gettoptracks JSON payload: artist embedded as {name}."""
    return {"tracks": {"track": [
        {"name": t["track"], "artist": {"name": t["artist"]},
         "listeners": str(t["listeners"])}
        for t in tracks
    ]}}


@pytest.mark.asyncio
async def test_tag_top_tracks_success_mapping(monkeypatch):
    """tag.gettoptracks success: [{artist, track, listeners}] with int
    listeners, in the tag-chart rank order (the chart IS the relevance)."""
    from services.metadata import MetadataService
    svc = MetadataService("k")

    async def fake_get(url, params=None):
        assert params["method"] == "tag.gettoptracks"
        assert params["tag"] == "workout"
        assert params["limit"] == "6"
        return _FakeResp(payload=_lfm_tag_payload(
            {"artist": "Survivor", "track": "Eye of the Tiger", "listeners": 500},
            {"artist": "Eminem", "track": "Till I Collapse", "listeners": 900},
        ))
    monkeypatch.setattr(svc._http, "get", fake_get)

    out = await svc.tag_top_tracks("workout")
    assert out == [
        {"artist": "Survivor", "track": "Eye of the Tiger", "listeners": 500},
        {"artist": "Eminem", "track": "Till I Collapse", "listeners": 900},
    ]
    await svc.aclose()


@pytest.mark.asyncio
async def test_tag_top_tracks_failure_returns_empty(monkeypatch):
    """Non-200, a Last.fm error payload and a transport exception all -> []
    (failure-silence, the same contract as similar_artists/track_search)."""
    from services.metadata import MetadataService
    svc = MetadataService("k")

    async def http_500(url, params=None):
        return _FakeResp(status_code=500)
    monkeypatch.setattr(svc._http, "get", http_500)
    assert await svc.tag_top_tracks("workout") == []

    async def error_payload(url, params=None):
        return _FakeResp(payload={"error": 6, "message": "not found"})
    monkeypatch.setattr(svc._http, "get", error_payload)
    assert await svc.tag_top_tracks("workout") == []

    async def transport_boom(url, params=None):
        raise RuntimeError("network down")
    monkeypatch.setattr(svc._http, "get", transport_boom)
    assert await svc.tag_top_tracks("workout") == []
    await svc.aclose()


@pytest.mark.asyncio
async def test_tag_top_tracks_no_key_or_empty_tag():
    """Without an API key or with an empty tag the call short-circuits to []
    before any HTTP (the same gate as the other Last.fm methods)."""
    from services.metadata import MetadataService
    svc = MetadataService("")
    assert await svc.tag_top_tracks("workout") == []
    svc2 = MetadataService("k")
    assert await svc2.tag_top_tracks("   ") == []
    await svc.aclose()
    await svc2.aclose()


@pytest.mark.asyncio
async def test_tag_top_tracks_cached_single_http_call(monkeypatch):
    """Second call for the same tag (normalized cache key) -> a single HTTP
    call (the day-TTL cache, the pattern of test_lastfm_cache_single_network_call)."""
    from services.metadata import MetadataService
    svc = MetadataService("k")
    calls = {"n": 0}

    async def fake_get(url, params=None):
        calls["n"] += 1
        return _FakeResp(payload=_lfm_tag_payload(
            {"artist": "Survivor", "track": "Eye of the Tiger", "listeners": 500}))
    monkeypatch.setattr(svc._http, "get", fake_get)

    a = await svc.tag_top_tracks("Workout")
    b = await svc.tag_top_tracks("workout ")  # cache key is normalized
    assert a == b == [{"artist": "Survivor", "track": "Eye of the Tiger",
                       "listeners": 500}]
    assert calls["n"] == 1
    await svc.aclose()


# ════ MusicBrainz get_artist_bands (the B5 solo-career novelty probe) ════

_MB_UA = "music-search-mate/1.0 (github.com/vonavikon/music-search-mate)"
_GAHAN_MBID = "d547b148-cf1f-421b-9464-ce7411ba44b8"


def _mb_search_payload(name, mbid, score=100):
    """A MusicBrainz ws/2 artist-search JSON payload (top match only)."""
    return {"artists": [{"id": mbid, "name": name, "score": score}]}


def _mb_rels_payload():
    """The live shape of Dave Gahan's artist-rels: 'member of band' and
    'founder' both point at Depeche Mode, 'parent' at Stella Rose — only the
    membership relation must be mapped."""
    return {"relations": [
        {"type": "founder", "target-type": "artist", "direction": "forward",
         "artist": {"id": "b0pr9", "name": "Depeche Mode"}},
        {"type": "member of band", "target-type": "artist", "direction": "forward",
         "artist": {"id": "b0pr9", "name": "Depeche Mode"}},
        {"type": "member of band", "target-type": "artist", "direction": "forward",
         "artist": {"id": "s0uls", "name": "Soulsavers"}},
        {"type": "parent", "target-type": "artist", "direction": "forward",
         "artist": {"id": "st3ll", "name": "Stella Rose"}},
        {"type": "member of band", "target-type": "url", "direction": "forward"},
    ]}


def _mb_http(search_resp=None, rels_resp=None):
    """fake svc._http.get dispatching by URL: /artist?query=... -> the search
    response, /artist/{mbid}?inc=artist-rels -> the rels response. Counts the
    calls per endpoint and captures the headers (the UA etiquette contract)."""
    state = {"search": 0, "rels": 0, "headers": []}

    async def fake_get(url, params=None, headers=None):
        state["headers"].append(headers)
        if url.endswith("/artist") and "query" in (params or {}):
            state["search"] += 1
            return search_resp if search_resp is not None else _FakeResp(status_code=503)
        if "/artist/" in url and (params or {}).get("inc") == "artist-rels":
            state["rels"] += 1
            return rels_resp if rels_resp is not None else _FakeResp(status_code=503)
        return _FakeResp(status_code=404)
    return fake_get, state


@pytest.mark.asyncio
async def test_get_artist_bands_maps_member_of_band(monkeypatch):
    """Dave Gahan -> {'Depeche Mode', 'Soulsavers'}: only 'member of band'
    relations count (founder/parent/url-target ignored), and every request
    carries the descriptive MusicBrainz User-Agent."""
    from services.metadata import MetadataService
    svc = MetadataService("k")
    fake_get, state = _mb_http(
        search_resp=_FakeResp(payload=_mb_search_payload("Dave Gahan", _GAHAN_MBID)),
        rels_resp=_FakeResp(payload=_mb_rels_payload()))
    monkeypatch.setattr(svc._http, "get", fake_get)

    out = await svc.get_artist_bands("Dave Gahan")
    assert out == {"Depeche Mode", "Soulsavers"}
    assert state["search"] == 1 and state["rels"] == 1
    assert all(h and h.get("User-Agent") == _MB_UA for h in state["headers"])
    await svc.aclose()


@pytest.mark.asyncio
async def test_get_artist_bands_score_gate_no_rels_call(monkeypatch):
    """A loose name match (score < 90) is a namesake risk: no rels call at
    all, empty set (never attribute somebody else's memberships)."""
    from services.metadata import MetadataService
    svc = MetadataService("k")
    fake_get, state = _mb_http(
        search_resp=_FakeResp(payload=_mb_search_payload("Gahan", "other-mbid", score=80)))
    monkeypatch.setattr(svc._http, "get", fake_get)

    assert await svc.get_artist_bands("Gahan") == set()
    assert state["search"] == 1 and state["rels"] == 0
    await svc.aclose()


@pytest.mark.asyncio
async def test_get_artist_bands_fail_soft(monkeypatch):
    """Non-200 search, transport error, non-200 rels and an empty name all ->
    an empty set (the probe degrades to a no-op, never raises)."""
    from services.metadata import MetadataService
    svc = MetadataService("k")

    fake_get, _s = _mb_http(search_resp=_FakeResp(status_code=503))
    monkeypatch.setattr(svc._http, "get", fake_get)
    assert await svc.get_artist_bands("Dave Gahan") == set()

    async def boom(url, params=None, headers=None):
        raise RuntimeError("network down")
    monkeypatch.setattr(svc._http, "get", boom)
    assert await svc.get_artist_bands("Dave Gahan") == set()

    fake_get2, _s2 = _mb_http(
        search_resp=_FakeResp(payload=_mb_search_payload("Dave Gahan", _GAHAN_MBID)),
        rels_resp=_FakeResp(status_code=503))
    monkeypatch.setattr(svc._http, "get", fake_get2)
    assert await svc.get_artist_bands("Dave Gahan") == set()

    assert await svc.get_artist_bands("   ") == set()
    await svc.aclose()


@pytest.mark.asyncio
async def test_get_artist_bands_cached_single_lookup(monkeypatch):
    """TTL cache: the second lookup of the same artist (normalized key) fires
    ZERO new HTTP calls — the whole search+rels pair ran once."""
    from services.metadata import MetadataService
    svc = MetadataService("k")
    fake_get, state = _mb_http(
        search_resp=_FakeResp(payload=_mb_search_payload("Dave Gahan", _GAHAN_MBID)),
        rels_resp=_FakeResp(payload=_mb_rels_payload()))
    monkeypatch.setattr(svc._http, "get", fake_get)

    a = await svc.get_artist_bands("Dave Gahan")
    b = await svc.get_artist_bands("dave gahan ")  # cache key is normalized
    assert a == b == {"Depeche Mode", "Soulsavers"}
    assert state["search"] == 1 and state["rels"] == 1
    await svc.aclose()


@pytest.mark.asyncio
async def test_artist_from_title_split(monkeypatch):
    """"Metrik - Gravity" -> the artist comes from the left part, validated via Last.fm."""
    from services.metadata import MetadataService
    svc = MetadataService("k")

    async def fake_getinfo(artist):
        return {"artist": "Metrik"} if artist == "Metrik" else None
    monkeypatch.setattr(svc, "_artist_getinfo", fake_getinfo)

    assert await svc._artist_from_title("Metrik - Gravity") == "Metrik"
    # a pseudo left part is rejected
    assert await svc._artist_from_title("Various Artists - Hits") == ""
    # no split -> empty result
    assert await svc._artist_from_title("Просто название трека") == ""
    await svc.aclose()


# ════ pretty_title ════

class TestPrettyTitle:
    def test_garbage_to_pretty(self):
        from services.metadata import pretty_title
        assert pretty_title("jerry lee lewis great balls of fire") == "Jerry Lee Lewis Great Balls of Fire"
        assert pretty_title("Chuck Berry, Chuck Berry") == "Chuck Berry"
        assert pretty_title("Elvis_Presley_Blue_Shoes") == "Elvis Presley Blue Shoes"
        assert pretty_title("  Rock n roll  ") == "Rock n roll"
        assert pretty_title("Queen - The Show Must Go On -") == "Queen - The Show Must Go On"
        # careful handling: already-pretty titles are untouched, minor words are not capitalized
        assert pretty_title("Great Balls of Fire") == "Great Balls of Fire"
        assert pretty_title("the great balls of fire") == "The Great Balls of Fire"
        assert pretty_title("") == ""


@pytest.mark.asyncio
async def test_resolve_title_split_no_crash(monkeypatch):
    """Regression for a production crash (UnboundLocalError in sources): a full
    resolve() of a track without a Content ID artist, where the artist yields an "X - Y" title split."""
    from services.metadata import MetadataService
    svc = MetadataService("k")

    async def gi(artist):  # _artist_getinfo
        return {"artist": "Fonky Family"} if "onky" in artist else None
    async def ti(artist, track):  # _lastfm_getinfo
        return None
    async def tt(artist):  # _artist_toptags
        return "french hip hop"
    async def ir(artist, track, video_id):  # _itunes_resolve
        return {}
    for name, fn in [("_artist_getinfo", gi), ("_lastfm_getinfo", ti),
                     ("_artist_toptags", tt), ("_itunes_resolve", ir)]:
        monkeypatch.setattr(svc, name, fn)

    out = await svc.resolve(
        {"title": "Fonky Family - Mystère et suspense (Clip officiel)",
         "channel": "Fonky Family", "duration": 263},
        "0YhXGXdtBfM")
    assert out.artist == "Fonky Family"
    assert "title-split"  # the mechanism is alive (artist confirmed)
    await svc.aclose()


def test_pretty_title_strips_video_markers():
    """"Iam - Petit Frere (Clip Officiel)" -> the clean track title."""
    from services.metadata import pretty_title
    assert pretty_title("Petit Frère (Clip Officiel)") == "Petit Frère"
    assert pretty_title("Umbrella (official Video)") == "Umbrella"
    assert pretty_title("Song (HD Remastered 2011)") == "Song"
    # meaningful parenthesized parts are untouched
    assert pretty_title("Fade Into You (Remix)") == "Fade Into You (Remix)"


@pytest.mark.asyncio
async def test_artist_from_title_by_form():
    """"Freestate by Depeche Mode" -> Depeche Mode (the channel-became-artist
    case). Validated via artist.getInfo."""
    from unittest.mock import AsyncMock
    from services.metadata import MetadataService
    svc = MetadataService.__new__(MetadataService)
    svc._lastfm_key = "k"
    svc._artist_getinfo = AsyncMock(return_value={"artist": "Depeche Mode"})
    assert await svc._artist_from_title("Freestate by Depeche Mode") == "Depeche Mode"
    # "Stand by Me" does not turn "Me" into an artist: getInfo is empty -> no guess
    svc._artist_getinfo = AsyncMock(return_value=None)
    assert await svc._artist_from_title("Stand by Me") == ""
    # the dash form keeps working as before
    svc._artist_getinfo = AsyncMock(return_value={"artist": "Metrik"})
    assert await svc._artist_from_title("Metrik - Gravity") == "Metrik"


def test_bio_line_strips_markdown():
    """The bio is stripped of * and [12] markers."""
    from types import SimpleNamespace
    from bot.handler import _bio_line
    meta = SimpleNamespace(bio="В 2007 году *Ultra* был переиздан, **Freestate** "
                                "значится как трек[12][3].")
    line = _bio_line(meta)
    assert "*" not in line and "[12]" not in line and "[3]" not in line
    assert "Ultra" in line and "Freestate" in line


class TestStripVersions:
    """Version suffixes are stripped before the iTunes lookup; otherwise the
    artwork would come from third-party slowed/nightcore compilations."""

    def test_stripped(self):
        from services.metadata import _strip_versions
        assert _strip_versions("Upside Down (slowed reverb)") == "Upside Down"
        assert _strip_versions("Mask Off (sped up)") == "Mask Off"
        assert _strip_versions("Track - slowed + reverb") == "Track"
        assert _strip_versions("Track - Slowed Reverb") == "Track"

    def test_clean_untouched(self):
        from services.metadata import _strip_versions
        assert _strip_versions("Upside Down") == "Upside Down"
        assert _strip_versions("Back in Black") == "Back in Black"
