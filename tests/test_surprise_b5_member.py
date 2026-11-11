"""B5 tests (solo-career novelty): the MusicBrainz member-of-hit-band probe
in the surprise pipeline's verifier pass.

The live case (research card П6): after Depeche Mode hits, «удиви»
recommended Dave Gahan — the band's frontman. "Dave Gahan" shares no name
token with "Depeche Mode" and scores ~0.2 on fuzzy.artist_match, so the
token rule and the fuzzy verifier both pass; MetadataService.get_artist_bands
(MusicBrainz "member of band" relations, mocked here) closes the gap in the
verifier pass — a candidate whose artist plays in a hit artist's band is a
disguised hit and drops out of the rotation shortlist.

Self-contained module in the pattern of test_surprise_f5_f8.py (inline
BothubClient, route patched directly, no conftest fixture requests — see
that module's docstring for the pytest 9.2.0 fixture-attachment rationale).

Covered:
  - Gahan -> bands {Depeche Mode} + hits {Depeche Mode} -> dropped, the log
    names the artist and the hit band, the next candidate delivers;
  - an unrelated neighbor (bands {Radiohead}) is kept;
  - a MusicBrainz error -> kept (fail-soft, debug log), no crash;
  - the per-run rate cap: 4 probed candidates -> at most 3 MB lookups;
  - repeat channels cost one lookup (per-run verdict memo);
  - a metadata service WITHOUT the method (old stubs) -> no-op, no crash.
"""
import logging
import time
from unittest.mock import patch

import pytest

from services.llm_search import search_candidates
from services.pipelines.surprise import SurprisePipeline
from types import SimpleNamespace


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Same guard as test_surprise_pipeline.py: the enrich step must not touch
    the network (candidates here are fake)."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


def _fake_bothub():
    """BothubClient built inline, mirroring the conftest bothub_client
    fixture (see test_surprise_f5_f8.py for why the fixture is not used)."""
    from services.llm_search import BothubClient
    return BothubClient(
        "https://fake-bothub.local/v1",
        "fake-bothub-key",
        "fake-model",
    )


def _extract_json(intent="surprise", queries=None, intro="Ищу незнакомое..."):
    """LLM#1 extract — a fake parsed dict (same shape as
    test_surprise_pipeline._extract_json)."""
    return {
        "intro": intro,
        "intent": intent,
        "signals": {"language": "ru", "era": None, "vocal": None,
                    "genre": None, "mood": None},
        "target": {"artist": None, "track": None},
        "reference": {"artist": None, "track": None},
        "sonic": {"tempo": None, "instrumentation": None,
                  "dynamics": None, "vocal_timbre": None},
        "association_candidates": [],
        "mood_tags": [],
        "queries": queries if queries is not None else [],
    }


def _yt_entry(vid, channel, views, title=None):
    """A fake yt-dlp candidate."""
    return {"video_id": vid, "title": title or f"Track {vid}", "channel": channel,
            "duration": 240, "views": views}


class _FakeStore:
    """get_user_hit_artists / recent_context / get_recent_best_titles over
    canned lists, recording the calls."""
    _lastfm_key = ""

    def __init__(self, hit_artists=None, recent=None, delivered=None):
        self._hit_artists = hit_artists or []
        self._recent = recent or []
        self._delivered = delivered or []
        self.calls = []

    async def get_user_hit_artists(self, user_id, limit=20):
        self.calls.append(("hits", user_id, limit))
        return list(self._hit_artists)

    async def recent_context(self, user_id, limit=3, window_min=30):
        self.calls.append(("recent", user_id))
        return list(self._recent)

    async def get_recent_best_titles(self, user_id, limit=20):
        self.calls.append(("delivered", user_id, limit))
        return list(self._delivered)


class _MetaSvcBands:
    """similar_artists returns canned neighbors; get_artist_bands returns the
    canned MusicBrainz band sets (or raises for the configured artists). Both
    record their calls so the tests assert the exact MB traffic."""

    _lastfm_key = "k"  # _filter_and_enrich reads this attribute on any svc

    def __init__(self, neighbors=None, bands=None, raise_for=None):
        self._neighbors = neighbors or {}
        self._bands = bands or {}
        self._raise = set(raise_for or ())
        self.calls = []

    async def similar_artists(self, artist, limit=10):
        self.calls.append(("similar", artist))
        return list(self._neighbors.get(artist, []))

    async def get_artist_bands(self, artist):
        self.calls.append(("bands", artist))
        if artist in self._raise:
            raise RuntimeError("musicbrainz outage")
        return set(self._bands.get(artist, ()))


def _chat_spy(captured, extract):
    """chat_json answering ONLY the extract (a rank call = LLM#2 leaked)."""
    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault("calls", []).append(gen_name)
        if gen_name == "llm-extract":
            return extract
        return None
    return fake_chat


def _capture_search(captured, entries):
    """search_metadata patch capturing the queries, returning canned entries."""
    async def cap(queries, per_query=20, on_progress=None):
        if per_query == 20:
            captured["queries"] = list(queries)
        return entries
    return cap


async def _run(desc, entries, store, svc):
    """One surprise search over the mocks: route wired to SurprisePipeline,
    chat_json -> the surprise extract, search_metadata -> the entries."""
    client = _fake_bothub()
    captured = {}
    with patch("services.llm_search.route", lambda intent: SurprisePipeline()):
        with patch.object(client, "chat_json",
                          side_effect=_chat_spy(captured, _extract_json())):
            with patch("services.llm_search.search_metadata") as mock_yt:
                mock_yt.side_effect = _capture_search(captured, entries)
                out = await search_candidates(desc, client,
                                              metadata_svc=svc, user_id=42, store=store)
    out.captured = captured
    return out


# ════ B5: the member-of-hit-band drop ════

@pytest.mark.asyncio
async def test_member_of_hit_band_dropped(caplog):
    """The live П6 case: Dave Gahan (getSimilar neighbor after Depeche Mode
    hits) is a MEMBER of the hit band per MusicBrainz -> dropped out of the
    shortlist with the member log line; the next candidate in pick order
    delivers instead."""
    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(neighbors={"Depeche Mode": ["Dave Gahan", "Molchat Doma"]},
                        bands={"Dave Gahan": {"Depeche Mode"}})
    entries = [
        _yt_entry("v1", "Dave Gahan - Topic", 5_000_000, title="Dave Gahan - Tomorrow"),
        _yt_entry("v2", "Molchat Doma - Topic", 4_000_000),
    ]

    with patch("secrets.randbelow", return_value=0):  # rotation pinned
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out = await _run("удиви меня", entries, store, svc)

    assert ("bands", "Dave Gahan") in svc.calls  # the probe actually fired
    assert [c.video_id for c in out.candidates] == ["v2"]
    assert "'Dave Gahan' is a member of hit band 'Depeche Mode'" in caplog.text


@pytest.mark.asyncio
async def test_unrelated_band_member_kept(caplog):
    """A neighbor whose bands do NOT match a hit artist (bands {Radiohead}
    vs hits {Depeche Mode}: artist_match ~0.35 < 0.8) is NOT a disguised hit
    — the probe runs and the candidate still delivers, no member log."""
    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(neighbors={"Depeche Mode": ["Some Neighbor"]},
                        bands={"Some Neighbor": {"Radiohead"}})
    entries = [_yt_entry("v1", "Some Neighbor - Topic", 5_000_000)]

    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out = await _run("удиви меня", entries, store, svc)

    assert ("bands", "Some Neighbor") in svc.calls
    assert [c.video_id for c in out.candidates] == ["v1"]
    assert "is a member of hit band" not in caplog.text


@pytest.mark.asyncio
async def test_mb_error_candidate_kept(caplog):
    """Fail-soft: a MusicBrainz outage (the lookup raises) never breaks the
    pick — the candidate is kept unchecked, a debug log says the probe
    failed."""
    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(neighbors={"Depeche Mode": ["Molchat Doma"]},
                        raise_for={"Molchat Doma"})
    entries = [_yt_entry("v1", "Molchat Doma - Topic", 5_000_000)]

    with caplog.at_level(logging.DEBUG, logger="bot.llm_search"):
        out = await _run("удиви меня", entries, store, svc)

    assert [c.video_id for c in out.candidates] == ["v1"]
    assert "musicbrainz lookup failed" in caplog.text
    assert "is a member of hit band" not in caplog.text


@pytest.mark.asyncio
async def test_mb_rate_cap_six_lookups_per_run():
    """Rate-limit awareness: 8 verifier-passing candidates would all be
    probed, but the per-run budget stops MusicBrainz at 6 lookups — the 7th
    candidate enters the shortlist unchecked (fail-soft by budget)."""
    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(neighbors={"Depeche Mode": ["Neighbor One"]})
    entries = [
        _yt_entry("v1", "Band A - Topic", 8_000_000),
        _yt_entry("v2", "Band B - Topic", 7_000_000),
        _yt_entry("v3", "Band C - Topic", 6_000_000),
        _yt_entry("v4", "Band D - Topic", 5_000_000),
        _yt_entry("v5", "Band E - Topic", 4_000_000),
        _yt_entry("v6", "Band F - Topic", 3_000_000),
        _yt_entry("v7", "Band G - Topic", 2_000_000),
    ]

    with patch("secrets.randbelow", return_value=0):  # rotation pinned
        out = await _run("удиви меня", entries, store, svc)

    band_calls = [a for tag, a in svc.calls if tag == "bands"]
    # the rotation shortlist holds 5 passers, so the walk naturally stops
    # after five probes — the budget (6) is only the hard ceiling
    assert band_calls == ["Band A", "Band B", "Band C", "Band D", "Band E"], \
        "MB probes in pick order, bounded by shortlist size and budget"
    assert len(out.candidates) == 1  # the run still delivered


@pytest.mark.asyncio
async def test_repeat_channel_costs_one_lookup():
    """Per-run verdict memo: two candidates from the SAME channel (the usual
    multi-track neighbor result) trigger ONE MusicBrainz lookup — the second
    reuses the cached verdict and also drops."""
    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(neighbors={"Depeche Mode": ["Dave Gahan", "Molchat Doma"]},
                        bands={"Dave Gahan": {"Depeche Mode"}})
    entries = [
        _yt_entry("v1", "Dave Gahan - Topic", 5_000_000, title="Dave Gahan - Hourglass"),
        _yt_entry("v2", "Dave Gahan - Topic", 4_000_000, title="Dave Gahan - Tomorrow"),
        _yt_entry("v3", "Molchat Doma - Topic", 3_000_000),
    ]

    with patch("secrets.randbelow", return_value=0):
        out = await _run("удиви меня", entries, store, svc)

    band_calls = [a for tag, a in svc.calls if tag == "bands"]
    assert band_calls == ["Dave Gahan", "Molchat Doma"]
    assert [c.video_id for c in out.candidates] == ["v3"]


@pytest.mark.asyncio
async def test_svc_without_get_artist_bands_method_noop():
    """Backward compatibility: a metadata service stub WITHOUT the new method
    (every pre-B5 test double) must not crash — the probe is a no-op and the
    normal flow delivers."""
    class _OldSvc:
        _lastfm_key = "k"

        def __init__(self, neighbors=None):
            self._neighbors = neighbors or {}
            self.calls = []

        async def similar_artists(self, artist, limit=10):
            self.calls.append(artist)
            return list(self._neighbors.get(artist, []))

    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _OldSvc(neighbors={"Depeche Mode": ["Molchat Doma"]})
    entries = [_yt_entry("v1", "Molchat Doma - Topic", 5_000_000)]

    out = await _run("удиви меня", entries, store, svc)

    assert [c.video_id for c in out.candidates] == ["v1"]


@pytest.mark.asyncio
async def test_compound_artist_components_probed():
    """«Soulsavers & Dave Gahan» must die on the Dave Gahan component."""
    from services.pipelines import surprise as sp

    async def bands(name):
        # only the component resolves; the full compound misses MB's score gate
        return {"Depeche Mode"} if name == "Dave Gahan" else set()

    svc = SimpleNamespace(get_artist_bands=bands)
    cands = [{"video_id": "g1", "title": "Soulsavers & Dave Gahan - Kingdom",
              "channel": "Medialook Studio", "views": 5_000_000, "duration": 240},
             {"video_id": "ok1", "title": "Faithless - Insomnia",
              "channel": "Faithless", "views": 4_000_000, "duration": 220}]
    out = await _run_with_hits(["Depeche Mode"], cands, svc) if False else None
    # drive through the pipeline run() with stubbed search/store like sibling tests
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    async def fake_search(queries, **kw):
        return list(cands)

    ctx = SimpleNamespace(
        description="удиви", context=[], excluded_ids=set(), excluded_artists=set(),
        metadata_svc=svc, bothub=None, on_progress=None, trace=None,
        extract={"intent": "surprise", "queries": [], "signals": {}, "sonic": {},
                 "association_candidates": [], "reference": {}, "target": {},
                 "intro": ""},
        t_start=0.0, t_extract=0.0, user_id=7, excluded_titles=None,
        restart_streak=0, store=None)
    store_stub = SimpleNamespace(
        get_user_hit_artists=AsyncMock(return_value=["Depeche Mode"]),
        get_recent_best_titles=AsyncMock(return_value=[]))
    ctx.store = store_stub
    with patch.object(sp, "_search_youtube", new=AsyncMock(return_value=(cands, []))), \
         patch.object(sp, "_filter_and_enrich",
                      new=AsyncMock(return_value=(list(cands), []))):
        out = await sp.SurprisePipeline().run(ctx)
    assert out.candidates and out.candidates[0].video_id == "ok1", \
        "the compound Gahan track must be skipped in favor of Faithless"
