"""Featured-artist novelty tests (QA x2): the MusicBrainz member-of-hit-band
probe in the surprise pipeline's verifier pass also covers the artists
FEATURED in the candidate title.

The live case (QA-simulation 2026-11-06, case x2 — seeded Deep Purple +
Depeche Mode hits): the member-of filter cut the solo records of Dave Gahan
and Martin L. Gore, but "Goldfrapp - Ocean Feat. Dave Gahan" was delivered —
the exclusion key looked at the CHANNEL (GoldfrappTV) and nothing checked
the feat credit in the title. The fix extracts the featured names
(feat./ft./featuring/×, case-insensitive, bracket junk cut) and runs them
through the same member probe: same drop mechanism, same log format plus a
source note ("featured in title"), same per-run MusicBrainz budget and
verdict memo (one artist is never asked twice in a run, feat names included).

Self-contained module in the pattern of tests/test_surprise_b5_member.py
(inline BothubClient, route patched directly, no conftest fixture requests —
see test_surprise_f5_f8.py for the pytest 9.2.0 fixture-attachment
rationale). The _FakeStore / _MetaSvcBands / _run helpers are that module's,
trimmed to what these tests need.

Covered:
  - (a) the x2 case: feat member -> dropped with the member log line naming
    the FEATURED artist, the next candidate in pick order delivers;
  - (b) a feat artist with no hit-band ties -> the candidate delivers;
  - (c) "ft. Alpha & Bravo": every name is checked, the first proven member
    (Bravo) drops the candidate;
  - (d) budget: artist + feat probes over 4 candidates never push the
    MusicBrainz call count past _MB_CALL_BUDGET;
  - (e) a title WITHOUT a feat marker behaves exactly as before: the exact
    pre-fix MB call sequence, nothing extra;
  - the per-run memo: the same feat artist on two candidates costs ONE
    lookup;
  - the _feat_artists parser itself (markers, junk cuts, multi-names).
"""
import logging
from unittest.mock import patch

import pytest

from services.llm_search import search_candidates
from services.pipelines.surprise import SurprisePipeline


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Same guard as test_surprise_b5_member.py: the enrich step must not touch
    the network (candidates here are fake)."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


def _fake_bothub():
    """BothhubClient built inline, mirroring the conftest bothub_client
    fixture (see test_surprise_f5_f8.py for why the fixture is not used)."""
    from services.llm_search import BothubClient
    return BothubClient(
        "https://fake-bothub.local/v1",
        "fake-bothub-key",
        "fake-model",
    )


def _extract_json(intent="surprise", queries=None, intro="Ищу незнакомое..."):
    """LLM#1 extract — a fake parsed dict (same shape as
    test_surprise_b5_member._extract_json)."""
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
    canned lists (test_surprise_b5_member._FakeStore, verbatim)."""
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
    canned MusicBrainz band sets. Both record their calls so the tests
    assert the exact MB traffic (test_surprise_b5_member._MetaSvcBands,
    without the raise_for knob these tests do not need)."""

    _lastfm_key = "k"  # _filter_and_enrich reads this attribute on any svc

    def __init__(self, neighbors=None, bands=None):
        self._neighbors = neighbors or {}
        self._bands = bands or {}
        self.calls = []

    async def similar_artists(self, artist, limit=10):
        self.calls.append(("similar", artist))
        return list(self._neighbors.get(artist, []))

    async def get_artist_bands(self, artist):
        self.calls.append(("bands", artist))
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


def _band_calls(svc):
    """The MusicBrainz lookup names, in call order."""
    return [a for tag, a in svc.calls if tag == "bands"]


# ════ (a) the x2 case: a feat member drops the candidate ════

@pytest.mark.asyncio
async def test_feat_member_of_hit_band_drops_candidate(caplog):
    """(a) The x2 case: 'Goldfrapp - Ocean Feat. Dave Gahan (Official
    Video)' after Depeche Mode hits. The channel (GoldfrappTV) is clean and
    the title-parsed artist Goldfrapp has no hit ties, but the feat credit
    is not clean: the probe fires for the artist first, then for the
    featured Dave Gahan, the candidate drops with the member log line
    naming Dave Gahan (with the feat source note), and the next candidate
    in pick order delivers."""
    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(neighbors={"Depeche Mode": ["Goldfrapp", "Molchat Doma"]},
                        bands={"Dave Gahan": {"Depeche Mode"}})
    entries = [
        _yt_entry("v1", "GoldfrappTV", 5_000_000,
                  title="Goldfrapp - Ocean Feat. Dave Gahan (Official Video)"),
        _yt_entry("v2", "Molchat Doma - Topic", 4_000_000),
    ]

    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out = await _run("удиви меня", entries, store, svc)

    assert _band_calls(svc) == ["Goldfrapp", "Dave Gahan", "Molchat Doma"], \
        "artist probe first, then the feat name, then the next candidate"
    assert [c.video_id for c in out.candidates] == ["v2"]
    assert "'Dave Gahan' is a member of hit band 'Depeche Mode'" in caplog.text
    assert "(featured in title)" in caplog.text  # the source note


# ════ (b) a feat artist with no hit-band ties delivers ════

@pytest.mark.asyncio
async def test_feat_artist_without_hit_ties_delivers(caplog):
    """(b) A feat artist whose bands hold no hit artist (bands empty) is not
    a disguised hit: the feat side IS probed (the lookup happened) and the
    candidate still delivers, no member log."""
    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(neighbors={"Depeche Mode": ["Goldfrapp"]}, bands={})
    entries = [
        _yt_entry("v1", "GoldfrappTV", 5_000_000,
                  title="Goldfrapp - Ocean Feat. Some Guest"),
    ]

    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out = await _run("удиви меня", entries, store, svc)

    assert _band_calls(svc) == ["Goldfrapp", "Some Guest"]
    assert [c.video_id for c in out.candidates] == ["v1"]
    assert "is a member of hit band" not in caplog.text


# ════ (c) several feat names: check until the first member ════

@pytest.mark.asyncio
async def test_multiple_feat_names_checked_until_member(caplog):
    """(c) 'Anthem ft. Alpha & Bravo': both names are extracted and probed
    in order; the FIRST proven member (Bravo) drops the candidate — Alpha
    is checked and cleared on the way."""
    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(
        neighbors={"Depeche Mode": ["Neighbour Band", "Molchat Doma"]},
        bands={"Bravo": {"Depeche Mode"}})
    entries = [
        _yt_entry("v1", "NeighbourTV", 5_000_000,
                  title="Neighbour Band - Anthem ft. Alpha & Bravo"),
        _yt_entry("v2", "Molchat Doma - Topic", 4_000_000),
    ]

    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out = await _run("удиви меня", entries, store, svc)

    assert _band_calls(svc) == ["Neighbour Band", "Alpha", "Bravo",
                                "Molchat Doma"]
    assert [c.video_id for c in out.candidates] == ["v2"]
    assert "'Bravo' is a member of hit band 'Depeche Mode'" in caplog.text
    assert "(featured in title)" in caplog.text


# ════ (d) the feat checks share the per-run MB budget ════

@pytest.mark.asyncio
async def test_feat_checks_share_the_mb_call_budget():
    """(d) Budget: artist + feat probes over 4 passing candidates would ask
    MusicBrainz 8 times; the shared per-run budget caps the run at
    _MB_CALL_BUDGET (6) — the 4th candidate enters the shortlist unchecked
    (fail-soft by budget), and no name is ever asked twice."""
    from services.pipelines.surprise import _MB_CALL_BUDGET

    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(neighbors={"Depeche Mode": ["Neighbor One"]}, bands={})
    entries = [
        _yt_entry("v1", "Band One - Topic", 8_000_000,
                  title="Band One - Song Feat. Guest One"),
        _yt_entry("v2", "Band Two - Topic", 7_000_000,
                  title="Band Two - Song Feat. Guest Two"),
        _yt_entry("v3", "Band Three - Topic", 6_000_000,
                  title="Band Three - Song Feat. Guest Three"),
        _yt_entry("v4", "Band Four - Topic", 5_000_000,
                  title="Band Four - Song Feat. Guest Four"),
    ]

    out = await _run("удиви меня", entries, store, svc)

    calls = _band_calls(svc)
    assert len(calls) == _MB_CALL_BUDGET  # 8 wanted, 6 allowed
    assert len(calls) == len(set(calls))  # no name asked twice
    assert len(out.candidates) == 1  # the run still delivered


# ════ (e) no feat marker: byte-for-byte the old MB traffic ════

@pytest.mark.asyncio
async def test_no_feat_title_exact_old_mb_calls(caplog):
    """(e) A title WITHOUT a feat marker behaves exactly as before the fix:
    the exact pre-fix MusicBrainz call sequence (the title-parsed artists
    alone, in pick order — the configured bands entry for Dave Gahan proves
    nothing but the title was asked), no member drop, no feat log."""
    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(neighbors={"Depeche Mode": ["Goldfrapp", "Molchat Doma"]},
                        bands={"Dave Gahan": {"Depeche Mode"}})
    entries = [
        _yt_entry("v1", "GoldfrappTV", 5_000_000,
                  title="Goldfrapp - Ocean (Official Video)"),
        _yt_entry("v2", "Molchat Doma - Topic", 4_000_000),
    ]

    with patch("secrets.randbelow", return_value=0):  # rotation pinned
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out = await _run("удиви меня", entries, store, svc)

    assert _band_calls(svc) == ["Goldfrapp", "Molchat Doma"], \
        "the pre-fix sequence: title artists only, in pick order"
    assert [c.video_id for c in out.candidates] == ["v1"]
    assert "is a member of hit band" not in caplog.text
    assert "(featured in title)" not in caplog.text


# ════ the per-run memo across feat probes ════

@pytest.mark.asyncio
async def test_same_feat_artist_costs_one_lookup():
    """Memo: the same featured artist on TWO candidates triggers ONE
    MusicBrainz lookup — the second candidate reuses the per-run verdict
    and drops by the memo, and the next clean candidate delivers."""
    store = _FakeStore(hit_artists=["Depeche Mode"])
    svc = _MetaSvcBands(neighbors={"Depeche Mode": ["Goldfrapp", "Molchat Doma"]},
                        bands={"Dave Gahan": {"Depeche Mode"}})
    entries = [
        _yt_entry("v1", "GoldfrappTV", 5_000_000,
                  title="Goldfrapp - Ocean Feat. Dave Gahan"),
        _yt_entry("v2", "Molchat Doma - Topic", 4_000_000,
                  title="Molchat Doma - Sudno Feat. Dave Gahan"),
        _yt_entry("v3", "Molchat Doma - Topic", 3_000_000),
    ]

    out = await _run("удиви меня", entries, store, svc)

    assert _band_calls(svc) == ["Goldfrapp", "Dave Gahan", "Molchat Doma"], \
        "the second Feat. Dave Gahan reuses the memo, no re-ask"
    assert [c.video_id for c in out.candidates] == ["v3"]


# ════ the title feat parser itself ════

def test_feat_artists_parser():
    """Markers (feat./ft./featuring/×, any case, optional dot and bracket
    opening), junk cuts (brackets, a next dash part, pipes), multi-name
    splits, and the silent skip of garbage."""
    from services.pipelines.surprise import _feat_artists
    # the x2 title: bracket junk after the name is cut
    assert _feat_artists("Ocean Feat. Dave Gahan (Official Video)") == ["Dave Gahan"]
    assert _feat_artists("Goldfrapp - Ocean Feat. Dave Gahan") == ["Dave Gahan"]
    # marker variants, case-insensitive, optional dot, bracketed credit
    assert _feat_artists("Song ft. Dave Gahan") == ["Dave Gahan"]
    assert _feat_artists("Song FEATURING Dave Gahan") == ["Dave Gahan"]
    assert _feat_artists("Ocean × Dave Gahan") == ["Dave Gahan"]
    assert _feat_artists("Song feat Dave Gahan") == ["Dave Gahan"]
    assert _feat_artists("Song (feat. Dave Gahan)") == ["Dave Gahan"]
    # several names after one marker: every significant component survives
    assert _feat_artists("Song ft. Alpha & Bravo") == ["Alpha", "Bravo"]
    assert _feat_artists("Song Feat. Alpha, Bravo") == ["Alpha", "Bravo"]
    # a next dash-separated part ends the segment
    assert _feat_artists("Song ft. Dave Gahan - Remastered") == ["Dave Gahan"]
    # junk, skipped silently: no marker, bare marker, bracket-only tail,
    # single-letter names, "ft." inside a word, a feat-stem word
    assert _feat_artists("Goldfrapp - Ocean") == []
    assert _feat_artists("Song (feat.)") == []
    assert _feat_artists("Ocean Feat. (Official Video)") == []
    assert _feat_artists("") == []
    assert _feat_artists("Song ft. A & B") == []
    assert _feat_artists("Soft. Rock Song") == []
    assert _feat_artists("Features of a Song") == []
