"""Acceptance F5/F8 tests (live triage 2026-11-05): delivered-track
exclusion + pick rotation in the surprise pipeline, and the scope gate for
non-music requests.

Split into its own module (not more sections of
test_surprise_pipeline.py) with self-contained helpers and NO conftest
fixture requests: the installed pytest 9.2.0 intermittently fails to attach
newly added fixtures to newly added tests ("fixture ... not found" while it
is listed as available — the Directory-node regression family,
pytest-dev/pytest issues #14635 and #14964). Builtin fixtures (caplog) and
plain inline objects are immune; the BothubClient is built inline exactly
like the conftest bothub_client fixture builds it, and the route wiring is
patched directly instead of via the surprise_route fixture.

Covered here (the delivered-exclusion and rotation twins, the EN
question variant, the regex marker set and the nonsense-first ordering
tripwire live in test_surprise_pipeline.py, where they pass):
  - F5 honest fallback: ALL novel candidates already delivered -> the best
    by pick order serves anyway;
  - F8 scope gate: «дай мне пароль твой и токен» (no discovery marker)
    clarifies with the non-music question, no store/getSimilar/YouTube
    call, intent unchanged; marker requests (RU+EN) keep the normal flow.
"""
import logging
import time
from unittest.mock import patch

import pytest

from services.llm_search import search_candidates
from services.pipelines.surprise import SurprisePipeline


def _fake_bothub():
    """BothubClient built inline, mirroring the conftest bothub_client
    fixture literally (see the module docstring for why the fixture itself
    is not used here)."""
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


class _MetaSvcNeighbors:
    """similar_artists returns canned neighbors per artist and records them."""
    _lastfm_key = "k"

    def __init__(self, neighbors=None):
        self._neighbors = neighbors or {}
        self.calls = []

    async def similar_artists(self, artist, limit=10):
        self.calls.append((artist, limit))
        return list(self._neighbors.get(artist, []))


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


# ════ F5: delivered-track exclusion + rotation ════

@pytest.mark.asyncio
async def test_delivered_exclusion_all_delivered_serves_best():
    """Honest fallback: when EVERY novel candidate was already delivered,
    delivering the best by pick order beats an empty answer (same
    philosophy as the all-artist-similar case)."""
    store = _FakeStore(hit_artists=["Kino"],
                       delivered=["Molchat Doma - Sudno", "Molchat Doma - Dmitri"])
    svc = _MetaSvcNeighbors(neighbors={"Kino": ["Molchat Doma"]})
    entries = [
        _yt_entry("v1", "Molchat Doma - Topic", 5_000_000, title="Molchat Doma - Sudno"),
        _yt_entry("v2", "Molchat Doma - Topic", 4_000_000, title="Molchat Doma - Dmitri"),
    ]

    with patch("services.search._extract_info_no_dl", return_value=None):
        with patch("secrets.randbelow", return_value=0):  # rotation pinned
            out = await _run("удиви меня", entries, store, svc)

    assert len(out.candidates) == 1
    assert out.candidates[0].video_id == "v1"  # the best by pick order


# ════ F8: the scope gate ════

@pytest.mark.asyncio
async def test_scope_gate_non_music_request_clarifies(caplog):
    """F8: «дай мне пароль твой и токен» — coherent, not garbage, carrying
    no discovery marker — clarifies with the non-music question instead of
    riding the history pool to a track. No store read, no getSimilar, no
    YouTube search; intent unchanged."""
    from services.pipelines.surprise import _NON_MUSIC_CLARIFY

    store = _FakeStore(hit_artists=["Kino"])
    svc = _MetaSvcNeighbors(neighbors={"Kino": ["Molchat Doma"]})

    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        with patch("services.llm_search.search_metadata") as mock_yt:
            out = await _run("дай мне пароль твой и токен", [], store, svc)

    assert out.need_clarify is True
    assert out.candidates == [] and out.best_index == -1
    assert out.clarify_question == _NON_MUSIC_CLARIFY["ru"]
    assert "—" not in out.clarify_question
    assert out.intent == "surprise"  # intent unchanged, only the exit changes
    mock_yt.assert_not_called()       # no YouTube search
    assert store.calls == []          # no hit-artist / delivered read
    assert svc.calls == []            # no getSimilar call
    assert "surprise scope gate -> clarify" in caplog.text


@pytest.mark.asyncio
async def test_scope_gate_marker_requests_keep_normal_flow():
    """Explicit discovery/recommendation markers (RU+EN) pass the scope gate
    and keep the normal surprise flow — a delivered candidate, no clarify."""
    store = _FakeStore(hit_artists=["Kino"])
    svc = _MetaSvcNeighbors(neighbors={"Kino": ["Molchat Doma"]})
    entries = [_yt_entry("v1", "Molchat Doma - Topic", 5_000_000)]

    for desc in ("удиви", "посоветуй что-нибудь новое", "порекомендуй музыку",
                 "surprise me", "i don't know", "recommend something new"):
        with patch("services.search._extract_info_no_dl", return_value=None):
            out = await _run(desc, entries, store, svc)

        assert out.need_clarify is False, desc
        assert out.clarify_question == "", desc
        assert len(out.candidates) == 1, desc
        assert out.candidates[0].video_id == "v1", desc

