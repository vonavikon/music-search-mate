"""Artist-only scope, handler side (prod 14.11: «ввожу просто название группы —
хочу только её песни, а Не то даёт другую песень ЭТОЙ ЖЕ группы»).

The core (services/llm_search, another change) adds to SearchOut:
  scope_artist: str  — non-empty = the search is scoped to ONE artist's songs;
  target: dict       — {'artist': str|None, 'track': str|None}.
The handler must (1) persist target into the callback state, (2) keep the
target artist out of the channel ban the "Не то" restart builds (2-3 misses on
"Depeche Mode - Topic" must not ban Depeche Mode itself and flip the pool to
covers/foreign artists), (3) never auto-download in artist-only mode — the
user picks the song from the preview.

scope_artist is attached with setattr: the tests run both before and after the
core lands its field. Fixture/driver patterns follow tests/test_handler_anchor.py
(MagicMock/AsyncMock over real dicts; conftest isolation applies automatically).
"""
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import User

from bot.handler import _finish_search, handle_callback
from services.llm_search import Candidate, SearchOut, _ref_key

UID = 123  # the house test user (test_handler.py)


# ════ helpers (patterns lifted from tests/test_handler_anchor.py) ════

def _fake_out(**kw):
    """A SearchOut; the default is the 14.11 artist-only shape — a this_track
    search scoped to one band with a strong best (auto-download conditions met)."""
    defaults = {
        "intent": "this_track",
        "signals": {},
        "queries": ["depeche mode"],
        "assoc": [],
        "candidates": [
            Candidate("dm1", "Depeche Mode - Enjoy the Silence",
                      "Depeche Mode - Topic", 254, 10 ** 6,
                      score=0.95, match_score=0.9),
            Candidate("dm2", "Depeche Mode - Personal Jesus",
                      "Depeche Mode", 250, 10 ** 5,
                      score=0.7, match_score=0.6),
            Candidate("dm3", "Depeche Mode - Stripped",
                      "Depeche Mode - Topic", 233, 10 ** 4,
                      score=0.6, match_score=0.5),
        ],
        "best_index": 0,
        "confidence": "high",
        "intro": "Ищу: песни Depeche Mode.",
        "verdict": "OK",
        "need_clarify": False,
        "clarify_question": "",
        "reference": {},
        "target": {"artist": "Depeche Mode", "track": None},
    }
    defaults.update(kw)
    return SearchOut(**defaults)


def _scoped(out, artist):
    """Attach the core-contract scope_artist field; plain setattr works whether
    or not the dataclass declares it yet (the core lands it in parallel)."""
    setattr(out, "scope_artist", artist)
    return out


async def _run_finish(out):
    """Drive _finish_search directly (the _run_finish shape of
    test_handler_anchor.py) and capture insert_query/save_query_state calls.
    Returns (context, status, insert_query_mock, save_query_state_mock,
    get_cached_file_mock)."""
    update = MagicMock()
    update.effective_user = User(id=UID, is_bot=False, first_name="Test")
    update.effective_chat = MagicMock()
    update.effective_chat.id = 456
    context = MagicMock()
    context.bot_data = {"media": MagicMock()}
    context.chat_data = {}
    status = AsyncMock()
    mock_insert = AsyncMock(return_value=1)
    mock_save = AsyncMock()
    mock_cached = AsyncMock(return_value=None)
    with patch("bot.handler.insert_query", mock_insert), \
         patch("bot.handler.save_query_state", mock_save), \
         patch("bot.handler.incr_usage", AsyncMock()), \
         patch("bot.handler.get_cached_file", mock_cached), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")):
        await _finish_search(update, context, status, out, "depeche mode",
                             update.effective_user)
    return context, status, mock_insert, mock_save, mock_cached


def _dm_store(**extra):
    """The saved callback state of an artist-only «depeche mode» search after
    the user pressed «Не то» on every candidate (streak 3 >= 2 -> restart):
    two songs from the band's own channels + one foreign cover."""
    return {
        "candidates": [
            Candidate("dm1", "Depeche Mode - Enjoy the Silence",
                      "Depeche Mode - Topic", 254, 10 ** 6, score=0.95),
            Candidate("dm2", "Depeche Mode - Personal Jesus",
                      "Depeche Mode", 250, 10 ** 5, score=0.7),
            Candidate("cov1", "Enjoy the Silence (acoustic cover)",
                      "Acoustic Covers", 240, 10 ** 3, score=0.4),
        ],
        "confidence": "high", "best_index": 0, "verdict": "", "intro": "",
        "desc": "depeche mode", "rejected_ids": ["dm1", "dm2", "cov1"],
        "intent": "this_track", "reference": {}, "ts": time.time(),
        "target": {"artist": "Depeche Mode", "track": None},
        **extra,
    }


async def _run_fb_miss(store):
    """Press «Не то» (fb:m) on rank 0 of the saved query 1 and capture the
    restart _do_search call (the miss-streak re-search path)."""
    cq = MagicMock()
    cq.data = "fb:m:1:0"
    cq.answer = AsyncMock()
    cq.edit_message_reply_markup = AsyncMock()
    cq.message.reply_text = AsyncMock(return_value=AsyncMock())

    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=UID, is_bot=False, first_name="Test")
    update.effective_chat = MagicMock()

    context = MagicMock()
    context.bot_data = {"media": MagicMock()}
    context.chat_data = {"q:1": store}

    mock_do = AsyncMock(return_value=None)  # restart "failed" -> fall through
    with patch("bot.handler.insert_feedback", AsyncMock()), \
         patch("bot.handler.get_query_correlation", AsyncMock(return_value=None)), \
         patch("bot.handler.score_feedback", MagicMock()), \
         patch("bot.handler._sends_over", AsyncMock(return_value=False)), \
         patch("bot.handler.save_query_state", AsyncMock()), \
         patch("bot.handler.delete_query_state", AsyncMock()), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
         patch("bot.handler._do_search", mock_do):
        await handle_callback(update, context)  # must not raise
    return mock_do


# ════ 1. _save_state persists target ════

async def test_save_state_persists_target():
    """The finished search's out.target lands in the callback state (chat_data)
    and survives the JSON serialization _save_state sends to SQLite."""
    out = _scoped(_fake_out(), "Depeche Mode")
    context, status, _ins, mock_save, _gf = await _run_finish(out)

    state = context.chat_data["q:1"]
    assert state["target"] == {"artist": "Depeche Mode", "track": None}
    # the existing keys stay (the contract adds target, replaces nothing)
    for key in ("candidates", "confidence", "best_index", "verdict", "intro",
                "desc", "rejected_ids", "intent", "reference", "ts"):
        assert key in state, f"state lost its '{key}' key"
    # SQLite round-trip: _state_serialize must keep target JSON-clean
    mock_save.assert_awaited_once()
    persisted = json.loads(mock_save.await_args.args[1])
    assert persisted["target"]["artist"] == "Depeche Mode"


async def test_save_state_target_empty_dict_is_safe():
    """this_track searches without an artist target store an empty dict — never
    a crash, never a None that would break JSON (need_clarify stops the flow
    before the download path; here only the state write is under test)."""
    out = _fake_out(target={}, need_clarify=True, clarify_question="Какая песня?")
    context, _status, _ins, _save, _gf = await _run_finish(out)
    assert context.chat_data["q:1"]["target"] == {}


# ════ 2. the target artist is never banned by the «Не то» restart ════

async def test_miss_streak_never_bans_target_artist():
    """(2026-11-14) The 14.11 scenario: 3 «Не то» on «Depeche Mode - Topic» /
    «Depeche Mode» candidates with state.target.artist='Depeche Mode' — the
    restart's excluded_artists must NOT contain 'depeche mode' (the band
    banning itself pushes the pool into covers/foreign artists)."""
    mock_do = await _run_fb_miss(_dm_store())
    mock_do.assert_awaited_once()
    excluded = mock_do.await_args.kwargs["excluded_artists"]
    assert "depeche mode" not in excluded
    # the rejected video_ids and titles are still fully excluded
    assert mock_do.await_args.kwargs["excluded_ids"] == {"dm1", "dm2", "cov1"}


async def test_miss_streak_still_bans_foreign_cover():
    """The mixed scenario: 2 rejected DM songs + 1 rejected cover of another
    channel — only the cover's channel gets banned."""
    mock_do = await _run_fb_miss(_dm_store())
    excluded = mock_do.await_args.kwargs["excluded_artists"]
    assert excluded == {_ref_key("Acoustic Covers")}


async def test_protection_falls_back_to_scope_artist_key():
    """Legacy states saved before the 'target' key may only carry
    'scope_artist' — the protection must work from that fallback too."""
    store = _dm_store()
    del store["target"]
    store["scope_artist"] = "Depeche Mode"
    mock_do = await _run_fb_miss(store)
    excluded = mock_do.await_args.kwargs["excluded_artists"]
    assert "depeche mode" not in excluded
    assert _ref_key("Acoustic Covers") in excluded


async def test_this_track_regression_bans_still_work():
    """Regression: an ordinary this_track search (target.track set, no target
    artist) keeps the old behavior — every rejected channel is banned."""
    store = _dm_store(target={"artist": None, "track": "Enjoy the Silence"})
    mock_do = await _run_fb_miss(store)
    excluded = mock_do.await_args.kwargs["excluded_artists"]
    assert excluded == {_ref_key("Depeche Mode"), _ref_key("Acoustic Covers")}


async def test_old_state_without_target_does_not_crash():
    """Hostile: an old-format state with neither 'target' nor 'scope_artist' —
    no crash, previous behavior (all rejected channels banned)."""
    store = _dm_store()
    del store["target"]
    mock_do = await _run_fb_miss(store)
    excluded = mock_do.await_args.kwargs["excluded_artists"]
    assert excluded == {_ref_key("Depeche Mode"), _ref_key("Acoustic Covers")}


# ════ 3. artist-only never auto-downloads: preview always ════

async def test_artist_scope_disables_auto_and_shows_preview():
    """scope_artist='Depeche Mode' with the auto-download conditions met
    (this_track, match_score 0.9 >= 0.65, high confidence, no ambiguity):
    auto must be False, no download may start, and the preview (pk: pick
    buttons + the core's intro head line) is shown for the user to choose."""
    out = _scoped(_fake_out(), "Depeche Mode")
    context, status, mock_insert, _save, mock_cached = await _run_finish(out)

    # the decision is persisted as preview, not auto-download
    assert mock_insert.await_args.kwargs["auto_download"] is False
    # the preview replaced the download: pick buttons, no file fetch
    status.edit_text.assert_awaited()
    text, kwargs = (status.edit_text.await_args.args[0],
                    status.edit_text.await_args.kwargs)
    callbacks = [b.callback_data for row in kwargs["reply_markup"].inline_keyboard
                 for b in row]
    assert any(cb.startswith("pk:1:") for cb in callbacks), "no song pick buttons"
    # every DM song of the pool is offered (the incident's ask: her songs)
    assert "Enjoy the Silence" in text and "Personal Jesus" in text
    # the core's intro is passed through as the head line, nothing new added
    assert "Ищу: песни Depeche Mode." in text
    # no download was started for the "best" guess
    mock_cached.assert_not_awaited()


async def test_no_scope_keeps_auto():
    """scope_artist absent or '' — the previous behavior: the same conditions
    record auto_download=True (the download path itself is out of scope here;
    need_clarify stops the flow right after the decision is persisted)."""
    for out in (_fake_out(need_clarify=True, clarify_question="Какая песня?"),
                _scoped(_fake_out(need_clarify=True,
                                  clarify_question="Какая песня?"), "")):
        _ctx, _status, mock_insert, _save, _gf = await _run_finish(out)
        assert mock_insert.await_args.kwargs["auto_download"] is True, \
            "without an artist-only scope the auto decision must not change"


async def test_artist_scope_ambiguous_flag_still_recorded():
    """The ambiguous/auto bookkeeping keeps working in artist-only mode: the
    queries row still carries the real ambiguity verdict for the funnel."""
    out = _scoped(_fake_out(confidence="low"), "Depeche Mode")
    _ctx, _status, mock_insert, _save, _gf = await _run_finish(out)
    assert mock_insert.await_args.kwargs["ambiguous"] is True
    assert mock_insert.await_args.kwargs["auto_download"] is False
