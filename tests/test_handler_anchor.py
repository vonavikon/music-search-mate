"""Clarification memory (audit П1, prod 12.11 «мадагаскар 2»): the handler side.

A from_source search fixes the movie; the user's clarify answers are fragments
with no source in them. The handler must (1) remember the from_source reference
source per user (TTL 15 min), (2) supply it as source_anchor= to
search_candidates for every NEW user text, (3) never for refinement envelopes
(ORIGINAL_QUERY: ...), the sim button or restarts, (4) drop it on /new.

Storage is asserted on the module dict bot.handler._SOURCE_ANCHOR (user_id ->
(source, ts)); the supply — on the search_candidates mock kwargs. Fixture and
driver patterns follow tests/test_handler.py (MagicMock/AsyncMock over real
dicts, conftest meta/search-cache isolation applies automatically).
"""
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import User

import bot.handler as handler
from bot.handler import (
    _SOURCE_ANCHOR, _SOURCE_ANCHOR_TTL, _anchor_for, _finish_search,
    _remember_source_anchor, handle_message, handle_new,
)
from services.llm_search import Candidate, SearchOut

UID = 123  # the house test user (test_handler.py)


@pytest.fixture(autouse=True)
def _hermetic_usage(monkeypatch):
    """Zero the usage counters (same class as test_handler.py): a dev DB with
    accumulated counters would silently stop the path before _do_search."""
    monkeypatch.setattr(
        "bot.handler.get_usage",
        AsyncMock(return_value={"searches": 0, "urls": 0, "sends": 0,
                                "facts": 0, "bytes_sent": 0}))
    monkeypatch.setattr("bot.handler.get_global_usage",
                        AsyncMock(return_value={"searches": 0}))


@pytest.fixture(autouse=True)
def _isolated_anchor_store():
    """The anchor store is module-level state — never leak it between tests."""
    _SOURCE_ANCHOR.clear()
    yield
    _SOURCE_ANCHOR.clear()


# ════ helpers (patterns lifted from tests/test_handler.py) ════

def _fake_out(**kw):
    """A SearchOut; the default is the prod П1 shape — a from_source search that
    ends in a clarify question (the branch that triggered the 12.11 incident)."""
    defaults = {
        "intent": "from_source",
        "signals": {},
        "queries": ["madagascar 2 soundtrack"],
        "assoc": [],
        "candidates": [
            Candidate("v1", "Test Track", "Test Artist", 200, 500000,
                      score=0.85, match_score=0.72),
        ],
        "best_index": 0,
        "confidence": "high",
        "intro": "Ищу: саундтрек.",
        "verdict": "OK",
        "need_clarify": True,
        "clarify_question": "Из какого фильма, игры или рекламы трек?",
        "reference": {"source": "мадагаскар 2"},
        "target": {},
    }
    defaults.update(kw)
    return SearchOut(**defaults)


async def _run_finish(out):
    """Drive _finish_search directly (the shape of _run_finish_search in
    test_handler.py, cut to the clarify branch — no media/download needed)."""
    update = MagicMock()
    update.effective_user = User(id=UID, is_bot=False, first_name="Test")
    update.effective_chat = MagicMock()
    update.effective_chat.id = 456
    context = MagicMock()
    context.bot_data = {"media": MagicMock()}
    context.chat_data = {}
    status = AsyncMock()
    with patch("bot.handler.insert_query", AsyncMock(return_value=1)), \
         patch("bot.handler.save_query_state", AsyncMock()), \
         patch("bot.handler.incr_usage", AsyncMock()), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")):
        await _finish_search(update, context, status, out, "test query",
                             update.effective_user)
    return status


async def _drive_message(text, search_out, fake_config, *, chat_id=456,
                         refine=None):
    """Drive handle_message directly (the house pattern: MagicMock update and
    context over real dicts) with search_candidates faked. Returns the mock so
    tests assert on its call kwargs."""
    update = MagicMock()
    update.effective_user = User(id=UID, is_bot=False, first_name="Test")
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    update.message.text = text
    status = AsyncMock()
    update.message.reply_text = AsyncMock(return_value=status)

    context = MagicMock()
    context.bot_data = {"cfg": fake_config, "media": MagicMock(),
                        "bothub": MagicMock(), "metadata": MagicMock()}
    context.chat_data = {"refine": refine} if refine else {}

    mock_search = AsyncMock(return_value=search_out)
    with patch("bot.handler.count_recent", AsyncMock(return_value=0)), \
         patch("bot.handler.recent_context", AsyncMock(return_value=[])), \
         patch("bot.handler.search_candidates", mock_search), \
         patch("bot.handler.rerank_pool", AsyncMock(return_value=None)), \
         patch("bot.handler.get_session", AsyncMock(return_value=None)), \
         patch("bot.handler.insert_query", AsyncMock(return_value=1)), \
         patch("bot.handler.save_query_state", AsyncMock()), \
         patch("bot.handler.incr_usage", AsyncMock()), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")):
        await handle_message(update, context)
    return mock_search


# ════ 1. storage ════

async def test_from_source_result_stores_anchor():
    """A finished from_source search with reference.source stores the anchor
    per user — the exact prod shape (the search ends in a clarify question)."""
    before = time.time()
    await _run_finish(_fake_out())
    entry = _SOURCE_ANCHOR.get(UID)
    assert entry is not None, "the from_source reference source must be remembered"
    source, ts = entry
    assert source == "мадагаскар 2"
    assert ts >= before  # fresh ts (TTL runs from the store moment)


async def test_from_source_reference_name_fallback():
    """reference.name is the fallback source when .source is absent."""
    await _run_finish(_fake_out(reference={"name": "Такси"}))
    assert _SOURCE_ANCHOR[UID][0] == "Такси"


async def test_different_from_source_overwrites():
    """A later from_source search with a different source replaces the anchor."""
    await _run_finish(_fake_out(reference={"source": "мадагаскар 2"}))
    await _run_finish(_fake_out(reference={"source": "Такси"}))
    assert _SOURCE_ANCHOR[UID][0] == "Такси"
    assert len(_SOURCE_ANCHOR) == 1


async def test_anchor_store_cap_drops_oldest():
    """Past 64 users the oldest entry (by ts) is dropped, never a silent grow."""
    now = time.time()
    for uid in range(1000, 1064):  # 64 fresh entries, ts decreasing with uid
        _SOURCE_ANCHOR[uid] = (f"src{uid}", now - (uid - 1000) * 0.001)
    _remember_source_anchor(UID, _fake_out(reference={"source": "мадагаскар 2"}))
    assert len(_SOURCE_ANCHOR) == 64
    assert _SOURCE_ANCHOR[UID][0] == "мадагаскар 2"
    assert 1063 not in _SOURCE_ANCHOR  # the oldest (by ts) went first
    assert 1000 in _SOURCE_ANCHOR


# ════ 2. supply ════

async def test_next_text_search_receives_anchor(fake_config):
    """The 12.11 case: after «мадагаскар 2» the fragment answer («нет, ее
    включали когда угнали машину») is a NEW text -> its search_candidates call
    carries source_anchor='мадагаскар 2'."""
    _SOURCE_ANCHOR[UID] = ("мадагаскар 2", time.time())
    mock_search = await _drive_message(
        "нет, ее включали когда угнали машину", _fake_out(), fake_config,
        chat_id=457)
    mock_search.assert_awaited_once()
    assert mock_search.await_args.kwargs["source_anchor"] == "мадагаскар 2"


async def test_no_anchor_no_kwarg(fake_config):
    """With nothing remembered the call shape stays the pre-П1 one: no
    source_anchor kwarg at all."""
    mock_search = await _drive_message(
        "бодрое как AC/DC", _fake_out(intent="mood"), fake_config, chat_id=458)
    mock_search.assert_awaited_once()
    assert "source_anchor" not in mock_search.await_args.kwargs


# ════ 3. expiry ════

def test_anchor_for_expires_beyond_ttl():
    """An entry older than the TTL yields None and is dropped lazily."""
    _SOURCE_ANCHOR[UID] = ("мадагаскар 2", time.time() - _SOURCE_ANCHOR_TTL - 5)
    assert _anchor_for(UID, "это трек 80х") is None
    assert UID not in _SOURCE_ANCHOR  # dropped, not just masked


def test_anchor_for_fresh_returns_source():
    _SOURCE_ANCHOR[UID] = ("мадагаскар 2", time.time() - 60)
    assert _anchor_for(UID, "это трек 80х") == "мадагаскар 2"


async def test_expired_anchor_not_supplied(fake_config):
    """Integration: a stale anchor must not reach the search call."""
    _SOURCE_ANCHOR[UID] = ("мадагаскар 2", time.time() - _SOURCE_ANCHOR_TTL - 5)
    mock_search = await _drive_message(
        "это трек 80х, его там на кассете включают", _fake_out(), fake_config,
        chat_id=459)
    assert "source_anchor" not in mock_search.await_args.kwargs


# ════ 4. no storage for non-from_source; /new clears ════

async def test_non_from_source_does_not_store():
    """similar/mood results and from_source without a source never touch the
    store (the TTL is the only expiry — nothing else clears the anchor)."""
    await _run_finish(_fake_out(intent="similar", reference={}))
    await _run_finish(_fake_out(intent="mood", reference={}))
    await _run_finish(_fake_out(reference={}))            # from_source, no source
    await _run_finish(_fake_out(reference={"artist": "X"}))  # artist is not a source
    assert _SOURCE_ANCHOR == {}


async def test_new_command_clears_anchor():
    """/new starts a fresh search — the remembered source dies with the context."""
    _SOURCE_ANCHOR[UID] = ("мадагаскар 2", time.time())
    update = MagicMock()
    update.effective_user = User(id=UID, is_bot=False, first_name="Test")
    update.message.reply_text = AsyncMock()
    context = MagicMock()
    context.bot_data = {"cfg": None}  # empty allowlist -> _denied is False
    context.chat_data = {}
    with patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")):
        await handle_new(update, context)
    assert UID not in _SOURCE_ANCHOR


# ════ 5. the ORIGINAL_QUERY guard ════

def test_anchor_for_skips_refinement_envelope():
    """A refine rewrite carries its own reference — the guard must yield None
    even with a fresh anchor stored."""
    _SOURCE_ANCHOR[UID] = ("мадагаскар 2", time.time())
    envelope = ("ORIGINAL_QUERY: мадагаскар 2 трек\n\n"
                "REFINEMENT: нет, ее включали когда угнали машину\n\n"
                "(REFINEMENT корректирует ORIGINAL_QUERY; при конфликте приоритет "
                "за REFINEMENT)")
    assert _anchor_for(UID, envelope) is None
    # plain new text still gets it
    assert _anchor_for(UID, "нет, ее включали когда угнали машину") == "мадагаскар 2"


async def test_refine_path_gets_no_anchor(fake_config):
    """Integration: text inside the refine window merges into an ORIGINAL_QUERY
    envelope and its search_candidates call carries no source_anchor."""
    _SOURCE_ANCHOR[UID] = ("мадагаскар 2", time.time())
    refine = {"ts": time.monotonic(),
              "desc": "мадагаскар 2 трек на кассете",
              "rejected_ids": set()}
    mock_search = await _drive_message(
        "это трек 80х, его там на кассете включают", _fake_out(), fake_config,
        chat_id=460, refine=refine)
    mock_search.assert_awaited_once()
    assert "ORIGINAL_QUERY:" in mock_search.await_args.args[0]
    assert "source_anchor" not in mock_search.await_args.kwargs


# ════ robustness ════

def test_anchor_for_never_raises_on_garbage():
    """A failure to supply an anchor must not break a search: the helper
    swallows anything (here: a broken store entry)."""
    _SOURCE_ANCHOR[UID] = None  # type: ignore[assignment]  # corrupted entry
    assert _anchor_for(UID, "фрагмент") is None


def test_remember_never_raises_on_garbage_out():
    """A malformed SearchOut/reference must not blow up _finish_search."""
    _remember_source_anchor(UID, None)
    _remember_source_anchor(UID, object())
    _remember_source_anchor(UID, _fake_out(reference=None))  # type: ignore[arg-type]
    assert _SOURCE_ANCHOR == {}
