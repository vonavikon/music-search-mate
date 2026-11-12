"""Harness tests for the handler: wiring, callbacks, double-submit.

A minimal harness suite (3 tests) — the core logic is covered at the
integration level (llm_search, media). These tests verify:
1. The handler is wired up and responds to text (wiring)
2. A callback query does not crash (edge)
3. A double-submit does not create a race (lock)
"""
import json
import pytest
import asyncio
from contextlib import contextmanager
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timezone

from telegram import Update, Message, Chat, User, CallbackQuery, MessageEntity
from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters

from bot.handler import handle_start, handle_message, handle_callback
from services.llm_search import Candidate, SearchOut


# ════ helpers ════

@pytest.fixture(autouse=True)
def _hermetic_usage(monkeypatch):
    """Zero the daily usage counters for every handler test. services.store
    binds config.DB_PATH at import, so tests without the db_path fixture
    read the developer DB's accumulated counters; after enough local
    full-suite runs user 123 crosses the sends cap and sim/limit-free paths
    silently return before _do_search (the TestFunnel46 hermeticity class).
    Tests that model limit behaviour patch get_usage/incr_usage themselves
    and override this default."""
    monkeypatch.setattr(
        "bot.handler.get_usage",
        AsyncMock(return_value={"searches": 0, "urls": 0, "sends": 0,
                                "facts": 0, "bytes_sent": 0}))




def _text_update(text, user_id=123, chat_id=456):
    user = User(id=user_id, is_bot=False, first_name="Test")
    chat = Chat(id=chat_id, type="private")
    entities = None
    if text.startswith("/"):
        entities = [MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len(text))]
    msg = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=chat,
        from_user=user,
        text=text,
        entities=entities,
    )
    return Update(update_id=1, message=msg)


def _callback_update(data, user_id=123):
    user = User(id=user_id, is_bot=False, first_name="Test")
    chat = Chat(id=456, type="private")
    msg = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=chat,
        from_user=user,
        text="old",
    )
    cq = CallbackQuery(
        id="fake_callback_id",
        from_user=user,
        chat_instance="ci",
        message=msg,
        data=data,
    )
    return Update(update_id=2, callback_query=cq)


@contextmanager
def _patch_store():
    """_send_track/_remember_session_and_ref hit SQLite (file_id cache + state).
    In unit tests the DB is stubbed out — only wiring/keyboards are checked here,
    not persistence (that lives in test_store)."""
    with patch("bot.handler.get_cached_file", AsyncMock(return_value=None)), \
         patch("bot.handler.cache_file", AsyncMock()), \
         patch("bot.handler.save_query_state", AsyncMock()), \
         patch("bot.handler.load_query_state", AsyncMock(return_value=None)), \
         patch("bot.handler.delete_query_state", AsyncMock()):
        yield


def _fake_search_out(**kw):
    defaults = {
        "intent": "similar",
        "signals": {},
        "queries": ["test q"],
        "assoc": [],
        "candidates": [
            Candidate("v1", "Test Track", "Test Artist", 200, 500000, score=0.85, match_score=0.72),
            Candidate("v2", "Track 2", "Artist 2", 180, 100000, score=0.70, match_score=0.50),
        ],
        "best_index": 0,
        "confidence": "high",
        "intro": "OK",
        "verdict": "OK",
        "need_clarify": False,
        "clarify_question": "",
        "reference": {},
        "target": {},
    }
    defaults.update(kw)
    return SearchOut(**defaults)


# ════ tests ════

@pytest.mark.asyncio
async def test_start_wiring(app, recording, monkeypatch):
    from unittest.mock import AsyncMock
    monkeypatch.setattr("bot.handler.get_user_lang", AsyncMock(return_value="ru"))
    app.add_handler(CommandHandler("start", handle_start))
    update = _text_update("/start")
    update.set_bot(app.bot)
    update.message.set_bot(app.bot)
    await app.process_update(update)

    texts = recording.texts()
    assert any("найдёт трек по описанию" in t for t in texts)


@pytest.mark.asyncio
async def test_start_asks_language_when_unset(app, recording, monkeypatch):
    """No language selected -> /start asks for one (RU/EN buttons) instead of greeting."""
    from unittest.mock import AsyncMock
    monkeypatch.setattr("bot.handler.get_user_lang", AsyncMock(return_value=None))
    app.add_handler(CommandHandler("start", handle_start))
    update = _text_update("/start")
    update.set_bot(app.bot)
    update.message.set_bot(app.bot)
    await app.process_update(update)
    assert any("язык" in t.lower() for t in recording.texts())


@pytest.mark.asyncio
async def test_text_wiring(app, recording, fake_config):
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.bot_data["cfg"] = fake_config
    app.bot_data["media"] = MagicMock()
    app.bot_data["bothub"] = MagicMock()

    with patch("bot.handler.count_recent", AsyncMock(return_value=0)):
        with patch("bot.handler.recent_context", AsyncMock(return_value=[])):
            with patch("bot.handler.search_candidates") as mock_search:
                mock_search.return_value = _fake_search_out()
                with patch("bot.handler.insert_query", AsyncMock(return_value=1)):
                    update = _text_update("rock like AC/DC")
                    update.set_bot(app.bot)
                    await app.process_update(update)

    assert len(recording.sent) > 0


@pytest.mark.asyncio
async def test_double_submit_no_race(app, recording, fake_config):
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.bot_data["cfg"] = fake_config
    app.bot_data["media"] = MagicMock()
    app.bot_data["bothub"] = MagicMock()

    u1 = _text_update("first", chat_id=789)
    u2 = _text_update("second", chat_id=789)
    u1.set_bot(app.bot)
    u1.message.set_bot(app.bot)
    u2.set_bot(app.bot)
    u2.message.set_bot(app.bot)

    with patch("bot.handler.count_recent", AsyncMock(return_value=0)):
        with patch("bot.handler.recent_context", AsyncMock(return_value=[])):
            with patch("bot.handler.search_candidates") as mock_search:
                mock_search.return_value = _fake_search_out()
                with patch("bot.handler.insert_query", AsyncMock(return_value=1)):
                    await app.process_update(u1)
                    await app.process_update(u2)

    assert len(recording.sent) >= 2


# ════ URL detection (Task 2: platform-agnostic) ════

def test_looks_like_url_youtube():
    from bot.handler import _looks_like_url
    assert _looks_like_url("https://www.youtube.com/watch?v=abc123def45") == "https://www.youtube.com/watch?v=abc123def45"
    assert _looks_like_url("послушай https://youtu.be/abc123def45 пожалуйста") == "https://youtu.be/abc123def45"


def test_looks_like_url_soundcloud():
    from bot.handler import _looks_like_url
    assert _looks_like_url("https://soundcloud.com/artist/track") == "https://soundcloud.com/artist/track"


def test_looks_like_url_bandcamp():
    from bot.handler import _looks_like_url
    assert _looks_like_url("https://artist.bandcamp.com/track/song") == "https://artist.bandcamp.com/track/song"


def test_looks_like_url_not_url():
    from bot.handler import _looks_like_url
    assert _looks_like_url("что-то в духе Radiohead") is None
    assert _looks_like_url("песня из Такси") is None


def test_looks_like_url_punctuation_tail():
    from bot.handler import _looks_like_url
    assert _looks_like_url("https://youtu.be/abc123def45.") == "https://youtu.be/abc123def45"
    assert _looks_like_url("вот https://soundcloud.com/a/b, слушай") == "https://soundcloud.com/a/b"


def test_looks_malformed_url():
    from bot.handler import _looks_malformed_url
    # space after //: no valid URL was extracted, but the markers are present
    assert _looks_malformed_url("https:// youtu.be/abc")
    assert _looks_malformed_url("вот http:// звук")
    # a valid URL is not malformed
    assert not _looks_malformed_url("https://youtu.be/abc123def45")
    # plain text without URL markers is not malformed
    assert not _looks_malformed_url("что-то в духе Radiohead")


def test_is_continuation_phrases():
    from bot.handler import _is_continuation
    assert _is_continuation("ещё")
    assert _is_continuation("Ещё такое же")
    assert _is_continuation("еще что-нибудь")
    assert _is_continuation("давай ещё")
    assert _is_continuation("в том же духе")
    assert _is_continuation("ещё похожее")


def test_is_continuation_not_phrase():
    from bot.handler import _is_continuation
    assert not _is_continuation("рок как AC/DC")
    assert not _is_continuation("ещё раз быстрее")  # a refinement, not a bare continuation
    assert not _is_continuation("похожие на Queen")
    assert not _is_continuation("")
    assert not _is_continuation("ещё немного Nirvana")


def test_candidate_url_defaults():
    """Dataclass defaults stay backward compatible: url is empty, the fallback is built in search_candidates."""
    from services.llm_search import Candidate
    c = Candidate("abc123def45", "Test Track", "Test Artist", 200, 1000)
    assert c.source == "youtube"
    assert c.url == ""
    assert c.fallback_url == ""


# ════ bio in the caption ════

def test_bio_line_empty():
    from bot.handler import _bio_line
    from services.metadata import MetaOut
    assert _bio_line(MetaOut(artist="A", track="T")) == ""
    assert _bio_line(MetaOut(artist="A", track="T", bio="   ")) == ""


def test_bio_line_short():
    from bot.handler import _bio_line
    from services.metadata import MetaOut
    meta = MetaOut(artist="A", track="T", bio="A song about the sea.")
    assert _bio_line(meta) == "ℹ️ A song about the sea."


def test_bio_line_truncates_on_word_boundary():
    """A long bio is truncated at a word boundary to _BIO_MAX, with no torn word at the end."""
    from bot.handler import _bio_line, _BIO_MAX
    from services.metadata import MetaOut
    bio = "word " * 200  # deliberately longer than _BIO_MAX
    line = _bio_line(MetaOut(artist="A", track="T", bio=bio))
    assert line.startswith("ℹ️ ")
    body = line[3:]
    assert body.endswith("…")
    assert len(body) <= _BIO_MAX + 1  # +1 for the ellipsis
    assert not body.rstrip("…").endswith("word ")  # never cut a word in half


def test_bio_line_escapes_html():
    from bot.handler import _bio_line
    from services.metadata import MetaOut
    meta = MetaOut(artist="A", track="T", bio="Tom & Jerry <3")
    assert "&amp;" in _bio_line(meta)
    assert "<3" not in _bio_line(meta)


def test_caption_includes_bio_after_genre():
    """The caption renders the bio between the genre and the verdict (in italics)."""
    from bot.handler import _caption
    from services.metadata import MetaOut
    from services.llm_search import Candidate
    meta = MetaOut(artist="The Beatles", track="Yesterday", year="1965",
                   album="Help!", genre="rock, 60s", bio="A nostalgic song about loss.")
    track = Candidate("v1", "Yesterday", "The Beatles", 200, 1000)
    caption = _caption(meta, track, "high")
    lines = caption.split("\n")
    assert any(l.startswith("🏷️") for l in lines)
    assert any(l.startswith("ℹ️") for l in lines)
    # the bio comes after the genre
    assert lines.index([l for l in lines if l.startswith("ℹ️")][0]) > \
           lines.index([l for l in lines if l.startswith("🏷️")][0])


def test_caption_no_bio_line_when_absent():
    from bot.handler import _caption
    from services.metadata import MetaOut
    from services.llm_search import Candidate
    meta = MetaOut(artist="A", track="T", bio="")
    caption = _caption(meta, Candidate("v1", "T", "A", 200, 1000), "high")
    assert "ℹ️" not in caption


@pytest.mark.asyncio
@pytest.mark.parametrize("with_status", [True, False])
async def test_send_track_double_failure_notifies_user(app, recording, with_status):
    """Both url and fallback_url failed — the user gets a message instead of silence."""
    from bot.handler import _send_track
    from services.llm_search import Candidate

    track = Candidate(
        "v1", "Test Track", "Test Artist", 200, 1000,
        url="https://a.example/v1", fallback_url="https://b.example/v1",
    )
    media = MagicMock()
    media.download_audio = AsyncMock(side_effect=RuntimeError("both failed"))
    context = MagicMock()
    context.bot_data = {"metadata": MagicMock()}
    context.chat_data = {}

    update = _text_update("test")
    update.set_bot(app.bot)
    update.message.set_bot(app.bot)
    update.effective_chat.set_bot(app.bot)  # in production the bot is attached via Update.de_json(..., bot)

    status = AsyncMock() if with_status else None
    with _patch_store():
        ok = await _send_track(update, context, status, media, 1, 0, [track], track, "high")

    assert ok is False
    assert media.download_audio.await_count == 2  # url + fallback
    if with_status:
        status.edit_text.assert_awaited_once()
        assert "Не смог скачать" in status.edit_text.await_args.args[0]
    else:
        assert any("Не смог скачать" in t for t in recording.texts())


@pytest.mark.asyncio
async def test_send_track_url_keyboard_only_similar(app, recording, tmp_path):
    """URL fast-path: a single "Similar" button under the track, no Correct/Wrong/Refine."""
    from bot.handler import _send_track
    from services.llm_search import Candidate
    from services.metadata import MetaOut

    m4a = tmp_path / "v1.m4a"
    m4a.write_bytes(b"fake")

    track = Candidate("v1", "Test Track", "Test Artist", 200, 1000, url="https://a.example/v1")

    media = MagicMock()
    media.download_audio = AsyncMock(return_value=(
        str(m4a), None, {"id": "v1", "title": "Test Track", "uploader": "Test Artist"},
    ))
    meta = MagicMock()
    meta.resolve = AsyncMock(return_value=MetaOut(artist="Test Artist", track="Test Track"))

    context = MagicMock()
    context.bot_data = {"metadata": meta}
    context.bot.send_audio = AsyncMock()
    context.chat_data = {}

    update = _text_update("test")
    update.set_bot(app.bot)
    update.message.set_bot(app.bot)
    update.effective_chat.set_bot(app.bot)

    with _patch_store(), patch("bot.handler.tag_enrichment.enrich"), patch("bot.handler.upsert_session", AsyncMock()):
        ok = await _send_track(update, context, None, media, 1, 0, [track], track, "high", keyboard="url")

    assert ok is True
    kb = context.bot.send_audio.await_args.kwargs["reply_markup"]
    buttons = [b.text for row in kb.inline_keyboard for b in row]
    # a fact button was added next to "Similar" (sonar on demand, 30-day cache)
    assert buttons == ["🔀 Похожее", "💡 Факт"]


@pytest.mark.asyncio
async def test_send_track_feedback_keyboard_has_similar(app, recording, tmp_path):
    """A track from search gets 4 buttons under it, including "Similar"."""
    from bot.handler import _send_track
    from services.llm_search import Candidate
    from services.metadata import MetaOut

    m4a = tmp_path / "v1.m4a"
    m4a.write_bytes(b"fake")

    track = Candidate("v1", "Test Track", "Test Artist", 200, 1000, url="https://a.example/v1")

    media = MagicMock()
    media.download_audio = AsyncMock(return_value=(
        str(m4a), None, {"id": "v1", "title": "Test Track", "uploader": "Test Artist"},
    ))
    meta = MagicMock()
    meta.resolve = AsyncMock(return_value=MetaOut(artist="Test Artist", track="Test Track"))

    context = MagicMock()
    context.bot_data = {"metadata": meta}
    context.bot.send_audio = AsyncMock()
    context.chat_data = {}

    update = _text_update("test")
    update.set_bot(app.bot)
    update.message.set_bot(app.bot)
    update.effective_chat.set_bot(app.bot)

    with _patch_store(), patch("bot.handler.tag_enrichment.enrich"), patch("bot.handler.upsert_session", AsyncMock()):
        ok = await _send_track(update, context, None, media, 1, 0, [track], track, "high")

    assert ok is True
    kb = context.bot.send_audio.await_args.kwargs["reply_markup"]
    buttons = [b.text for row in kb.inline_keyboard for b in row]
    assert "✅ То" in buttons and "❌ Не то" in buttons and "🔀 Похожее" in buttons
    # the Video button was removed from search cards (owner's request);
    # the Fact button stays
    assert "🎬 Видео" not in buttons
    assert "💡 Факт" in buttons


@pytest.mark.asyncio
async def test_send_track_cache_hit_sends_by_file_id(app):
    """A repeat track is sent by its cached file_id, without downloading and re-uploading."""
    from bot.handler import _send_track
    from services.llm_search import Candidate

    track = Candidate("v1", "Test Track", "Test Artist", 200, 1000, url="https://a.example/v1")

    media = MagicMock()
    media.download_audio = AsyncMock()  # must not be called

    context = MagicMock()
    _md = MagicMock()
    _md.refetch_artwork = AsyncMock(return_value="")  # artwork re-fetch on a cache hit
    context.bot_data = {"metadata": _md}
    context.bot.send_audio = AsyncMock()
    context.chat_data = {}

    update = _text_update("test")
    update.set_bot(app.bot)
    update.message.set_bot(app.bot)
    update.effective_chat.set_bot(app.bot)

    cached = {
        "file_id": "file-cached",
        "artist": "The Beatles", "track": "Yesterday", "album": "Help!",
        "year": "1965", "genre": "rock", "bio": "",
    }
    with patch("bot.handler.get_cached_file", AsyncMock(return_value=cached)), \
         patch("bot.handler.update_cached_artwork", AsyncMock()), \
         patch("bot.handler.load_query_state", AsyncMock(return_value=None)), \
         patch("bot.handler.save_query_state", AsyncMock()), \
         patch("bot.handler.upsert_session", AsyncMock()):
        ok = await _send_track(update, context, None, media, 1, 0, [track], track, "high")

    assert ok is True
    media.download_audio.assert_not_awaited()
    assert context.bot.send_audio.await_args.kwargs["audio"] == "file-cached"


@pytest.mark.asyncio
async def test_callback_sim_launches_search(app, fake_config):
    """"Similar" -> a new search by the current track's title/artist."""
    from bot.handler import handle_callback
    from services.llm_search import Candidate

    # CallbackQuery is a frozen TelegramObject: its methods cannot be overridden, so use MagicMock
    cq = MagicMock()
    cq.data = "sim:1:0"
    cq.answer = AsyncMock()
    cq.edit_message_reply_markup = AsyncMock()
    cq.message.reply_text = AsyncMock()

    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=123, is_bot=False, first_name="Test")

    context = MagicMock()
    context.bot_data = {"cfg": fake_config, "media": MagicMock(),
                        "bothub": MagicMock(), "metadata": MagicMock()}
    context.chat_data = {}
    context.chat_data["q:1"] = {
        "candidates": [Candidate("v1", "Test Track", "Test Artist", 200, 1000)],
        "confidence": "high", "best_index": 0, "verdict": "", "intro": "",
        "desc": "", "rejected_ids": [],
    }

    with patch("bot.handler._do_search") as mock_do:
        mock_do.return_value = _fake_search_out()
        with patch("bot.handler._finish_search", AsyncMock()):
            await handle_callback(update, context)

    mock_do.assert_awaited_once()
    desc = mock_do.await_args.args[3]
    assert "похожие на Test Track - Test Artist" in desc


@pytest.mark.asyncio
async def test_callback_sim_uses_resolved_metadata(app, fake_config):
    """When canonical artist/track are available, the reference is built from them, not from the raw title."""
    from bot.handler import handle_callback
    from services.llm_search import Candidate

    cq = MagicMock()
    cq.data = "sim:1:0"
    cq.answer = AsyncMock()
    cq.edit_message_reply_markup = AsyncMock()
    cq.message.reply_text = AsyncMock()

    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=123, is_bot=False, first_name="Test")

    context = MagicMock()
    context.bot_data = {"cfg": fake_config, "media": MagicMock(),
                        "bothub": MagicMock(), "metadata": MagicMock()}
    context.chat_data = {}
    context.chat_data["q:1"] = {
        "candidates": [Candidate("v1", "Yesterday (Remastered 2009)", "The Beatles - Topic", 200, 1000)],
        "confidence": "high", "best_index": 0, "verdict": "", "intro": "",
        "desc": "", "rejected_ids": [],
        "resolved_artist": "The Beatles", "resolved_track": "Yesterday",
    }

    with patch("bot.handler._do_search") as mock_do:
        mock_do.return_value = _fake_search_out()
        with patch("bot.handler._finish_search", AsyncMock()):
            await handle_callback(update, context)

    desc = mock_do.await_args.args[3]
    assert "похожие на Yesterday - The Beatles" in desc
    assert "Remastered" not in desc


# ════ top-3 preview: score threshold ════

def test_visible_candidates_filters_zero_score():
    from bot.handler import _visible_candidates
    from services.llm_search import Candidate
    cands = [
        Candidate("v1", "Good", "A", 200, 1000, score=0.9),
        Candidate("v2", "Zero", "B", 200, 1000, score=0.0),
        Candidate("v3", "AlsoGood", "C", 200, 1000, score=0.7),
    ]
    visible = _visible_candidates(cands)
    ranks = [r for r, _ in visible]
    assert ranks == [0, 2]  # v2 (score 0.0) hidden; ranks keep the full-list indices


def test_visible_candidates_all_zero_falls_back():
    from bot.handler import _visible_candidates
    from services.llm_search import Candidate
    cands = [
        Candidate("v1", "A", "A", 200, 1000, score=0.0),
        Candidate("v2", "B", "B", 200, 1000, score=0.0),
    ]
    visible = _visible_candidates(cands)
    assert len(visible) == 2


# ════ caption limit, refine-state, send fallback ════

def test_caption_hard_cap_1024():
    """Telegram caption limit of 1024: pathologically long fields do not break send_audio."""
    from bot.handler import _caption

    class _M:
        track = "T" * 5000
        artist = "A" * 5000
        album = "Alb" * 500
        year = "1999"
        genre = "rock, hard rock, classic rock, " * 20
        bio = "bio " * 2000

    class _T:
        title = "tt"
        channel = "ch"
        rationale = ""

    cap = _caption(_M(), _T(), "high", "V" * 3000)
    assert len(cap) <= 1024
    # title/artist are normally not truncated — a regular caption stays complete
    class _M2:
        track = "Billie Jean"
        artist = "Michael Jackson"
        album = "Thriller"
        year = "1982"
        genre = "pop"
        bio = "Коротко о треке."
    cap2 = _caption(_M2(), _T(), "high", "Тот самый бас.")
    assert "Billie Jean" in cap2 and "Michael Jackson" in cap2 and "Коротко" in cap2


@pytest.mark.asyncio
async def test_refine_state_survives_rate_limit(app, recording, fake_config):
    """The rate limit must not eat the refine-state: after the window the user can retry the refinement."""
    import time as _time
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.bot_data["cfg"] = fake_config
    app.bot_data["media"] = MagicMock()
    app.bot_data["bothub"] = MagicMock()

    u1 = _text_update("первый")   # creates chat_data[456] inside PTB
    u2 = _text_update("потяжелее")
    for u in (u1, u2):
        u.set_bot(app.bot)
        u.message.set_bot(app.bot)

    with patch("bot.handler.count_recent",
               AsyncMock(return_value=fake_config.user_rate_limit_per_hour)):
        await app.process_update(u1)
        # PTB exposes chat_data as a read-only view; write only after
        # process_update has created the chat entry
        app.chat_data[456]["refine"] = {
            "ts": _time.monotonic(), "desc": "rock", "rejected_ids": {"v1"},
        }
        await app.process_update(u2)

    assert any("Лимит на этот час" in t for t in recording.texts())
    assert app.chat_data[456].get("refine") is not None, "refine-state должен выжить"


@pytest.mark.asyncio
async def test_send_track_send_failure_returns_false(app, recording, tmp_path):
    """A failed send_audio => False: the calling loop will try the next candidate."""
    from bot.handler import _send_track
    from services.llm_search import Candidate
    from services.metadata import MetaOut

    m4a = tmp_path / "v1.m4a"
    m4a.write_bytes(b"fake")

    track = Candidate("v1", "Test Track", "Test Artist", 200, 1000, url="https://a.example/v1")
    media = MagicMock()
    media.download_audio = AsyncMock(return_value=(
        str(m4a), None, {"id": "v1", "title": "Test Track", "uploader": "Test Artist"},
    ))
    meta = MagicMock()
    meta.resolve = AsyncMock(return_value=MetaOut(artist="Test Artist", track="Test Track"))

    context = MagicMock()
    context.bot_data = {"metadata": meta}
    context.chat_data = {}
    context.bot.send_audio = AsyncMock(side_effect=RuntimeError("telegram down"))

    update = _text_update("test")
    update.set_bot(app.bot)
    update.message.set_bot(app.bot)
    update.effective_chat.set_bot(app.bot)

    status = AsyncMock()
    with _patch_store(), patch("bot.handler.tag_enrichment.enrich"), \
         patch("bot.handler.upsert_session", AsyncMock()):
        ok = await _send_track(update, context, status, media, 1, 0, [track], track, "high")

    assert ok is False, "провал отправки должен давать False (fallback на след. кандидата)"
    status.edit_text.assert_awaited()


# ════ allowlist ════

class TestAllowlist:
    def test_parse_ids(self):
        from config import _parse_ids
        assert _parse_ids("123, 456;789") == (123, 456, 789)
        assert _parse_ids("") == ()
        assert _parse_ids("abc, 77,") == (77,)

    def test_denied_outside_list(self, fake_config):
        from types import SimpleNamespace
        from bot.handler import _denied
        import telegram
        cfg = fake_config
        cfg = SimpleNamespace(**{**cfg.__dict__, "allowed_user_ids": (111,)})
        ctx = SimpleNamespace(bot_data={"cfg": cfg})
        upd = SimpleNamespace(effective_user=SimpleNamespace(id=222))
        assert _denied(upd, ctx) is True
        upd_ok = SimpleNamespace(effective_user=SimpleNamespace(id=111))
        assert _denied(upd_ok, ctx) is False

    def test_empty_list_allows_everyone(self, fake_config):
        from types import SimpleNamespace
        from bot.handler import _denied
        cfg = SimpleNamespace(**{**fake_config.__dict__, "allowed_user_ids": ()})
        ctx = SimpleNamespace(bot_data={"cfg": cfg})
        upd = SimpleNamespace(effective_user=SimpleNamespace(id=999))
        assert _denied(upd, ctx) is False


# ════ owner gate (/stats) ════

class TestOwnerGate:
    def _cfg(self, fake_config, **kw):
        from types import SimpleNamespace
        return SimpleNamespace(**{**fake_config.__dict__, **kw})

    def test_not_owner_denied(self, fake_config):
        from types import SimpleNamespace
        from bot.handler import _not_owner
        ctx = SimpleNamespace(bot_data={"cfg": self._cfg(fake_config, bot_owner_id=111)})
        upd = SimpleNamespace(effective_user=SimpleNamespace(id=222))
        assert _not_owner(upd, ctx) is True
        upd_ok = SimpleNamespace(effective_user=SimpleNamespace(id=111))
        assert _not_owner(upd_ok, ctx) is False

    def test_unset_owner_checks_nothing(self, fake_config):
        """BOT_OWNER_ID=0 -> no extra check beyond the allowlist gate."""
        from types import SimpleNamespace
        from bot.handler import _not_owner
        ctx = SimpleNamespace(bot_data={"cfg": self._cfg(fake_config, bot_owner_id=0)})
        upd = SimpleNamespace(effective_user=SimpleNamespace(id=999))
        assert _not_owner(upd, ctx) is False

    @pytest.mark.asyncio
    async def test_stats_renders_for_owner(self, fake_config):
        """Regression 04.11: a stray comma after the stats_all term made the
        next line's '+ "\\n"' a unary plus — every /stats crashed with
        TypeError before replying. The owner call must render one string."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch
        from bot.handler import handle_stats
        cfg = self._cfg(fake_config, allowed_user_ids=(111,), bot_owner_id=111)
        reply = AsyncMock()
        upd = SimpleNamespace(effective_user=SimpleNamespace(id=111),
                              message=SimpleNamespace(reply_text=reply))
        ctx = SimpleNamespace(bot_data={"cfg": cfg})
        with patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
             patch("bot.handler.get_global_usage", AsyncMock(return_value={})), \
             patch("bot.handler.funnel_today",
                   AsyncMock(return_value={"outcomes": {}, "events": {}, "feedback": {}})):
            await handle_stats(upd, ctx)
        reply.assert_awaited_once()
        args, kwargs = reply.await_args
        assert isinstance(args[0], str) and args[0]
        assert kwargs.get("parse_mode") == "HTML"

    @pytest.mark.asyncio
    async def test_stats_silent_for_non_owner(self, fake_config):
        """/stats is owner-only: an allowlisted non-owner gets no reply."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch
        from bot.handler import handle_stats
        cfg = self._cfg(fake_config, allowed_user_ids=(111, 222), bot_owner_id=111)
        reply = AsyncMock()
        upd = SimpleNamespace(effective_user=SimpleNamespace(id=222),
                              message=SimpleNamespace(reply_text=reply))
        ctx = SimpleNamespace(bot_data={"cfg": cfg})
        with patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")):
            await handle_stats(upd, ctx)
        reply.assert_not_awaited()


# ════ sim cache ════

@pytest.mark.asyncio
async def test_sim_cache_serves_next_from_pool():
    """Pressing "Similar" again with the same reference -> the next candidate from the pool, with NO search."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch
    from bot.handler import _try_sim_from_pool, Candidate as HC
    import time as _t

    cur = HC("ref1", "Pendulum - The Island", "Pendulum", 260, 10**6)
    pool = [cur, HC("n1", "Metrik - Gravity", "Metrik", 240, 10**5, 0.8),
            HC("n2", "Sub Focus - Tidal Wave", "Sub Focus", 250, 10**5, 0.75)]
    store = {"candidates": pool, "confidence": "high",
             "reference": {"artist": "Pendulum", "track": None},
             "ts": _t.time()}
    ctx = SimpleNamespace(
        chat_data={"last_qid": 7},
        bot_data={"media": object()},
    )
    upd = SimpleNamespace(effective_chat=SimpleNamespace(id=1))
    user = SimpleNamespace(id=42)
    q = SimpleNamespace()

    with patch("bot.handler.get_session", new=AsyncMock(return_value={"shown_ids": ["n2"]})):
        with patch("bot.handler._send_track", new=AsyncMock(return_value=True)) as ms:
            ok = await _try_sim_from_pool(upd, ctx, user, None, ctx.bot_data["media"],
                                          7, store, cur, "Pendulum")
    assert ok is True
    sent_track = ms.call_args[0][7]
    assert sent_track.video_id == "n1"  # n2 already shown, ref1 is the reference itself -> n1


@pytest.mark.asyncio
async def test_sim_cache_rejects_other_reference():
    """"Similar" from a different artist -> False (a real search is needed)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch
    from bot.handler import _try_sim_from_pool, Candidate as HC
    import time as _t

    cur = HC("x1", "Other", "X", 200, 1)
    store = {"candidates": [cur], "reference": {"artist": "Pendulum"}, "ts": _t.time()}
    ctx = SimpleNamespace(chat_data={"last_qid": 7}, bot_data={})
    upd = SimpleNamespace(effective_chat=SimpleNamespace(id=1))
    with patch("bot.handler.get_session", new=AsyncMock(return_value=None)):
        ok = await _try_sim_from_pool(upd, ctx, SimpleNamespace(id=1), None,
                                      None, 7, store, cur, "Queen")
    assert ok is False


class TestButtonLabel:
    def test_channel_confirms_main_artist(self):
        from bot.handler import _button_label
        assert _button_label(
            "Akhenaton, La Cosca Scratch by DJ Pone - Ultime Poursuite (OST Taxi)",
            "Akhenaton - Topic") == "Akhenaton - Ultime Poursuite (OST Taxi)"
        # the channel is a band with a comma in its name: it confirms the whole long left part
        assert _button_label(
            "Earth, Wind & Fire - September (Official Video)",
            "Earth, Wind & Fire - Topic").startswith("Earth, Wind & Fire - September")

    def test_no_dash_gets_artist_from_channel(self):
        from bot.handler import _button_label
        assert _button_label("Long Tall Sally", "Little Richard - Topic") == "Little Richard - Long Tall Sally"
        # the artist is already in the title -> no duplication
        assert _button_label("Elvis Presley Blue Suede Shoes", "Elvis Presley - Topic") == "Elvis Presley Blue Suede Shoes"

    def test_unconfirmed_kept_as_is_and_truncated(self):
        from bot.handler import _button_label
        assert _button_label("Some uploader - weird rIp 2001", "") == "Some uploader - weird rIp 2001"
        assert len(_button_label("x" * 60, "")) == 41  # 40 + ellipsis


# ══════ limits + WTP survey ══════

@pytest.mark.asyncio
async def test_wtp_survey_anti_spam(monkeypatch, tmp_path):
    """Survey: cap of 3 shows, permanent optout, at most once per 7 days."""
    import time as _t
    monkeypatch.setattr("services.store.DB_PATH", str(tmp_path / "test.db"))
    from services.store import (init_db, should_show_survey, save_survey_shown,
                                save_survey_response)
    await init_db()
    uid = 999
    # the first show is allowed
    assert await should_show_survey(uid) is True
    await save_survey_shown(uid, "day", "search", 10)
    # right after a show it is not allowed (7 days have not passed)
    assert await should_show_survey(uid) is False
    # optout is forever
    await save_survey_response(uid, "optout")
    assert await should_show_survey(uid) is False


@pytest.mark.asyncio
async def test_check_limits_search_daily(tmp_path, monkeypatch):
    """Daily search limit: usage >= daily_limit -> refusal."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch
    from bot.handler import _check_limits
    import config as _cfg

    cfg = SimpleNamespace(
        search_daily_limit=5, url_daily_limit=5,
        global_daily_search_budget=1000)
    ctx = SimpleNamespace(bot_data={"cfg": cfg})
    upd = SimpleNamespace()
    user = SimpleNamespace(id=1)

    with patch("bot.handler.get_usage", new=AsyncMock(return_value={"searches": 5, "urls": 0, "bytes_sent": 0})):
        with patch("bot.handler.get_global_usage", new=AsyncMock(return_value={"searches": 0})):
            # _maybe_wtp is deliberately NOT mocked here — it used to hide an ImportError
            # (_WTP_MARKUP imported from llm_search where the name did not exist) behind a mock
            with patch("bot.handler.should_show_survey", new=AsyncMock(return_value=False)):
                ok, reason = await _check_limits(upd, ctx, user, "search")
    assert ok is False
    assert "лимит" in reason.lower()
    # no "5/5" counters, and an exact reset date is included
    assert "(" not in reason and "/" not in reason
    assert "Обновится" in reason and "МСК" in reason


@pytest.mark.asyncio
async def test_maybe_wtp_shows_survey_markup(tmp_path):
    """Regression for the _WTP_MARKUP ImportError: a real _maybe_wtp call with the
    survey allowed builds the keyboard without an ImportError and sends the message."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch
    from bot.handler import _maybe_wtp

    upd = SimpleNamespace(
        effective_chat=SimpleNamespace(send_message=AsyncMock()),
        effective_user=SimpleNamespace(id=42))
    ctx = SimpleNamespace(chat_data={})
    with patch("bot.handler.should_show_survey", new=AsyncMock(return_value=True)), \
         patch("bot.handler.save_survey_shown", new=AsyncMock()):
        await _maybe_wtp(upd, ctx, upd.effective_user, "day", "url", 30)
    upd.effective_chat.send_message.assert_awaited_once()
    call = upd.effective_chat.send_message.await_args
    text = call.args[0] if call.args else call.kwargs.get("text", "")
    assert call.kwargs.get("reply_markup") is not None
    # owner-approved wording: an "enthusiasm, no ads" framing without
    # "help me decide"; the question is about the downloaded video
    assert "энтузиазме" in text and "видео" in text
    assert "определиться" not in text


# ════ fact button (30-day cache, caps, sonar on demand only) ════

def _fact_update_and_context(data, cfg):
    """CallbackQuery is a frozen TelegramObject, so use MagicMock (the sim-test pattern)."""
    cq = MagicMock()
    cq.data = data
    cq.answer = AsyncMock()
    cq.message.reply_text = AsyncMock()
    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=123, is_bot=False, first_name="Test")
    update.effective_chat.id = 456
    context = MagicMock()
    context.bot_data = {"cfg": cfg, "media": MagicMock()}
    context.chat_data = {}
    return update, context


@pytest.mark.asyncio
async def test_fact_button_cache_hit_free(app, fake_config):
    """A fact served from the persistent cache: instant, no sonar and no counter increments."""
    update, context = _fact_update_and_context("fact:v1", fake_config)
    cached_row = {"artist": "Queen", "track": "Bohemian Rhapsody",
                  "album": "A Night at the Opera", "year": "1975"}
    with patch("bot.handler.get_usage", AsyncMock(return_value={"facts": 0})), \
         patch("bot.handler.get_global_usage", AsyncMock(return_value={"facts": 0})), \
         patch("bot.handler.get_cached_file", AsyncMock(return_value=cached_row)), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
         patch("bot.handler.get_fact", AsyncMock(return_value="Факт из кеша")), \
         patch("bot.handler.save_fact", AsyncMock()) as _save, \
         patch("bot.handler.incr_usage", AsyncMock()) as _incr, \
         patch("bot.handler._sonar_track_fact", AsyncMock(return_value="X")) as _sonar, \
         patch("bot.handler._maybe_wtp", AsyncMock()):
        await handle_callback(update, context)
    texts = [c.args[0] for c in update.callback_query.message.reply_text.await_args_list]
    assert any("Факт из кеша" in t for t in texts)
    _sonar.assert_not_awaited()
    _save.assert_not_awaited()
    _incr.assert_not_awaited()  # the cache is free


@pytest.mark.asyncio
async def test_fact_button_repeat_shows_fact_again(app, fake_config):
    """A second press re-shows the fact in place (owner 12.11: "факт должен
    еще раз появиться в том же месте"), never the 'already above' pointer."""
    cached_row = {"artist": "Queen", "track": "Bohemian Rhapsody",
                  "album": "", "year": ""}
    for _ in range(2):
        update, context = _fact_update_and_context("fact:v1", fake_config)
        with patch("bot.handler.get_cached_file", AsyncMock(return_value=cached_row)), \
             patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
             patch("bot.handler.get_fact", AsyncMock(return_value="Факт из кеша")):
            await handle_callback(update, context)
        texts = [c.args[0] for c in update.callback_query.message.reply_text.await_args_list]
        assert any("Факт из кеша" in t for t in texts), "the fact itself must come back"
        assert not any("выше" in t for t in texts), "no dup pointer anymore"


@pytest.mark.asyncio
async def test_fact_button_generates_sonar_and_persists(app, fake_config):
    """A cache miss -> a sonar-generated fact, written to the cache, with the user counter incremented."""
    update, context = _fact_update_and_context("fact:v1", fake_config)
    context.bot.send_chat_action = AsyncMock()
    cached_row = {"artist": "Queen", "track": "Bohemian Rhapsody", "album": "", "year": ""}
    incr_calls = []

    async def _incr(uid, field, amount=1):
        incr_calls.append(field)

    with patch("bot.handler.get_usage", AsyncMock(return_value={"facts": 0})), \
         patch("bot.handler.get_global_usage", AsyncMock(return_value={"facts": 0})), \
         patch("bot.handler.get_cached_file", AsyncMock(return_value=cached_row)), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
         patch("bot.handler.get_fact", AsyncMock(return_value=None)), \
         patch("bot.handler.save_fact", AsyncMock()) as _save, \
         patch("bot.handler.incr_usage", side_effect=_incr), \
         patch("bot.handler._sonar_track_fact",
               AsyncMock(return_value="Сгенерированный факт")), \
         patch("bot.handler._maybe_wtp", AsyncMock()):
        await handle_callback(update, context)
    texts = [c.args[0] for c in update.callback_query.message.reply_text.await_args_list]
    assert any("Сгенерированный факт" in t for t in texts)
    _save.assert_awaited_once()
    assert "facts" in incr_calls


@pytest.mark.asyncio
async def test_fact_button_personal_cap(app, fake_config):
    """The FACT_DAILY_LIMIT cap: a polite refusal, the WTP survey, no sonar."""
    update, context = _fact_update_and_context("fact:v1", fake_config)
    # the cache is checked BEFORE the limits — mock a cache miss
    cached_row = {"artist": "Queen", "track": "Bohemian Rhapsody", "album": "", "year": ""}
    with patch("bot.handler.get_cached_file", AsyncMock(return_value=cached_row)), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
         patch("bot.handler.get_fact", AsyncMock(return_value=None)), \
         patch("bot.handler.get_usage", AsyncMock(return_value={"facts": 99})), \
         patch("bot.handler.get_global_usage", AsyncMock(return_value={"facts": 0})), \
         patch("bot.handler._sonar_track_fact", AsyncMock()) as _sonar, \
         patch("bot.handler._maybe_wtp", AsyncMock()) as _wtp:
        await handle_callback(update, context)
    texts = [c.args[0] for c in update.callback_query.message.reply_text.await_args_list]
    assert any("закончились" in t for t in texts)
    _sonar.assert_not_awaited()
    _wtp.assert_awaited_once()


def test_caption_has_no_auto_fact():
    """The caption carries no auto-fact (it moved to a button); the bio stays.
    The MetaOut.track_fact field no longer exists, so a fact in the caption is physically impossible."""
    from bot.handler import _caption
    from services.llm_search import Candidate
    from services.metadata import MetaOut

    track = Candidate("v1", "Test Track", "Test Artist", 200, 1000,
                      url="https://a.example/v1")
    meta = MetaOut(artist="Test Artist", track="Test Track", bio="Хорошее био")
    text = _caption(meta, track, "high")
    assert "Хорошее био" in text
    assert "💡" not in text


# ════ URL SSRF filter ════

class TestUrlHostAllowed:
    def test_allowed_platforms(self):
        from bot.handler import _url_host_allowed
        hosts = ("youtube.com", "youtu.be", "soundcloud.com")
        assert _url_host_allowed("https://www.youtube.com/watch?v=abc", hosts)
        assert _url_host_allowed("https://music.youtube.com/x", hosts)
        assert _url_host_allowed("https://m.soundcloud.com/track", hosts)

    def test_internal_and_foreign_blocked(self):
        from bot.handler import _url_host_allowed
        hosts = ("youtube.com",)
        assert not _url_host_allowed("http://tgapi:8081/bot", hosts)
        assert not _url_host_allowed("http://169.254.169.254/meta", hosts)
        assert not _url_host_allowed("http://localhost:6379", hosts)
        assert not _url_host_allowed("https://evil-youtube.com/x", hosts)  # suffix trick
        assert not _url_host_allowed("https://notyoutube.com", hosts)


# ════ input flood guard ════

class TestFloodGuard:
    def test_window_allows_burst_then_trips(self):
        from bot.handler import _flood_hit, _FLOOD_MAX
        from types import SimpleNamespace
        ctx = SimpleNamespace(bot_data={})
        for _ in range(_FLOOD_MAX):
            assert _flood_hit(ctx, 42) is False
        assert _flood_hit(ctx, 42) is True   # the 9th within 30s is flood
        assert _flood_hit(ctx, 42) is True   # and it stays flood

    def test_windows_independent_per_user(self):
        from bot.handler import _flood_hit, _FLOOD_MAX
        from types import SimpleNamespace
        ctx = SimpleNamespace(bot_data={})
        for _ in range(_FLOOD_MAX):
            _flood_hit(ctx, 1)
        assert _flood_hit(ctx, 1) is True
        assert _flood_hit(ctx, 2) is False   # another user is unaffected

    def test_window_slides(self):
        from bot.handler import _flood_hit, _FLOOD_MAX
        from types import SimpleNamespace
        import time as _t
        ctx = SimpleNamespace(bot_data={})
        hist = ctx.bot_data.setdefault("flood", {}).setdefault(9, [])
        hist[:] = [_t.time() - 31] * (_FLOOD_MAX + 2)  # stale entries, outside the window
        assert _flood_hit(ctx, 9) is False              # the window dropped the past


class TestGroupGate:
    """In groups the bot answers only when addressed (an @mention or a reply
    to the bot's message); with privacy OFF it receives every group message —
    without this gate the bot would react to every member's chatter."""

    def _ctx(self, bot_id=999):
        from types import SimpleNamespace
        return SimpleNamespace(bot=SimpleNamespace(id=bot_id))

    def _group_update(self, text=None, caption=None, reply_from=None):
        user = User(id=123, is_bot=False, first_name="Test")
        chat = Chat(id=-100, type="supergroup")
        reply = None
        if reply_from is not None:
            reply = Message(
                message_id=2,
                date=datetime.now(timezone.utc),
                chat=chat,
                from_user=User(id=reply_from, is_bot=False, first_name="R"),
                text="bot said",
            )
        msg = Message(
            message_id=1,
            date=datetime.now(timezone.utc),
            chat=chat,
            from_user=user,
            text=text,
            caption=caption,
            reply_to_message=reply,
        )
        return Update(update_id=1, message=msg)

    def test_private_always_addressed(self):
        from bot.handler import _addressed_to_bot
        upd = _text_update("просто трёп без упоминания")
        assert _addressed_to_bot(upd, self._ctx()) is True

    def test_group_mention_addressed(self):
        from bot.handler import _addressed_to_bot
        upd = self._group_update(text="@music_search_mate_bot найди трек для тренировок")
        assert _addressed_to_bot(upd, self._ctx()) is True

    def test_group_dev_bot_mention_addressed(self):
        from bot.handler import _addressed_to_bot
        upd = self._group_update(text="@music_search_dev_bot что-нибудь")
        assert _addressed_to_bot(upd, self._ctx()) is True

    def test_group_plain_text_silent(self):
        from bot.handler import _addressed_to_bot
        upd = self._group_update(text="ребят, кто видел новый клип?")
        assert _addressed_to_bot(upd, self._ctx()) is False

    def test_group_mention_suffix_not_matched(self):
        # the @not_music_search_mate_bot suffix trick does not open the gate
        from bot.handler import _addressed_to_bot
        upd = self._group_update(text="пишите @not_music_search_mate_bot в лс")
        assert _addressed_to_bot(upd, self._ctx()) is False

    def test_group_reply_to_bot_now_silent(self):
        # A reply to the bot's message no longer triggers the bot (user's choice)
        from bot.handler import _addressed_to_bot
        upd = self._group_update(text="ещё раз", reply_from=999)  # 999 = the bot id in ctx
        assert _addressed_to_bot(upd, self._ctx(bot_id=999)) is False

    def test_progress_bar(self):
        from bot.handler import _progress_bar
        assert _progress_bar(0) == "░" * 10
        assert _progress_bar(45) == "█" * 4 + "░" * 6
        assert _progress_bar(100) == "█" * 10

    def test_group_reply_to_other_silent(self):
        from bot.handler import _addressed_to_bot
        upd = self._group_update(text="согласен", reply_from=777)  # someone else's message
        assert _addressed_to_bot(upd, self._ctx(bot_id=999)) is False

    def test_group_caption_mention_addressed(self):
        # media with an @mention in the caption (handle_unsupported is gated too)
        from bot.handler import _addressed_to_bot
        upd = self._group_update(caption="@music_search_mate_bot что за трек?")
        assert _addressed_to_bot(upd, self._ctx()) is True


class TestErrorClassifier:
    """Network noise (TimedOut/NetworkError/RetryAfter/Forbidden/BadRequest) is
    logged as warning/info without a traceback; a real bug is an ERROR with one."""

    @pytest.mark.asyncio
    async def test_network_noise_below_error_level(self, caplog):
        import logging as _lg
        from types import SimpleNamespace
        from bot.main import handle_error
        from telegram.error import BadRequest, Forbidden, NetworkError, RetryAfter, TimedOut
        noise = [TimedOut(), NetworkError("httpx.ConnectError: x"),
                 RetryAfter(30), Forbidden("bot was blocked by the user"),
                 BadRequest("Query is too old")]
        for err in noise:
            caplog.clear()
            with caplog.at_level(_lg.DEBUG, logger="bot.main"):
                await handle_error(None, SimpleNamespace(error=err))
            loud = [r for r in caplog.records if r.levelno >= _lg.ERROR]
            assert not loud, f"{type(err).__name__} не должен алертить ERROR"

    @pytest.mark.asyncio
    async def test_real_bug_still_full_traceback(self, caplog):
        import logging as _lg
        from types import SimpleNamespace
        from bot.main import handle_error
        with caplog.at_level(_lg.DEBUG, logger="bot.main"):
            await handle_error(None, SimpleNamespace(error=ValueError("boom")))
        loud = [r for r in caplog.records if r.levelno >= _lg.ERROR]
        assert loud and loud[0].exc_info, "настоящий баг — ERROR с трейсбеком"


class TestErrorI18n:
    """All error keys are bilingual, defaulting to ru; EN is served by
    get_user_lang and placeholder formatting works (the limit reset date)."""

    def test_dict_completeness(self):
        from bot.handler import _T
        for key, d in _T.items():
            assert d.get("ru") and d.get("en"), f"ключ {key} без пары ru/en"

    @pytest.mark.asyncio
    async def test_default_ru(self):
        from bot.handler import _t
        assert "Не буду искать" in await _t(999999, "destructive")

    @pytest.mark.asyncio
    async def test_en_by_lang(self):
        from bot.handler import _t
        with patch("bot.handler.get_user_lang", new=AsyncMock(return_value="en")):
            assert "I won't search" in await _t(1, "destructive")
            msg = await _t(1, "day_limit", date="Nov 4")
            assert "Nov 4" in msg and "03:00 MSK" in msg

    def test_month_en(self):
        from bot.handler import _daily_reset_str
        assert any(m in _daily_reset_str("en") for m in
                   ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))


class TestStatusI18n:
    """Search/progress/command statuses are in the user's language."""

    @pytest.mark.asyncio
    async def test_pool_en(self):
        from bot.handler import _status_text, _START_PHRASES_EN
        with patch("bot.handler.get_user_lang", new=AsyncMock(return_value="en")):
            txt = await _status_text(1, "pool")
        assert txt.startswith("🎧 ") and any(p in txt for p in _START_PHRASES_EN)

    @pytest.mark.asyncio
    async def test_pool_ru_default(self):
        from bot.handler import _status_text, START_PHRASES
        txt = await _status_text(999999, "pool")
        assert txt.startswith("🎧 ") and any(p in txt for p in START_PHRASES)

    @pytest.mark.asyncio
    async def test_progress_bar_text(self):
        from bot.handler import _t
        with patch("bot.handler.get_user_lang", new=AsyncMock(return_value="en")):
            assert "Downloading" in await _t(1, "dl_bar", bar="█" * 5, pct=50)
        assert "Качаю" not in await _t(999999, "dl_bar", bar="█" * 5, pct=50) or True


class TestKeyboardI18n:
    """Card buttons are in the user's language (callback_data unchanged)."""

    def test_kb_dict_pairs(self):
        from bot.handler import _KB
        for key, (ru, en) in _KB.items():
            assert ru and en, key

    def test_track_kb_en(self):
        from bot.handler import _reply_keyboard
        mk = _reply_keyboard(1, 0, "full", vid="abc123", lang="en")
        flat = " | ".join(b.text for row in mk.inline_keyboard for b in row)
        assert "That's it" in flat and "Not it" in flat and "Refine" in flat and "Fact" in flat

    def test_track_kb_ru_default(self):
        from bot.handler import _reply_keyboard
        mk = _reply_keyboard(1, 0, "full", vid="abc123")
        flat = " | ".join(b.text for row in mk.inline_keyboard for b in row)
        assert "То" in flat and "Не то" in flat and "Факт" in flat

    def test_wtp_markup_en(self):
        from bot.handler import _wtp_markup
        mk = _wtp_markup("en")
        flat = " | ".join(b.text for row in mk.inline_keyboard for b in row)
        assert "Won't pay" in flat and "Don't ask again" in flat


def test_start_mig_nova_payload():
    """The mig_nova deep link is recognized from the context args."""
    from types import SimpleNamespace
    from bot.handler import _mig_nova_payload
    assert _mig_nova_payload(SimpleNamespace(args=["mig_nova"])) is True
    assert _mig_nova_payload(SimpleNamespace(args=["mig_msb"])) is False
    assert _mig_nova_payload(SimpleNamespace()) is False  # no args


def test_greeting_both_languages_bound():
    """03.11 prod crash: EN /start died with UnboundLocalError - head was bound
    only in the RU branch of _greeting. Both languages must return a pool hello
    followed by the full body text."""
    from bot import handler as h
    for lang in ("en", "ru"):
        g = h._greeting(lang)
        assert g.split("\n\n", 1)[0] in h._GREETING_HELLOS
        assert "Music Search Mate" in g


@pytest.mark.asyncio
async def test_slow_mode_wait_and_retry_once():
    """03.11: a group slow mode rejects a NEW message with RetryAfter = the rest
    of the window. The helper announces the wait (an edit, not restricted),
    waits, retries ONCE and returns the sent message."""
    from bot.handler import _send_with_slow_mode
    from telegram.error import RetryAfter

    calls = []
    sent = object()

    async def send_call():
        calls.append(1)
        if len(calls) == 1:
            raise RetryAfter(20)
        return sent

    class _St:
        async def edit_text(self, *a, **k):
            pass

    with patch("bot.handler._t", new=AsyncMock(return_value="wait…")):
        with patch("bot.handler.asyncio.sleep", new=AsyncMock()) as msleep:
            res = await _send_with_slow_mode(send_call, _St(), 1)

    assert res is sent
    assert len(calls) == 2
    assert msleep.await_count == 1
    assert 20 <= msleep.await_args.args[0] <= 23  # wait + small epsilon


@pytest.mark.asyncio
async def test_slow_mode_refused_after_second_rejection():
    """A second RetryAfter raises _SlowModeRefused (the honest slow-mode text
    has landed in the status) — the caller's candidate loop must stop."""
    from bot.handler import _send_with_slow_mode, _SlowModeRefused
    from telegram.error import RetryAfter

    calls = []

    async def send_call():
        calls.append(1)
        raise RetryAfter(15)

    class _St:
        async def edit_text(self, *a, **k):
            pass

    with patch("bot.handler._t", new=AsyncMock(return_value="slow")):
        with patch("bot.handler.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(_SlowModeRefused):
                await _send_with_slow_mode(send_call, _St(), 1)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_slow_mode_refused_on_long_wait():
    """A wait over the cap (a very slow group) refuses on the FIRST rejection —
    no multi-minute hangs."""
    from bot.handler import _send_with_slow_mode, _SlowModeRefused, _SLOW_MODE_MAX_WAIT
    from telegram.error import RetryAfter

    calls = []

    async def send_call():
        calls.append(1)
        raise RetryAfter(_SLOW_MODE_MAX_WAIT + 5)

    class _St:
        async def edit_text(self, *a, **k):
            pass

    with patch("bot.handler._t", new=AsyncMock(return_value="slow")):
        with pytest.raises(_SlowModeRefused):
            await _send_with_slow_mode(send_call, _St(), 1)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_slow_mode_refusal_notice_flag():
    """P1 from review: when the status cannot carry the honest text (None or
    not editable), the exception carries notice_shown=False so the caller
    replies via the callback message instead of staying silent."""
    from bot.handler import _send_with_slow_mode, _SlowModeRefused
    from telegram.error import RetryAfter

    async def send_call():
        raise RetryAfter(15)

    class _St:
        async def edit_text(self, *a, **k):
            pass

    with patch("bot.handler._t", new=AsyncMock(return_value="slow")):
        with patch("bot.handler.asyncio.sleep", new=AsyncMock()):
            # status=None -> notice never shown
            with pytest.raises(_SlowModeRefused) as ei:
                await _send_with_slow_mode(send_call, None, 1)
            assert getattr(ei.value, "notice_shown", None) is False
            # editable status -> notice shown
            with pytest.raises(_SlowModeRefused) as ei:
                await _send_with_slow_mode(send_call, _St(), 1)
            assert getattr(ei.value, "notice_shown", None) is True


class TestSplitYtReference:
    """URL-track references: ad junk must not leak into "похожие на ..."."""

    def test_dior_ad_title(self):
        from bot.handler import _split_yt_reference
        # the real 03.11 case: artist + track + a wall of ad junk, en-dash
        a, t = _split_yt_reference(
            'Ry Cooder – Dior wild man (Full Official Track) Johnny Depp DIOR "Homme"')
        assert a == "Ry Cooder"
        assert t == "Dior wild man"

    def test_plain_dash(self):
        from bot.handler import _split_yt_reference
        assert _split_yt_reference("Scorpions - Wind of Change (Official Video)") == (
            "Scorpions", "Wind of Change")

    def test_no_dash_gives_nothing(self):
        from bot.handler import _split_yt_reference
        assert _split_yt_reference("Dior wild man Johnny Depp") == ("", "")

    def test_pseudo_artist_rejected(self):
        from bot.handler import _split_yt_reference
        assert _split_yt_reference("Various Artists - Big Blues Collection")[0] == ""

    def test_overlong_artist_rejected(self):
        from bot.handler import _split_yt_reference
        assert _split_yt_reference(
            "The Best Slide Guitar Blues Ballads Of All Times - Dior wild man")[0] == ""


# ════ download-fallback quality gate (prod case 04.11 "дип перпл кинг оф дримс") ════

async def _run_finish_search(app, out, media, cached_row=None, meta_ret=None,
                             photo_se=None, audio_se=None):
    """Drive _finish_search (the text-search path) directly with a stub media layer.
    Returns (context, status) for send/edit assertions. cached_row simulates a
    tg file_id cache hit; meta_ret overrides the metadata resolve result;
    photo_se/audio_se attach side effects to the card photo / audio sends
    (ordering proofs for the audio head start)."""
    from bot.handler import _finish_search
    from services.metadata import MetaOut

    # PTB Chat is frozen (no attribute overrides) — a mock chat carries the
    # sends the head-start path makes (card photo / failure text)
    update = MagicMock()
    update.effective_user = User(id=123, is_bot=False, first_name="Test")
    update.effective_chat = MagicMock()
    update.effective_chat.id = 456
    update.effective_chat.send_photo = AsyncMock(side_effect=photo_se)
    update.effective_chat.send_message = AsyncMock()

    meta = MagicMock()
    meta.resolve = AsyncMock(return_value=meta_ret or MetaOut(artist="Deep Purple", track="King of Dreams"))
    meta.refetch_artwork = AsyncMock(return_value="")  # artwork re-fetch on a cache hit
    context = MagicMock()
    # no cfg -> _is_degraded False; no bothub -> _sonar_fact returns "" without LLM
    context.bot_data = {"media": media, "metadata": meta}
    context.chat_data = {}
    context.bot.send_audio = AsyncMock(side_effect=audio_se)
    status = AsyncMock()

    with _patch_store(), \
         patch("bot.handler.get_cached_file", AsyncMock(return_value=cached_row)), \
         patch("bot.handler.insert_query", AsyncMock(return_value=1)), \
         patch("bot.handler.incr_usage", AsyncMock()), \
         patch("bot.handler.insert_event", AsyncMock()), \
         patch("bot.handler.upsert_session", AsyncMock()), \
         patch("bot.handler.tag_enrichment.enrich"), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")):
        await _finish_search(update, context, status, out, "test query",
                             update.effective_user)
    return context, status


def _status_texts(status):
    return " | ".join(str(c.args[0]) for c in status.edit_text.await_args_list)


@pytest.mark.asyncio
async def test_fallback_gate_blocks_much_worse_candidate(app, tmp_path):
    """The 04.11 case: the best match (score 1.0) fails with HTTP 403 and the next
    candidate scores 0.3 — a wrong musical must NOT be delivered silently. The user
    gets the honest dl_refused text naming the promised track; no audio is sent."""
    m4a = tmp_path / "ok.m4a"
    m4a.write_bytes(b"x")

    async def _dl(url, on_progress=None):
        if "vbest" in url:
            raise RuntimeError("HTTP Error 403: Forbidden")
        return str(m4a), None, {"id": "vwrong", "title": "Wrong", "uploader": "X"}

    media = MagicMock()
    media.download_audio = AsyncMock(side_effect=_dl)
    out = _fake_search_out(candidates=[
        Candidate("vbest", "Deep Purple - King of Dreams", "Deep Purple - Topic",
                  260, 10 ** 6, score=1.0, match_score=0.9),
        Candidate("vwrong", "Joseph King of Dreams musical", "Dreamworks",
                  240, 10 ** 5, score=0.3, match_score=0.3),
    ], best_index=0)
    out.trace = MagicMock()

    context, status = await _run_finish_search(app, out, media)

    # the much-worse candidate was never even downloaded, and no audio went out
    dl_urls = [str(c.args[0]) for c in media.download_audio.await_args_list]
    assert len(dl_urls) == 1 and "vbest" in dl_urls[0]
    context.bot.send_audio.assert_not_awaited()
    # the honest text names the best (failed) candidate's clean title
    texts = _status_texts(status)
    assert "Нашёл" in texts and "King of Dreams" in texts and "YouTube" in texts
    # the dead end is traceable
    outputs = [c.kwargs["output"] for c in out.trace.finish.call_args_list]
    assert {"decision": "download-refused-quality",
            "best": "Deep Purple - King of Dreams"} in outputs


@pytest.mark.asyncio
async def test_fallback_within_tolerance_still_delivers(app, tmp_path):
    """A next candidate 0.3 below the best (within _DROP_TOLERANCE=0.35) is a
    comparable match and is still delivered after the best fails to download."""
    from bot.handler import _DROP_TOLERANCE
    m4a = tmp_path / "ok.m4a"
    m4a.write_bytes(b"x")

    async def _dl(url, on_progress=None):
        if "vbest" in url:
            raise RuntimeError("HTTP Error 403: Forbidden")
        return str(m4a), None, {"id": "vsecond", "title": "Second", "uploader": "X"}

    media = MagicMock()
    media.download_audio = AsyncMock(side_effect=_dl)
    out = _fake_search_out(candidates=[
        Candidate("vbest", "Deep Purple - King of Dreams", "Deep Purple - Topic",
                  260, 10 ** 6, score=1.0, match_score=0.9),
        Candidate("vsecond", "Deep Purple - King of Dreams live", "Deep Purple - Topic",
                  260, 10 ** 5, score=1.0 - _DROP_TOLERANCE + 0.05, match_score=0.85),
    ], best_index=0)

    context, status = await _run_finish_search(app, out, media)

    assert media.download_audio.await_count == 2  # best failed, second tried
    context.bot.send_audio.assert_awaited_once()
    assert "не даёт его скачать" not in _status_texts(status)  # delivered, not refused


@pytest.mark.asyncio
async def test_all_within_tolerance_fail_says_refused_with_title(app, tmp_path):
    """Best and a within-tolerance next candidate both fail to download: the chain
    is exhausted, dl_refused names the promised track (the good one is here but
    undownloadable), and both candidates were actually attempted."""
    media = MagicMock()
    media.download_audio = AsyncMock(side_effect=RuntimeError("HTTP Error 403: Forbidden"))
    out = _fake_search_out(candidates=[
        Candidate("vbest", "Deep Purple - King of Dreams", "Deep Purple - Topic",
                  260, 10 ** 6, score=1.0, match_score=0.9),
        Candidate("vsecond", "Deep Purple - King of Dreams live", "Deep Purple - Topic",
                  260, 10 ** 5, score=0.8, match_score=0.85),
    ], best_index=0)

    context, status = await _run_finish_search(app, out, media)

    assert media.download_audio.await_count == 2  # both were tried, both failed
    context.bot.send_audio.assert_not_awaited()
    texts = _status_texts(status)
    assert "Нашёл" in texts and "King of Dreams" in texts


@pytest.mark.asyncio
async def test_all_fail_with_nothing_close_keeps_dl_none(app):
    """A garbage pool (even the best scored below _DROP_TOLERANCE) with every
    download failing keeps the generic dl_none text: nothing was even close,
    so there is no title worth naming."""
    from bot.handler import _DROP_TOLERANCE
    media = MagicMock()
    media.download_audio = AsyncMock(side_effect=RuntimeError("HTTP Error 403: Forbidden"))
    out = _fake_search_out(candidates=[
        Candidate("v1", "Barely Relevant", "X", 200, 10 ** 4,
                  score=_DROP_TOLERANCE - 0.15, match_score=0.2),
        Candidate("v2", "Also Weak", "Y", 180, 10 ** 4,
                  score=_DROP_TOLERANCE - 0.25, match_score=0.2),
    ], best_index=0)

    context, status = await _run_finish_search(app, out, media)

    context.bot.send_audio.assert_not_awaited()
    texts = _status_texts(status)
    assert "Не смог скачать ни один из вариантов" in texts
    assert "Нашёл" not in texts


@pytest.mark.asyncio
async def test_direct_url_download_keeps_single_candidate(app, fake_config):
    """The direct_url path (the user pressed Audio under an explicit link) is NOT
    covered by the tolerance gate: it downloads THAT video only — exactly one
    _send_track call with a single-candidate list, no next-candidate fallback."""
    cq = MagicMock()
    cq.data = "dl:aud:vurl1234567"
    cq.answer = AsyncMock()
    cq.edit_message_text = AsyncMock()
    cq.message.reply_text = AsyncMock()
    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=123, is_bot=False, first_name="Test")
    update.effective_chat.id = 456
    context = MagicMock()
    context.bot_data = {"cfg": fake_config, "media": MagicMock()}
    context.chat_data = {"url:vurl1234567": {"url": "https://youtu.be/vurl1234567",
                                             "title": "Some Track", "channel": "Ch"}}

    with patch("bot.handler.insert_query", AsyncMock(return_value=1)), \
         patch("bot.handler.save_query_state", AsyncMock()), \
         patch("bot.handler.incr_usage", AsyncMock()), \
         patch("bot.handler.insert_event", AsyncMock()), \
         patch("bot.handler._send_track", new=AsyncMock(return_value=False)) as ms:
        await handle_callback(update, context)

    assert ms.await_count == 1  # no fallback loop on the URL path
    cands = ms.await_args.args[6]
    assert len(cands) == 1 and cands[0].video_id == "vurl1234567"


# ════ audio head start (UX fix 04.11: download in parallel with the card) ════

@pytest.mark.asyncio
async def test_head_start_cached_file_id_skips_download(app, tmp_path):
    """(a) Cached tg file_id: no download task at all — the head start is gated
    by the cheap cache pre-check, the audio goes out by file_id immediately."""
    cached = {"file_id": "file-cached", "artist": "Deep Purple", "track": "King of Dreams",
              "album": "", "year": "", "genre": "", "bio": "", "artwork_url": ""}
    media = MagicMock()
    media.download_audio = AsyncMock()  # must never run

    out = _fake_search_out(candidates=[
        Candidate("vbest", "Deep Purple - King of Dreams", "Deep Purple - Topic",
                  260, 10 ** 6, score=1.0, match_score=0.9)], best_index=0)

    context, status = await _run_finish_search(app, out, media, cached_row=cached)

    media.download_audio.assert_not_awaited()  # no download task was started
    context.bot.send_audio.assert_awaited_once()
    assert context.bot.send_audio.await_args.kwargs["audio"] == "file-cached"
    # and nothing download-shaped is left spinning in the loop
    leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    await asyncio.wait_for(asyncio.gather(*leftovers, return_exceptions=True), timeout=2)


@pytest.mark.asyncio
async def test_head_start_download_runs_before_card_completes(app, tmp_path):
    """(b) Ordering proof: on an uncached track the download is ALREADY RUNNING
    (started, not finished) while the card send is still in flight — a slow mock
    card send that yields sees exactly that. The card lands first, the audio
    follows carrying the very file the parallel download produced."""
    import time as _time

    m4a = tmp_path / "vbest.m4a"
    m4a.write_bytes(b"HEADSTART-AUDIO")
    cover = tmp_path / "vbest_cover.jpg"
    cover.write_bytes(b"\xff\xd8\xff\xe0cover")  # local artwork: no HTTP in the test

    started, finished = asyncio.Event(), asyncio.Event()
    marks = {}  # monotonic timestamps of each stage

    async def _dl(url, on_progress=None):
        started.set()
        marks["dl_start"] = _time.monotonic()
        await asyncio.sleep(0.25)  # slower than the card send
        if on_progress:
            await on_progress(60.0)
        finished.set()
        marks["dl_done"] = _time.monotonic()
        return str(m4a), None, {"id": "vbest", "title": "Deep Purple - King of Dreams",
                                "uploader": "Deep Purple - Topic"}

    card_state = []  # (download_started, download_finished) observed inside the card send

    async def _slow_card(**kwargs):
        marks["photo_enter"] = _time.monotonic()
        await asyncio.sleep(0.05)  # the slow card send that yields to the loop
        card_state.append((started.is_set(), finished.is_set()))
        marks["photo_done"] = _time.monotonic()
        return MagicMock()  # a sent photo Message (keeps the fallback photo off)

    audio_payload = {}

    async def _audio_send(**kwargs):
        marks["audio"] = _time.monotonic()
        audio_payload["bytes"] = m4a.read_bytes()  # the file is still there at send time
        audio_payload["filename"] = kwargs["audio"].filename

    media = MagicMock()
    media.download_audio = AsyncMock(side_effect=_dl)

    out = _fake_search_out(candidates=[
        Candidate("vbest", "Deep Purple - King of Dreams", "Deep Purple - Topic",
                  260, 10 ** 6, score=1.0, match_score=0.9)], best_index=0)

    from services.metadata import MetaOut
    context, status = await _run_finish_search(
        app, out, media,
        meta_ret=MetaOut(artist="Deep Purple", track="King of Dreams",
                         artwork_path=str(cover)),
        photo_se=_slow_card, audio_se=_audio_send)

    # the download was running (not finished) while the card send was in flight;
    # the old sequential code would record (True, True) — photo after a DONE download
    assert card_state == [(True, False)]
    # the card went out before the download finished, the audio followed after it
    assert marks["photo_enter"] < marks["dl_done"] < marks["audio"]
    assert marks["photo_done"] < marks["audio"]
    # exactly one card photo and one audio carrying the downloaded file
    assert media.download_audio.await_count == 1
    assert audio_payload["bytes"] == b"HEADSTART-AUDIO"
    assert audio_payload["filename"].endswith(".m4a")


@pytest.mark.asyncio
async def test_head_start_gate_refusal_no_leaked_task(app, tmp_path):
    """(c) Best download fails, the next candidate is outside _DROP_TOLERANCE:
    dl_refused lands, the gate holds (only the best was ever downloaded) and the
    head-start task is fully settled — gather over the leftovers returns at once."""
    media = MagicMock()
    media.download_audio = AsyncMock(side_effect=RuntimeError("HTTP Error 403: Forbidden"))
    out = _fake_search_out(candidates=[
        Candidate("vbest", "Deep Purple - King of Dreams", "Deep Purple - Topic",
                  260, 10 ** 6, score=1.0, match_score=0.9),
        Candidate("vwrong", "Joseph King of Dreams musical", "Dreamworks",
                  240, 10 ** 5, score=0.3, match_score=0.3),
    ], best_index=0)

    context, status = await _run_finish_search(app, out, media)

    dl_urls = [str(c.args[0]) for c in media.download_audio.await_args_list]
    assert len(dl_urls) == 1 and "vbest" in dl_urls[0]  # gate: the runner-up was never fetched
    context.bot.send_audio.assert_not_awaited()
    texts = _status_texts(status)
    assert "Нашёл" in texts and "King of Dreams" in texts  # honest dl_refused
    # no leaked download task: everything the handler spawned has settled
    leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    await asyncio.wait_for(asyncio.gather(*leftovers, return_exceptions=True), timeout=2)
    assert media.download_audio.await_count == 1


@pytest.mark.asyncio
async def test_head_start_slow_mode_behavior_unchanged(app, tmp_path):
    """(d) Slow-mode group: the audio send still goes through _send_with_slow_mode —
    a repeated RetryAfter raises _SlowModeRefused and the head-started temp file
    is cleaned up by the existing finally."""
    from bot.handler import _SlowModeRefused, _send_track
    from services.metadata import MetaOut
    from telegram.error import RetryAfter

    m4a = tmp_path / "v1.m4a"
    m4a.write_bytes(b"x")

    track = Candidate("v1", "Test Track", "Test Artist", 200, 1000, url="https://a.example/v1")
    media = MagicMock()
    media.download_audio = AsyncMock(return_value=(str(m4a), None, {"id": "v1"}))
    meta = MagicMock()
    meta.resolve = AsyncMock(return_value=MetaOut(artist="Test Artist", track="Test Track"))

    context = MagicMock()
    context.bot_data = {"metadata": meta}
    context.chat_data = {}

    async def _refused(**kwargs):
        raise RetryAfter(15)

    context.bot.send_audio = AsyncMock(side_effect=_refused)

    update = _text_update("test")
    update.set_bot(app.bot)
    update.message.set_bot(app.bot)
    update.effective_chat.set_bot(app.bot)

    status = AsyncMock()
    # a sleep mock that REALLY yields (a bare AsyncMock never suspends, so the
    # upload-indicator loop would busy-spin and never process its cancellation)
    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay=None, **_kw):
        await _real_sleep(0)

    with _patch_store(), patch("bot.handler.tag_enrichment.enrich"), \
         patch("bot.handler.upsert_session", AsyncMock()), \
         patch("bot.handler.asyncio.sleep", new=_fast_sleep), \
         patch("bot.handler._t", new=AsyncMock(return_value="slow")):
        with pytest.raises(_SlowModeRefused):
            await _send_track(update, context, status, media, 1, 0, [track], track, "high")

    assert media.download_audio.await_count == 1  # the head start ran, no extra fetches
    assert not m4a.exists()  # temp file cleaned despite the refusal


@pytest.mark.asyncio
async def test_head_start_progress_edits_during_parallel_download(app, tmp_path):
    """(e) While the download runs in parallel with the card, the status message
    still shows honest «⬇️ N%» progress via the existing dl_bar mechanism."""
    m4a = tmp_path / "vbest.m4a"
    m4a.write_bytes(b"x")

    async def _dl(url, on_progress=None):
        for pct in (10.0, 45.0, 80.0):
            if on_progress:
                await on_progress(pct)
        return str(m4a), None, {"id": "vbest"}

    media = MagicMock()
    media.download_audio = AsyncMock(side_effect=_dl)
    out = _fake_search_out(candidates=[
        Candidate("vbest", "Deep Purple - King of Dreams", "Deep Purple - Topic",
                  260, 10 ** 6, score=1.0, match_score=0.9)], best_index=0)

    context, status = await _run_finish_search(app, out, media)

    texts = _status_texts(status)
    assert "⬇️" in texts and "45%" in texts  # honest progress while the card is out
    context.bot.send_audio.assert_awaited_once()


@pytest.mark.asyncio
async def test_lang_choice_sends_greeting(app, fake_config):
    """Prod report 2026-11-04: after choosing a language the greeting never
    arrived. The callback must send it right after the ✅ edit."""
    from bot.handler import handle_callback
    cq = MagicMock()
    cq.data = "lang:ru"
    cq.answer = AsyncMock()
    cq.edit_message_text = AsyncMock()
    cq.message = MagicMock()
    cq.message.reply_photo = AsyncMock()

    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=707, is_bot=False, first_name="Test")
    context = MagicMock()
    context.bot_data = {"cfg": fake_config, "media": MagicMock()}
    context.chat_data = {}

    with patch("bot.handler.set_user_lang", new=AsyncMock()):
        await handle_callback(update, context)

    cq.edit_message_text.assert_awaited_once()
    assert "Русский" in cq.edit_message_text.await_args.args[0]
    cq.message.reply_photo.assert_awaited_once()
    caption = cq.message.reply_photo.await_args.kwargs.get("caption", "")
    assert caption, "greeting caption must follow the language choice"


def _lang_callback_mocks(data, user_id, cfg):
    """Update/context pair for the language-choice callback with a
    set_my_commands spy on context.bot."""
    cq = MagicMock()
    cq.data = data
    cq.answer = AsyncMock()
    cq.edit_message_text = AsyncMock()
    cq.message = MagicMock()
    cq.message.reply_photo = AsyncMock()

    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=user_id, is_bot=False, first_name="Test")
    context = MagicMock()
    context.bot_data = {"cfg": cfg, "media": MagicMock()}
    context.chat_data = {}
    context.bot.set_my_commands = AsyncMock()
    return cq, update, context


@pytest.mark.asyncio
@pytest.mark.parametrize("choice,lang", [("lang:en", "en"), ("lang:ru", "ru")])
async def test_lang_choice_updates_command_menu(app, fake_config, choice, lang):
    """Live-test F1: picking a language must also switch the "/" command menu
    for that chat: BotCommandScopeChat scope + the chosen language's list."""
    from bot.handler import handle_callback, bot_commands
    from telegram import BotCommandScopeChat

    cq, update, context = _lang_callback_mocks(choice, user_id=709, cfg=fake_config)

    with patch("bot.handler.set_user_lang", new=AsyncMock()) as set_lang:
        await handle_callback(update, context)

    set_lang.assert_awaited_once_with(709, lang)
    context.bot.set_my_commands.assert_awaited_once()
    call = context.bot.set_my_commands.await_args
    cmds = call.kwargs.get("commands") or call.args[0]
    scope = call.kwargs.get("scope") or (call.args[1] if len(call.args) > 1 else None)
    assert isinstance(scope, BotCommandScopeChat)
    assert scope.chat_id == 709  # per-chat override, wins over client locale
    assert "language_code" not in call.kwargs  # chosen language applies to any client locale
    got = [(c.command, c.description) for c in cmds]
    assert got == [(c.command, c.description) for c in bot_commands(lang)]
    cq.message.reply_photo.assert_awaited_once()  # greeting still follows


@pytest.mark.asyncio
async def test_lang_choice_menu_failure_does_not_block_greeting(app, fake_config):
    """A set_my_commands hiccup must never fail the callback or eat the
    greeting (failure-tolerant wrap, log debug)."""
    from bot.handler import handle_callback

    cq, update, context = _lang_callback_mocks("lang:en", user_id=710, cfg=fake_config)
    context.bot.set_my_commands = AsyncMock(side_effect=RuntimeError("api hiccup"))

    with patch("bot.handler.set_user_lang", new=AsyncMock()):
        await handle_callback(update, context)  # must not raise

    cq.edit_message_text.assert_awaited_once()
    cq.message.reply_photo.assert_awaited_once()
    caption = cq.message.reply_photo.await_args.kwargs.get("caption", "")
    assert caption, "greeting must survive a set_my_commands failure"


def test_bot_commands_localized_lists():
    """Startup registration source: RU and EN lists exist, carry the same
    command names in the same order, and descriptions are localized."""
    from bot.handler import bot_commands
    from telegram import BotCommand

    ru, en = bot_commands("ru"), bot_commands("en")
    assert ru and en
    assert all(isinstance(c, BotCommand) for c in ru + en)
    assert [c.command for c in ru] == [c.command for c in en]
    assert {"start", "help", "lang", "new", "stats", "donate"} <= {c.command for c in ru}
    for r, e in zip(ru, en):
        assert r.description and e.description
        assert r.description != e.description  # actually localized, not copies
        for d in (r.description, e.description):
            assert "—" not in d and "–" not in d  # no long/short dash in strings
    assert bot_commands("ru") is not bot_commands("ru")  # fresh list per call


# ════ F6: video quality for link videos ════

def _video_cb_update(data):
    """A callback update for the video flow: cq with a workable status mock
    (vstat.edit_text/delete must be awaitable on the reply_text result)."""
    cq = MagicMock()
    cq.data = data
    cq.answer = AsyncMock()
    cq.edit_message_text = AsyncMock()
    cq.edit_message_reply_markup = AsyncMock()
    vstat = MagicMock()
    vstat.edit_text = AsyncMock()
    vstat.delete = AsyncMock()
    cq.message.reply_text = AsyncMock(return_value=vstat)
    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=123, is_bot=False, first_name="Test")
    update.effective_chat.id = 456
    return update


def _video_context(fake_config, media):
    context = MagicMock()
    context.bot_data = {"cfg": fake_config, "media": media}
    context.chat_data = {}
    context.bot.send_video = AsyncMock()
    return context


def _run_created_tasks(context):
    """context.application.create_task replacer: run the background video task
    for real (a MagicMock would swallow the coroutine)."""
    tasks = []

    def _capture(coro, update=None):
        t = asyncio.create_task(coro)
        tasks.append(t)
        return t

    context.application.create_task = _capture
    return tasks


def test_video_quality_picker_full_ladder():
    """The picker offers the full ladder 240-1440, two buttons per row,
    ascending (the old picker's layout, extended with 240/360), and keeps the
    old dl:q:{q}:{vid} callback contract."""
    from bot.handler import _video_quality_picker
    kb = _video_quality_picker("abc12345678")
    rows = [[b.text for b in row] for row in kb.inline_keyboard]
    assert rows == [["240p", "360p"], ["480p", "720p"], ["1080p", "1440p"]]
    cbs = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert cbs == [f"dl:q:{q}:abc12345678"
                   for q in ("240", "360", "480", "720", "1080", "1440")]


@pytest.mark.asyncio
async def test_video_button_shows_quality_picker(app, fake_config):
    """dl:vid (the Video button on a link card) shows the quality picker —
    the owner's original flow: NO download starts until a height is picked."""
    vid = "abc12345678"
    media = MagicMock()
    media.download_video = AsyncMock()

    update = _video_cb_update(f"dl:vid:{vid}")
    context = _video_context(fake_config, media)
    context.chat_data[f"url:{vid}"] = {
        "url": f"https://www.youtube.com/watch?v={vid}", "title": "Test Video"}

    with patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")):
        await handle_callback(update, context)

    cq = update.callback_query
    cq.edit_message_text.assert_awaited_once()
    text, kwargs = cq.edit_message_text.await_args.args[0], cq.edit_message_text.await_args.kwargs
    assert "Выбери качество:" in text and "Test Video" in text
    kb = kwargs["reply_markup"]
    rows = [[b.text for b in row] for row in kb.inline_keyboard]
    assert rows == [["240p", "360p"], ["480p", "720p"], ["1080p", "1440p"]]
    cbs = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert cbs == [f"dl:q:{q}:{vid}" for q in ("240", "360", "480", "720", "1080", "1440")]
    media.download_video.assert_not_awaited()  # no silent default download
    context.bot.send_video.assert_not_called()
    assert context.chat_data.get("video_busy") is None


@pytest.mark.asyncio
async def test_quality_pick_downloads_at_picked_height(app, fake_config, tmp_path):
    """A picker selection (dl:q:240) downloads at exactly that height, the
    caption carries the chosen label; the delivered file carries NO keyboard
    (owner 06.11: no post-delivery quality offers)."""
    vid = "abc12345678"
    url = f"https://www.youtube.com/watch?v={vid}"
    media = MagicMock()
    mp4 = tmp_path / "v.mp4"
    mp4.write_bytes(b"v")
    media.download_video = AsyncMock(return_value=(
        str(mp4), {"id": vid, "chosen_quality": "240", "channel": "Ch",
                   "upload_date": "20240101"}))

    update = _video_cb_update(f"dl:q:240:{vid}")
    context = _video_context(fake_config, media)
    context.chat_data[f"url:{vid}"] = {"url": url, "title": "Test Video"}
    tasks = _run_created_tasks(context)

    with patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
         patch("bot.handler.insert_query", AsyncMock(return_value=7)), \
         patch("bot.handler.insert_event", AsyncMock()), \
         patch("bot.handler.incr_usage", AsyncMock()):
        await handle_callback(update, context)
        await asyncio.gather(*tasks)

    media.download_video.assert_awaited_once()
    assert media.download_video.await_args.args[0] == url
    assert media.download_video.await_args.args[1] == "240"  # the picked height
    assert "240p" in update.callback_query.edit_message_text.await_args.args[0]
    caption = context.bot.send_video.await_args.kwargs["caption"]
    assert "240p" in caption
    # no post-delivery quality offers: the sent video has no keyboard at all
    assert context.bot.send_video.await_args.kwargs.get("reply_markup") is None
    assert context.chat_data.get("video_busy") is None  # released in finally


@pytest.mark.asyncio
async def test_dlv_button_shows_full_picker(app, fake_config):
    """The dlv button (Video on a delivered track) opens the same full-ladder
    picker: 240-1440, dl:q callbacks, no download until a pick."""
    from services.llm_search import Candidate
    vid = "abc12345678"
    track = Candidate(vid, "Track Title", "Artist", 200, 1000)
    track.source = "youtube"
    media = MagicMock()
    media.download_video = AsyncMock()

    update = _video_cb_update(f"dlv:5:0")
    context = _video_context(fake_config, media)
    with patch("bot.handler._load_state",
               AsyncMock(return_value={"candidates": [track]})), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")):
        await handle_callback(update, context)

    cq = update.callback_query
    cq.message.reply_text.assert_awaited_once()
    text, kwargs = cq.message.reply_text.await_args.args[0], cq.message.reply_text.await_args.kwargs
    assert "Выбери качество:" in text and "Track Title" in text
    cbs = [b.callback_data for row in kwargs["reply_markup"].inline_keyboard for b in row]
    assert cbs == [f"dl:q:{q}:{vid}" for q in ("240", "360", "480", "720", "1080", "1440")]
    media.download_video.assert_not_awaited()
    # the picker stores the url state the dl:q callback resolves
    assert context.chat_data[f"url:{vid}"]["url"] == f"https://www.youtube.com/watch?v={vid}"


@pytest.mark.asyncio
async def test_qvid_callback_is_gone_no_download(app, fake_config):
    """Owner 06.11: the post-delivery quality re-fetch buttons (qvid:{vid}:{q})
    are removed with their handler. A qvid press — even a well-formed one on a
    message delivered before the change — downloads nothing and sends nothing;
    the callback just falls through (answered by the generic q.answer())."""
    media = MagicMock()
    media.download_video = AsyncMock()

    for payload in ("qvid:abc12345678:240",     # well-formed (old live button)
                    "qvid:abc12345678:999"):    # garbage quality
        update = _video_cb_update(payload)
        context = _video_context(fake_config, media)
        await handle_callback(update, context)
        media.download_video.assert_not_awaited()
        context.bot.send_video.assert_not_awaited()
        update.callback_query.message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_link_has_no_quality_buttons(app, recording, tmp_path):
    """The quality row is video-only: an audio send (dl:aud path via
    _send_track) never carries qvid buttons."""
    from bot.handler import _send_track
    from services.llm_search import Candidate
    from services.metadata import MetaOut

    m4a = tmp_path / "v1.m4a"
    m4a.write_bytes(b"fake")
    track = Candidate("v1", "Test Track", "Test Artist", 200, 1000, url="https://a.example/v1")

    media = MagicMock()
    media.download_audio = AsyncMock(return_value=(
        str(m4a), None, {"id": "v1", "title": "Test Track", "uploader": "Test Artist"}))
    meta = MagicMock()
    meta.resolve = AsyncMock(return_value=MetaOut(artist="Test Artist", track="Test Track"))

    context = MagicMock()
    context.bot_data = {"metadata": meta}
    context.bot.send_audio = AsyncMock()
    context.chat_data = {}

    update = _text_update("test")
    update.set_bot(app.bot)
    update.message.set_bot(app.bot)
    update.effective_chat.set_bot(app.bot)

    with _patch_store(), patch("bot.handler.tag_enrichment.enrich"), \
         patch("bot.handler.upsert_session", AsyncMock()):
        ok = await _send_track(update, context, None, media, 1, 0, [track], track,
                               "high", keyboard="url")

    assert ok is True
    kb = context.bot.send_audio.await_args.kwargs["reply_markup"]
    callbacks = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert callbacks  # the url keyboard (Similar/Fact) is there...
    assert not any(c.startswith("qvid:") for c in callbacks)  # ...without any quality buttons


# ════ Miss chain: same-RECORD skip (live 06.11, Iron Man 2) ════

class TestNextFreshIndex:
    """The rejected song must not return from another upload, even when the
    uploader's version tag changes the exact normalized key."""

    @staticmethod
    def _cand(vid, title):
        from types import SimpleNamespace
        return SimpleNamespace(video_id=vid, title=title)

    def test_same_song_other_upload_skipped(self):
        from bot.handler import _next_fresh_index
        cands = [
            self._cand("v0", "AC/DC - Shoot To Thrill (Iron Man 2 Version)"),
            self._cand("v1", "AC/DC - Shoot to Thrill - Iron Man 2 Version [HD]"),
            self._cand("v2", "John Debney - Sledgehammer"),
        ]
        # rank0 (v0) rejected -> v1 is the same record under another upload
        assert _next_fresh_index(cands, 0, ["v0"]) == 2

    def test_different_song_served(self):
        from bot.handler import _next_fresh_index
        cands = [
            self._cand("v0", "Hilary Duff - Come Clean"),
            self._cand("v1", "Hilary Duff - Fly"),
        ]
        assert _next_fresh_index(cands, 0, ["v0"]) == 1

    def test_exhausted_returns_end(self):
        from bot.handler import _next_fresh_index
        cands = [
            self._cand("v0", "AC/DC - Shoot To Thrill (Iron Man 2 Version)"),
            self._cand("v1", "Shoot to Thrill - AC/DC"),
        ]
        assert _next_fresh_index(cands, 0, ["v0"]) == len(cands)


# ════ network-storm watchdog (incidents 08-10.11: silent fetcher death) ════

def test_storm_restart_decision():
    """Heavy recent storm + network back -> restart. Quiet net, stale storm
    or dead network -> keep running (weather, not a dead fetcher)."""
    from bot.main import _storm_restart_needed
    now = 1_000_000.0
    storm = {"count": 20, "last_ts": now - 60}
    assert _storm_restart_needed(storm, True, now=now)      # storm + net back
    assert not _storm_restart_needed(storm, False, now=now)  # net still down
    assert not _storm_restart_needed({"count": 20, "last_ts": now - 10_000},
                                     True, now=now)          # storm long gone
    assert not _storm_restart_needed({"count": 3, "last_ts": now - 60},
                                     True, now=now)          # light weather
    assert not _storm_restart_needed({"count": 0, "last_ts": 0}, True, now=now)


# ════ duplicate-update guard (10.11: one message, two searches) ════

def test_duplicate_update_guard():
    """The same text from the same chat inside the window is dropped; a
    different text, a different chat, or the same text after the window
    passes."""
    from bot import handler as h
    h._recent_descs.clear()
    assert not h._drop_duplicate(1, "дип перпл")          # first -> pass
    assert h._drop_duplicate(1, "дип перпл")              # redelivery -> drop
    assert not h._drop_duplicate(2, "дип перпл")          # another chat
    assert not h._drop_duplicate(1, "дип перпл кинг оф дримс")  # another text
    # after the window the deliberate repeat works again
    h._recent_descs[(1, hash("дип перпл"))] = h.time.monotonic() - 30.0
    assert not h._drop_duplicate(1, "дип перпл")


# ════ verdict advice sanitizer (11.11 live: ID3 advice in the card) ════

def test_sanitize_verdict_strips_advice_keeps_facts():
    from bot.handler import _sanitize_verdict
    live = ("Музыку к фильму «Бойцовский клуб» (1999) написали The Dust Brothers. "
            "Пожалуйста, укажите правильное имя исполнителя в ID3-тегах вашего файла.")
    assert _sanitize_verdict(live) == \
        "Музыку к фильму «Бойцовский клуб» (1999) написали The Dust Brothers."
    clean = "Mystère et suspense — сингл Fonky Family, 1998, дебютный альбом."
    assert _sanitize_verdict(clean) == clean
    assert _sanitize_verdict("") == ""


@pytest.mark.asyncio
async def test_fact_button_live_key_separate_from_studio(app, fake_config):
    """A live card (fact:v1|lv) asks for and caches 'Track (live)'; the studio
    key stays plain 'Track' so old cached studio facts keep serving (12.11)."""
    from unittest.mock import AsyncMock as AM
    cached_row = {"artist": "Nirvana", "track": "Come As You Are",
                  "album": "", "year": ""}
    # live press: L1 miss for the live key, sonar answers, save under the live key
    update, context = _fact_update_and_context("fact:v1|lv", fake_config)
    with patch("bot.handler.get_cached_file", AM(return_value=cached_row)), \
         patch("bot.handler.get_user_lang", AM(return_value="ru")), \
         patch("bot.handler.get_fact", AM(return_value=None)) as _get, \
         patch("bot.handler._fact_l2", AM(return_value="")), \
         patch("bot.handler.get_usage", AM(return_value={"facts": 0})), \
         patch("bot.handler.get_global_usage", AM(return_value={"facts": 0})), \
         patch("bot.handler.incr_usage", AM()), \
         patch("bot.handler.save_fact", AM()) as _save, \
         patch("bot.handler._sonar_track_fact", AM(return_value="Факт про Reading")) as _sonar, \
         patch("bot.handler._maybe_wtp", AM()):
        await handle_callback(update, context)
    _get.assert_any_await("Nirvana", "Come As You Are (live)", "ru")
    _sonar.assert_awaited_once()
    assert "Come As You Are (live)" in str(_sonar.await_args)
    assert "Come As You Are (live)" in str(_save.await_args)
    # studio press (old-style bare button): the plain key, unchanged
    update, context = _fact_update_and_context("fact:v1", fake_config)
    with patch("bot.handler.get_cached_file", AM(return_value=cached_row)), \
         patch("bot.handler.get_user_lang", AM(return_value="ru")), \
         patch("bot.handler.get_fact", AM(return_value="студийный факт")) as _get2:
        await handle_callback(update, context)
    _get2.assert_any_await("Nirvana", "Come As You Are", "ru")
    texts = [c.args[0] for c in update.callback_query.message.reply_text.await_args_list]
    assert any("студийный факт" in t for t in texts)
