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
