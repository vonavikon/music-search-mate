"""L2 persistent metadata cache (services.meta_cache) wiring in bot/handler.

Covers the four integration points of the cache layer:
  a) facts:  store.get_fact (L1) miss + mc.get_fact hit -> the fact is served
             for free (before the caps) and L1 is healed via save_fact;
  b) bio RU: mc.get_bio hit -> no LLM translation call;
  c) card:   mc.get_track hit -> metadata_svc.resolve is not called at all;
  d) artwork: mc.artwork_path hit -> the cover is sent as a local file, not a URL.

meta_cache functions are patched at the services.meta_cache module level
(handler holds the module reference), following the MagicMock/AsyncMock
harness style of tests/test_handler.py.
"""
import asyncio

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from telegram import InputFile, User

from bot.handler import handle_callback, _send_track
from services.llm_search import Candidate
from services.metadata import MetaOut


EN_BIO = ("Deep Purple formed in Hertford in 1968 and went on to pioneer "
          "hard rock and heavy metal.")
RU_BIO = "Deep Purple образовались в Хартфорде в 1968 году."
FACT = "Deep Purple recorded the Machine Head album in Montreux in 1971."


def _mk_update():
    """MagicMock update (CallbackQuery/Chat methods are not overridable on
    frozen TelegramObjects, so plain mocks with explicit AsyncMocks)."""
    update = MagicMock()
    update.effective_user = User(id=123, is_bot=False, first_name="Test")
    update.effective_chat.id = 456
    update.effective_chat.send_photo = AsyncMock()
    update.effective_chat.send_message = AsyncMock()
    return update


def _mk_context(bot_data=None):
    context = MagicMock()
    context.bot_data = bot_data if bot_data is not None else {}
    context.chat_data = {}
    context.bot.send_audio = AsyncMock()
    context.bot.send_chat_action = AsyncMock()
    return context


def _track():
    return Candidate("v1", "Smoke on the Water", "Deep Purple Official",
                     200, 1000, url="https://www.youtube.com/watch?v=v1")


# ════ a) facts: L2 hit is free and heals L1 ════

@pytest.mark.asyncio
async def test_fact_l2_hit_backfills_store_and_skips_caps(fake_config):
    """The store 30d cache misses, the L2 persistent cache hits: the fact is
    replied before the caps (no usage read, no counter bump) and save_fact
    backfills L1. The cached artist carries channel litter ("Deep Purple
    Official") -> both cache keys must use the cleaned "Deep Purple"."""
    cq = MagicMock()
    cq.data = "fact:v1"
    cq.answer = AsyncMock()
    cq.message.reply_text = AsyncMock()
    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=123, is_bot=False, first_name="Test")
    context = _mk_context({"cfg": fake_config, "media": MagicMock()})

    cached_row = {"artist": "Deep Purple Official",
                  "track": "Smoke on the Water",
                  "album": "Machine Head", "year": "1972"}
    with patch("bot.handler.insert_event", AsyncMock()), \
         patch("bot.handler.get_cached_file", AsyncMock(return_value=cached_row)), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
         patch("bot.handler.get_fact", AsyncMock(return_value=None)), \
         patch("bot.handler.save_fact", AsyncMock()) as _save, \
         patch("bot.handler.get_usage", AsyncMock()) as _usage, \
         patch("bot.handler.incr_usage", AsyncMock()) as _incr, \
         patch("services.meta_cache.get_fact", AsyncMock(return_value=FACT)) as _mc_get:
        await handle_callback(update, context)

    # L2 lookup and the L1 backfill share the cleaned artist key
    _mc_get.assert_awaited_once_with("Deep Purple", "Smoke on the Water", "ru")
    _save.assert_awaited_once_with("Deep Purple", "Smoke on the Water", "ru", FACT)
    texts = [c.args[0] for c in cq.message.reply_text.await_args_list]
    assert any(FACT in t for t in texts)
    # a cached fact is free: no limits, no counters
    _usage.assert_not_awaited()
    _incr.assert_not_awaited()


# ════ b) bio RU: an L2 translation hit skips the LLM ════

@pytest.mark.asyncio
async def test_bio_ru_l2_hit_no_llm_call():
    """A cached-file card with an EN bio: mc.get_bio(artist, 'ru') answers
    the translation, bothub.chat_text is never awaited, and the RU text lands
    in the audio caption."""
    update = _mk_update()
    bothub = MagicMock()
    bothub.chat_text = AsyncMock()
    _md = MagicMock()
    _md.refetch_artwork = AsyncMock(return_value="")
    context = _mk_context({"metadata": _md, "bothub": bothub})
    media = MagicMock()
    media.download_audio = AsyncMock()  # must not be reached (file_id path)

    cached = {"file_id": "file-cached", "artist": "Deep Purple Official",
              "track": "Smoke on the Water", "bio": EN_BIO}
    with patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
         patch("bot.handler.get_cached_file", AsyncMock(return_value=cached)), \
         patch("bot.handler.insert_event", AsyncMock()), \
         patch("bot.handler.upsert_session", AsyncMock()), \
         patch("bot.handler.load_query_state", AsyncMock(return_value=None)), \
         patch("bot.handler.save_query_state", AsyncMock()), \
         patch("bot.handler.cache_file", AsyncMock()), \
         patch("services.meta_cache.get_bio", AsyncMock(return_value=RU_BIO)) as _mc_bio, \
         patch("services.meta_cache.artwork_path", AsyncMock(return_value=None)):
        ok = await _send_track(update, context, None, media, 1, 0,
                               [_track()], _track(), "high")

    assert ok is True
    media.download_audio.assert_not_awaited()
    # the key is the cleaned artist, the call is for the ru translation
    _mc_bio.assert_awaited_once_with("Deep Purple", "ru")
    bothub.chat_text.assert_not_awaited()
    caption = context.bot.send_audio.await_args.kwargs["caption"]
    assert RU_BIO[:24] in caption


# ════ c) card: an mc.get_track hit replaces metadata_svc.resolve ════

@pytest.mark.asyncio
async def test_card_l2_hit_skips_resolve(tmp_path):
    """mc.get_track(video_id) hit -> the early and the post-download resolves
    are both skipped; the card is built from the cached dict; no put_track
    (a read is not re-written)."""
    update = _mk_update()
    m4a = tmp_path / "v1.m4a"
    m4a.write_bytes(b"fake")

    media = MagicMock()
    media.download_audio = AsyncMock(return_value=(
        str(m4a), None, {"id": "v1", "title": "Smoke on the Water",
                         "channel": "Deep Purple Official"}))
    metadata_svc = MagicMock()
    metadata_svc.resolve = AsyncMock(
        return_value=MetaOut(artist="Deep Purple", track="Smoke on the Water"))
    context = _mk_context({"metadata": metadata_svc})  # no bothub -> no sonar

    card_cache = {"artist": "Deep Purple", "title": "Smoke on the Water",
                  "album": "Machine Head", "year": "1972", "genre": "hard rock"}
    with patch("bot.handler.get_cached_file", AsyncMock(return_value=None)), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
         patch("bot.handler.insert_event", AsyncMock()), \
         patch("bot.handler.incr_usage", AsyncMock()), \
         patch("bot.handler.upsert_session", AsyncMock()), \
         patch("bot.handler.load_query_state", AsyncMock(return_value=None)), \
         patch("bot.handler.save_query_state", AsyncMock()), \
         patch("bot.handler.cache_file", AsyncMock()), \
         patch("bot.handler.tag_enrichment.enrich"), \
         patch("services.meta_cache.get_track", AsyncMock(return_value=card_cache)) as _mc_track, \
         patch("services.meta_cache.put_track", AsyncMock()) as _mc_put, \
         patch("services.meta_cache.get_bio", AsyncMock(return_value=None)), \
         patch("services.meta_cache.artwork_path", AsyncMock(return_value=None)), \
         patch("services.meta_cache.put_bio", AsyncMock()):
        ok = await _send_track(update, context, None, media, 1, 0,
                               [_track()], _track(), "high")

    assert ok is True
    metadata_svc.resolve.assert_not_awaited()
    # both call sites (the early card and the post-download resolve) asked L2
    assert _mc_track.await_count >= 2
    assert all(c.args == ("v1",) for c in _mc_track.await_args_list)
    _mc_put.assert_not_awaited()  # a cache hit is not written back
    caption = context.bot.send_audio.await_args.kwargs["caption"]
    assert "Smoke on the Water" in caption  # the card came from the cached dict


# ════ d) artwork: a local L2 file beats the URL ════

@pytest.mark.asyncio
async def test_artwork_l2_file_sent_instead_of_url(tmp_path):
    """mc.artwork_path returns a local cover -> send_photo gets a file
    (InputFile), never the URL string; the fresh resolve result goes to
    put_track with the non-empty fields only."""
    update = _mk_update()
    m4a = tmp_path / "v1.m4a"
    m4a.write_bytes(b"fake")
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"\xff\xd8\xff fakejpeg")

    media = MagicMock()
    media.download_audio = AsyncMock(return_value=(
        str(m4a), None, {"id": "v1", "title": "Smoke on the Water",
                         "channel": "Deep Purple Official"}))
    metadata_svc = MagicMock()
    metadata_svc.resolve = AsyncMock(return_value=MetaOut(
        artist="Deep Purple", track="Smoke on the Water",
        album="Machine Head", artwork_url="https://example.com/cover.jpg"))
    context = _mk_context({"metadata": metadata_svc})  # no bothub -> no sonar

    with patch("bot.handler.get_cached_file", AsyncMock(return_value=None)), \
         patch("bot.handler.get_user_lang", AsyncMock(return_value="ru")), \
         patch("bot.handler.insert_event", AsyncMock()), \
         patch("bot.handler.incr_usage", AsyncMock()), \
         patch("bot.handler.upsert_session", AsyncMock()), \
         patch("bot.handler.load_query_state", AsyncMock(return_value=None)), \
         patch("bot.handler.save_query_state", AsyncMock()), \
         patch("bot.handler.cache_file", AsyncMock()), \
         patch("bot.handler.tag_enrichment.enrich"), \
         patch("services.meta_cache.get_track", AsyncMock(return_value=None)), \
         patch("services.meta_cache.put_track", AsyncMock()) as _mc_put, \
         patch("services.meta_cache.get_bio", AsyncMock(return_value=None)), \
         patch("services.meta_cache.artwork_path", AsyncMock(return_value=cover)) as _mc_art, \
         patch("services.meta_cache.store_artwork", AsyncMock()):
        ok = await _send_track(update, context, None, media, 1, 0,
                               [_track()], _track(), "high")
        await asyncio.sleep(0)  # let the fire-and-forget store_artwork task run

    assert ok is True
    photos = [c.kwargs.get("photo") for c in
              update.effective_chat.send_photo.await_args_list]
    assert photos, "the cover card must be sent"
    assert all(not isinstance(p, str) for p in photos)  # no URL sends
    assert any(isinstance(p, InputFile) for p in photos)
    # the L2 artwork was asked with the resolved artist/album
    assert any(c.args == ("Deep Purple", "Machine Head")
               for c in _mc_art.await_args_list)
    # the fresh resolve was persisted with the non-empty fields only
    _mc_put.assert_awaited_once()
    vid, base = _mc_put.await_args.args
    assert vid == "v1"
    assert base["artist"] == "Deep Purple"
    assert base["artwork_url"] == "https://example.com/cover.jpg"
    assert all(v for v in base.values())
