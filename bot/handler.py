"""Telegram handlers: search pipeline + URL fast-path + feedback/similar/refinement.

Flow (updated per review 27c):
  message -> [refine active?] refine the previous query excluding rejected ones
           -> rate-limit ->
    [URL?] media.extract_url_meta -> Audio/Video buttons
    [else] recent_context -> search_candidates ->
      [clarify?] clarifying question (no download)
      [else]    auto best -> download -> send (single+download path,
                preview carousel removed)
  callback:
    fb:h/m (miss accumulates rejected_ids + the next candidate),
    sim: similar (pool cache or a new search), rn/rf/rfx: refinement,
    dlv: video from a track, dl:aud/vid/q: download by link,
    fact: a track fact (sonar on demand, 30d cache, caps)
"""
import asyncio
import json
import logging
import os
import re
import tempfile
import time
from html import escape
from pathlib import Path
from uuid import uuid4

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

import random

from services.llm_search import search_candidates, rerank_pool, Candidate, _ref_key, _START_PHRASES as START_PHRASES, WTP_PRICES, destructive_prefilter
from services.match_score import should_auto_download, is_ambiguous
from services.store import (
    insert_query, insert_feedback, count_recent, recent_context, insert_event, funnel_today,
    get_usage, incr_usage, get_global_usage, should_show_survey, save_survey_shown, save_survey_response, get_user_lang, set_user_lang,
    upsert_session, get_session, save_query_state, load_query_state,
    delete_query_state, cache_file, get_cached_file, get_query_correlation,
    update_cached_artwork, get_fact, save_fact,
)
from services.tracing import SearchTrace, score_feedback
from services.metadata import MetaOut, pretty_title
from services.media import cleanup_temp
from services import tag_enrichment

log = logging.getLogger("bot.handler")
_CLEANUP_EVERY = 10
_SIM_REUSE_WINDOW = 900  # sec: cached "similar" window (same reference -> next from pool)
_REUSE_BLOCKERS = ("другое", "другой", "другая", "вообще", "не это", "не этот", "совсем")  # these words mark the refinement as a "new search"
_REFINE_TTL = 600  # sec: window where text after "Уточнить" counts as a query edit (spec: 10-30 min)

# per-chat pipeline serialization: two back-to-back messages in one chat must not
# run parallel searches (race on chat_data / status / q:{id} / rejected_ids). The
# second waits for the first. The lock lives in process memory — lost on restart (fine for an MVP).
_chat_locks: dict[int, asyncio.Lock] = {}


def _chat_lock(chat_id: int) -> asyncio.Lock:
    lock = _chat_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_locks[chat_id] = lock
    return lock


def _denied(update: Update, context) -> bool:
    """Personal allowlist ALLOWED_USER_IDS. The bot token is publicly reachable;
    without the list, anyone who finds the bot drives LLM and yt-dlp on the
    owner's tab. Empty list = no restrictions (deliberate: test runs)."""
    cfg = context.bot_data.get("cfg")
    ids = getattr(cfg, "allowed_user_ids", ()) or ()
    if not ids or update.effective_user is None:
        return False
    if update.effective_user.id in ids:
        return False
    log.info("access denied: user=%s outside allowlist (%d id(s))",
             update.effective_user.id, len(ids))
    return True


# All user-facing errors/refusals are bilingual. Key -> {ru, en}; the language
# comes from get_user_lang (the user's /lang choice), default ru. Search
# statuses ("Ищу...") are not errors and stay RU pending a separate decision.
_T = {
    "guest": {"ru": "Бот частный, им пользуются по приглашению.",
              "en": "The bot is private, invite only."},
    "flood": {"ru": "Слишком много сообщений подряд — притормози немного.",
              "en": "Too many messages in a row, slow down a bit."},
    "too_long": {"ru": "Слишком длинный запрос. Опиши, что ищешь, в паре предложений.",
                 "en": "The request is too long. Describe what you're looking for in a sentence or two."},
    "destructive": {"ru": "Не буду искать по такому запросу.",
                    "en": "I won't search for that."},
    "hourly": {"ru": "Лимит на этот час исчерпан. Почасовое окно обновится само, напиши чуть позже.",
               "en": "Hourly limit reached. The window resets on its own, try a bit later."},
    "day_limit": {"ru": "Дневной лимит исчерпан. Обновится {date} в 03:00 МСК.",
                  "en": "Daily limit reached. Resets on {date} at 03:00 MSK."},
    "global_budget": {"ru": "Общий бюджет поисков исчерпан. Обновится {date} в 03:00 МСК.",
                      "en": "Shared search budget is used up. Resets on {date} at 03:00 MSK."},
    "host_not_allowed": {"ru": "Эту площадку я не качаю. Поддерживаю: YouTube, SoundCloud, Bandcamp, VK. Пришли ссылку оттуда или опиши трек словами.",
                         "en": "I don't download from this platform. I support YouTube, SoundCloud, Bandcamp, VK. Send a link from there or describe the track."},
    "malformed_url": {"ru": "Ссылка выглядит битой (пробел или опечатка). Пришли рабочую ссылку или опиши трек словами.",
                      "en": "The link looks broken (a space or a typo). Send a working link or describe the track."},
    "not_found": {"ru": "Не получилось найти. Попробуй иначе: припев, исполнителя или ссылку.",
                  "en": "Couldn't find it. Try again: a chorus line, the artist, or a link."},
    "nothing_found": {"ru": "Ничего не нашёл. Уточни описание.",
                      "en": "Nothing found. Make the description more specific."},
    "variants_over": {"ru": "Варианты кончились. Скажи, что не так, или опиши поиск иначе.",
                      "en": "Out of options. Tell me what's wrong, or describe the search differently."},
    "last_variant": {"ru": "Это был последний вариант. Скажи, что не так.",
                     "en": "That was the last one. Tell me what's wrong."},
    "search_cold": {"ru": "Тот поиск уже остыл. Напиши, что искать.",
                    "en": "That search has gone cold. Tell me what to look for."},
    "stale_state": {"ru": "Прошлый поиск уже не актуален.",
                    "en": "The previous search is no longer relevant."},
    "empty_feedback": {"ru": "Пусто. Опиши словами, что не так, или дай новый запрос.",
                       "en": "Empty. Tell me in words what's wrong, or send a new request."},
    "link_read_fail": {"ru": "Не удалось прочитать ссылку. Если это поддерживаемая платформа, попробуй позже.",
                       "en": "Couldn't read the link. If it's a supported platform, try later."},
    "link_read_hint": {"ru": "Не удалось прочитать ссылку. Опиши трек словами или пришли рабочую ссылку.",
                   "en": "Couldn't read the link. Describe the track in words or send a working link."},
    "link_read_desc": {"ru": "Не удалось прочитать ссылку. Пробую понять описание…",
                       "en": "Couldn't read the link. Trying to understand the description…"},
    "dl_none": {"ru": "Не смог скачать ни один из вариантов. Попробуй позже или кинь ссылку.",
                "en": "Couldn't download any of the variants. Try later or send a link."},
    "dl_tg_reject": {"ru": "Трек скачался, но Telegram не принял файл. Пробую другой вариант…",
                     "en": "The track downloaded, but Telegram rejected the file. Trying another variant…"},
    "dl_track_fail": {"ru": "Не смог скачать {title}. Попробуй другую ссылку.",
                      "en": "Couldn't download {title}. Try another link."},
    "dl_fail_generic": {"ru": "Не смог скачать. Попробуй ещё раз или пришли ссылку на трек.",
                        "en": "Couldn't download. Try again or send a link to the track."},
    "dl_failed_note": {"ru": "(скачивание не удалось, попробуй другой)",
                       "en": "(download failed, try another)"},
    "unsupported": {"ru": "Понимаю только текст. Опиши трек словами или пришли ссылку.",
                    "en": "I only understand text. Describe the track in words or send a link."},
    "link_stale": {"ru": "Ссылка устарела. Пришли заново.",
                   "en": "The link has expired. Send it again."},
    "video_yt_only": {"ru": "Видео доступно только для YouTube-ссылок.",
                      "en": "Video is only available for YouTube links."},
    "video_busy": {"ru": "Видео уже качается — дождись, оно придёт.",
                   "en": "A video is already downloading, wait for it to arrive."},
    "video_cap": {"ru": "Видео на сегодня закончились. Аудио и поиск работают как обычно.",
                "en": "No more videos today. Audio and search work as usual."},
    "video_fail": {"ru": "Не смог скачать или отправить видео. Попробуй качество ниже или аудио.",
                   "en": "Couldn't download or send the video. Try a lower quality or audio."},
    "fact_ctx_lost": {"ru": "Контекст трека потерялся. Попроси этот трек ещё раз — тогда расскажу и факт.",
                      "en": "The track context is lost. Ask for the track again and I'll share the fact."},
    "fact_dup": {"ru": "Этот факт уже чуть выше ⬆️",
                 "en": "This fact is just above ⬆️"},
    "fact_cap_user": {"ru": "Факты на сегодня закончились — попробуй завтра.",
                      "en": "No more facts today, come back tomorrow."},
    "fact_cap_global": {"ru": "Факты на сегодня закончились у всех — попробуй завтра.",
                        "en": "Facts are used up for today for everyone, come back tomorrow."},
    "fact_none": {"ru": "Про этот трек факта не нашлось.",
                  "en": "Couldn't find a fact about this track."},
    "st_refine": {"ru": "🎧 Учитываю уточнение, ищу заново…",
                  "en": "🎧 Taking your edit into account, searching again…"},
    "st_url": {"ru": "🎧 Читаю ссылку…",
               "en": "🎧 Reading the link…"},
    "st_similar": {"ru": "🎧 Ищу похожее…",
                   "en": "🎧 Looking for similar…"},
    "st_other": {"ru": "🎧 Ищу другой…",
                 "en": "🎧 Looking for another one…"},
    "st_processing": {"ru": "🎧 Готовлю файл…",
                    "en": "🎧 Preparing the file…"},
    "dl_bar": {"ru": "⬇️ {bar} {pct}%",
               "en": "⬇️ Downloading {bar} {pct}%"},
    "video_start": {"ru": "🎬 Качаю видео {quality}: {title}…",
                    "en": "🎬 Downloading video {quality}: {title}…"},
    "video_start_short": {"ru": "🎬 Качаю видео {quality}…",
                          "en": "🎬 Downloading video {quality}…"},
    "new_reset": {"ru": "Начинаем с чистого листа. Что ищем?",
                  "en": "Fresh start. What are we looking for?"},
    "wtp_thanks": {"ru": "Спасибо! Учёл: ⭐ {stars:.0f} за скачивание.",
                   "en": "Thanks! Noted: ⭐ {stars:.0f} per download."},
    "payment_thanks": {"ru": "Спасибо! Звёзды идут на сервер, LLM и трафик — бот остаётся без рекламы.",
                       "en": "Thank you! Stars go to the server, LLM and traffic — the bot stays ad-free."},
    "wtp_survey": {"ru": "Проект сделан на энтузиазме и против какой-либо рекламы. "
                         "В связи с его ростом поддерживать его всё труднее. "
                         "Мы хотели бы узнать, сколько бы пользователи были готовы платить "
                         "за каждое скачанное видео?",
                   "en": "This project runs on enthusiasm and against any advertising. "
                         "As it grows, keeping it up takes more and more. "
                         "We'd like to know how much users would be ready to pay "
                         "per downloaded video?"},
    "stats_head": {"ru": "Сегодня", "en": "Today"},
    "stats_you": {"ru": "Ты: поисков {s} · выдач {f} · фактов {fa} · видео {v} · {mb} МБ",
                  "en": "You: searches {s} · sends {f} · facts {fa} · videos {v} · {mb} MB"},
    "stats_funnel": {"ru": "Воронка: {q} зап. · находки {ok} · уточни {c} · пусто {e} · блок {b} · лимит {l} · скачано {d} · То {h} из {fb}",
                   "en": "Funnel: {q} req · found {ok} · clarify {c} · empty {e} · blocked {b} · limit {l} · downloaded {d} · hit {h} of {fb}"},
    "stats_all": {"ru": "Все: поисков {s} · выдач {f} · фактов {fa} · видео {v} · {mb} МБ",
                  "en": "Everyone: searches {s} · sends {f} · facts {fa} · videos {v} · {mb} MB"},
    "sends_over": {"ru": "Лимит скачиваний на сегодня вышел. Поиск остаётся, файлы вернутся завтра.",
                   "en": "Daily download limit reached. Search still works, files will be back tomorrow."},
}


async def _lang_of(user_id) -> str:
    """User language with a safe fallback (no record/DB -> ru)."""
    try:
        return (await get_user_lang(user_id)) or "ru"
    except Exception:
        return "ru"


async def _t(user_id, key: str, **fmt) -> str:
    """Error/refusal text in the user's language (default ru)."""
    d = _T.get(key) or {}
    s = d.get(await _lang_of(user_id)) or d.get("ru") or key
    return s.format(**fmt) if fmt else s




# ---------- persistent callback state (SQLite on top of chat_data) ----------

def _state_serialize(state: dict) -> str:
    """Candidate objects -> dicts for JSON. state must not be mutated (chat_data holds it)."""
    d = dict(state)
    d["candidates"] = [c.__dict__ for c in state.get("candidates", [])]
    return json.dumps(d, ensure_ascii=False)


def _state_deserialize(raw: str) -> dict:
    d = json.loads(raw)
    d["candidates"] = [Candidate(**c) for c in d.get("candidates", [])]
    return d


async def _save_state(context, query_id: int, state: dict) -> None:
    """Save callback state to chat_data (first-level cache) + SQLite (survives restarts)."""
    context.chat_data[f"q:{query_id}"] = state
    try:
        await save_query_state(query_id, _state_serialize(state))
    except Exception:
        log.exception("save_query_state failed q=%s", query_id)


async def _load_state(context, query_id: int) -> dict | None:
    """Callback state: chat_data, on a miss — SQLite (after a restart)."""
    state = context.chat_data.get(f"q:{query_id}")
    if state is not None:
        return state
    try:
        raw = await load_query_state(query_id)
    except Exception:
        log.exception("load_query_state failed q=%s", query_id)
        return None
    if raw is None:
        return None
    try:
        state = _state_deserialize(raw)
    except Exception:
        log.exception("state deserialize failed q=%s", query_id)
        return None
    context.chat_data[f"q:{query_id}"] = state  # restored to memory
    return state


async def _evict_state(context, query_id: int) -> None:
    """Drop the state from memory and DB (track found / variants exhausted)."""
    context.chat_data.pop(f"q:{query_id}", None)
    try:
        await delete_query_state(query_id)
    except Exception:
        log.exception("delete_query_state failed q=%s", query_id)


# Inline button labels in the user's language (callback_data unchanged)
_KB = {
    "hit": ("✅ То", "✅ That's it"),
    "miss": ("❌ Не то", "❌ Not it"),
    "refine": ("✏️ Уточнить", "✏️ Refine"),
    "refine_query": ("✏️ Уточнить запрос", "✏️ Refine request"),
    "similar": ("🔀 Похожее", "🔀 Similar"),
    "fact": ("💡 Факт", "💡 Fact"),
    "audio": ("🎵 Аудио", "🎵 Audio"),
    "video": ("🎬 Видео", "🎬 Video"),
    "open_yt": ("Открыть на YouTube", "Open on YouTube"),
    "cancel": ("🚫 Отмена", "🚫 Cancel"),
    "wtp_no": ("Не буду платить", "Won't pay"),
    "wtp_optout": ("Больше не спрашивать", "Don't ask again"),
}


def _kbt(lang: str, key: str) -> str:
    ru, en = _KB[key]
    return en if lang == "en" else ru


def _reply_keyboard(query_id: int, rank: int, keyboard: str, vid: str = "",
                    lang: str = "ru") -> InlineKeyboardMarkup:
    """Keyboard under a track. URL fast-path — only "Похожее" (the user brought
    the track themselves); search — То/Не то/Уточнить/Похожее. The 💡 fact button
    costs a sonar call ($0.008) and is generated only on demand (30d cache is free)."""
    fact_btn = InlineKeyboardButton(
        _kbt(lang, "fact"), callback_data=f"fact:{vid}") if vid else None
    if keyboard == "url":
        rows = [[InlineKeyboardButton(_kbt(lang, "similar"), callback_data=f"sim:{query_id}:{rank}")]]
        if fact_btn:
            rows[0].append(fact_btn)
        return InlineKeyboardMarkup(rows)
    rows = [
        [
            InlineKeyboardButton(_kbt(lang, "hit"), callback_data=f"fb:h:{query_id}:{rank}"),
            InlineKeyboardButton(_kbt(lang, "miss"), callback_data=f"fb:m:{query_id}:{rank}"),
        ],
        [
            InlineKeyboardButton(_kbt(lang, "refine"), callback_data=f"rf:{query_id}:{rank}"),
            InlineKeyboardButton(_kbt(lang, "similar"), callback_data=f"sim:{query_id}:{rank}"),
        ],
    ]
    # Owner request: the 🎬 Video button was removed from search cards —
    # video is link-only now. The dlv handler stays for old buttons.
    if fact_btn:
        rows.append([fact_btn])
    return InlineKeyboardMarkup(rows)

# URL fast-path: deterministic link detection BEFORE the LLM. Platform-agnostic —
# catches any http(s) URL (YouTube, SoundCloud, Bandcamp, Yandex Music, ...);
# yt-dlp deals with the platform. Strip trailing punctuation (a period/comma/
# bracket at the end of a sentence). Independent of the LLM, hence stable.
_URL_RE = re.compile(r"https?://[^\s]+", re.I)

_CONF_BADGE = {"high": "🎯", "medium": "🔎", "low": "🤔"}  # log-only; not shown on the card


def _looks_like_url(text: str) -> str | None:
    """Extract a URL from text. Platform-agnostic — yt-dlp handles the rest.
    Returns the URL or None."""
    m = _URL_RE.search(text)
    if not m:
        return None
    url = m.group(0)
    # strip trailing punctuation (period/comma/bracket at the end of a sentence)
    url = url.rstrip(".,;:!?)")
    return url


def _url_host_allowed(url: str, allowed_hosts) -> bool:
    """SSRF filter (audit finding). yt-dlp would take any URL from the
    message and go anywhere from the container (tgapi:8081, link-local).
    Only music platforms are allowed; subdomains pass via an endswith match."""
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    return any(host == h or host.endswith("." + h) for h in allowed_hosts)


def _looks_malformed_url(text: str) -> bool:
    """Malformed link: URL markers are present but no valid URL was extracted.
    Example: "https:// youtu.be/abc" (space after //) — _looks_like_url returns
    None, and we don't want to search the leftover "abc" as text."""
    low = text.strip().lower()
    if not any(m in low for m in ("http://", "https://", "youtu.be",
                                  "youtube.com", "soundcloud.com", "bandcamp.com")):
        return False
    return _looks_like_url(text) is None


# Bare "ещё такое же" ("more like this") requests with no musical content.
# Exact match against known forms (after normalization) — anything else is a new query, not a continuation.
_CONTINUATION_PHRASES = {
    "ещё", "еще", "ещё такое же", "еще такое же", "такое же",
    "ещё что-нибудь", "еще что-нибудь", "что-нибудь ещё", "что-нибудь еще",
    "ещё похожее", "еще похожее", "похожее ещё", "похожее еще",
    "есть ли ещё", "есть ли еще", "ещё есть", "еще есть",
    "в том же духе", "в том же стиле", "аналогичное", "подобное",
    "давай ещё", "давай еще", "дай ещё", "дай еще",
    "дай схожее", "дай похожее", "давай похожее", "давай схожее",
    "схожее", "похожее дай", "найди похожее", "найди схожее",
    "а дай схожее", "а дай похожее", "а давай похожее", "а давай схожее",
    "ещё в этом духе", "еще в этом духе", "хочу похожее", "хочу схожее",
    # variants with "ещё" in the middle (prod case: Alice in Chains)
    "дай еще похожее", "дай ещё похожее", "давай еще похожее", "давай ещё похожее",
    "а дай еще похожее", "а дай ещё похожее", "а давай еще похожее", "а давай ещё похожее",
    "еще дай похожее", "ещё дай похожее", "найди еще похожее", "найди ещё похожее",
}

# Regex fallback: a short phrase (<=5 words) made only of continuation words
# catches variants not listed in the exact list ("а дай ЕЩЁ похожее")
_CONTINUATION_WORDS = {"ещё", "еще", "похожее", "похожих", "схожее", "схожих",
                       "давай", "дай", "найди", "хочу", "мне", "ну", "а",
                       "такое", "же", "в", "том", "духе", "стиле",
                       "подобное", "аналогичное", "есть", "ли", "что-нибудь"}


def _is_continuation(text: str) -> bool:
    """A reference to the current track, not a standalone music query.
    Regex fallback for short phrases made entirely of continuation words —
    the exact list can't cover every variant ("а дай ещё похожее" prod case)."""
    t = re.sub(r"[^\w\sа-яё-]", " ", (text or "").lower(), flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    if t in _CONTINUATION_PHRASES:
        return True
    words = t.split()
    if 1 <= len(words) <= 5 and all(w in _CONTINUATION_WORDS for w in words):
        return True
    return False


_LANG_MARKUP = InlineKeyboardMarkup([[
    InlineKeyboardButton("Русский", callback_data="lang:ru"),
    InlineKeyboardButton("English", callback_data="lang:en"),
]])



# greeting word pool (the language is asked first; greet in that language)
_GREETING_HELLOS = [
    # ONLY clean greetings in different languages (owner: no "Меломан, привет"
    # or other add-ons); the language only picks the text that follows
    "Привет!", "Здравствуй!", "Здравствуйте!", "Добрый день!", "Доброе утро!",
    "Добрый вечер!", "Приветствую!", "Салют!", "Хай!",
    "Hello!", "Hi!", "Hey!", "Hello there!", "Hey there!", "Hiya!",
    "Greetings!", "Welcome!", "Howdy!",
    "Hola!", "Bonjour!", "Ciao!", "Salut!", "Hallo!", "Olá!", "Hej!",
    "Cześć!", "Merhaba!", "Aloha!", "Namaste!", "Salam!", "Jambo!",
]

def _greeting(lang: str) -> str:
    if lang == "en":
        return (head + "\n\n<b>Music Search Mate will find a track from a "
                "scene description</b>, or pick one in the same style but, "
                "say, in another language.\n\n"
                "Examples:\n"
                "• 'energetic like AC/DC'\n"
                "• 'song from that car chase in a Mercedes ad'\n"
                "• 'similar to Moody Blues — Nights in White Satin'\n\n"
                "<b>Or just send a link</b> from YouTube, SoundCloud, "
                "and I'll download it.")
    # Random greeting word in the chosen language + the owner's fixed text;
    # the picture goes in the next message
    import random as _r
    hellos = _GREETING_HELLOS
    head = _r.choice(hellos)
    return (head + "\n\n<b>Music Search Mate найдёт трек по описанию сцены, "
            "где он играл</b>, или подберёт в том же стиле, но, например, "
            "на другом языке.\n\n"
            "Примеры запросов:\n"
            "• «AC/DC, но что-то современное, и на немецком языке»\n"
            "• «песня из погони такси на мерседесах»\n"
            "• «похожее на трек Moody Blues — Nights in White Satin»\n\n"
            "<b>Или просто кинь ссылку</b> из YouTube, SoundCloud, "
            "и я помогу скачать.")


def _mig_nova_payload(context) -> bool:
    """/start arrived with a mig_nova deep-link (the old bot's broadcast)."""
    return any((a or "").startswith("mig_nova")
               for a in (getattr(context, "args", None) or []))


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user  # fix: user was undefined here (NameError for guests)
    if _denied(update, context):
        await update.message.reply_text(await _t(user.id, 'guest'))
        return
    # mig_nova deep-link from the old bot's broadcast — conversion tracking
    # and a smart re-broadcast (excluding those who already came over)
    if _mig_nova_payload(context):
        await insert_event(user_id=user.id, kind="mig_nova")
        log.info("mig_nova arrival user=%s", user.id)
    # Owner decision: ask for the info language on /start, store it per user
    try:
        lang = await get_user_lang(update.effective_user.id)
    except Exception:
        lang = "ru"  # no table / DB down - default; must not block the question
    if lang is None:
        await update.message.reply_text(
            "На каком языке показывать справку о треках?\nWhat language for track info?",
            reply_markup=_LANG_MARKUP)
        return
    from telegram.constants import ParseMode as _PM
    # /start with the project picture (assets/start.png in the image)
    _img = Path(__file__).resolve().parent.parent / "assets" / "start.png"
    if _img.exists():
        with open(_img, "rb") as _ph:
            await update.message.reply_photo(
                photo=_ph, caption=_greeting(lang), parse_mode=_PM.HTML)
        return
    await update.message.reply_text(_greeting(lang), parse_mode=_PM.HTML)


async def handle_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/new — explicitly start a fresh search; past context is not carried over
    (refactor after the "BB King -> stivie wonder" case): clear refine mode,
    the shown-tracks session, and set the "context starts here" marker."""
    if _denied(update, context):
        return
    context.chat_data.pop("refine", None)
    context.chat_data["ctx_after"] = time.time()
    await update.message.reply_text(await _t(update.effective_user.id, "new_reset"))


async def handle_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Change the info language: /lang or the buttons after /start."""
    if _denied(update, context):
        return
    await update.message.reply_text(
        "Язык справки / Info language:", reply_markup=_LANG_MARKUP)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/help — mechanics reference (unlike /start, it explains HOW).
    HTML formatting with bullet points, no dashes, RU/EN by user language."""
    if _denied(update, context):
        return
    from telegram.constants import ParseMode as _PM
    try:
        lang = await get_user_lang(update.effective_user.id) or "ru"
    except Exception:
        lang = "ru"
    if lang == "en":
        text = (
            "<b>How to search</b>\n"
            "• Describe the track in words: \"energetic like AC/DC\", "
            "\"song from that ad, female vocal\"\n"
            "• Or send a link: YouTube, SoundCloud, Bandcamp\n\n"
            "<b>Buttons under the track</b>\n"
            "• \"То\" or \"Не то\": hit or the next option\n"
            "• \"Уточнить\": refine the request in words\n"
            "• \"Похожее\": similar tracks\n"
            "• 💡 \"Факт\": a fact about the track\n\n"
            "Want a live version, a cover or a slowed edit? "
            "Just say so in the request.")
    else:
        text = (
            "<b>Как искать</b>\n"
            "• Опиши трек словами: «бодрое как AC/DC», "
            "«песня из рекламы, поёт девушка»\n"
            "• Или кинь ссылку: YouTube, SoundCloud, Bandcamp\n\n"
            "<b>Кнопки под треком</b>\n"
            "• «То» или «Не то»: попадание или следующий вариант\n"
            "• «Уточнить»: правка запроса словами\n"
            "• «Похожее»: соседние треки от выданного\n"
            "• 💡 «Факт»: интересный факт о треке\n\n"
            "Нужна живая версия, кавер или slowed? "
            "Просто напиши это в запросе: найду именно её.")
    await update.message.reply_text(text, parse_mode=_PM.HTML)


# EN pool of search statuses (the RU pool is _START_PHRASES in llm_search)
_START_PHRASES_EN = [
    "Studying the request…",
    "Checking the catalog…",
    "Cross-checking sources…",
    "Searching by description…",
    "Collecting options…",
    "Looking through discographies…",
    "Filtering finds…",
    "Nailing down details…",
    "Checking releases…",
    "Looking for matches…",
    "Matching descriptions…",
    "Opening the reference…",
]


async def _status_text(user_id, kind: str = "pool") -> str:
    """Search status in the user's language. kind=pool -> a random phrase from the pool."""
    if kind == "pool":
        lang = await _lang_of(user_id)
        phrases = _START_PHRASES_EN if lang == "en" else START_PHRASES
        return f"🎧 {random.choice(phrases)}"
    return await _t(user_id, kind)


async def _log_outcome(user_id, desc, outcome: str, latency_ms: int | None = None) -> None:
    """Refusal outcome into the funnel (silent) — backfills points missed earlier."""
    try:
        await insert_query(user_id=user_id, correlation_id=uuid4().hex,
                           description=desc, intent=None, signals=None, queries=None,
                           candidates=None, best_video_id=None, best_title=None,
                           best_score=None, confidence=None, match_score=None,
                           outcome=outcome, latency_ms=latency_ms)
    except Exception:
        pass


def _progress_bar(pct: float) -> str:
    """Download animation with a fill bar."""
    filled = max(0, min(10, int(pct // 10)))
    return "\u2588" * filled + "\u2591" * (10 - filled)


# Groups: privacy mode OFF hands the bot EVERY group message — without a gate
# the bot would answer every little thing members say. Addressed = an @mention
# in the text/caption OR a reply to the bot's own message. In DMs the gate is transparent.
_GROUP_BOT_MENTION_RE = re.compile(r"@(?:music_search_mate_bot|music_search_dev_bot)\b")


def _addressed_to_bot(update: Update, context) -> bool:
    """True = the message is addressed to the bot (in groups); always True in DMs."""
    chat = update.effective_chat
    if chat is None or chat.type not in ("group", "supergroup"):
        return True
    msg = update.message
    text = (msg.text or msg.caption or "") if msg else ""
    # A reply to the bot's message no longer invokes the bot (user preference) —
    # in groups only an explicit @mention does
    return bool(_GROUP_BOT_MENTION_RE.search(text))


async def handle_unsupported(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Voice/photo/sticker/files — silence before, an honest reply now.
    Sits behind anti-flood — a flood of files no longer generates a flood of
    replies. In groups — only when addressed (@mention/reply to the bot)."""
    if not _addressed_to_bot(update, context):
        return
    user = update.effective_user
    if _flood_hit(context, user.id if user else 0):
        await _flood_warn(update, context, user.id if user else 0)
        return
    if _denied(update, context):
        return
    await update.message.reply_text(await _t(user.id, "unsupported"))


# Ingress anti-flood. Sliding window in bot_data (memory — reset by a restart,
# fine for anti-flood): shields against storms of any message type BEFORE
# the pipeline and BEFORE replying (file handler and guests included).
_FLOOD_WINDOW = 30   # sec
_FLOOD_MAX = 8       # messages per window
_FLOOD_WARN_EVERY = 60  # warn at most once a minute


def _flood_hit(context, user_id) -> bool:
    """True = message over the window limit (flood)."""
    uid = user_id or 0
    now = time.time()
    hist = context.bot_data.setdefault("flood", {}).setdefault(uid, [])
    hist[:] = [t for t in hist if now - t < _FLOOD_WINDOW]
    hist.append(now)
    return len(hist) > _FLOOD_MAX


async def _flood_warn(update, context, user_id) -> None:
    """One warning per window — don't answer spam with spam."""
    uid = user_id or 0
    now = time.time()
    warns = context.bot_data.setdefault("flood_warn", {})
    if now - warns.get(uid, 0) > _FLOOD_WARN_EVERY:
        warns[uid] = now
        try:
            await update.effective_chat.send_message(await _t(uid, "flood"))
        except Exception:
            pass


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # In groups react only when addressed (@mention of / reply to the bot) —
    # the gate sits BEFORE anti-flood so group chatter doesn't eat the user's window.
    if not _addressed_to_bot(update, context):
        return
    user = update.effective_user
    if _flood_hit(context, user.id if user else 0):
        await _flood_warn(update, context, user.id if user else 0)
        return
    if _denied(update, context):
        await update.message.reply_text(await _t(user.id, "guest"))
        return
    desc = (update.message.text or "").strip()
    # Groups: a reply/mention arrives with the @username in the text — strip it,
    # or the bot's name leaks into the search query
    desc = re.sub(r"@(?:music_search_mate_bot|music_search_dev_bot)\b", "", desc).strip()
    if not desc:
        return
    if len(desc) > 500:
        # Don't go silent on long input — ask the user to rephrase.
        await update.message.reply_text(await _t(user.id, "too_long"))
        return

    # Moderation layer 1: destructive queries are cut BEFORE the pipeline (no
    # LLM/yt calls spent). The reply is short, no lecturing; subtle cases are
    # caught by layer 2 (the block field in LLM#1) inside _do_search.
    _dcat = destructive_prefilter(desc)
    if _dcat:
        log.warning("destructive query blocked (regex) cat=%s user=%s",
                    _dcat, user.id if user else 0)
        await _log_outcome(user.id, desc, "blocked")
        await update.message.reply_text(await _t(user.id, "destructive"))
        return

    # per-chat lock: two back-to-back messages in one chat must not start parallel
    # pipelines — otherwise a race on chat_data / status / q:{id} / rejected_ids.
    # The second message waits for the first to finish.
    async with _chat_lock(update.effective_chat.id):
        # refinement intercept: within the TTL window after "Уточнить" any text counts as
        # an edit of the previous query. The state is one-shot, but we pop() it only when
        # actually entering refinement: rate-limit is checked BEFORE extraction, otherwise
        # a refusal silently dropped the refinement text and rejected_ids (review 2026-09-03).
        refine = context.chat_data.get("refine")
        refine_armed = bool(refine) and (time.monotonic() - refine.get("ts", 0.0)) <= _REFINE_TTL
        if refine and not refine_armed:
            context.chat_data.pop("refine", None)  # TTL expired - clean it out
            refine = None

        cfg = context.bot_data["cfg"]
        media = context.bot_data["media"]

        n = await count_recent(user.id, hours=1)
        if n >= cfg.user_rate_limit_per_hour:
            await _log_outcome(user.id, desc, "limit")
            await update.message.reply_text(await _t(user.id, "hourly"))
            return

        if refine_armed and not _looks_like_url(desc):
            # A link inside the refine window = a NEW query, not an edit of the old one
            context.chat_data.pop("refine", None)
            await _handle_refine(update, context, desc, refine)
            return

        # periodic temp cleanup (off the event loop)
        context.bot_data["req_counter"] = context.bot_data.get("req_counter", 0) + 1
        if context.bot_data["req_counter"] % _CLEANUP_EVERY == 0:
            await asyncio.to_thread(cleanup_temp, cfg.temp_ttl_minutes)
            # PII retention + expired-facts purge, once a day
            _lp = context.bot_data.get("last_purge") or 0
            if time.time() - _lp > 86400:
                context.bot_data["last_purge"] = time.time()
                try:
                    from services.store import purge_old_data
                    purged = await purge_old_data(cfg.data_retention_days)
                    log.info("retention purge: %s", purged)
                except Exception:
                    log.exception("retention purge failed")
                # monitor the tgapi directory (local Bot API): growth is a signal
                try:
                    _tg = sum(
                        os.path.getsize(os.path.join(r, f))
                        for r, _, fs in os.walk("/app/data/tgapi") for f in fs)
                    if _tg > 512 * 1024 * 1024:
                        log.warning("tgapi dir %.0f MB, time to investigate", _tg / 1e6)
                except OSError:
                    pass

        # URL fast-path: link -> straight to yt-dlp, bypassing the LLM.
        # SSRF filter (platform allowlist) runs before yt-dlp.
        url = _looks_like_url(desc)
        if url:
            if not _url_host_allowed(url, cfg.url_allowed_hosts):
                await _log_outcome(user.id, desc, "ssrf")
                await update.message.reply_text(await _t(user.id, "host_not_allowed"))
                return
            await _handle_url(update, context, media, user, url, desc)
            return
        # Malformed link (space/typo) — don't search its remains as text.
        if _looks_malformed_url(desc):
            try:  # log the outcome to the funnel
                await insert_query(user_id=user.id, correlation_id=uuid4().hex,
                                   description=desc, intent=None, signals=None, queries=None,
                                   candidates=None, best_video_id=None, best_title=None,
                                   best_score=None, confidence=None, match_score=None,
                                   outcome="bad_url")
            except Exception:
                pass
            await update.message.reply_text(await _t(user.id, "malformed_url"))
            return

        # A bare "ещё такое же" request with a live context refers to the current track.
        # Turn it into "similar to <track> — <artist>" and run the regular pipeline
        # (LLM#1 + reference ban). Without a session "ещё" falls into a plain search and clarify.
        if _is_continuation(desc):
            sess = await get_session(user.id)
            if sess and sess.get("track"):
                artist = sess.get("artist") or sess["track"]
                log.info("session continuation user=%s -> similar to %s", user.id, sess["track"])
                desc = f"похожие на {sess['track']} - {artist}"

        # A lively status from the first second — a random phrase from the
        # audiophile pool (llm_search._START_PHRASES). Before, the phrase flashed between
        # "Понимаю запрос…" and the intro (a duplicated start from the ux-review) —
        # the user couldn't catch it.
        status = await update.message.reply_text(await _status_text(update.effective_user.id))
        out = await _do_search(update, context, status, desc)
        if out is None:
            return
        await _finish_search(update, context, status, out, desc, user)


def _wtp_markup(lang: str = "ru") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(f"⭐ {p}", callback_data=f"wtp:{p}") for p in WTP_PRICES]] +
        [[InlineKeyboardButton(_kbt(lang, "wtp_no"), callback_data="wtp:0"),
          InlineKeyboardButton(_kbt(lang, "wtp_optout"), callback_data="wtp:optout")]]
    )


# Daily-limit reset date in refusal captions. Usage counters are keyed by the
# container's UTC date (incr_usage: datetime.date.today()); reset = midnight UTC
# = 03:00 MSK. A bare "tomorrow" confuses a user who is already on their "tomorrow".
_MONTHS_EN = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTHS_RU = ("января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря")


def _daily_reset_str(lang: str = "ru") -> str:
    import datetime as _dt
    d = _dt.date.today() + _dt.timedelta(days=1)
    if lang == "en":
        return f"{_MONTHS_EN[d.month - 1]} {d.day}"
    return f"{d.day} {_MONTHS_RU[d.month - 1]}"


async def _check_limits(update, context, user, kind: str) -> tuple[bool, str]:
    """Unified limit gate. kind='search' (LLM pipeline) | 'url' (no LLM).
    Returns (allowed, reason). Daily limits + global budget; the hourly
    anti-flood is separate — user_rate_limit_per_hour via count_recent in
    callers. Shows the WTP survey on the first limit hit (anti-spam in store)."""
    cfg = context.bot_data["cfg"]
    usage = await get_usage(user.id)
    glob = await get_global_usage()

    if kind == "search":
        # hourly search/url_hourly_limit were cut — they were configured but never
        # enforced (the real anti-flood is count_recent earlier in the pipeline)
        daily = usage.get("searches", 0)
        d_limit = cfg.search_daily_limit
        g_limit = cfg.global_daily_search_budget
        g_used = glob.get("searches", 0)
    else:
        daily = usage.get("urls", 0)
        d_limit = cfg.url_daily_limit
        g_limit = 0  # URLs don't count against the global LLM budget
        g_used = 0

    # daily limit (the "3/2" counters were removed from the text — confusing;
    # "tomorrow" was replaced with the exact reset date)
    if daily >= d_limit:
        await _maybe_wtp(update, context, user, "day", kind, daily)
        return False, await _t(user.id, "day_limit", date=_daily_reset_str(await _lang_of(user.id)))

    # global budget (search only)
    if kind == "search" and g_used >= g_limit:
        await _maybe_wtp(update, context, user, "global", kind, daily)
        return False, await _t(user.id, "global_budget", date=_daily_reset_str(await _lang_of(user.id)))

    return True, ""


async def _sends_over(update, context, user) -> bool:
    """Daily cap on audio sends. "Не то"/"Похожее" chains bypass the search/URL
    limits — every send costs a yt-dlp run plus megabytes of upload; the cap
    closes the abuse vector (fb:m is only capped at +3 per query)."""
    cfg = context.bot_data["cfg"]
    try:
        usage = await get_usage(user.id)
    except Exception:
        return False
    n = usage.get("sends") or 0
    if n < cfg.sends_daily_limit:
        return False
    await _maybe_wtp(update, context, user, "sends", "sends", n)
    await q_reply_sends_over(update)
    return True


async def q_reply_sends_over(update) -> None:
    try:
        await update.effective_chat.send_message(
            await _t(update.effective_user.id, "sends_over"))
    except Exception:
        pass


async def _maybe_wtp(update, context, user, trigger: str, feature: str, usage_n: int) -> None:
    """WTP survey on hitting a limit (at most once per 7 days, cap 3, optout is forever)."""
    if not await should_show_survey(user.id):
        return
    await save_survey_shown(user.id, trigger, feature, usage_n)
    # _WTP_MARKUP is already built at module level (:388); a separate import from
    # llm_search was an ImportError-P0 (the name doesn't exist there) — it crashed
    # _check_limits on the very first limit hit
    try:
        # Owner's wording: an honest frame instead of "помоги определиться" —
        # enthusiasm, no ads, growing load; paid content is video only (GB of traffic)
        await update.effective_chat.send_message(
            await _t(user.id, "wtp_survey"), reply_markup=_wtp_markup(await _lang_of(user.id)))
    except Exception:
        pass


async def _do_search(update, context, status, desc, excluded_ids=None):
    """Streaming status + search_candidates. Returns SearchOut or None (error already shown).

    excluded_ids — rejected video_ids (refinement), passed to search_candidates for
    deterministic exclusion before ranking. Shared path for new searches and refinements."""
    bothub = context.bot_data["bothub"]
    _t_search0 = time.monotonic()  # query latency for the queries table
    # /new cuts off history - take only queries after the marker
    ctx_after = context.chat_data.get("ctx_after", 0.0)
    ctx_history = await recent_context(update.effective_user.id)
    if ctx_after:
        ctx_history = [c for c in ctx_history if c.get("ts", 0) > ctx_after]

    # streaming: on_progress edits the status per stage. prog keeps the last text
    # and timestamp — in pauses between stages a pulse animates the status with dots
    # so a long ytsearch/LLM call doesn't look like a hang.
    prog = {"intro": "", "last": 0.0, "text": "", "alive": True}

    async def on_progress(stage: str, text: str):
        if stage == "intro":
            prog["intro"] = text
        if stage in ("start", "intro"):
            full = text
        else:
            intro = prog["intro"]
            full = f"{intro}\n\n{text}" if intro else text
        prog["text"] = full
        # edit throttle: at most once per 1.2s (except the first stage)
        now = time.monotonic()
        if stage != "start" and now - prog["last"] < 1.2:
            return
        prog["last"] = now
        try:
            await status.edit_text(full)
        except Exception as e:
            if "retry" in str(e).lower() or "RetryAfter" in type(e).__name__:
                await asyncio.sleep(1.0)  # Telegram Flood control
            # MessageNotModified / throttle — ignored

    # pulse: if a stage hasn't updated for >2s (long ytsearch/LLM#2), append dots
    # to the last text — a live signal that the bot is working, not stuck.
    async def _pulse():
        dots = 0
        while prog["alive"]:
            await asyncio.sleep(2.2)
            if not prog["alive"] or not prog["text"]:
                continue
            if time.monotonic() - prog["last"] >= 2.0:
                dots = (dots + 1) % 4
                try:
                    await status.edit_text(prog["text"] + "·" * dots)
                except Exception:
                    pass

    pulse = asyncio.create_task(_pulse())
    # Telegram-side "bot is alive" indicator (typing dots in the chat) while the
    # search runs: send_chat_action lives ~5s — send in a loop every 4.5s.
    async def _action_loop():
        try:
            while True:
                try:
                    await context.bot.send_chat_action(
                        chat_id=update.effective_chat.id, action="typing")
                except Exception:
                    pass
                await asyncio.sleep(4.5)
        except asyncio.CancelledError:
            pass

    actions = asyncio.create_task(_action_loop())
    # limit gate before any network call
    allowed, reason = await _check_limits(update, context, update.effective_user, "search")
    if not allowed:
        await _log_outcome(update.effective_user.id, desc, "limit",
                           latency_ms=int((time.monotonic() - _t_search0) * 1000))
        # without cancel both indicator tasks lived forever
        # (typing every 4.5s indefinitely, stacking on every refusal)
        prog["alive"] = False
        pulse.cancel()
        actions.cancel()
        await status.edit_text(reason)
        return None

    # v1.10: one search = one Langfuse trace. trace_id = correlation_id (uuid4.hex),
    # the same value goes to insert_query — the "log <-> DB <-> trace" link. Without
    # keys SearchTrace is a no-op; the pipeline never notices.
    trace = SearchTrace(uuid4().hex, update.effective_user.id,
                        update.effective_chat.id if update.effective_chat else None, desc)
    try:
        out = await search_candidates(desc, bothub, context=ctx_history,
                                      excluded_ids=excluded_ids, on_progress=on_progress,
                                      metadata_svc=context.bot_data.get("metadata"),
                                      trace=trace)
    except Exception:
        prog["alive"] = False
        pulse.cancel()
        actions.cancel()
        log.exception("search_candidates failed")
        trace.finish(output={"error": "search_candidates raised"})
        try:
            await insert_query(user_id=update.effective_user.id,
                               correlation_id=trace.trace_id, description=desc,
                               intent=None, signals=None, queries=None, candidates=None,
                               best_video_id=None, best_title=None, best_score=None,
                               confidence=None, match_score=None, outcome="error",
                               latency_ms=int((time.monotonic() - _t_search0) * 1000))
        except Exception:
            pass
        await status.edit_text(await _t(update.effective_user.id, "not_found"))
        return None
    prog["alive"] = False
    pulse.cancel()
    actions.cancel()
    # Moderation layer 2: LLM#1 classified the query as destructive —
    # the same short refusal; YouTube is never touched.
    if getattr(out, "block", ""):
        log.warning("destructive query blocked (llm) cat=%s user=%s",
                    out.block, update.effective_user.id if update.effective_user else 0)
        try:
            await insert_query(user_id=update.effective_user.id,
                               correlation_id=(getattr(out.trace, "trace_id", "") or uuid4().hex),
                               description=desc, intent=out.intent, signals=None, queries=None,
                               candidates=None, best_video_id=None, best_title=None,
                               best_score=None, confidence=None, match_score=None,
                               outcome="blocked",
                               latency_ms=int((time.monotonic() - _t_search0) * 1000))
        except Exception:
            pass
        await status.edit_text(await _t(update.effective_user.id, "destructive"))
        return None
    out.latency_ms = int((time.monotonic() - _t_search0) * 1000)
    return out


async def _finish_search(update, context, status, out, desc, user):
    """SearchOut post-processing: insert_query + session + clarify/auto-download/preview.
    Shared by new searches and refinement - they differ only in desc and excluded_ids (in _do_search)."""
    media = context.bot_data["media"]

    if not out.candidates and not (out.need_clarify and out.clarify_question):
        head = (out.intro + "\n\n") if out.intro else ""
        await status.edit_text(head + await _t(update.effective_user.id, "nothing_found"))
        await _log_outcome(update.effective_user.id, desc, "empty",
                           latency_ms=getattr(out, "latency_ms", None))
        if getattr(out, "trace", None):
            out.trace.finish(output={"decision": "empty"})
        return

    # auto-download is decided by the rule-based match-score, not LLM confidence.
    # should_auto_download: from_source is always preview (1/8 hit); this_track at
    # score>=0.65, similar at >=0.70. Thresholds per intent — this_track is softer
    # (exact comparison target + both+fid bonus), similar is stricter (we trust the LLM rank).
    best = out.candidates[out.best_index] if 0 <= out.best_index < len(out.candidates) else None
    # Even past the auto-download threshold, an ambiguous choice (a close second
    # candidate or low confidence) shows the top-3 instead of downloading one version.
    ambiguous = is_ambiguous(out.candidates, out.best_index, out.confidence)
    auto = bool(best) and should_auto_download(out.intent, best.match_score) and not ambiguous

    # Persist match_score and the auto/preview decision to queries so Store logs can
    # measure hit-rate per threshold (which match_score actually earns a "То").
    # Since v1.10 correlation_id is the Langfuse trace_id (uuid4.hex); before v1.10 it was str(user.id).
    trace = getattr(out, "trace", None)
    # query outcome for the funnel (ok/clarify/empty) + latency from _do_search
    _outcome = ("clarify" if (out.need_clarify and out.clarify_question)
                else ("empty" if not out.candidates else "ok"))
    query_id = await insert_query(
        user_id=user.id, correlation_id=(trace.trace_id if trace else str(user.id)),
        description=desc, intent=out.intent,
        signals=out.signals, queries=out.queries,
        candidates=[c.__dict__ for c in out.candidates[:10]],
        best_video_id=(out.candidates[out.best_index].video_id if 0 <= out.best_index < len(out.candidates) else None),
        best_title=(out.candidates[out.best_index].title if 0 <= out.best_index < len(out.candidates) else None),
        best_score=(out.candidates[out.best_index].score if 0 <= out.best_index < len(out.candidates) else None),
        confidence=out.confidence,
        match_score=(best.match_score if best else None),
        auto_download=(auto if best else None),
        ambiguous=ambiguous,
        outcome=_outcome,
        latency_ms=getattr(out, "latency_ms", None),
    )
    # desc and rejected_ids are needed for refinement: desc is the original query to
    # merge with the edit; rejected_ids accumulate on "Не то" and feed the next search's exclude.
    # State goes to chat_data + SQLite (survives restarts), not memory alone.
    await _save_state(context, query_id, {
        "candidates": out.candidates, "confidence": out.confidence,
        "best_index": out.best_index, "verdict": out.verdict, "intro": out.intro,
        "desc": desc, "rejected_ids": [],
        # for the "Похожее" cache (same reference) and refinement reranking
        "intent": out.intent, "reference": out.reference, "ts": time.time(),
    })
    context.chat_data["last_qid"] = query_id
    # usage accounting (LLM search)
    await incr_usage(user.id, "searches")

    # branch 1: the model asks for a clarification (query too vague)
    if out.need_clarify and out.clarify_question:
        head = (out.intro + "\n\n") if out.intro else ""
        await status.edit_text(head + "🤔 " + out.clarify_question)
        if trace:
            trace.finish(output={"decision": "clarify", "question": out.clarify_question})
        return

    log.info("decide user=%s intent=%s score=%.2f amb=%s tid=%s",
             user.id, out.intent, best.match_score if best else 0.0, ambiguous,
             trace.trace_id[:8] if trace else "-")
    # Owner decision 2026-11-01: ONE result delivered immediately with the track
    # downloaded; the send is always "card (cover + facts) -> tagged audio".
    # Preview/carousel was removed from the main path: quality is worked through via
    # buttons on the audio ("Не то" = the next candidate via the miss chain, "Уточнить", "Похожее").
    sp_decide = trace.span("decide", input={"intent": out.intent,
                                            "match_score": best.match_score if best else None}) if trace else None
    # Designed with the owner: the card is ONE audio message with the cover in the
    # player (thumbnail + ID3) and a structured caption; a separate photo card is
    # not sent (complaint: "сперва картинка, потом песня - должно быть одним").
    if sp_decide:
        sp_decide.finish(output={"decision": "single+download"})
    if trace:
        trace.finish(output={"decision": "single+download", "best": best.title if best else None})
    if best:
        # download fallback: on error (403, etc.) try the next candidate, up to 2 attempts
        for offset in range(min(2, len(out.candidates))):
            idx = out.best_index + offset
            if idx >= len(out.candidates):
                break
            track = out.candidates[idx]
            ok = await _send_track(update, context, status, media, query_id, idx,
                                   out.candidates, track, out.confidence,
                                   out.verdict if idx == out.best_index else track.rationale)
            if ok:
                return
        # all attempts failed
        if status:
            await status.edit_text(await _t(update.effective_user.id, "dl_none"))
        return
    if status:
        await status.edit_text(await _t(update.effective_user.id, "nothing_found"))


async def _try_sim_from_pool(update, context, user, status, media,
                              query_id, store, cur, artist) -> bool:
    """A repeated "Похожее" for the same reference -> the next unseen candidate
    from the saved pool, without LLM or ytsearch (2-3s instead of 25-38s).
    False = cache miss; the caller runs a full search."""
    last_qid = context.chat_data.get("last_qid")
    if not last_qid or last_qid != query_id:
        return False
    ts = store.get("ts") or 0
    if time.time() - ts > _SIM_REUSE_WINDOW:
        return False
    prev_ref = (store.get("reference") or {}).get("artist") or ""
    if not prev_ref or _ref_key(prev_ref) != _ref_key(artist):
        return False  # "Похожее" for a different track - a real search is needed
    sess = await get_session(user.id)
    shown = set((sess or {}).get("shown_ids") or [])
    shown.add(cur.video_id)
    cands = [c for c in store.get("candidates", [])
             if c.video_id and c.video_id not in shown]
    if not cands:
        return False  # pool exhausted - a new neighbor takes a full search
    nxt = cands[0]
    log.info("sim cache hit: ref=%r -> next from pool %r (shown=%d)",
             artist, nxt.title[:40], len(shown))
    # rank must be nxt's index in store["candidates"] — with rank=0
    # the fb/sim/dlv buttons addressed candidates[0] (a different track)
    try:
        nxt_rank = store["candidates"].index(nxt)
    except ValueError:
        nxt_rank = 0
    ok = await _send_track(update, context, status, media, last_qid, nxt_rank,
                           store["candidates"], nxt, store.get("confidence", "low"),
                           nxt.rationale or "Из тех же поисков, следующий по похожести.")
    if ok:
        pass  # the cache path carries no Langfuse trace: with no LLM stages there is nothing to trace
    return ok


async def _handle_refine(update, context, comment, refine):
    """Refine the previous query: merge the original desc + the comment,
    exclude rejected video_ids, run a new search. Rate-limit is already checked
    by the caller (handle_message) BEFORE the refine state is extracted — no need
    to duplicate it here: the state is only taken once the limit passes (review 2026-09-03)."""
    user = update.effective_user
    original_desc = (refine.get("desc") or "").strip()
    excluded = set(refine.get("rejected_ids") or set())
    # Exclude the shown cache (session shown_ids) together with the explicitly rejected —
    # "a different variant" must not return already delivered tracks, including ones from earlier refinement rounds.
    sess = await get_session(user.id)
    if sess:
        excluded |= set(sess.get("shown_ids") or [])
    comment = (comment or "").strip()
    if not comment:
        await update.message.reply_text(await _t(update.effective_user.id, "empty_feedback"))
        return
    # Structure it: explicit ORIGINAL_QUERY / REFINEMENT markers, the edit wins.
    # Otherwise LLM#1 may read the two parts as two contradictory queries (flagship cross-check).
    combined = (f"ORIGINAL_QUERY: {original_desc}\n\n"
                f"REFINEMENT: {comment}\n\n"
                f"(REFINEMENT корректирует ORIGINAL_QUERY; при конфликте приоритет за REFINEMENT)")
    status = await update.message.reply_text(await _status_text(update.effective_user.id, "st_refine"))

    # A constraining refinement ("no vocals", "heavier") is almost always about the same
    # candidate pool -> rerank without new fetching (~12s instead of ~24s, no LLM#1+ytsearch).
    # "Something entirely different" in the user's words = a full new search.
    pool = refine.get("candidates") or []
    if (pool and len(excluded) < len(pool)
            and not any(w in comment.lower() for w in _REUSE_BLOCKERS)):
        bothub = context.bot_data["bothub"]
        rtrace = SearchTrace(uuid4().hex, user.id,
                             update.effective_chat.id if update.effective_chat else None,
                             combined)
        out = await rerank_pool(
            combined, bothub, pool, excluded_ids=excluded,
            reference=refine.get("reference"), intent=refine.get("intent"),
            trace=rtrace)
        if out is not None:
            await _finish_search(update, context, status, out, combined, user)
            return

    out = await _do_search(update, context, status, combined, excluded_ids=excluded)
    if out is None:
        # The search failed, but refine was already popped in handle_message. Restore the
        # state (without comment) so the user can retry the refinement instead of retyping (flagship cross-check).
        refine.update({"ts": time.monotonic()})
        context.chat_data["refine"] = refine
        return
    await _finish_search(update, context, status, out, combined, user)


async def _handle_url(update, context, media, user, url, desc):
    # URL limit gate
    allowed, reason = await _check_limits(update, context, user, "url")
    if not allowed:
        await _log_outcome(user.id, desc, "limit")
        await update.message.reply_text(reason)
        return
    status = await update.message.reply_text(await _status_text(update.effective_user.id, "st_url"))
    try:
        meta = await media.extract_url_meta(url)
    except Exception:
        log.exception("extract_url_meta failed: %s", url)
        # Don't creatively interpret a broken URL as from_source. If the message has
        # meaningful words besides the link — search those; otherwise clarify.
        rest = _URL_RE.sub("", desc).strip()
        if len(rest) < 3:
            await status.edit_text(await _t(update.effective_user.id, "link_read_hint"))
            return
        await status.edit_text(await _t(update.effective_user.id, "link_read_desc"))
        out = await _do_search(update, context, status, rest)
        if out is not None:
            await _finish_search(update, context, status, out, rest, user)
        return

    if not meta:
        await status.edit_text(await _t(update.effective_user.id, "link_read_fail"))
        return

    vid = meta.get("video_id") or meta.get("id") or ""
    title = meta.get("title") or url
    dur = meta.get("duration") or 0

    # Owner's variant A: URL -> Audio/Video buttons
    # (like nova, familiar to migrants). Audio = m4a as now, Video = quality picker.
    _lang = await _lang_of(update.effective_user.id)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(_kbt(_lang, "audio"), callback_data=f"dl:aud:{vid}"),
         InlineKeyboardButton(_kbt(_lang, "video"), callback_data=f"dl:vid:{vid}")],
    ])
    dur_str = f"{dur // 60}:{dur % 60:02d}" if dur else "?"
    await status.edit_text(
        f"🔗 {title[:200]}\n⏱ {dur_str}",
        reply_markup=kb)
    # state for callback handling
    context.chat_data[f"url:{vid}"] = {"url": url, "title": title}


def _button_label(title: str, channel: str) -> str:
    """Pretty "Artist - Track" label for top-3 buttons and statuses. The
    YouTube title is cleaned up and rebuilt: the channel (minus "- Topic")
    confirms the main artist on the left ("Akhenaton, La Cosca Scratch
    by DJ Pone - Ultime Poursuite" -> "Akhenaton - Ultime Poursuite"), and a
    title missing the artist gets it from the channel ("Long Tall Sally" +
    Little Richard). Channel not confirmed -> show as is: a wrong swap is
    worse than the clutter."""

    t = pretty_title(title).strip()
    ch = pretty_title((channel or "").removesuffix(" - Topic").strip())
    if " - " in t:
        left, right = (x.strip() for x in t.split(" - ", 1))
        if ch and 3 <= len(ch) <= len(left) and ch.lower() in left.lower():
            left = ch  # main artist confirmed by the channel
        t = f"{left} - {right}"
    elif ch and ch.lower() not in t.lower():
        t = f"{ch} - {t}"
    return (t[:40] + "…") if len(t) > 40 else t


def _visible_candidates(candidates):
    """Top-3 preview: candidates with a nonzero score (explicitly rated relevant
    by LLM#2). If all scores are 0.0 (the LLM failed) — show them all. Returns
    [(rank, candidate)] where rank is the index in the full candidates list
    (for the pk callback), not the position in the preview."""
    scored = [(i, c) for i, c in enumerate(candidates) if c.score > 0.0]
    return (scored or list(enumerate(candidates)))[:3]


def _render_preview(query_id, candidates, verdict="", intro="", lang="ru"):
    """Text + keyboard of the top-3 preview. Shared by the initial send and by
    RESTORING the preview after the chosen download failed (before, the buttons
    were burned and the user got stuck)."""
    top = _visible_candidates(candidates)
    lines = [(intro + "\n\n") if intro else ""]
    lines.append("Вот что нашёл. Какой качаем?")
    for n, (_, c) in enumerate(top):
        pt = _button_label(c.title, c.channel)
        lines.append(f"{n + 1}. {pt}")
    if verdict:
        lines.append(f"\n{verdict}")
    rows = []
    for rank, c in top:
        label = _button_label(c.title, c.channel)
        rows.append([InlineKeyboardButton(label, callback_data=f"pk:{query_id}:{rank}")])
    rows.append([InlineKeyboardButton(_kbt(lang, "refine_query"), callback_data=f"rn:{query_id}")])
    return "\n".join(lines), InlineKeyboardMarkup(rows)


# The zombie carousel chain was removed (_card_text/_carousel_markup/
# _send_card/_show_top3 + the cr: callback below) — the path is single+download,
# no new cr: buttons are created. Living neighbors: _render_preview/
# _visible_candidates (pk fallback) are untouched.


async def _remember_session_and_ref(context, update, query_id, track, artist, track_name):
    """Canonical artist/track for the MLT reference + the session.

    Single point for the download path and the cache-hit path (there meta is
    unavailable, but canonical fields arrive from file_cache). "Похожее" takes
    the clean reference (not raw title/channel); "ещё такое же" on the next
    message resolves through the session via get_session."""
    last_query = ""
    if query_id:
        st = await _load_state(context, query_id)
        if st is not None:
            st["resolved_artist"] = artist
            st["resolved_track"] = track_name
            await _save_state(context, query_id, st)
            last_query = st.get("desc", "")
    try:
        await upsert_session(
            user_id=update.effective_user.id,
            artist=artist, track=track_name,
            video_id=track.video_id, last_query=last_query,
        )
    except Exception:
        log.exception("upsert_session failed user=%s", update.effective_user.id)


async def _is_degraded(context) -> bool:
    """Degraded mode — the global budget is exhausted.
    Search still works; a link is sent instead of the file. Graceful when cfg is absent."""
    try:
        cfg = context.bot_data.get("cfg")
        if not cfg:
            return False
        glob = await get_global_usage()
        return glob.get("searches", 0) >= cfg.global_daily_search_budget
    except Exception:
        return False


async def _send_track(update, context, status, media, query_id, rank, candidates, track, confidence, verdict="", keyboard="feedback"):
    metadata_svc = context.bot_data["metadata"]
    kb = _reply_keyboard(query_id, rank, keyboard, vid=track.video_id,
                               lang=await _lang_of(update.effective_user.id))

    # Telegram draws an upload indicator while we download/tag/send
    async def _upload_action():
        try:
            while True:
                try:
                    await context.bot.send_chat_action(
                        chat_id=update.effective_chat.id, action="upload_document")
                except Exception:
                    pass
                await asyncio.sleep(4.5)
        except asyncio.CancelledError:
            pass

    _ua = asyncio.create_task(_upload_action())

    # tg_file_id cache keyed by video_id. The track was sent before — send by file_id,
    # no repeated download (yt-dlp/ffmpeg) or re-upload. Canonical metadata sits next to the file_id.
    if track.video_id:
        try:
            cached = await get_cached_file(track.video_id)
        except Exception:
            log.exception("get_cached_file failed %s", track.video_id)
            cached = None
        if cached and cached.get("file_id"):
            cached_meta = MetaOut(
                artist=cached.get("artist") or track.channel,
                track=cached.get("track") or track.title,
                album=cached.get("album") or "",
                year=cached.get("year") or "",
                genre=cached.get("genre") or "",
                bio=cached.get("bio") or "",
                bio_about="artist" if cached.get("bio") else "",
            )
            # Auto-facts were removed — the sonar fact is button-only (💡;
            # $0.008 a call; 30d cache). Sonar stays ONLY as a fallback for an empty bio.
            if not _clean_bio(cached_meta.bio):
                cached_meta.bio = ""
                cached_meta.bio = await _sonar_fact(
                    context, update.effective_user.id,
                    cached_meta.artist, cached_meta.track, cached_meta.year)
            elif cached_meta.bio:
                cached_meta.bio = await _translate_bio(
                    context, update.effective_user.id, cached_meta.bio)
            # photo card before the audio (a thumbnail doesn't render)
            # album cover only (iTunes), no YouTube screenshots
            _aw = cached.get("artwork_url") or ""
            if not _aw and cached_meta.artist and cached_meta.track:
                # Old cache rows (from before artwork_url existed) lack the URL —
                # refetch the cover by the canonical pair and persist it to the cache
                # so the self-heal doesn't repeat on every hit
                _aw = await metadata_svc.refetch_artwork(
                    cached_meta.artist, cached_meta.track, track.video_id)
                if _aw:
                    try:
                        await update_cached_artwork(track.video_id, _aw)
                    except Exception:
                        log.exception("update_cached_artwork failed %s", track.video_id)
            if _aw:
                try:
                    await update.effective_chat.send_photo(photo=_aw)
                except Exception:
                    pass
            try:
                await context.bot.send_audio(
                    chat_id=update.effective_chat.id,
                    audio=cached["file_id"],
                    caption=_caption(cached_meta, track, confidence, verdict),
                    parse_mode=ParseMode.HTML, reply_markup=kb,
                    performer=(cached_meta.artist or "")[:64],
                    title=(cached_meta.track or "")[:64],
                )
                if status:
                    await status.delete()
            except Exception:
                # file_id went stale / network — fall through to the regular download below
                log.exception("send_audio cached failed (fallback to download): %s", track.video_id)
            else:
                await insert_event(user_id=update.effective_user.id, kind="cache_hit",
                                   query_id=query_id, video_id=track.video_id)
                await _remember_session_and_ref(
                    context, update, query_id, track,
                    cached_meta.artist, cached_meta.track)
                _ua.cancel()
                return True

    m4a_path, thumb, art = None, None, None
    # Degraded mode — send a link instead of the file
    if await _is_degraded(context):
        _btn = _button_label(track.title, track.channel)
        _url = track.url or track.fallback_url or ""
        _yt_lang = await _lang_of(update.effective_user.id)
        if _url:
            await update.effective_chat.send_message(
                f"🔗 {_btn}\n\nБюджет скачиваний на сегодня исчерпан — вот ссылка:",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton(_kbt(_yt_lang, "open_yt"), url=_url)]]))
            if status:
                await status.delete()
            _ua.cancel()  # an upload_document loop would run forever without cancel
            return True
        _ua.cancel()
        return False

    async def _dl_progress(pct: float) -> None:
        # live download progress in the status message
        if status is None:
            return
        try:
            await status.edit_text(await _t(update.effective_user.id, "dl_bar",
                                          bar=_progress_bar(pct), pct=int(pct)))
        except Exception:
            pass

    try:
        try:
            url = track.url or f"https://www.youtube.com/watch?v={track.video_id}"
            m4a_path, thumb, info = await media.download_audio(
                url, on_progress=_dl_progress if status else None)
            # After 100% comes convert+meta+tags (10-25s without updates) —
            # switch the status so the bar doesn't look frozen
            if status:
                try:
                    await status.edit_text(await _t(update.effective_user.id, "st_processing"))
                except Exception:
                    pass
        except Exception as e:
            log.warning("download failed %s: %s", track.video_id, str(e)[:200])
            if track.fallback_url:
                try:
                    log.info("trying fallback_url: %s", track.fallback_url)
                    m4a_path, thumb, info = await media.download_audio(track.fallback_url)
                except Exception as e2:
                    log.warning("fallback also failed: %s", str(e2)[:200])
                    if status:
                        await status.edit_text(await _t(update.effective_user.id, "dl_track_fail", title=pretty_title(track.title)[:60]))
                    else:
                        await update.effective_chat.send_message(await _t(update.effective_user.id, "dl_fail_generic"))
                    _ua.cancel()
                    return False
            else:
                if status:
                    await status.edit_text(await _t(update.effective_user.id, "dl_track_fail", title=pretty_title(track.title)[:60]))
                else:
                    await update.effective_chat.send_message(await _t(update.effective_user.id, "dl_fail_generic"))
                _ua.cancel()
                return False
        meta = None
        try:
            meta = await metadata_svc.resolve(info, track.video_id)
            art = meta.artwork_path
            # enrich runs ffmpeg synchronously (subprocess.run up to 120s) — in a
            # thread only, otherwise it blocks the event loop and every chat (review 2026-09-03)
            await asyncio.to_thread(
                tag_enrichment.enrich,
                m4a_path, track=meta.track, artist=meta.artist, album=meta.album,
                year=meta.year, artwork=meta.artwork_path, fallback_thumb=thumb,
            )
        except Exception:
            log.exception("metadata/enrich failed for %s", track.video_id)

        if meta:
            await _remember_session_and_ref(context, update, query_id, track,
                                            meta.artist, meta.track)

        # The auto track fact was removed (💡 button under the card).
        # Sonar stays as a fallback: Last.fm wiki empty -> a bio fact.
        if meta and not meta.bio:
            meta.bio = await _sonar_fact(context, update.effective_user.id,
                                         meta.artist, meta.track, meta.year)
        if meta and meta.bio:
            meta.bio = await _translate_bio(context, update.effective_user.id, meta.bio)
        caption = _caption(meta, track, confidence, verdict)
        try:
            with open(m4a_path, "rb") as f:
                _meta_title = (meta.track if meta else None) or pretty_title(track.title)
                _meta_artist = (meta.artist if meta else None) or pretty_title(track.channel)

                # Owner decision after live tests: a photo card with the cover
                # as a SEPARATE message before the audio. A thumbnail in
                # send_audio does NOT render as a visible photo (confirmed by
                # tests 1-3); the embedded cover is only visible inside the player.
                # Album cover from iTunes ONLY.
                # YouTube screenshots were removed (prod case GTA VI: a random frame instead
                # of the cover). No album cover = no photo; better without a picture.
                _aw_url = getattr(meta, "artwork_url", "") if meta else ""
                if _aw_url:
                    try:
                        await update.effective_chat.send_photo(photo=_aw_url)
                    except Exception:
                        pass
                elif meta and meta.artwork_path and Path(meta.artwork_path).exists():
                    try:
                        with open(meta.artwork_path, "rb") as _cf:
                            await update.effective_chat.send_photo(
                                photo=InputFile(_cf, filename="cover.jpg"))
                    except Exception:
                        pass

                # Thumbnail = album cover (like nova; the owner confirmed
                # Telegram shows it in the player). The fd is closed in finally (it leaked
                # on a send failure); the download goes through async with (the client
                # wasn't closed); the file is unique (a fixed /tmp/tg_thumb.jpg raced
                # between parallel chats).
                _thumb_f = None
                _aw_t = getattr(meta, "artwork_url", "") if meta else ""
                _aw_p = getattr(meta, "artwork_path", "") if meta else ""
                try:
                    if _aw_p and Path(_aw_p).exists() and Path(_aw_p).stat().st_size < 200_000:
                        _thumb_f = open(_aw_p, "rb")
                    elif _aw_t:
                        import httpx as _hx2
                        async with _hx2.AsyncClient(timeout=10) as _cl:
                            _r2 = await _cl.get(_aw_t)
                        if _r2.status_code == 200 and len(_r2.content) < 200_000:
                            _fd, _tp_name = tempfile.mkstemp(suffix=".jpg", prefix="tg_thumb_")
                            try:
                                os.write(_fd, _r2.content)
                            except Exception:
                                pass
                            finally:
                                # the raw fd doesn't leak if write fails
                                try:
                                    os.close(_fd)
                                except OSError:
                                    pass
                            if Path(_tp_name).exists() and Path(_tp_name).stat().st_size > 0:
                                _thumb_f = open(_tp_name, "rb")
                    sent = await context.bot.send_audio(
                        chat_id=update.effective_chat.id,
                        audio=InputFile(f, filename=_safe_name(track, ext=Path(m4a_path).suffix)),
                        caption=caption, parse_mode=ParseMode.HTML,
                        performer=_meta_artist[:64], title=_meta_title[:64],
                        thumbnail=(InputFile(_thumb_f, filename="cover.jpg") if _thumb_f else None),
                        reply_markup=kb, read_timeout=120, write_timeout=120, connect_timeout=30,
                    )
                    await insert_event(user_id=update.effective_user.id, kind="dl_audio",
                                       query_id=query_id, video_id=track.video_id)
                finally:
                    if _thumb_f:
                        _thumb_f.close()
            # audio traffic accounting (MB; cache hits by file_id don't count).
            # send counter — caps "Не то"/"Похожее" chains.
            try:
                _mb = int(Path(m4a_path).stat().st_size / 1_048_576)
                await incr_usage(update.effective_user.id, "bytes_sent", _mb)
            except Exception:
                pass
            try:
                await incr_usage(update.effective_user.id, "sends")
            except Exception:
                pass
            # cache the tg file_id (the same file is later sent by it, without a download).
            file_id = getattr(getattr(sent, "audio", None), "file_id", None)
            if file_id:
                try:
                    await cache_file(
                        video_id=track.video_id, file_id=file_id,
                        artist=meta.artist if meta else "",
                        track=meta.track if meta else "",
                        album=meta.album if meta else "",
                        year=meta.year if meta else "",
                        genre=meta.genre if meta else "",
                        bio=meta.bio if meta else "",
                        artwork_url=meta.artwork_url if meta else "",
                    )
                except Exception:
                    log.exception("cache_file failed %s", track.video_id)
            if status:
                await status.delete()
        except Exception:
            # a send failure = a candidate failure: the caller tries the next one
            # (it used to return True here and the fallback silently gave up — review 2026-09-03)
            log.exception("send_audio failed")
            if status:
                await status.edit_text(await _t(update.effective_user.id, "dl_tg_reject"))
            _ua.cancel()
            return False
        _ua.cancel()
        return True
    finally:
        # clean temp files right after sending; don't wait for the TTL cleanup
        for p in (m4a_path, thumb, art):
            if p:
                try:
                    await asyncio.to_thread(os.remove, p)
                except OSError:
                    pass


def _album_line(meta) -> str:
    parts = []
    if meta.year:
        parts.append(_esc(meta.year))
    if meta.album:
        parts.append(_esc(meta.album))
    if not parts:
        return ""
    return "📀 " + " · ".join(parts)


def _strip_md(text):
    """Markdown (**, __, `) and footnote links [1], [11], [note 1] from
    Last.fm/sonar — they don't render in TG and clutter the text."""
    t = re.sub(r'\*{1,2}|_{1,2}|`{1,3}', '', str(text or ''))
    t = re.sub(r'\[\d{1,2}\]|\[note \d+\]|\[citation needed\]', '', t)
    return t.strip()


def _genre_line(meta) -> str:
    # genre from Last.fm toptags (track-level, else an artist-level fallback): "pop, 80s, ..."
    if not meta.genre:
        return ""
    return "🏷️ " + _esc(_strip_md(meta.genre))


_BIO_MAX = 500  # the owner asked for longer (220 cut mid-word)


_BIO_GARBAGE_RE = re.compile(
    r"^(неверный|ошибка|invalid|error|not found|не найден|нет данных|no data|"
    r"unknown|неизвестн|\s*$)", re.I)


def _clean_bio(bio):
    """Garbage bio ("Неверный тег для...") -> empty."""
    b = (bio or "").strip()
    if not b or len(b) < 30 or _BIO_GARBAGE_RE.match(b):
        return ""
    return b


def _bio_line(meta) -> str:
    """An "about the track" line from meta.bio (Last.fm wiki summary, track or
    artist). Cut at a word boundary - the caption must not bloat. Empty = no line.
    Markdown/footnote cleanup (Freestate case: *Ultra* and [12][3] arrived raw
    in a sonar bio — the earlier cleanup covered only the fact, not the bio)."""

    if not meta.bio:
        return ""
    raw = _strip_md(str(meta.bio).strip())
    if not raw:
        return ""
    if len(raw) > _BIO_MAX:
        cut = raw[:_BIO_MAX]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        raw = cut.rstrip(" ,.;:!?()[]-–-") + "…"
    prefix = "Об артисте: " if getattr(meta, "bio_about", "") == "artist" else ""
    return "ℹ️ " + prefix + _esc(raw)


_TR_CACHE: dict = {}  # (hash(...), kind, lang) -> (ts, text): bio translations
# and sonar facts repeat (24h in memory; facts are doubled by the persistent store.facts cache, 30d)


async def _sonar_fact(context, user_id: int, artist: str, track: str, year: str) -> str:
    """A fun fact about the track via sonar (web) when the Last.fm wiki is missing.
    24h cache keyed by (artist+track, language); error/disabled = empty (card without ℹ️)."""
    bothub = context.bot_data.get("bothub")
    if not bothub or not bothub.sonar_enabled or not (artist and track):
        return ""
    try:
        lang = await get_user_lang(user_id) or "ru"
    except Exception:
        lang = "ru"
    key = (hash((artist.lower(), track.lower())), "fact", lang)
    hit = _TR_CACHE.get(key)
    if hit and time.time() - hit[0] < 86400:
        return hit[1]
    lang_name = {"ru": "русском", "en": "English"}.get(lang, "русском")
    y = f", {year}" if year else ""
    out = await bothub.chat_text(
        "Ты музыкальный справочник. Найди в интернете ОДИН интересный факт "
        "об этом треке или его создании. Ответь на " + lang_name +
        " одним-двумя предложениями, только факт, без вступлений. "
        "Если факт найти нельзя - верни пустую строку.",
        f"{artist} - {track}{y}", max_tokens=200, model=bothub._sonar_model)
    if out and len(out) > 10:
        if len(_TR_CACHE) > 256:
            _TR_CACHE.clear()
        _TR_CACHE[key] = (time.time(), out)
        return out
    return ""


async def _sonar_track_fact(context, user_id: int, artist: str, track: str,
                            album: str = "", year: str = "") -> str:
    """A fun fact about the SPECIFIC track/album (not an artist bio).
    sonar in the user's language. 24h cache. Empty = no fact (doesn't block the send)."""
    bothub = context.bot_data.get("bothub")
    if not bothub or not bothub.sonar_enabled or not (artist and track):
        return ""
    try:
        lang = await get_user_lang(user_id) or "ru"
    except Exception:
        lang = "ru"
    key = (hash((artist.lower(), track.lower())), "trackfact", lang)
    hit = _TR_CACHE.get(key)
    if hit and time.time() - hit[0] < 86400:
        return hit[1]
    lang_name = {"ru": "русском", "en": "English"}.get(lang, "русском")
    ctx = f" (альбом: {album})" if album else ""
    y = f", {year}" if year else ""
    out = await bothub.chat_text(
        "Ты музыкальный справочник. Найди в интернете ОДИН интересный факт "
        "о создании, записи, истории или культурном влиянии ИМЕННО ЭТОГО трека "
        "или альбома (не биографию исполнителя). Ответь на " + lang_name +
        " двумя-тремя предложениями. Без вступлений. "
        "Если факт найти нельзя - верни пустую строку.",
        f"{artist} - {track}{y}{ctx}", max_tokens=300, model=bothub._sonar_model)
    if out and len(out) > 10:
        if len(_TR_CACHE) > 256:
            _TR_CACHE.clear()
        _TR_CACHE[key] = (time.time(), out)
        return out
    return ""


async def _translate_bio(context, user_id: int, bio: str) -> str:
    """The Last.fm bio arrives in English - translate it into the user's language
    (asked at /start). The translation is cached; any error = the original, it never blocks the send."""
    try:
        lang = await get_user_lang(user_id)
    except Exception:
        lang = None
    if lang != "ru" or not bio or not any(c.isascii() and c.isalpha() for c in bio[:80]):
        return bio  # en user, or text already non-English - as is
    key = (hash(bio), "ru")
    hit = _TR_CACHE.get(key)
    if hit and time.time() - hit[0] < 86400:
        return hit[1]
    bothub = context.bot_data.get("bothub")
    if bothub is None:
        return bio
    out = await bothub.chat_text(
        "Ты переводчик музыкальных справок. Переведи текст на русский язык. "
        "Имена, названия треков и альбомов оставь в оригинале. "
        "Верни только перевод, без комментариев.",
        bio, max_tokens=400)
    if out:
        if len(_TR_CACHE) > 256:
            _TR_CACHE.clear()
        _TR_CACHE[key] = (time.time(), out)
        return out
    return bio


def _caption(meta, track, confidence, verdict=""):
    # Telegram hard-cuts captions > 1024 chars — send_audio fails wholesale
    # (review 2026-09-03). Hence: per-field limits at build time + a safety
    # drop of secondary lines, and only in a pathological case a plain-text cut.
    def _cut(v, n):
        v = str(v or "").strip()
        return v if len(v) <= n else v[:n - 1].rstrip() + "…"

    if meta:
        title = _cut(meta.track or track.title, 200)
        artist = _cut(meta.artist or track.channel, 150)
        album_line = _album_line(meta)
        genre_line = _genre_line(meta)
        bio_line = _bio_line(meta)
    else:
        title = _cut(pretty_title(track.title), 200)
        artist = _cut(pretty_title(track.channel), 150)
        album_line = ""
        genre_line = ""
        bio_line = ""
    # Designed with the owner: blocks separated by a blank line - title/facts/
    # bio/"why this one" read separately; a 💬 marker on the rationale line.
    head = [f"🎵 <b>{_esc(title)}</b>", f"🎤 {_esc(artist)}"]
    facts = [x for x in (album_line, genre_line) if x]
    blocks = [head]
    if facts:
        blocks.append([_cut(x, 90) for x in facts])
    if bio_line:
        blocks.append([_cut(bio_line, _BIO_MAX)])
    # The 💬 LLM verdict was dropped. The 💡 fact moved from the caption to
    # a button (sonar calls are expensive, generated only on demand) — only the ℹ️ bio stays here.
    lines = []
    for b in blocks:
        if lines:
            lines.append("")
        lines.extend(b)
    text = "\n".join(lines)
    if len(text) > 1024:
        # safety net: cut the less important lines, never touch title/artist
        for i in range(len(lines) - 1, 1, -1):
            text = "\n".join(lines[:i])
            if len(text) <= 1024:
                return text
        # total pathology (enormous title+artist): plain text, tags stripped —
        # broken HTML would kill the send more reliably than the length
        plain = re.sub(r"<[^>]+>", "", "\n".join(lines))
        return plain[:1023] + "…"
    return text


def _safe_name(track, ext: str = "m4a"):
    """User-facing file name: "Artist - Track.ext" with spaces (underscores
    looked like junk in players - a prod complaint). Only characters invalid
    for the filesystem are stripped."""
    base = _button_label(track.title, track.channel)
    base = re.sub(r'[\\/:*?"<>|]+', "", base)[:60].strip(" .") or "track"
    return f"{base}.{ext.lstrip('.') or 'm4a'}"


def _esc(s):
    return escape(str(s))


def _refine_prompt(reply_text: str, lang: str = "ru") -> InlineKeyboardMarkup:
    """Keyboard with a cancel button for the force-reply message in refine mode."""
    return InlineKeyboardMarkup([[InlineKeyboardButton(_kbt(lang, "cancel"), callback_data="rfx")]])


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass  # callback too old — ignore and keep processing
    if _denied(update, context):
        return  # foreign buttons are silently ignored: buttons on old messages say nothing about the bot
    data = (q.data or "").split(":")
    kind = data[0] if data else ""
    media = context.bot_data["media"]
    user = update.effective_user

    def _ints(*idx):
        out = []
        for i in idx:
            try:
                out.append(int(data[i]))
            except (ValueError, IndexError):
                return None
        return out

    # refinement cancel (button on the force-reply message)
    if kind == "rfx":
        context.chat_data.pop("refine", None)
        try:
            await q.edit_message_text("Отменил. Жду новый запрос.")
        except Exception:
            pass
        return

    # The 💡 fact button on a card. A sonar call is $0.008, hence: a 30d
    # persistent cache (free repeats), a per-user cap, a global budget for everyone.
    # The cache is checked BEFORE the limits (a cached fact is always free);
    # a negative result (sonar empty) is billed too — otherwise repeated clicks
    # on "не нашлось" burn sonar past the caps; the id is cut with split(":", 1) — foreign ":" inside the id.
    if kind == "fact":
        vid = (q.data or "").split(":", 1)[1] if ":" in (q.data or "") else ""
        if vid:
            await insert_event(user_id=update.effective_user.id, kind="fact", video_id=vid)
        if not vid:
            return
        # Complaint: "дает просто повтор". A cache hit on a second press looked like
        # a duplicate message. Now a second press within an hour gets a short
        # pointer up the thread, not a duplicate fact.
        _fs = context.chat_data.get(f"factshown:{vid}")
        if _fs and time.time() - _fs < 3600:
            await q.message.reply_text(await _t(update.effective_user.id, "fact_dup"))
            return
        try:
            row = await get_cached_file(vid) or {}
        except Exception:
            log.exception("get_cached_file failed %s", vid)
            row = {}
        _f_artist, _f_track = row.get("artist") or "", row.get("track") or ""
        if not (_f_artist and _f_track):
            await q.message.reply_text(await _t(update.effective_user.id, "fact_ctx_lost"))
            return
        try:
            _f_lang = await get_user_lang(user.id) or "ru"
        except Exception:
            _f_lang = "ru"
        _f_cached = await get_fact(_f_artist, _f_track, _f_lang)
        if _f_cached:
            # the cache is free: no sonar, no increments, limits not checked
            context.chat_data[f"factshown:{vid}"] = time.time()
            await q.message.reply_text(f"💡 {_strip_md(_f_cached)}")
            return
        cfg = context.bot_data["cfg"]
        usage = await get_usage(user.id)
        glob = await get_global_usage()
        if (usage.get("facts") or 0) >= cfg.fact_daily_limit:
            await _maybe_wtp(update, context, user, "fact", "fact",
                             usage.get("facts") or 0)
            await q.message.reply_text(await _t(update.effective_user.id, "fact_cap_user"))
            return
        if (glob.get("facts") or 0) >= cfg.global_daily_fact_budget:
            await q.message.reply_text(await _t(update.effective_user.id, "fact_cap_global"))
            return
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action="typing")
        except Exception:
            pass
        # the call itself costs money — increment before the call, whatever the outcome
        await incr_usage(user.id, "facts")
        fact = await _sonar_track_fact(
            context, user.id, _f_artist, _f_track,
            row.get("album") or "", row.get("year") or "")
        if not fact:
            await q.message.reply_text(await _t(update.effective_user.id, "fact_none"))
            return
        try:
            await save_fact(_f_artist, _f_track, _f_lang, fact)
        except Exception:
            log.exception("save_fact failed")
        context.chat_data[f"factshown:{vid}"] = time.time()
        await q.message.reply_text(f"💡 {_strip_md(fact)}")
        return

    # "Уточнить" from the top-3 preview: rephrase the query (without the rejected track)
    if kind == "rn":
        await insert_event(user_id=update.effective_user.id, kind="refine")
        nums = _ints(1)
        if nums is None:
            return
        (query_id,) = nums
        store = await _load_state(context, query_id)
        if not store:
            await q.message.reply_text(await _t(update.effective_user.id, "search_cold"))
            return
        context.chat_data["refine"] = {
            "query_id": query_id,
            "desc": store.get("desc", ""),
            "rejected_ids": set(),
            "ts": time.monotonic(),
            # pool + context of the original search for a rerank without new fetching
            "candidates": store.get("candidates", []),
            "reference": store.get("reference") or {},
            "intent": store.get("intent"),
        }
        try:
            await q.edit_message_text(
                "Окей, что добавить или поменять? Жанр, настроение, темп, слова, мелодия.",
                reply_markup=_refine_prompt("", lang=await _lang_of(update.effective_user.id)),
            )
        except Exception:
            pass
        return

    # "Уточнить" on a delivered track: refinement with the rejected excluded.
    # rejected = everything rejected via "Не то" earlier + the currently shown clip.
    if kind == "rf":
        await insert_event(user_id=update.effective_user.id, kind="refine")
        nums = _ints(1, 2)
        if nums is None:
            return
        query_id, rank = nums
        store = await _load_state(context, query_id)
        if not store or rank < 0 or rank >= len(store["candidates"]):
            await q.message.reply_text(await _t(update.effective_user.id, "search_cold"))
            return
        cur = store["candidates"][rank]
        rejected = set(store.get("rejected_ids", []))
        rejected.add(cur.video_id)
        context.chat_data["refine"] = {
            "query_id": query_id,
            "desc": store.get("desc", ""),
            "rejected_ids": rejected,
            "ts": time.monotonic(),
            # pool + context of the original search for a rerank without new fetching
            "candidates": store.get("candidates", []),
            "reference": store.get("reference") or {},
            "intent": store.get("intent"),
        }
        try:
            await q.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        try:
            await q.message.reply_text(
                "Что не так с треком? Например:\n"
                "• другой артист\n• быстрее или медленнее\n"
                "• оригинал, а не кавер\n• без вокала / инструментал",
                reply_markup=_refine_prompt("", lang=await _lang_of(update.effective_user.id)),
            )
        except Exception:
            pass
        return

    # top-3 choice
    # the 🎬 Video button on a delivered track -> quality picker
    if kind == "dlv":
        nums = _ints(1, 2)
        if nums is None:
            return
        query_id, rank = nums
        store = await _load_state(context, query_id)
        cands = store["candidates"] if store else []
        if not store or rank < 0 or rank >= len(cands):
            await q.message.reply_text(await _t(update.effective_user.id, "stale_state"))
            return
        track = cands[rank]
        vid = track.video_id
        if not vid or track.source != "youtube":
            await q.message.reply_text(await _t(update.effective_user.id, "video_yt_only"))
            return
        yt_url = f"https://www.youtube.com/watch?v={vid}"
        qualities = InlineKeyboardMarkup([
            [InlineKeyboardButton("480p", callback_data=f"dl:q:480:{vid}"),
             InlineKeyboardButton("720p", callback_data=f"dl:q:720:{vid}")],
            [InlineKeyboardButton("1080p", callback_data=f"dl:q:1080:{vid}"),
             InlineKeyboardButton("1440p", callback_data=f"dl:q:1440:{vid}")],
        ])
        try:
            await q.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        await q.message.reply_text(
            f"🎬 {track.title[:200]}\nВыбери качество:",
            reply_markup=qualities)
        context.chat_data[f"url:{vid}"] = {"url": yt_url, "title": track.title}
        return

    # Audio/Video download buttons for a URL
    if kind == "dl":
        # Callback formats: dl:aud:{vid} | dl:vid:{vid} | dl:q:{quality}:{vid}
        # (quality has 4 segments — video_id sits at index 3, not 2!)
        if len(data) == 4:
            # dl:q:720:{vid} — quality
            action = f"q:{data[2]}"  # "q:720"
            vid = data[3]            # video_id
        else:
            action = data[1] if len(data) >= 3 else ""
            vid = data[2] if len(data) >= 3 else ""
        url_state = context.chat_data.get(f"url:{vid}")
        if url_state:
            url = url_state["url"]
            title = url_state["title"]
        elif vid and len(vid) == 11:
            # chat_data lives in memory — a container restart (deploy) loses it.
            # A YouTube video_id is enough to rebuild the URL.
            url = f"https://www.youtube.com/watch?v={vid}"
            title = vid
        else:
            await q.message.reply_text(await _t(update.effective_user.id, "link_stale"))
            return
        media_svc = context.bot_data["media"]

        if action == "aud":
            # Audio: the standard path, _handle_url-like logic
            try:
                await q.edit_message_text(f"🎵 Качаю аудио: {title[:60]}…")
            except Exception:
                pass
            cand = Candidate(vid, title, "", 0, 0)
            cand.url = url
            query_id = await insert_query(
                user_id=user.id, correlation_id=uuid4().hex,
                description=f"url: {title}", intent="direct_url",
                signals=None, queries=[url], candidates=[cand.__dict__],
                best_video_id=vid, best_title=title, best_score=None, confidence="high")
            await incr_usage(user.id, "urls")
            await _send_track(update, context, q.message, media_svc, query_id, 0,
                              [cand], cand, "high", "Лови трек по ссылке.", keyboard="url")
            return

        if action == "vid":
            # Video: show the quality picker
            qualities = InlineKeyboardMarkup([
                [InlineKeyboardButton("480p", callback_data=f"dl:q:480:{vid}"),
                 InlineKeyboardButton("720p", callback_data=f"dl:q:720:{vid}")],
                [InlineKeyboardButton("1080p", callback_data=f"dl:q:1080:{vid}"),
                 InlineKeyboardButton("1440p", callback_data=f"dl:q:1440:{vid}")],
            ])
            try:
                await q.edit_message_text(
                    f"🎬 {title[:200]}\nВыбери качество:",
                    reply_markup=qualities)
            except Exception:
                pass
            return

        if action.startswith("q:"):
            # Video gets dedicated gates (GB-sized uploads are not let through unchecked) + a busy-guard.
            # The upload goes into create_task — the PTB dispatcher is sequential,
            # and a 2GB video (up to ~20 min) would otherwise stall the update queue of the WHOLE bot.
            cfg = context.bot_data["cfg"]
            v_usage = await get_usage(user.id)
            if (v_usage.get("videos") or 0) >= cfg.video_daily_limit:
                await _maybe_wtp(update, context, user, "video", "video",
                                 v_usage.get("videos") or 0)
                await q.message.reply_text(
                    await _t(update.effective_user.id, "video_cap"))
                return
            _busy = context.chat_data.get("video_busy")
            if _busy and time.time() - _busy < 600:
                await q.message.reply_text(await _t(update.effective_user.id, "video_busy"))
                return
            context.chat_data["video_busy"] = time.time()
            quality = data[2].rstrip("p") if len(data) >= 4 else "720"
            quality = f"{quality}p"
            try:
                await q.edit_message_text(await _t(update.effective_user.id, "video_start",
                                              quality=quality, title=title[:50]))
            except Exception:
                pass

            async def _bg_video():
                path = None
                vstat = None
                try:
                    try:
                        vstat = await q.message.reply_text(
                            await _t(update.effective_user.id, "video_start_short", quality=quality))
                    except Exception:
                        vstat = None

                    async def _vpct(p: float) -> None:
                        if vstat is None:
                            return
                        try:
                            await vstat.edit_text(await _t(update.effective_user.id, "dl_bar",
                                              bar=_progress_bar(p), pct=int(p)))
                        except Exception:
                            pass

                    path, info = await media_svc.download_video(
                        url, quality, on_progress=_vpct if vstat else None)
                    if vstat:  # convert/merge/send have no updates — switch the stage
                        try:
                            await vstat.edit_text(await _t(update.effective_user.id, "st_processing"))
                        except Exception:
                            pass
                    fname = _button_label(title, info.get("channel", ""))[:60]
                    fname = re.sub(r'[\\/:*?"<>|]+', "", fname) or "video"
                    with open(path, "rb") as vf:
                        await context.bot.send_video(
                            chat_id=update.effective_chat.id,
                            video=InputFile(vf, filename=f"{fname}.mp4"),
                            caption=f"🎬 {title[:200]}\n📅 {info.get('upload_date', '?')[:4] or '?'}",
                            supports_streaming=True,
                            read_timeout=600, write_timeout=600, connect_timeout=30)
                    await incr_usage(user.id, "videos")
                    await insert_event(user_id=user.id, kind="dl_video", video_id=vid)
                    try:
                        _mb = int(Path(path).stat().st_size / 1_048_576)
                        await incr_usage(user.id, "bytes_sent", _mb)
                    except Exception:  # non-OSError from store too
                        pass
                except Exception as e:
                    log.exception("video download failed: %s", str(e)[:100])
                    try:
                        await q.message.reply_text(
                            await _t(update.effective_user.id, "video_fail"))
                    except Exception:
                        pass
                finally:
                    # the mp4 is deleted even when send fails (it used to live until TTL)
                    if path:
                        try:
                            await asyncio.to_thread(Path(path).unlink, True)
                        except Exception:
                            pass
                    if vstat is not None:
                        try:
                            await vstat.delete()
                        except Exception:
                            pass
                    context.chat_data.pop("video_busy", None)

            # update= — a background send error reaches the error handler with the
            # user's live context; stop() on deploy waits out a 2GB send
            context.application.create_task(_bg_video(), update=update)
            return

    # donate button -> XTR invoice
    if kind == "don":
        try:
            stars = int(data[1]) if len(data) > 1 and data[1].isdigit() else 0
        except ValueError:
            stars = 0
        if stars not in _DONATE_STARS:
            return
        try:
            await q.answer()
        except Exception:
            pass
        from telegram import LabeledPrice
        await context.bot.send_invoice(
            chat_id=update.effective_chat.id,
            title=f"Поддержать проект — {stars} ⭐",
            description="Music Search Mate: поиск музыки по описанию, без рекламы.",
            payload=f"donate{stars}", currency="XTR",
            prices=[LabeledPrice(label=f"Звёзды ({stars})", amount=stars)])
        return

    # WTP survey: button answer
    if kind == "wtp":
        val = data[1] if len(data) == 2 else ""
        try:
            await q.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        if val == "optout":
            await save_survey_response(user.id, "optout")
            await q.message.reply_text("Понял, больше не спрошу.")
        elif val.isdigit():
            stars = float(val)
            await save_survey_response(user.id, "button", stars_value=stars)
            await q.message.reply_text(await _t(update.effective_user.id, "wtp_thanks", stars=stars))
        return

    # info language choice
    if kind == "lang":
        lang = data[1] if len(data) == 2 and data[1] in ("ru", "en") else None
        if lang:
            await set_user_lang(user.id, lang)
            try:
                await q.edit_message_text(f"✅ {('Русский' if lang == 'ru' else 'English')}")
            except Exception:
                pass
        return

    # preview carousel: paging cards without server-side state (index in the callback)
    # The carousel cr: callback was removed together with the chain above

    if kind == "pk":
        nums = _ints(1, 2)
        if nums is None:
            return
        query_id, rank = nums
        store = await _load_state(context, query_id)
        cands = store["candidates"] if store else []
        if not store or rank < 0 or rank >= len(cands):
            await q.message.reply_text(await _t(update.effective_user.id, "search_cold"))
            return
        if context.chat_data.get(f"busy:{query_id}"):  # double-click protection
            return
        context.chat_data[f"busy:{query_id}"] = True
        try:
            track = cands[rank]
            try:
                await q.edit_message_text(f"Качаю: {_button_label(track.title, track.channel)}…")
            except Exception:
                pass
            ok = await _send_track(update, context, None, media, query_id, rank,
                                   cands, track, store["confidence"], store.get("verdict", ""))
            if not ok:
                # a download failure must not burn the choice — restore the preview
                text, markup = _render_preview(query_id, cands,
                                               store.get("verdict", ""), store.get("intro", ""),
                                               lang=await _lang_of(update.effective_user.id))
                try:
                    await q.message.edit_text(text + "\n\n" + await _t(update.effective_user.id, "dl_failed_note"),
                                              reply_markup=markup)
                except Exception:
                    pass
        finally:
            context.chat_data[f"busy:{query_id}"] = False
        return

    # "Похожее": a new search for tracks similar to the current one. reference is the canonical
    # artist/track from Store (set in _send_track after metadata.resolve), else raw
    # title/channel. LLM#1 generates the association; the reference ban in search_candidates never
    # returns the reference itself. The button is available on every search result and on URL tracks.
    if kind == "sim":
        await insert_event(user_id=update.effective_user.id, kind="similar")
        nums = _ints(1, 2)
        if nums is None:
            return
        query_id, rank = nums
        store = await _load_state(context, query_id)
        cands = store["candidates"] if store else []
        if not store or rank < 0 or rank >= len(cands):
            await q.message.reply_text(await _t(update.effective_user.id, "search_cold"))
            return
        cur = cands[rank]
        try:
            await q.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        status = await q.message.reply_text(await _status_text(update.effective_user.id, "st_similar"))
        if await _sends_over(update, context, user):
            return
        artist = store.get("resolved_artist") or cur.channel
        track = store.get("resolved_track") or cur.title
        # Same reference within _SIM_REUSE_WINDOW and unseen candidates of the same
        # kind left in the pool -> the next one from the pool, without the pipeline
        # (a repeated "Похожее" used to rerun a full 25-38s search).
        media = context.bot_data["media"]
        if await _try_sim_from_pool(update, context, user, status, media,
                                    query_id, store, cur, artist):
            return
        desc = f"похожие на {track} - {artist}"
        out = await _do_search(update, context, status, desc)
        if out is None:
            return
        await _finish_search(update, context, status, out, desc, user)
        return

    # "То / Не то" feedback
    if kind == "fb" and len(data) == 4:
        action = data[1]
        nums = _ints(2, 3)
        if nums is None:
            return
        query_id, rank = nums
        store = await _load_state(context, query_id)
        candidates = store["candidates"] if store else None

        try:
            await q.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass

        cur = candidates[rank] if candidates and 0 <= rank < len(candidates) else None
        if cur:
            await insert_feedback(query_id=query_id, user_id=user.id, video_id=cur.video_id,
                                  title=cur.title, feedback=("hit" if action == "h" else "miss"),
                                  shown_rank=rank)
            log.info("feedback user=%s q=%s %s rank=%d %s",
                     user.id, query_id, "hit" if action == "h" else "miss", rank,
                     cur.title[:40])
            # v1.10: feedback -> a score on the search trace (name=user_feedback, hit=1/miss=0);
            # the query's correlation_id = trace_id. Old records (str(user.id)) are silently skipped.
            cid = await get_query_correlation(query_id)
            score_feedback(cid, action == "h", rank, cur.title)

        if action == "h":
            await q.message.reply_text("👍 Принято.")
            await _evict_state(context, query_id)  # track found — buttons no longer needed
            return

        # miss: accumulate the rejected video_id in the session — it lands in the exclude of a future refinement
        if store is not None and cur and cur.video_id not in store.get("rejected_ids", []):
            store.setdefault("rejected_ids", []).append(cur.video_id)
            await _save_state(context, query_id, store)  # rejected_ids survives a restart

        # miss -> the next candidate (limit: best + 3 more, DoS protection via "Не то")
        if not candidates or rank + 1 >= len(candidates):
            await q.message.reply_text(await _t(update.effective_user.id, "variants_over"))
            await _evict_state(context, query_id)
            return
        best_idx = store.get("best_index", 0) if store else 0
        if rank + 1 > best_idx + 3:
            await q.message.reply_text(await _t(update.effective_user.id, "last_variant"))
            await _evict_state(context, query_id)
            return
        nxt = candidates[rank + 1]
        if await _sends_over(update, context, user):
            return
        status = await q.message.reply_text(await _status_text(update.effective_user.id, "st_other"))
        await _send_track(update, context, status, media, query_id, rank + 1, candidates,
                          nxt, store["confidence"], nxt.rationale)


async def _stats_funnel(user_id: int) -> str:
    """Today's funnel line from queries/events/feedback."""
    f = await funnel_today()
    o = f.get("outcomes", {})
    e = f.get("events", {})
    fb = f.get("feedback", {})
    dl = e.get("dl_audio", 0) + e.get("cache_hit", 0) + e.get("dl_video", 0)
    return await _t(
        user_id, "stats_funnel",
        q=sum(o.values()), ok=o.get("ok", 0), c=o.get("clarify", 0),
        e=o.get("empty", 0), b=o.get("blocked", 0), l=o.get("limit", 0),
        d=dl, h=fb.get("hit", 0), fb=fb.get("hit", 0) + fb.get("miss", 0))


async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/stats (owner): today for yourself + globally — live analytics
    without trips to Langfuse."""
    if _denied(update, context):
        return
    user = update.effective_user
    try:
        u = await get_usage(user.id)
        g = await get_global_usage()
    except Exception:
        u, g = {}, {}
    def f(d, k):
        return (d or {}).get(k) or 0
    from telegram.constants import ParseMode as _PM
    await update.message.reply_text(
        f"<b>{await _t(user.id, 'stats_head')}</b>\n"
        + await _t(user.id, "stats_you", s=f(u,'searches'), f=f(u,'sends'),
                   fa=f(u,'facts'), v=f(u,'videos'), mb=f(u,'bytes_sent'))
        + "\n"
        + await _t(user.id, "stats_all", s=f(g,'searches'), f=f(g,'sends'),
                   fa=f(g,'facts'), v=f(g,'videos'), mb=f(g,'bytes_sent')),
        + "\n" + await _stats_funnel(user.id),
        parse_mode=_PM.HTML)


# ════ /donate with Stars (a port of nova's invoice, XTR) ════

_DONATE_STARS = (50, 100, 300)


async def handle_donate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _denied(update, context):
        return
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton(f"⭐ {n}", callback_data=f"don:{n}")
        for n in _DONATE_STARS]])
    from telegram.constants import ParseMode as _PM
    await update.message.reply_text(
        "Если этот проект как-то сделал вашу жизнь лучше, "
        "<b>поддержите нас звёздочками</b>:", parse_mode=_PM.HTML,
        reply_markup=kb)


async def handle_precheckout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Required for XTR invoices: without an answer Telegram cancels the payment."""
    try:
        await update.pre_checkout_query.answer(ok=True)
    except Exception:
        log.exception("precheckout failed")


async def handle_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    p = getattr(update.effective_message, "successful_payment", None)
    if not p:
        return
    log.info("donate ok user=%s stars=%s", update.effective_user.id,
             getattr(p, "total_amount", "?"))
    await update.message.reply_text(
        await _t(update.effective_user.id, "payment_thanks"))
