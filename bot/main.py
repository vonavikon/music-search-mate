"""Music Search Bot entry point.

python -m bot.main
"""
import asyncio
import logging
import os
import time

from telegram.error import BadRequest, Forbidden, NetworkError, RetryAfter, TimedOut
from telegram.ext import (AIORateLimiter, Application, CallbackQueryHandler,
                          CommandHandler, MessageHandler, PreCheckoutQueryHandler, filters)

from config import load_config, ensure_dirs, LOGS_DIR
from services import tracing
from services.store import init_db
from services.llm_search import BothubClient
from services.media import MediaService
from services.metadata import MetadataService
from bot.handler import (handle_start, handle_help, handle_lang, handle_new, handle_message,
                        handle_callback, handle_unsupported, handle_stats, handle_donate,
                        handle_precheckout, handle_payment, bot_commands)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for noisy in ("httpx", "telegram", "urllib3", "workqueue"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("bot.main")


async def post_init(app: Application) -> None:
    await init_db()
    # Telegram command menu (the "/" button left of the input field).
    # Both languages are registered on the default scope via language_code:
    # the client picks the list by ITS interface locale. A third-locale
    # client sees no hints (an RU/EN bot; the old behavior showed RU to
    # everyone — deliberate change). A user who picks a language in /start
    # gets a per-chat override (see the lang callback in bot/handler.py):
    # the chat scope outranks the default-scope lists.
    for code in ("ru", "en"):
        try:
            await app.bot.set_my_commands(bot_commands(code), language_code=code)
        except Exception as e:
            log.warning("set_my_commands(%s) failed: %s", code, e)
    log.info("db ready")
    # plain asyncio task (not app.create_task): at post_init the application
    # is not yet running and PTB would refuse to track it; the watchdog is
    # self-contained and outlives nothing — the loop owns it
    asyncio.create_task(_network_watchdog(app))


# ---- fetcher-death watchdog (incident 08-10.11: mac twice, prod once) ----
# After a long enough network storm the PTB update fetcher dies SILENTLY:
# the process lives, the log goes quiet, users get no answers until a manual
# restart. We cannot see the fetcher task from the public API, so we act on
# the storm itself: count fetcher/update network errors, and once the storm
# was heavy AND the network is back (getMe answers) — restart the process
# preemptively. Docker (restart: always) and the dev wrapper loop bring it
# back in seconds with a clean fetcher. Exit code 70 = "infrastructure,
# restart me", same convention as EX_SOFTWARE friends.
_NET_STORM = {"count": 0, "last_ts": 0.0}
_STORM_TRIGGER = 15      # network errors after which the fetcher is suspect
_STORM_MAX_AGE = 900.0   # ...and the storm must be recent (15 min)


def _storm_restart_needed(storm: dict, get_me_ok: bool,
                          now: float | None = None) -> bool:
    """Pure decision: heavy recent storm + network back -> restart."""
    now = now if now is not None else time.time()
    recent = now - storm.get("last_ts", 0.0) <= _STORM_MAX_AGE
    return bool(storm.get("count", 0) >= _STORM_TRIGGER and recent and get_me_ok)


async def _network_watchdog(app: Application) -> None:
    # black box for the silent-hang hunt (10.11): every pass snapshots the
    # pending task set; a task PENDING for 3 consecutive passes (3+ minutes)
    # that is not a known long-lived service gets its stack logged. The
    # timeouts alone did not catch the 23:23 hang — this names the line.
    prev_seen: dict[int, int] = {}
    while True:
        await asyncio.sleep(60)
        try:
            get_me_ok = bool(await app.bot.get_me())
        except Exception:
            get_me_ok = False  # network still down — weather, wait it out
        if _storm_restart_needed(_NET_STORM, get_me_ok):
            log.critical(
                "watchdog: %d network errors in the last %.0fs and getMe "
                "answers — the update fetcher is presumed dead, self-restarting",
                _NET_STORM["count"], _STORM_MAX_AGE)
            os._exit(70)
        if get_me_ok:
            _NET_STORM["count"] = 0  # healthy network resets the storm gauge
        log.debug("watchdog ok: get_me=%s storm=%d", get_me_ok, _NET_STORM["count"])
        # hang black box
        seen_now: dict[int, int] = {}
        for t in asyncio.all_tasks():
            if t.done() or t is asyncio.current_task():
                continue
            name = repr(t.get_coro())
            # NB: create_task_callback is PTB's wrapper for USER update
            # processing — it must stay VISIBLE (the 10.11 hang lives there)
            if any(s in name for s in ("_network_watchdog", "_pulse", "_action_loop",
                                        "run_forever", "network_retry_loop",
                                        "polling_action_cb", "_update_fetcher")):
                continue  # long-lived service tasks
            key = hash(name)
            seen_now[key] = prev_seen.get(key, 0) + 1
            if seen_now[key] == 3:
                frames = [f"{f.f_code.co_filename.split('/')[-1]}:{f.f_lineno}"
                          for f in t.get_stack(limit=4)]
                log.warning("watchdog: task pending %d+ min: %s at %s",
                            seen_now[key], name[:120], " <- ".join(frames) or "?")
        prev_seen = seen_now


async def handle_error(update: object, context) -> None:
    """Without an error_handler PTB swallows unhandled exceptions silently —
    the user gets no reply and the log only shows 'No error handlers'. We log
    the full traceback so every new critical failure is immediately visible
    in the logs.
    Network noise is classified: PTB never retries send calls, so TimedOut/
    RetryAfter on a flaky network is weather, not an alert. Full tracebacks
    remain for real bugs only."""
    err = context.error
    upd_name = (update or {}).__class__.__name__
    if isinstance(err, TimedOut):
        _NET_STORM["count"] += 1
        _NET_STORM["last_ts"] = time.time()
        log.warning("network timeout (update=%s): %s", upd_name, err)
        return
    if isinstance(err, NetworkError):  # TimedOut is already filtered out above
        _NET_STORM["count"] += 1
        _NET_STORM["last_ts"] = time.time()
        log.warning("network error (update=%s): %s", upd_name, err)
        return
    if isinstance(err, RetryAfter):
        log.warning("telegram 429, pausing %ss (update=%s)", err.retry_after, upd_name)
        return
    if isinstance(err, Forbidden):
        # user blocked the bot — routine situation, not an alert
        log.info("bot blocked by user (update=%s)", upd_name)
        return
    if isinstance(err, BadRequest):
        # "Query is too old" and the like — user pressed the button hours later
        log.warning("bad request (update=%s): %s", upd_name, err)
        return
    log.error("unhandled exception (update=%s): %s",
              upd_name, err, exc_info=err)


async def post_shutdown(app: Application) -> None:
    bothub: BothubClient = app.bot_data["bothub"]
    await bothub.aclose()
    metadata: MetadataService = app.bot_data["metadata"]
    await metadata.aclose()
    # SQLite caches hold non-daemon aiosqlite worker threads: unclosed, the
    # process HANGS after main() returns (15:06 incident: a silent fetcher
    # death shut the app down gracefully, the zombie process kept the
    # container "running" and the bot deaf until a manual restart). Closed
    # here, the process exits and docker's restart policy revives the bot.
    for _mod_name in ("services.meta_cache", "services.search_cache"):
        try:
            _mod = __import__(_mod_name, fromlist=["aclose"])
            await _mod.aclose()
        except Exception:
            pass  # not initialized or already closed - nothing to do
    tracing.shutdown()  # flush trailing traces before the process dies
    log.info("shutdown done")


def main():
    cfg = load_config()
    ensure_dirs()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    bothub = BothubClient(cfg.bothub_base, cfg.bothub_api_key, cfg.llm_model,
                          sonar_model=cfg.llm_model_sonar)
    media = MediaService(max_concurrent=cfg.max_concurrent_downloads)
    metadata = MetadataService(cfg.lastfm_api_key)
    # after load_config: the langfuse SDK must be initialized after env is loaded
    tracing.init(cfg.langfuse_public_key, cfg.langfuse_secret_key, cfg.langfuse_host,
                 environment=cfg.langfuse_environment)

    builder = Application.builder().token(cfg.bot_token)
    # Explicit timeouts for EVERY environment (10.11 incident: a VPN flap
    # half-closed the TCP connections and a cached-file send hung FOREVER —
    # process alive, log clean, the track never delivered; the same silent
    # class killed the fetcher twice). Defaults are short but not every
    # request carried them; now they always do.
    builder = builder.connect_timeout(10.0).read_timeout(30.0) \
                      .write_timeout(30.0).pool_timeout(10.0) \
                      .get_updates_connect_timeout(10.0) \
                      .get_updates_read_timeout(25.0) \
                      .get_updates_write_timeout(10.0)
    # local Bot API: files up to 2 GB (the standard API caps at 50 MB).
    # PTB 22.8: media_write_timeout finally exists (in 20.7 multipart had a
    # hardcoded 20s and builder-write had no effect) — now the 2GB upload is
    # configurable the intended way. read 600 stays: tgapi's reply to an
    # upload takes minutes.
    if cfg.local_bot_api_url and cfg.tg_api_id:
        builder = (
            builder
            .base_url(f"{cfg.local_bot_api_url}/bot")
            .read_timeout(600.0)
            .media_write_timeout(600.0)
            # The poll client is SEPARATE with a short read timeout.
            # Effective getUpdates read = read_timeout + poll_timeout: with the
            # shared 600 a broken long-poll would take up to 610s to detect
            # (10 minutes of blindness).
            .get_updates_read_timeout(10.0)
        )
        log.info("using LOCAL Bot API: %s (2GB file limit)", cfg.local_bot_api_url)
    # Telegram 429: auto-pause all requests for retry_after
    # (PTB itself does not wait out RetryAfter in send calls)
    builder = builder.rate_limiter(AIORateLimiter())
    app = (
        builder
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.bot_data["cfg"] = cfg
    app.bot_data["bothub"] = bothub
    app.bot_data["media"] = media
    app.bot_data["metadata"] = metadata

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("lang", handle_lang))
    app.add_handler(CommandHandler("new", handle_new))
    app.add_handler(CommandHandler("stats", handle_stats))
    app.add_handler(CommandHandler("donate", handle_donate))
    app.add_handler(PreCheckoutQueryHandler(handle_precheckout))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, handle_payment))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # voice/photo/sticker/video/file no longer met with silence
    app.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO | filters.PHOTO | filters.VIDEO
        | filters.Sticker.ALL | filters.Document.ALL | filters.ANIMATION,
        handle_unsupported))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_error_handler(handle_error)

    log.info("starting music-search-bot (model=%s)", cfg.llm_model)
    # bootstrap_retries=-1 is deliberate: on a compose rebuild tgapi comes up
    # slower than the bot, and an infinite startup retry is better than an
    # instant crash (the 20.7 default is already -1; from 21.11 it changes to 0).
    app.run_polling(bootstrap_retries=-1,
                    allowed_updates=["message", "callback_query", "pre_checkout_query"])


if __name__ == "__main__":
    main()
