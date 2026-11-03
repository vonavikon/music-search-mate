"""Music Search Bot entry point.

python -m bot.main
"""
import logging

from telegram import BotCommand
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
                        handle_precheckout, handle_payment)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for noisy in ("httpx", "telegram", "urllib3", "workqueue"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("bot.main")


async def post_init(app: Application) -> None:
    await init_db()
    # Telegram command menu (the "/" button left of the input field)
    try:
        await app.bot.set_my_commands([
            BotCommand("start", "найти трек по описанию или ссылке"),
            BotCommand("help", "как искать и что значат кнопки"),
            BotCommand("lang", "язык ответов"),
            BotCommand("new", "начать поиск с чистого листа"),
            BotCommand("stats", "цифры за сегодня"),
            BotCommand("donate", "поддержать проект"),
        ])
    except Exception as e:
        log.warning("set_my_commands failed: %s", e)
    log.info("db ready")


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
        log.warning("network timeout (update=%s): %s", upd_name, err)
        return
    if isinstance(err, NetworkError):  # TimedOut is already filtered out above
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
