"""Configuration from env. Paths to data/temp/logs."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv  # noqa: F401  (loads .env into os.environ)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "music-search.db"


@dataclass(frozen=True)
class Config:
    bot_token: str
    bothub_api_key: str
    bothub_base: str
    llm_model: str
    llm_model_sonar: str  # web-grounded model (Perplexity sonar via bothub); empty = layer off
    max_concurrent_downloads: int
    temp_ttl_minutes: int
    user_rate_limit_per_hour: int
    lastfm_api_key: str  # optional; empty = Last.fm layer is skipped (graceful)
    langfuse_public_key: str  # optional; empty = tracing disabled (services/tracing.py)
    langfuse_secret_key: str
    langfuse_host: str
    langfuse_environment: str  # production/development: keeps test traces out of prod dashboards
    # local Bot API (files up to 2 GB instead of 50 MB)
    tg_api_id: int
    tg_api_hash: str
    local_bot_api_url: str
    allowed_user_ids: tuple  # (int, ...); empty = no restrictions (personal bot: must be set)
    # Limits (generous for personal mode; a public deployment tightens them via env)
    search_daily_limit: int
    url_daily_limit: int
    global_daily_search_budget: int  # shared across all users
    # Button-triggered facts (sonar at $0.008/call) and videos are counted per item, not in gigabytes
    fact_daily_limit: int            # sonar facts per user per day (cache is free)
    global_daily_fact_budget: int    # across all users: ~$1/day at $0.008/call
    video_daily_limit: int           # video files per user per day
    sends_daily_limit: int           # audio sends per user per day (anti-abuse for "Not it"/"Similar" chains)
    # SSRF filter: yt-dlp only goes to allowed hosts
    # (without the filter any URL = a request from the container to anywhere)
    url_allowed_hosts: tuple
    data_retention_days: int         # retention for queries/feedback (personal data)


def _parse_ids(raw: str) -> tuple:
    """ALLOWED_USER_IDS="123, 456" -> (123, 456). Garbage entries are skipped silently."""
    out = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return tuple(out)


def load_config() -> Config:
    load_dotenv()  # .env in the project root (local runs; in docker env comes from env_file)
    return Config(
        bot_token=os.environ["BOT_TOKEN"],
        bothub_api_key=os.environ["BOTHUB_API_KEY"],
        bothub_base=os.getenv("BOTHUB_BASE", "https://openai.bothub.chat/v1"),
        llm_model=os.getenv("LLM_MODEL", "deepseek-v4-pro"),
        llm_model_sonar=os.getenv("LLM_MODEL_SONAR", "sonar"),
        max_concurrent_downloads=int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "2")),
        temp_ttl_minutes=int(os.getenv("TEMP_TTL_MINUTES", "30")),
        user_rate_limit_per_hour=int(os.getenv("USER_RATE_LIMIT_PER_HOUR", "20")),
        lastfm_api_key=os.getenv("LASTFM_API_KEY", ""),
        # LANGFUSE_BASE_URL is the name used in the Langfuse UI; the SDK expects LANGFUSE_HOST, accept both
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        langfuse_host=os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "")),
        langfuse_environment=os.getenv("LANGFUSE_ENVIRONMENT", "development"),
        # or 0: an empty TG_API_ID= (from .env.example) would otherwise raise ValueError at startup
        tg_api_id=int(os.getenv("TG_API_ID") or 0),
        tg_api_hash=os.getenv("TG_API_HASH", ""),
        local_bot_api_url=os.getenv("LOCAL_BOT_API_URL", ""),
        allowed_user_ids=_parse_ids(os.getenv("ALLOWED_USER_IDS", "")),
        search_daily_limit=int(os.getenv("SEARCH_DAILY_LIMIT", "200")),
        url_daily_limit=int(os.getenv("URL_DAILY_LIMIT", "200")),
        global_daily_search_budget=int(os.getenv("GLOBAL_DAILY_SEARCH_BUDGET", "1000")),
        fact_daily_limit=int(os.getenv("FACT_DAILY_LIMIT", "2")),
        global_daily_fact_budget=int(os.getenv("GLOBAL_DAILY_FACT_BUDGET", "125")),
        video_daily_limit=int(os.getenv("VIDEO_DAILY_LIMIT", "30")),
        sends_daily_limit=int(os.getenv("SENDS_DAILY_LIMIT", "100")),
        url_allowed_hosts=tuple(h.strip().lower() for h in
                                os.getenv("URL_ALLOWED_HOSTS",
                                          "youtube.com,youtu.be,music.youtube.com,"
                                          "soundcloud.com,bandcamp.com,vk.com").split(",") if h.strip()),
        data_retention_days=int(os.getenv("DATA_RETENTION_DAYS", "180")),
    )


def ensure_dirs() -> None:
    for d in (DATA_DIR, TEMP_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
