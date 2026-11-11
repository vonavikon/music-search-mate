"""Configuration from env. Paths to data/temp/logs."""
import os
from dataclasses import dataclass, field
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
    bot_owner_id: int  # owner-only commands (/stats); 0 = unset (the allowlist gate stays the only check)
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
    # Per-pipeline LLM models (bake-off 2026-11-04: swaps must be config-only).
    # Maps the five canonical pipeline intents to model names; the global
    # llm_model is the default for missing keys. Empty = every pipeline runs
    # on the global model (zero behavior change).
    llm_model_pipelines: dict = field(default_factory=dict)


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


# The default global model (LLM_MODEL). Kept as a named constant so
# load_config and the call-time pipeline_model helper (services/llm_search)
# share one source for the fallback value.
LLM_MODEL_DEFAULT = "deepseek-v4-pro"

# The five intent pipelines (services/pipelines/) that accept a per-pipeline
# LLM model override via LLM_MODEL_PIPELINES. The legacy fallback and unknown
# intents always run on the global llm_model.
PIPELINE_MODEL_INTENTS = ("this_track", "similar", "mood", "from_source", "surprise")


def parse_pipeline_models(raw: str, default: str) -> dict:
    """LLM_MODEL_PIPELINES="this_track=m1,similar=m2" -> {intent: model} for
    the five pipeline intents, missing keys filled with the global `default`
    (empty/absent env -> every pipeline equals the global model). Garbage
    entries (no '=' or an empty side) are skipped silently, the same
    tolerance as _parse_ids; keys are lowercased to match the normalized
    pipeline names."""
    overrides: dict = {}
    for part in (raw or "").replace(";", ",").split(","):
        name, sep, model = part.partition("=")
        name, model = name.strip().lower(), model.strip()
        if sep and name and model:
            overrides[name] = model
    return {intent: overrides.get(intent, default) for intent in PIPELINE_MODEL_INTENTS}


def load_config() -> Config:
    load_dotenv()  # .env in the project root (local runs; in docker env comes from env_file)
    llm_model = os.getenv("LLM_MODEL", LLM_MODEL_DEFAULT)
    return Config(
        bot_token=os.environ["BOT_TOKEN"],
        bothub_api_key=os.environ["BOTHUB_API_KEY"],
        bothub_base=os.getenv("BOTHUB_BASE", "https://openai.bothub.chat/v1"),
        llm_model=llm_model,
        # Per-pipeline overrides; missing keys are the global model above.
        llm_model_pipelines=parse_pipeline_models(
            os.getenv("LLM_MODEL_PIPELINES", ""), llm_model),
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
        # or 0: an empty BOT_OWNER_ID= would otherwise raise ValueError at startup
        bot_owner_id=int(os.getenv("BOT_OWNER_ID") or 0),
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
