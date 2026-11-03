"""Pytest fixtures providing a mocked test pipeline."""
from dataclasses import dataclass
from pathlib import Path

import pytest
from telegram._bot import Bot
from telegram.ext import ApplicationBuilder
from telegram.request import BaseRequest

from config import Config


@dataclass
class RecordedMessage:
    method: str
    kwargs: dict
    text_sent: str = ""


class RecordingRequest(BaseRequest):
    def __init__(self, token: str = "test:token"):
        super().__init__()
        self._token = token
        self.sent: list[RecordedMessage] = []

    async def do_request(self, *args, **kwargs):
        return 200, b"{}"

    async def _request_wrapper(self, *args, **kwargs):
        return 200, b"{}"

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    @property
    def base_url(self) -> str:
        return "http://fake-telegram.local/bot"

    @property
    def base_file_url(self) -> str:
        return "http://fake-telegram.local/file/bot"

    @property
    def token(self) -> str:
        return self._token

    # PTB 22.x: BaseRequest gained abstract timeout properties — stub them
    @property
    def read_timeout(self):
        return 5.0

    @property
    def write_timeout(self):
        return 5.0

    @property
    def connect_timeout(self):
        return 5.0

    @property
    def pool_timeout(self):
        return 1.0

    @property
    def media_write_timeout(self):
        return 20.0

    async def post(self, url: str, request_data=None, read_timeout=None,
                   write_timeout=None, connect_timeout=None, pool_timeout=None):
        """Core hook: intercepts bot._post and records the call."""
        endpoint = url.rsplit("/", 1)[-1] if "/" in url else url

        data = {}
        if request_data and hasattr(request_data, "parameters"):
            params = request_data.parameters  # property returning a dict {name: value}
            if isinstance(params, dict):
                data = dict(params)
            else:
                for p in params:
                    try:
                        data[p.name] = p.value
                    except Exception:
                        data[p.name] = str(p.value)

        text = str(data.get("text", data.get("caption", "")))
        self.sent.append(RecordedMessage(
            method=endpoint,
            kwargs=data,
            text_sent=text,
        ))
        if endpoint == "sendMessage":
            return {
                "message_id": 999, "date": 1,
                "chat": {"id": 123, "type": "private"},
                "text": text,
            }
        if endpoint == "sendAudio":
            return {
                "message_id": 1000, "date": 1,
                "chat": {"id": 123, "type": "private"},
                "audio": {"file_id": "fake_audio_id", "duration": 180},
            }
        if endpoint in ("editMessageText", "editMessageCaption"):
            return True
        if endpoint == "answerCallbackQuery":
            return True
        if endpoint == "getMe":
            return {
                "id": 123456, "is_bot": True, "first_name": "TestBot",
                "username": "test_bot",
            }
        return True

    def find(self, method_prefix: str) -> list[RecordedMessage]:
        return [m for m in self.sent if m.method.startswith(method_prefix)]

    def texts(self) -> list[str]:
        return [m.text_sent for m in self.sent if m.text_sent]


@pytest.fixture
def fake_config():
    return Config(
        bot_token="test:fake_token",
        bothub_api_key="fake-bothub-key",
        bothub_base="https://fake-bothub.local/v1",
        llm_model="fake-model",
        llm_model_sonar="",
        max_concurrent_downloads=2,
        temp_ttl_minutes=30,
        user_rate_limit_per_hour=100,
        lastfm_api_key="",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="",
        langfuse_environment="development",
        allowed_user_ids=(),
        search_daily_limit=500,
        url_daily_limit=500,
        global_daily_search_budget=1000,
        fact_daily_limit=2,
        global_daily_fact_budget=125,
        video_daily_limit=30,
        sends_daily_limit=100,
        url_allowed_hosts=("youtube.com", "youtu.be", "soundcloud.com", "bandcamp.com", "vk.com"),
        data_retention_days=180,
        tg_api_id=0,
        tg_api_hash="",
        local_bot_api_url="",
    )


@pytest.fixture
def recording():
    return RecordingRequest()


@pytest.fixture
async def bot(fake_config, recording):
    b = Bot(token=fake_config.bot_token, request=recording)
    yield b
    try:
        await b.shutdown()
    except Exception:
        pass


@pytest.fixture
async def app(fake_config, bot):
    builder = ApplicationBuilder()
    builder.bot(bot)
    builder.updater(None)
    app = builder.build()
    await app.initialize()
    yield app
    try:
        await app.shutdown()
    except Exception:
        pass


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    db_file = tmp_path / "test-music-search.db"
    monkeypatch.setattr("config.DB_PATH", str(db_file))
    monkeypatch.setattr("config.DATA_DIR", str(tmp_path))
    return db_file


@pytest.fixture
def bothub_client(fake_config):
    from services.llm_search import BothubClient
    return BothubClient(
        fake_config.bothub_base,
        fake_config.bothub_api_key,
        fake_config.llm_model,
    )
