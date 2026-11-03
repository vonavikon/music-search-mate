"""Тесты веб-API (tests/web/). Без сети: движок и метаданные замоканы."""
import asyncio

import pytest
from fastapi.testclient import TestClient

import web.app as web_app
import web.pipeline as web_pipeline
from services.llm_search import Candidate, SearchOut
from services.metadata import MetaOut


def _cand(vid="abc12345678", title="Artist - Track", channel="Artist - Topic",
          duration=240, views=1000, score=0.9, source="youtube",
          year="", album=""):
    return Candidate(vid, title, channel, duration, views, score,
                     source=source, canonical_year=year, canonical_album=album)


def _out(**kw):
    base = dict(intent="this_track", signals={}, queries=["q"], assoc=[],
                candidates=[], best_index=-1, confidence="high")
    base.update(kw)
    return SearchOut(**base)


@pytest.fixture
def engine(monkeypatch):
    """Подмена search_candidates: записывает аргументы, возвращает заготовленный SearchOut."""
    state = {"out": _out(), "raise": None, "calls": []}

    async def fake_search(description, bothub, context=None, excluded_ids=None,
                          on_progress=None, metadata_svc=None, trace=None):
        state["calls"].append({
            "description": description,
            "context": context,
            "excluded_ids": excluded_ids,
            "metadata_svc": metadata_svc,
        })
        if state["raise"] is not None:
            raise state["raise"]
        return state["out"]

    monkeypatch.setattr(web_pipeline, "search_candidates", fake_search)
    return state


@pytest.fixture
def card_meta(monkeypatch):
    """Подмена каскада карточки (Last.fm/iTunes): без сети."""
    state = {"meta": None}

    async def fake_resolve(self, info, video_id):
        return state["meta"]

    monkeypatch.setattr("services.metadata.MetadataService.resolve", fake_resolve)
    return state


@pytest.fixture
def client(monkeypatch):
    """Приложение с пустыми ключами (детерминированный /api/health) и без рейт-лимитов."""
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("BOTHUB_API_KEY", "")
    monkeypatch.setenv("LASTFM_API_KEY", "")
    monkeypatch.setenv("WEB_RATE_LIMIT_PER_HOUR", "100000")
    app = web_app.create_app()
    with TestClient(app) as c:
        yield c


# ---------- /api/health ----------

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["llm"] is False   # BOTHUB_API_KEY пуст
    assert body["lastfm"] is False


def test_health_llm_flag(monkeypatch, engine, card_meta):
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("BOTHUB_API_KEY", "some-key")
    monkeypatch.setenv("LASTFM_API_KEY", "")
    monkeypatch.setenv("WEB_RATE_LIMIT_PER_HOUR", "100000")
    with TestClient(web_app.create_app()) as c:
        body = c.get("/api/health").json()
    assert body["llm"] is True and body["lastfm"] is False


# ---------- /api/search: маппинг статусов ----------

def test_search_ok(client, engine, card_meta):
    engine["out"] = _out(candidates=[_cand(year="1998", album="Debut")], best_index=0)
    r = client.get("/api/search", params={"q": "французский рэп 90х"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["confidence"] == "high"
    assert body["intent"] == "this_track"
    assert body["question"] == "" and body["picks"] == []
    card = body["card"]
    assert card["video_id"] == "abc12345678"
    assert card["duration"] == 240 and card["views"] == 1000
    assert card["score"] == 0.9 and card["source"] == "youtube"
    assert card["year"] == "1998" and card["album"] == "Debut"
    # enrichment не дал artist -> канал без "- Topic"; artwork -> ytimg fallback
    assert card["artist"] == "Artist"
    assert card["artwork_url"] == "https://i.ytimg.com/vi/abc12345678/hqdefault.jpg"
    assert isinstance(body["took_ms"], int) and body["took_ms"] >= 0


def test_search_ok_with_meta(client, engine, card_meta):
    card_meta["meta"] = MetaOut(artist="Fonky Family", track="Mystère",
                                album="Si Dieu Veut", year="1998",
                                bio="Французская рэп-группа из Марселя. " * 20,
                                artwork_url="https://itunes/x600.jpg")
    engine["out"] = _out(candidates=[_cand()], best_index=0)
    body = client.get("/api/search", params={"q": "mystere"}).json()
    card = body["card"]
    assert card["artist"] == "Fonky Family"
    assert card["artwork_url"] == "https://itunes/x600.jpg"
    assert card["album"] == "Si Dieu Veut" and card["year"] == "1998"
    assert len(card["bio"]) <= 241 and card["bio"].endswith("…")


def test_search_clarify(client, engine, card_meta):
    engine["out"] = _out(need_clarify=True, clarify_question="Похожее на что?")
    body = client.get("/api/search", params={"q": "похожее"}).json()
    assert body["status"] == "clarify"
    assert body["question"] == "Похожее на что?"
    assert body["card"] is None and body["picks"] == []


def test_search_blocked(client, engine, card_meta):
    engine["out"] = _out(block="drugs")
    body = client.get("/api/search", params={"q": "как приготовить"}).json()
    assert body["status"] == "blocked"
    assert body["card"] is None and body["question"] == ""


def test_search_degraded(client, engine, card_meta):
    engine["out"] = _out(candidates=[_cand(vid=f"v{i}") for i in range(12)],
                         best_index=0, degraded=True, confidence="low")
    body = client.get("/api/search", params={"q": "beatles"}).json()
    assert body["status"] == "degraded"
    assert body["card"] is None
    assert len(body["picks"]) == 8  # максимум 8
    assert set(body["picks"][0]) == {"video_id", "title", "channel", "duration", "artwork_url"}


def test_search_empty(client, engine, card_meta):
    engine["out"] = _out(candidates=[], best_index=-1)
    body = client.get("/api/search", params={"q": "несуществующий трек"}).json()
    assert body["status"] == "empty"
    assert body["card"] is None and body["picks"] == []


def test_search_pick_list(client, engine, card_meta):
    engine["out"] = _out(candidates=[_cand(vid=f"p{i}", score=0.3) for i in range(10)],
                         best_index=-1)
    body = client.get("/api/search", params={"q": "российская электроника"}).json()
    assert body["status"] == "pick_list"
    assert body["card"] is None
    assert len(body["picks"]) == 8


def test_search_soundcloud_no_ytimg(client, engine, card_meta):
    engine["out"] = _out(candidates=[_cand(source="soundcloud", vid="sc-1")], best_index=0)
    body = client.get("/api/search", params={"q": "sc track"}).json()
    assert body["card"]["artwork_url"] == ""  # ytimg для soundcloud не подходит


# ---------- прокидывание параметров в движок ----------

def test_search_exclude_and_context(client, engine, card_meta):
    engine["out"] = _out(candidates=[], best_index=-1)
    r = client.get("/api/search", params={"q": "ещё", "context": "Daft Punk",
                                          "exclude": "vid1, vid2,,vid3"})
    assert r.status_code == 200
    call = engine["calls"][0]
    assert call["excluded_ids"] == {"vid1", "vid2", "vid3"}
    # формат контекста движка (как recent_context в handler.py)
    assert call["context"] == [{"description": "Daft Punk", "intent": None,
                                "best_title": "", "ts": call["context"][0]["ts"],
                                "outcome": "none"}]
    assert isinstance(call["metadata_svc"], object)  # движок получает сервис


def test_search_no_context_no_exclude(client, engine, card_meta):
    engine["out"] = _out(candidates=[], best_index=-1)
    client.get("/api/search", params={"q": "test"})
    call = engine["calls"][0]
    assert call["context"] is None and call["excluded_ids"] is None


# ---------- валидация и ошибки ----------

def test_search_empty_q_400(client, engine):
    r = client.get("/api/search", params={"q": "   "})
    assert r.status_code == 400 and "detail" in r.json()


def test_search_missing_q_400(client, engine):
    r = client.get("/api/search")
    assert r.status_code == 400


def test_search_too_long_q_400(client, engine):
    r = client.get("/api/search", params={"q": "a" * 501})
    assert r.status_code == 400 and "detail" in r.json()


def test_search_lang_en(client, engine):
    r = client.get("/api/search", params={"q": "", "lang": "en"})
    assert r.status_code == 400
    assert r.json()["detail"] == "Empty query. Describe what to search for."


def test_search_lang_ru_default(client, engine):
    r = client.get("/api/search", params={"q": ""})
    assert r.status_code == 400
    assert r.json()["detail"] == "Пустой запрос. Опиши, что искать."


def test_search_internal_error_500(client, engine, card_meta):
    engine["raise"] = RuntimeError("boom")
    r = client.get("/api/search", params={"q": "anything"})
    assert r.status_code == 500
    assert "detail" in r.json() and "boom" not in r.json()["detail"]


def test_search_timeout_503(client, engine, card_meta, monkeypatch):
    async def slow(description, bothub, **kw):
        await asyncio.sleep(0.3)
        return _out()

    monkeypatch.setattr(web_pipeline, "search_candidates", slow)
    monkeypatch.setattr(web_app, "SEARCH_TIMEOUT_SEC", 0.05)
    r = client.get("/api/search", params={"q": "slow"})
    assert r.status_code == 503 and "detail" in r.json()


# ---------- rate limit ----------

def test_rate_limit(monkeypatch, engine, card_meta):
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("BOTHUB_API_KEY", "")
    monkeypatch.setenv("WEB_RATE_LIMIT_PER_HOUR", "3")
    engine["out"] = _out(candidates=[], best_index=-1)
    with TestClient(web_app.create_app()) as c:
        codes = [c.get("/api/search", params={"q": "x"}).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        assert "detail" in c.get("/api/search", params={"q": "x"}).json()


def test_rate_limit_not_shared_between_apps(monkeypatch, engine, card_meta):
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("BOTHUB_API_KEY", "")
    monkeypatch.setenv("WEB_RATE_LIMIT_PER_HOUR", "2")
    engine["out"] = _out(candidates=[], best_index=-1)
    with TestClient(web_app.create_app()) as c1:
        assert c1.get("/api/search", params={"q": "x"}).status_code == 200
        assert c1.get("/api/search", params={"q": "x"}).status_code == 200
        assert c1.get("/api/search", params={"q": "x"}).status_code == 429
    # новый инстанс приложения = новое окно
    with TestClient(web_app.create_app()) as c2:
        assert c2.get("/api/search", params={"q": "x"}).status_code == 200


# ---------- статика ----------

def test_static_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Music Search Mate" in r.text


def test_api_not_shadowed_by_static(client, engine):
    r = client.get("/api/search", params={"q": ""})
    assert r.status_code == 400  # api отвечает, а не раздаёт index.html
