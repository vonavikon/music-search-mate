"""Tests for the per-pipeline model config (LLM_MODEL_PIPELINES, bake-off
2026-11-04: model swaps must be config-only).

Contract under test:
  - config parsing (config.parse_pipeline_models via load_config): a full
    mapping carries every intent; a PARTIAL mapping fills the missing intents
    with the global llm_model; absent/empty env maps every pipeline to the
    global model (zero behavior change). Garbage entries are skipped.
  - the pipeline_model helper (services.llm_search): normalization matches
    routing ("Similar Tracks" -> the similar key); a configured-but-unlisted
    intent returns the global default; unknown intents and None return None
    (the client's own model); an absent env returns None everywhere.
  - chat_json model override called-through: the request body carries the
    override for that single call, the client's own model otherwise.
  - e2e wiring: with LLM_MODEL=model-global, LLM_MODEL_PIPELINES=similar=
    model-sim and a client built on model-global (as load_config does), the
    llm-extract call runs on model-global (the intent does not exist before
    LLM#1 answers) and the llm-rank call inside the shared _rank_and_decide
    runs on model-sim — two spied calls, two distinct configured models.

Mocking follows tests/test_llm_search.py: chat_json patched on the
bothub_client fixture (already-parsed dicts, dispatched by gen_name),
search_metadata patched in services.llm_search, no network.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from config import load_config, parse_pipeline_models
from services.llm_search import pipeline_model, search_candidates


_ALL_GLOBAL = {"this_track": "global-m", "similar": "global-m", "mood": "global-m",
               "from_source": "global-m", "surprise": "global-m"}


def _env_only(monkeypatch, llm_model=None, pipelines=None):
    """Deterministic env for load_config: .env loading disabled, the two vars
    under test pinned (a developer's shell or repo .env must not leak in),
    the two required keys present."""
    monkeypatch.setattr("config.load_dotenv", lambda: None)
    monkeypatch.setenv("BOT_TOKEN", "t")
    monkeypatch.setenv("BOTHUB_API_KEY", "k")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL_PIPELINES", raising=False)
    if llm_model is not None:
        monkeypatch.setenv("LLM_MODEL", llm_model)
    if pipelines is not None:
        monkeypatch.setenv("LLM_MODEL_PIPELINES", pipelines)


# ════ 1. config parsing ════

def test_config_parsing_full(monkeypatch):
    """All five intents configured: the dict carries each model verbatim."""
    _env_only(monkeypatch, llm_model="global-m", pipelines=(
        "this_track=m1,similar=m2,mood=m3,from_source=m4,surprise=m5"))
    cfg = load_config()
    assert cfg.llm_model == "global-m"
    assert cfg.llm_model_pipelines == {
        "this_track": "m1", "similar": "m2", "mood": "m3",
        "from_source": "m4", "surprise": "m5",
    }


def test_config_parsing_partial_defaults_to_global(monkeypatch):
    """A partial mapping: only similar is swapped, every other intent (and
    the global field) keeps the global model."""
    _env_only(monkeypatch, llm_model="global-m", pipelines="similar=m2")
    cfg = load_config()
    assert cfg.llm_model == "global-m"
    assert cfg.llm_model_pipelines == {**_ALL_GLOBAL, "similar": "m2"}


def test_config_parsing_absent_all_global(monkeypatch):
    """No LLM_MODEL_PIPELINES in the env: every pipeline equals the global
    model — the config-only swap changes nothing until it is set."""
    _env_only(monkeypatch, llm_model="global-m")
    cfg = load_config()
    assert cfg.llm_model_pipelines == _ALL_GLOBAL


def test_config_parsing_empty_string_all_global(monkeypatch):
    """LLM_MODEL_PIPELINES= (empty): same as absent — all global."""
    _env_only(monkeypatch, llm_model="global-m", pipelines="")
    assert load_config().llm_model_pipelines == _ALL_GLOBAL


def test_parse_pipeline_models_skips_garbage():
    """Garbage entries (no '=', empty side) are skipped silently, the same
    tolerance as _parse_ids; keys are lowercased to the pipeline names."""
    assert parse_pipeline_models(
        "SIMILAR=m2, mood = m3 ,bogus,,=x,noequals", "global-m") == {
        **_ALL_GLOBAL, "similar": "m2", "mood": "m3",
    }


# ════ 2. the pipeline_model helper ════

def test_pipeline_model_normalized_intent(monkeypatch):
    """The lookup key is the NORMALIZED intent (the same key route() and
    rank_prompt_for use): a variant spelling hits the configured model."""
    monkeypatch.setenv("LLM_MODEL", "global-m")
    monkeypatch.setenv("LLM_MODEL_PIPELINES", "similar=m2")
    assert pipeline_model("similar") == "m2"
    assert pipeline_model("Similar Tracks") == "m2"
    assert pipeline_model(" SIMILAR_MUSIC ") == "m2"


def test_pipeline_model_unlisted_intent_gets_global(monkeypatch):
    """A configured-but-unlisted intent falls back to the global model (the
    parser fills it); unknown intents and None map to None = the client's
    own model, no override."""
    monkeypatch.setenv("LLM_MODEL", "global-m")
    monkeypatch.setenv("LLM_MODEL_PIPELINES", "similar=m2")
    assert pipeline_model("mood") == "global-m"
    assert pipeline_model("unknown_intent") is None
    assert pipeline_model(None) is None


def test_pipeline_model_absent_env_is_none(monkeypatch):
    """No LLM_MODEL_PIPELINES: None for every intent — the client's own model
    everywhere (zero behavior change)."""
    monkeypatch.delenv("LLM_MODEL_PIPELINES", raising=False)
    assert pipeline_model("similar") is None
    assert pipeline_model("this_track") is None


# ════ 3. chat_json model override called-through (request body) ════

@pytest.mark.asyncio
async def test_chat_json_model_override_in_request_body(bothub_client):
    """The real chat_json against a mocked HTTP layer: the request body's
    "model" field carries the override for THAT call only — the calls before
    and after keep the client's own model."""
    assert bothub_client._model == "fake-model"
    content = json.dumps({"ok": True})

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    with patch.object(bothub_client._http, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _FakeResp()

        await bothub_client.chat_json("s", "u")                          # client default
        await bothub_client.chat_json("s", "u", model="override-m")     # single override
        await bothub_client.chat_json("s", "u")                          # default again

    bodies = [call.kwargs["json"] for call in mock_post.await_args_list]
    assert [b["model"] for b in bodies] == ["fake-model", "override-m", "fake-model"]


# ════ 4. e2e: extract keeps the global model, rank takes the pipeline's ════

@pytest.mark.asyncio
async def test_e2e_similar_extract_global_rank_per_pipeline(bothub_client, monkeypatch):
    """End-to-end similar search with LLM_MODEL=model-global and
    LLM_MODEL_PIPELINES=similar=model-sim, the client built on model-global
    (exactly what load_config + main.py do): the extract call runs on
    model-global — the intent does not exist before LLM#1 answers — and the
    rank call inside the shared _rank_and_decide runs on model-sim."""
    monkeypatch.setenv("LLM_MODEL", "model-global")
    monkeypatch.setenv("LLM_MODEL_PIPELINES", "similar=model-sim")
    bothub_client._model = "model-global"  # emulate load_config wiring

    extract = {
        "intro": "Ищем в этом вайбе...",
        "intent": "similar",
        "signals": {"language": "ru", "genre": "rock"},
        "target": {"artist": None, "track": None},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": ["test query 1", "test query 2"],
    }
    rank = {
        "ranked": [
            {"video_id": "v1", "score": 0.95, "rationale": "Отличный грув."},
            {"video_id": "v2", "score": 0.80, "rationale": "Близко по жанру."},
        ],
        "best_index": 0,
        "confidence": "high",
        "verdict": "Этот трек — то что надо.",
        "need_clarify": False,
        "clarify_question": "",
    }
    entries = [
        {"video_id": "v1", "title": "Cool Rock Song", "channel": "Rock Band",
         "duration": 240, "views": 500000},
        {"video_id": "v2", "title": "Another Rock Track", "channel": "Other Band",
         "duration": 300, "views": 100000},
    ]
    captured: list[tuple[str, str]] = []

    async def spy_chat(system, user, max_tokens=None, trace=None,
                       gen_name="llm", model=None):
        # record the EFFECTIVE model: the override or the client's own
        captured.append((gen_name, model or bothub_client._model))
        return extract if gen_name == "llm-extract" else rank

    with patch("services.search._extract_info_no_dl", return_value=None):
        with patch.object(bothub_client, "chat_json", side_effect=spy_chat):
            with patch("services.llm_search.search_metadata", return_value=entries):
                out = await search_candidates("песня как Nights in White Satin",
                                               bothub_client)

    assert out.intent == "similar"
    assert out.candidates[out.best_index].video_id == "v1"
    # two spied calls, two distinct configured models
    assert captured == [("llm-extract", "model-global"), ("llm-rank", "model-sim")]
