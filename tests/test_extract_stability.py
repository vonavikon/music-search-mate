"""Extract stability (acceptance 17.11): the SAME self-contained text must
read the SAME way across runs.

Two deterministic layers over LLM#1 in services/llm_search.py:
  1. the extract-verdict cache — the first successful reading of a fresh
     text is persisted as a fact ("xtract", <key>, "cls") and every later
     identical text reuses it verbatim, LLM#1 skipped. Contextual
     refinements (context / excluded_* / source_anchor) are never cached;
     block and clarify verdicts are never cached and never served;
     store=None and every cache error degrade to the plain LLM#1 flow;
  2. the text variant pre-detector (_text_variant) — a variant modifier
     NAMED IN THE USER'S TEXT overrides the extract's target.variant (the
     text wins in one direction only: no marker in the text -> the LLM's
     variant stands).

Conventions follow tests/test_variant_modifiers.py: BothubClient.chat_json
is mocked with a side_effect returning ALREADY parsed dicts;
services.llm_search.search_metadata is patched; the this_track
confident-match shortcut is disabled (FUZZY_SHORTCUT_MIN) so the LLM#2
rank path runs; the store stub mirrors the services.store
get_fact/save_fact contract (see _sonar_ground_cached in
services/pipelines/from_source.py). No network.
"""
import copy
import json
import logging
from unittest.mock import patch

import pytest

from services.llm_search import (
    _extract_cache_key,
    _extract_verdict_cacheable,
    _text_variant,
    search_candidates,
)


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Enrich step offline (same as tests/test_llm_search.py): the top-5
    enrich would be a real yt-dlp call; the stub returns None."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers ════

def _extract_json(intent="this_track", target=None, queries=None, block=None):
    """LLM#1 extract — a fake already-parsed answer."""
    out = {
        "intro": "Ищу редакт-версию трека...",
        "intent": intent,
        "signals": {"language": "en", "era": None, "vocal": None,
                    "genre": None, "mood": None},
        "target": target or {"artist": None, "track": None, "variant": None},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries if queries is not None else ["timbaland the way i are"],
    }
    if block is not None:
        out["block"] = block
    return out


def _rank_json(scores, best_index=0):
    """LLM#2 rank — scores is a list of (video_id, score) in ranked order."""
    return {
        "ranked": [{"video_id": vid, "score": sc, "rationale": "ok"}
                   for vid, sc in scores],
        "best_index": best_index,
        "confidence": "high",
        "verdict": "Timbaland — The Way I Are (2007).",
        "need_clarify": False,
        "clarify_question": "",
    }


def _way_i_are_pool():
    """The incident pool (test_variant_modifiers.py): the official original
    plus slowed/sped-up fan edits of the same record."""
    return [
        {"video_id": "v1", "title": "Timbaland - The Way I Are",
         "channel": "TimbalandVEVO", "duration": 250, "views": 90_000_000},
        {"video_id": "v2", "title": "Timbaland - The Way I Are (Slowed)",
         "channel": "Chill Vibes", "duration": 260, "views": 500_000},
        {"video_id": "v3", "title": "Timbaland - The Way I Are (Sped Up)",
         "channel": "Fan Edits", "duration": 200, "views": 300_000},
    ]


class _MemoryStore:
    """The services.store contract the cache needs (get_fact/save_fact, the
    signatures _sonar_ground_cached uses), in memory: the first successful
    verdict persists, the second run hits. read_error/write_error make the
    corresponding op raise — the cache must degrade, never break search."""

    def __init__(self, read_error=None, write_error=None):
        self.facts = {}
        self.get_calls = []
        self.save_calls = []
        self._read_error = read_error
        self._write_error = write_error

    async def get_fact(self, artist, track, lang):
        self.get_calls.append((artist, track, lang))
        if self._read_error:
            raise self._read_error
        return self.facts.get((artist, track, lang))

    async def save_fact(self, artist, track, lang, fact):
        self.save_calls.append((artist, track, lang, fact))
        if self._write_error:
            raise self._write_error
        self.facts[(artist, track, lang)] = fact


def _chat_stub(extract_answer, calls):
    """A chat_json side_effect: llm-extract -> the given answer (a single
    dict, or a callable for per-call variance), everything else -> a plain
    rank answer. Records gen_name into `calls`."""

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        calls.append(gen_name)
        if gen_name == "llm-extract":
            return extract_answer() if callable(extract_answer) else extract_answer
        return _rank_json([("v1", 0.95), ("v2", 0.6), ("v3", 0.5)],
                          best_index=0)

    return fake_chat


async def _run(desc, bothub, store=None, **kwargs):
    """search_candidates with the offline conventions of this module."""
    with patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = _way_i_are_pool()
        return await search_candidates(desc, bothub, store=store, **kwargs)


# ════ 1. the key and the verdict-eligibility contract (units) ════

def test_extract_cache_key_folds_text_and_marks_context():
    """The key is "xv1:" + sha1 of the fuzzy-normalized text: case,
    punctuation and whitespace fold (the same text always hits). Every
    contextual argument adds the "|ctx" suffix — such keys are structurally
    unequal to fresh keys and are never read or written."""
    fresh = _extract_cache_key("Black Coffee and Cifgarute!")
    assert fresh.startswith("xv1:")
    assert "|ctx" not in fresh
    assert _extract_cache_key("  black coffee   AND cifgarute ") == fresh
    ctx = [{"description": "песня из фильма", "outcome": "hit"}]
    assert _extract_cache_key("Black Coffee and Cifgarute!",
                              context=ctx).startswith("xv1:")
    assert _extract_cache_key("Black Coffee and Cifgarute!",
                              context=ctx) == fresh + "|ctx"
    assert _extract_cache_key("Black Coffee and Cifgarute!",
                              excluded_ids={"v1"}) == fresh + "|ctx"
    assert _extract_cache_key("Black Coffee and Cifgarute!",
                              excluded_artists={"a"}) == fresh + "|ctx"
    assert _extract_cache_key("Black Coffee and Cifgarute!",
                              excluded_titles={"t"}) == fresh + "|ctx"
    assert _extract_cache_key("Black Coffee and Cifgarute!",
                              source_anchor="Мадагаскар 2") == fresh + "|ctx"
    # no text -> no key (a stop-signal, not a key)
    assert _extract_cache_key("") == ""
    assert _extract_cache_key("   ") == ""


def test_extract_verdict_cacheable_matrix():
    """Usable search answers cache (any intent, surprise's empty queries
    included); block and clarify verdicts never enter the cache."""
    # a normal verdict
    assert _extract_verdict_cacheable(_extract_json()) is True
    # block null-words are the normal "no category"
    assert _extract_verdict_cacheable(_extract_json(block="null")) is True
    # a real destructive category: fresh classification every time
    assert _extract_verdict_cacheable(_extract_json(block="drugs")) is False
    # clarify verdict (no queries, no target) on a non-surprise intent
    assert _extract_verdict_cacheable(
        _extract_json(intent="mood", queries=[])) is False
    # surprise legitimately carries empty queries — cacheable
    assert _extract_verdict_cacheable(
        _extract_json(intent="surprise", queries=[])) is True
    # degradation / junk
    assert _extract_verdict_cacheable(None) is False
    assert _extract_verdict_cacheable({}) is False
    assert _extract_verdict_cacheable(["junk"]) is False


# ════ 2. the cache roundtrip: the same text pins its reading ════

async def test_cache_roundtrip_second_run_skips_llm1(bothub_client, caplog):
    """THE flake fix. Run 1 (cold cache): LLM#1 classifies once and the
    verdict is saved under ("xtract", <xv1 key>, "cls") as JSON. Run 2,
    SAME text: the verdict comes from the cache — LLM#1 is never called
    (only llm-rank remains), a hit is not re-saved, and the result is
    identical (intent/target/queries/signals from the cache)."""
    store = _MemoryStore()
    calls = []
    extract_answer = _extract_json(
        target={"artist": "Timbaland", "track": "The Way I Are",
                "variant": "slowed"},
        queries=["timbaland the way i are slowed"])
    # the cache stores the VERBATIM LLM verdict, before the deterministic
    # layers (variant mirroring etc.) mutate their shared dicts downstream
    verbatim = copy.deepcopy(extract_answer)

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_stub(extract_answer, calls)):
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out1 = await _run("timbaland the way i are slowed",
                              bothub_client, store=store)

    # run 1: exactly one extract call, the verdict persisted
    assert calls.count("llm-extract") == 1
    assert len(store.save_calls) == 1
    artist, key, lang, payload = store.save_calls[0]
    assert (artist, lang) == ("xtract", "cls")
    assert key.startswith("xv1:") and "|ctx" not in key
    assert json.loads(payload) == verbatim
    assert store.get_calls == [(artist, key, lang)]
    # no hit yet — the first run was the write
    assert not any("extract verdict cache hit" in r.getMessage()
                   for r in caplog.records)

    calls.clear()
    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_stub(extract_answer, calls)):
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out2 = await _run("timbaland the way i are slowed",
                              bothub_client, store=store)

    # run 2: LLM#1 skipped, LLM#2 still ranks, the hit is logged, no re-save
    assert "llm-extract" not in calls
    assert calls.count("llm-rank") == 1
    assert any("extract verdict cache hit" in r.getMessage()
               for r in caplog.records)
    assert len(store.save_calls) == 1
    # identical reading: intent/target/queries/signals come from the cache
    assert out2.intent == out1.intent == "this_track"
    assert out2.target == out1.target
    assert out2.queries == out1.queries
    assert out2.signals.get("variant") == out1.signals.get("variant") \
        == "slowed"
    assert [c.video_id for c in out2.candidates] == \
        [c.video_id for c in out1.candidates]


async def test_contextual_calls_bypass_the_cache(bothub_client):
    """A contextual refinement (context= or excluded_ids=) is never READ and
    never WRITTEN: its meaning depends on the dialog state, so every such
    call classifies fresh."""
    store = _MemoryStore()
    calls = []
    ctx = [{"description": "песня из фильма", "outcome": "hit"}]

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_stub(_extract_json(), calls)):
        await _run("timbaland the way i are", bothub_client, store=store,
                   context=ctx)
        await _run("timbaland the way i are", bothub_client, store=store,
                   excluded_ids={"v1"})

    assert store.get_calls == []       # not read
    assert store.save_calls == []      # not written
    assert calls.count("llm-extract") == 2   # both runs classified fresh


async def test_block_verdict_not_cached(bothub_client):
    """A destructive-content verdict never enters the cache: it must
    re-classify fresh on every run (moderation drifts, the block decision is
    too important to pin to a first reading)."""
    store = _MemoryStore()
    calls = []
    blocked = _extract_json(block="drugs")

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_stub(blocked, calls)):
        out = await _run("музыка для варки", bothub_client, store=store)

    assert out.block == "drugs"
    assert store.save_calls == []


async def test_poisoned_block_row_never_served(bothub_client):
    """A block verdict sitting in the store (a corrupt/hand-written row) is
    not SERVED either: the read path applies the same eligibility, treats
    the row as a miss, classifies fresh — and the good verdict overwrites
    the poison (the cache self-heals)."""
    store = _MemoryStore()
    calls = []
    key = _extract_cache_key("timbaland the way i are")
    store.facts[("xtract", key, "cls")] = json.dumps(
        _extract_json(block="violence"))

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_stub(_extract_json(), calls)):
        out = await _run("timbaland the way i are", bothub_client, store=store)

    assert calls.count("llm-extract") == 1   # the poison was a miss
    assert out.block == ""                    # the fresh verdict delivered
    # self-healed: the stored row is the fresh non-block verdict now
    assert not json.loads(store.facts[("xtract", key, "cls")]).get("block")


async def test_clarify_verdict_not_cached(bothub_client):
    """A clarify extract (no queries, no target — the unstable garbage the
    clarify-early guard drops) never enters the cache."""
    store = _MemoryStore()
    calls = []
    clarify = _extract_json(intent="mood", queries=[])

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_stub(clarify, calls)):
        out = await _run("ну такое", bothub_client, store=store)

    assert out.need_clarify is True
    assert store.save_calls == []


async def test_store_none_previous_behavior_no_cache_logs(
        bothub_client, caplog):
    """store=None (legacy callers, tests): everything works exactly as
    before — the LLM#1 flow runs and no cache log line appears."""
    calls = []

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_stub(_extract_json(), calls)):
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out = await _run("timbaland the way i are", bothub_client)

    assert calls.count("llm-extract") == 1
    assert out.intent == "this_track"
    assert out.candidates
    assert not any("extract verdict cache" in r.getMessage()
                   for r in caplog.records)


async def test_degraded_extract_none_not_written(bothub_client):
    """LLM#1 degradation (extract None — bothub dead): the raw-search
    fallback runs and NOTHING enters the cache (a dead reading would poison
    the text for 30 days)."""
    store = _MemoryStore()
    calls = []

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        calls.append(gen_name)
        return None if gen_name == "llm-extract" else _rank_json(
            [("v1", 0.9)], best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        out = await _run("timbaland the way i are", bothub_client, store=store)

    assert out.degraded is True
    assert store.get_calls          # the read was attempted (a miss)
    assert store.save_calls == []   # the degradation was not written


async def test_cache_read_error_degrades_to_llm(bothub_client):
    """A store that raises on read (db locked): plain LLM#1 flow, the search
    is unaffected — the auxiliary-layer doctrine."""
    store = _MemoryStore(read_error=RuntimeError("db locked"))
    calls = []

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_stub(_extract_json(), calls)):
        out = await _run("timbaland the way i are", bothub_client, store=store)

    assert calls.count("llm-extract") == 1
    assert out.candidates
    assert out.candidates[out.best_index].video_id == "v1"  # plain delivery


# ════ 3. the text variant pre-detector (units) ════

@pytest.mark.parametrize("text,want", [
    # the canonical positive set, EN markers and RU stems alike
    ("timbaland the way i are slowed", "slowed"),
    ("дай замедленную версию", "slowed"),
    ("timbaland the way i are sped up", "sped_up"),
    (" Speed Up версия", "sped_up"),
    ("ускоренная версия", "sped_up"),
    ("хочу nightcore", "nightcore"),
    ("найткор плз", "nightcore"),
    ("сделай 8d", "nightcore"),
    ("с ревербом", "reverb"),
    ("ремикс этого трека", "remix"),
    ("акустичную версию", "acoustic"),
    ("инструментал", "instrumental"),
    ("extended mix please", "extended"),
    ("хочу вживую", "live"),
    # boundaries must hold on USER TEXT exactly as on titles
    ("музыка для бега", None),
    ("слоу данс", None),
    ("Slowdive", None),
    ("slow dance", None),
    ("Live and Let Die", None),
    ("Revolution", None),
    ("", None),
])
def test_text_variant_markers_and_boundaries(text, want):
    assert _text_variant(text) == want


# ════ 4. the override in search_candidates ════

async def test_text_marker_fills_missing_llm_variant(bothub_client, caplog):
    """«timbaland the way i are slowed» while LLM#1 dropped the variant
    (variant=None — the flake): the text marker fills it — target.variant,
    signals["variant"], the «Вариант-запрос:» rank note and the choke-point
    guard all see the FINAL slowed value (the slowed upload wins the pool)."""
    calls = []
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        calls.append(gen_name)
        if gen_name == "llm-extract":
            return _extract_json(
                target={"artist": "Timbaland", "track": "The Way I Are",
                        "variant": None},
                queries=["timbaland the way i are slowed"])
        captured.setdefault(gen_name, []).append(user)
        return _rank_json([("v1", 0.95), ("v2", 0.6), ("v3", 0.5)],
                          best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out = await _run("timbaland the way i are slowed", bothub_client)

    assert out.signals.get("variant") == "slowed"
    assert out.target.get("variant") == "slowed"
    assert "Вариант-запрос: slowed" in captured["llm-rank"][0]
    assert any("text variant override" in r.getMessage()
               for r in caplog.records)
    # the choke-point guard pressed the slowed upload to the top
    assert out.candidates[out.best_index].video_id == "v2"


async def test_text_marker_wins_over_conflicting_llm_variant(
        bothub_client, caplog):
    """The text says «sped up» while LLM#1 answered slowed (the flake's
    wrong-variant form): the TEXT wins — the override replaces the extract's
    variant before the canonical point, the rank note and the pool follow
    sped_up."""
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(
                target={"artist": "Timbaland", "track": "The Way I Are",
                        "variant": "slowed"},
                queries=["timbaland the way i are sped up"])
        captured.setdefault(gen_name, []).append(user)
        return _rank_json([("v1", 0.95), ("v2", 0.6), ("v3", 0.5)],
                          best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out = await _run("timbaland the way i are sped up", bothub_client)

    assert out.signals.get("variant") == "sped_up"
    assert "Вариант-запрос: sped_up" in captured["llm-rank"][0]
    assert any("text variant override: slowed -> sped_up" in r.getMessage()
               for r in caplog.records)
    assert out.candidates[out.best_index].video_id == "v3"


async def test_llm_variant_kept_when_text_has_no_marker(bothub_client,
                                                        caplog):
    """No marker in the text, variant from LLM#1: NEVER touched — the model
    may have understood the wish from context («замедли ту песню» names it,
    a dialog-driven wish may resolve through signals)."""
    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(
                target={"artist": "Timbaland", "track": "The Way I Are",
                        "variant": "slowed"},
                queries=["timbaland the way i are slowed"])
        return _rank_json([("v1", 0.95), ("v2", 0.6), ("v3", 0.5)],
                          best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out = await _run("timbaland the way i are", bothub_client)

    assert out.signals.get("variant") == "slowed"
    assert not any("text variant override" in r.getMessage()
                   for r in caplog.records)


# ════ 5. cache + pre-detector together ════

async def test_cached_variant_none_plus_text_marker(bothub_client):
    """The flake end-to-end: run 1 reads the text with variant=None from
    LLM#1 — the CACHE stores the verbatim verdict (None), while the final
    result of run 1 already carries the overridden slowed. Run 2 hits the
    cached variant=None verdict and the override re-applies to it: the
    deterministic layer runs after the cache, so the final variant is
    slowed in BOTH runs."""
    store = _MemoryStore()
    calls = []
    extract_answer = _extract_json(
        target={"artist": "Timbaland", "track": "The Way I Are",
                "variant": None},
        queries=["timbaland the way i are slowed"])

    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_stub(extract_answer, calls)):
        out1 = await _run("timbaland the way i are slowed",
                          bothub_client, store=store)
    # the cache pinned the verbatim LLM verdict — variant stayed None there
    assert json.loads(store.save_calls[0][3])["target"]["variant"] is None
    # ...but run 1's FINAL variant was already overridden from the text
    assert out1.signals.get("variant") == "slowed"

    calls.clear()
    with patch.object(bothub_client, "chat_json",
                      side_effect=_chat_stub(extract_answer, calls)):
        out2 = await _run("timbaland the way i are slowed",
                          bothub_client, store=store)

    assert "llm-extract" not in calls          # the cached verdict served
    assert out2.signals.get("variant") == "slowed"   # override re-applied
    assert out2.candidates[out2.best_index].video_id == "v2"


# ===== miss-valve (18.11, agent B) =====
# The "Не то" (miss) valve (2026-11-18): a miss press invalidates the
# extract-verdict cache row of the search's OWN description, so a poisoned
# first reading is not replayed for 30 days. Covered here: store.delete_fact
# (the row DELETE the invalidation needs) and the handler wiring. The helper
# itself (invalidate_extract_verdict) lives in services/llm_search (agent A)
# — it is patched in, never imported for real.


def _mv_state(desc, rejected=()):
    """A saved query state as _finish_search/_save_state writes it. desc is
    the search's own text; on a refine chain it already carries the "Не то:"
    tail — the helper simply won't match such keys (its concern, not ours)."""
    from services.llm_search import Candidate
    return {
        "candidates": [
            Candidate("v1", "Test Track", "Test Artist", 200, 500000,
                      score=0.85, match_score=0.72),
            Candidate("v2", "Track 2", "Artist 2", 180, 100000,
                      score=0.70, match_score=0.50),
        ],
        "confidence": "high", "best_index": 0, "verdict": "OK",
        "intro": "OK", "desc": desc, "rejected_ids": list(rejected),
        "intent": "this_track", "reference": {}, "ts": 0.0, "target": {},
    }


def _mv_update_context(data):
    """A fb-callback (update, context) — the MagicMock pattern of
    tests/test_handler.py (CallbackQuery is a frozen TelegramObject)."""
    from unittest.mock import AsyncMock, MagicMock
    from telegram import User
    cq = MagicMock()
    cq.data = data
    cq.answer = AsyncMock()
    cq.edit_message_reply_markup = AsyncMock()
    cq.message.reply_text = AsyncMock()
    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=123, is_bot=False, first_name="Test")
    update.effective_chat.id = 456
    context = MagicMock()
    context.bot_data = {"media": MagicMock()}
    context.chat_data = {}
    return update, context


def _mv_flow_patches():
    """Everything the fb-miss path touches besides the valve: feedback
    logging + trace scoring, the SQLite state mirror, the sends limit, the
    next-candidate delivery. Returns (mocks, patches) for an ExitStack."""
    from unittest.mock import AsyncMock, MagicMock, patch
    mocks = {
        "insert_feedback": AsyncMock(),
        "get_query_correlation": AsyncMock(return_value=None),
        "score_feedback": MagicMock(),
        "save_query_state": AsyncMock(),
        "delete_query_state": AsyncMock(),
        "get_user_lang": AsyncMock(return_value="ru"),
        "_sends_over": AsyncMock(return_value=False),
        "_send_track": AsyncMock(),
    }
    patches = [patch(f"bot.handler.{name}", mock)
               for name, mock in mocks.items()]
    return mocks, patches


async def test_mv_delete_fact_roundtrip(tmp_path, monkeypatch):
    """store.delete_fact removes exactly one (artist, track, lang) row:
    save -> cache hit -> delete True -> get None -> delete again False.
    The key is normalized exactly like get_fact/save_fact (case/whitespace)."""
    from services import store as store_mod
    monkeypatch.setattr("services.store.DB_PATH", str(tmp_path / "mv-facts.db"))
    monkeypatch.setattr("services.store.DATA_DIR", tmp_path)
    await store_mod.init_db()

    await store_mod.save_fact("Queen", "Bohemian Rhapsody", "ru", "факт")
    assert await store_mod.get_fact("Queen", "Bohemian Rhapsody", "ru") == "факт"
    # dirty casing/whitespace resolves to the same normalized key
    assert await store_mod.delete_fact("  queen ", "BOHEMIAN RHAPSODY", "ru") \
        is True
    assert await store_mod.get_fact("Queen", "Bohemian Rhapsody", "ru") is None
    # nothing left to delete -> False, no raise
    assert await store_mod.delete_fact("Queen", "Bohemian Rhapsody", "ru") \
        is False


async def test_mv_miss_press_invalidates_extract_verdict(monkeypatch, caplog):
    """A "Не то" press calls invalidate_extract_verdict with the search's OWN
    desc (from the saved state) and the store module — exactly once per
    press, and the True result is logged. A desc already carrying the "Не то:"
    tail (a refine chain) is passed as is: whether it matches a cacheable key
    is the helper's concern (it returns False there), not the handler's."""
    import logging
    from contextlib import ExitStack
    from unittest.mock import AsyncMock
    from services import store as store_mod
    from services import llm_search as lls
    from bot.handler import handle_callback

    helper = AsyncMock(return_value=True)
    monkeypatch.setattr(lls, "invalidate_extract_verdict", helper,
                        raising=False)

    for desc in ("black coffee and cifgarute",
                 "black coffee and cifgarute\nНе то: Test Track"):
        helper.reset_mock()
        update, context = _mv_update_context("fb:m:7:0")
        context.chat_data["q:7"] = _mv_state(desc)
        mocks, patches = _mv_flow_patches()
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with caplog.at_level(logging.INFO, logger="bot.handler"):
                await handle_callback(update, context)

        # the valve fired once, with the state's desc and the store module
        helper.assert_awaited_once()
        args, kwargs = helper.await_args
        assert args == (desc,)
        assert kwargs.get("store") is store_mod
        assert any("extract verdict invalidated (miss valve)" in r.getMessage()
                   for r in caplog.records)
        # the miss chain is intact: feedback registered, next variant sent
        mocks["insert_feedback"].assert_awaited_once()
        mocks["_send_track"].assert_awaited_once()


async def test_mv_valve_error_never_breaks_miss_chain(monkeypatch, caplog):
    """The valve is auxiliary: the helper raising degrades to a warning —
    the miss chain still registers the feedback and delivers the next
    candidate."""
    import logging
    from contextlib import ExitStack
    from unittest.mock import AsyncMock
    from services import llm_search as lls
    from bot.handler import handle_callback

    helper = AsyncMock(side_effect=RuntimeError("cache down"))
    monkeypatch.setattr(lls, "invalidate_extract_verdict", helper,
                        raising=False)

    update, context = _mv_update_context("fb:m:7:0")
    context.chat_data["q:7"] = _mv_state("black coffee and cifgarute")
    mocks, patches = _mv_flow_patches()
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with caplog.at_level(logging.WARNING, logger="bot.handler"):
            await handle_callback(update, context)

    helper.assert_awaited_once()
    assert any("miss valve failed" in r.getMessage() for r in caplog.records)
    mocks["insert_feedback"].assert_awaited_once()
    mocks["_send_track"].assert_awaited_once()


async def test_mv_missing_helper_miss_still_works(monkeypatch, caplog):
    """Before agent A merges: no invalidate_extract_verdict in
    services.llm_search — the lazy import raises ImportError, the valve is
    skipped SILENTLY (no warning) and the miss chain is fully intact."""
    import logging
    from contextlib import ExitStack
    from services import llm_search as lls
    from bot.handler import handle_callback

    monkeypatch.delattr(lls, "invalidate_extract_verdict", raising=False)

    update, context = _mv_update_context("fb:m:7:0")
    context.chat_data["q:7"] = _mv_state("black coffee and cifgarute")
    mocks, patches = _mv_flow_patches()
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with caplog.at_level(logging.WARNING, logger="bot.handler"):
            await handle_callback(update, context)

    assert not any("miss valve" in r.getMessage() for r in caplog.records)
    mocks["insert_feedback"].assert_awaited_once()
    mocks["_send_track"].assert_awaited_once()


async def test_mv_hit_press_does_not_invalidate(monkeypatch):
    """A "То" (hit) press is the opposite signal — the valve must stay
    silent (the cached reading was CONFIRMED correct)."""
    from contextlib import ExitStack
    from unittest.mock import AsyncMock
    from services import llm_search as lls
    from bot.handler import handle_callback

    helper = AsyncMock(return_value=True)
    monkeypatch.setattr(lls, "invalidate_extract_verdict", helper,
                        raising=False)

    update, context = _mv_update_context("fb:h:7:0")
    context.chat_data["q:7"] = _mv_state("black coffee and cifgarute")
    mocks, patches = _mv_flow_patches()
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        await handle_callback(update, context)

    helper.assert_not_awaited()
    mocks["insert_feedback"].assert_awaited_once()


# ════ 6. deterministic tie-break (18.11 acceptance) ════
# The 18.11 acceptance flake: three candidates scored 0.40 by the ranker and
# the winner jumped between runs — Python sort is stable, so with equal
# scores the order was whatever YouTube's result order fed in. Every score
# sort in services/llm_search.py now carries the (-score, video_id) key.

def _tie_slowed_pool(order):
    """Three slowed uploads of one record (all matching the «slowed»
    variant), video_ids picked so the alphabetical order (avid < mvid < zvid)
    differs from the first pool orders below."""
    base = {
        "zvid": {"video_id": "zvid", "title": "Timbaland - The Way I Are (Slowed)",
                 "channel": "Chill Vibes", "duration": 260, "views": 500_000},
        "mvid": {"video_id": "mvid",
                 "title": "Timbaland - The Way I Are Slowed + Reverb",
                 "channel": "Slowed World", "duration": 270, "views": 400_000},
        "avid": {"video_id": "avid",
                 "title": "Timbaland The Way I Are Slowed Edit",
                 "channel": "Edit Lab", "duration": 280, "views": 300_000},
    }
    return [dict(base[v]) for v in order]


async def test_tiebreak_variant_guard_equal_scores(bothub_client):
    """End-to-end through search_candidates: the ranker scores all three
    slowed uploads 0.40, the variant guard re-sorts (the LAST score sort of
    the choke point). With the pool fed z-first in one run and a-first in
    the other, the FINAL order is identical — by video_id — and the best
    stays on the LLM's picked video in both runs."""
    from services.llm_search import search_candidates as _sc

    async def _search(pool_order):
        calls = []
        extract_answer = _extract_json(
            target={"artist": "Timbaland", "track": "The Way I Are",
                    "variant": None},
            queries=["timbaland the way i are slowed"])

        async def fake_chat(system, user, max_tokens=None, trace=None,
                            gen_name="llm", model=None):
            calls.append(gen_name)
            if gen_name == "llm-extract":
                return extract_answer
            return _rank_json([("zvid", 0.40), ("mvid", 0.40), ("avid", 0.40)],
                              best_index=0)

        with patch.object(bothub_client, "chat_json",
                          side_effect=fake_chat):
            with patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
                 patch("services.llm_search.search_metadata") as mock_yt:
                mock_yt.return_value = _tie_slowed_pool(pool_order)
                return await _sc("timbaland the way i are slowed", bothub_client)

    out_z_first = await _search(["zvid", "mvid", "avid"])
    out_a_first = await _search(["avid", "zvid", "mvid"])

    # the same final order regardless of the pool's input order
    assert [c.video_id for c in out_z_first.candidates] == \
        ["avid", "mvid", "zvid"]
    assert [c.video_id for c in out_a_first.candidates] == \
        ["avid", "mvid", "zvid"]
    # the best is the LLM's picked video (tracked by id), in both runs
    assert out_z_first.candidates[out_z_first.best_index].video_id == "zvid"
    assert out_a_first.candidates[out_a_first.best_index].video_id == "zvid"


async def test_tiebreak_rerank_pool_equal_scores_with_reference(bothub_client):
    """rerank_pool's own sorts (score + the reference-ban composite): three
    candidates all at 0.40, one of them the reference — the non-reference
    pair orders by video_id ahead of the reference, same for both input
    orders of the pool."""
    from services.llm_search import Candidate, rerank_pool

    def _pool(order):
        base = {
            "zvid": Candidate("zvid", "Natural Blues", "Moby - Topic",
                              250, 9_000_000),
            "mvid": Candidate("mvid", "Porcelain", "MobyVEVO",
                              240, 8_000_000),
            "avid": Candidate("avid", "Go", "A Records", 230, 7_000_000),
        }
        return [base[v] for v in order]

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        return _rank_json([("zvid", 0.40), ("mvid", 0.40), ("avid", 0.40)],
                          best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        out_z_first = await rerank_pool(
            "calm moby-ish track", bothub_client, _pool(["zvid", "mvid", "avid"]),
            reference={"artist": "Moby"})
        out_a_first = await rerank_pool(
            "calm moby-ish track", bothub_client, _pool(["avid", "zvid", "mvid"]),
            reference={"artist": "Moby"})

    expected = ["avid", "mvid", "zvid"]   # non-refs by video_id, ref last
    assert [c.video_id for c in out_z_first.candidates] == expected
    assert [c.video_id for c in out_a_first.candidates] == expected
    # the reference never wins the best slot, whatever the input order was
    assert out_z_first.candidates[out_z_first.best_index].video_id == "avid"
    assert out_a_first.candidates[out_a_first.best_index].video_id == "avid"


# ════ 7. invalidate_extract_verdict (the miss-valve helper, units) ════

class _DeleteStore(_MemoryStore):
    """_MemoryStore plus the delete_fact row DELETE the miss valve needs:
    counts calls, deletes (and reports) only rows that actually exist."""

    def __init__(self):
        super().__init__()
        self.delete_calls = []

    async def delete_fact(self, artist, track, lang):
        self.delete_calls.append((artist, track, lang))
        key = (artist, track, lang)
        if key in self.facts:
            del self.facts[key]
            return True
        return False


async def test_invalidate_deletes_the_cached_row():
    """A fresh text's verdict row is deleted under the SAME ("xtract",
    <xv1 key>, "cls") triple the cache write used -> True."""
    from services.llm_search import invalidate_extract_verdict

    store = _DeleteStore()
    desc = "timbaland the way i are slowed"
    key = _extract_cache_key(desc)
    store.facts[("xtract", key, "cls")] = json.dumps(_extract_json())

    assert await invalidate_extract_verdict(desc, store=store) is True
    assert store.delete_calls == [("xtract", key, "cls")]
    assert ("xtract", key, "cls") not in store.facts   # the row is gone
    # the same normalized text hits the same (now absent) row -> False
    assert await invalidate_extract_verdict(
        "  Timbaland The Way I Are SLOWED ", store=store) is False


async def test_invalidate_contextual_desc_is_a_noop():
    """A desc carrying the handler's «Не то:» tail is a DIFFERENT text: its
    sha differs (no |ctx suffix — the suffix only comes from the contextual
    PARAMETERS, which stay None here), so the delete aims at a row that was
    never cached and returns False. The call DID happen — with the other
    key, never the fresh text's one."""
    from services.llm_search import invalidate_extract_verdict

    store = _DeleteStore()
    fresh = "black coffee and cifgarute"
    key = _extract_cache_key(fresh)
    store.facts[("xtract", key, "cls")] = json.dumps(_extract_json())

    contextual = fresh + "\nНе то: Test Track"
    ctx_key = _extract_cache_key(contextual)
    assert ctx_key.startswith("xv1:") and "|ctx" not in ctx_key
    assert ctx_key != key                       # another sha, never cached

    assert await invalidate_extract_verdict(contextual, store=store) is False
    assert store.delete_calls == [("xtract", ctx_key, "cls")]  # other key
    assert ("xtract", key, "cls") in store.facts  # the real row survived


async def test_invalidate_store_without_delete_fact():
    """A store stub that predates delete_fact (getattr guard): False, no
    AttributeError, nothing else attempted."""
    from services.llm_search import invalidate_extract_verdict

    store = _MemoryStore()                      # no delete_fact on purpose
    assert not hasattr(store, "delete_fact")
    assert await invalidate_extract_verdict(
        "timbaland the way i are", store=store) is False


async def test_invalidate_store_none_and_errors():
    """store=None -> False; a delete_fact that raises (db locked) -> False,
    never an exception up the miss chain."""
    from services.llm_search import invalidate_extract_verdict

    assert await invalidate_extract_verdict("some text") is False
    assert await invalidate_extract_verdict("some text", store=None) is False

    class _Boom:
        async def delete_fact(self, *a, **kw):
            raise RuntimeError("db locked")

    assert await invalidate_extract_verdict(
        "some text", store=_Boom()) is False


async def test_invalidate_empty_desc_is_a_stop_signal():
    """No text -> no key ("" is a stop-signal, not a key): False, and the
    store is never touched."""
    from services.llm_search import invalidate_extract_verdict

    store = _DeleteStore()
    assert await invalidate_extract_verdict("   ", store=store) is False
    assert store.delete_calls == []
