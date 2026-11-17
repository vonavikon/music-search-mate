"""Tests for the 16.11 officials work (2026-11-16) in FromSourcePipeline
plus the variant-aware should_auto_download.

Prod case 16.11: «песня где гимли и арагон бегут по горам» — the sonar
ground HAD the answer (Howard Shore "The Riders of Rohan") with sure=false,
but the deployed llm_search._sonar_ground keeps sure-only answers, so the
guess was dropped; the soundtrack seed is sure-only too; discovery ran on
descriptive queries, fan rips with film footage (TheLotrTV, jediking12)
won, and the user got "videos". The contracts under test:

  - UNSURE GROUND PROBE (before the ytsearch): on a ground-MISS,
    SOURCE-LESS, plain (non-analog) from_source search, ONE tolerant sonar
    call (gen_name="sonar-ground-probe", ctx-latched) may put the guessed
    "{artist} - {track}" at queries[0]. Only sure is explicitly False and
    both artist and track parse; sure=true adds NOTHING (the concrete path
    owns sure answers); errors/garbage change nothing. The probe phrase is
    strictly a ytsearch query: never saved into the source-fact cache.
  - OFFICIALS-ONLY GATE (after the rank tail, the scene-clip demotion and
    the fact-stop — the last guard before delivery): a plain from_source
    pool with ZERO official-quality candidates ("- Topic" channel / VEVO
    channel / "Official" title / channel == the candidate's canonical
    artist / a known soundtrack label channel) does NOT deliver — the
    honest clarify («...фанатские рипы...», EN by signals.language),
    reference.source preserved. ONE official anywhere in the pool delivers
    as today; an EMPTY pool keeps the previous empty path (NOT a clarify);
    the analog chain (rank_intent="similar") is never gated.
  - VARIANT-AWARE AUTO-DOWNLOAD: want_variant set + a NON-variant best
    candidate -> False (the original never downloads silently under a
    variant ask); want_variant empty, a variant candidate, or the
    llm_search.is_variant_upload helper not landed yet -> the previous
    threshold behavior.

Mocking follows tests/test_from_source_factstop.py: BothubClient.chat_json
patched on the bothub_client fixture (dispatched by gen_name),
search_metadata patched in services.llm_search, no network.
"""
import logging
import time
from unittest.mock import patch

import pytest

from services.pipelines.base import PipelineContext
from services.pipelines.from_source import (
    FromSourcePipeline,
    _is_official_upload,
    _is_scene_clip,
    _probe_phrase,
)
from services.match_score import should_auto_download


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Same guard as test_from_source_factstop.py: the enrich step must not
    touch the network (candidates here are fake)."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers (the test_from_source_factstop.py patterns) ════

def _extract_json(queries=None, source=None, lang="ru"):
    """LLM#1 extract — a fake parsed dict (from_source flavor). source lands
    in extract["reference"]["source"] (the anchor carry point the fact-stop
    gate reads; None = the source-less 16.11 incident shape)."""
    extract = {
        "intro": "Ищем: трек из фильма.",
        "intent": "from_source",
        "signals": {"language": lang, "era": None, "vocal": None,
                    "genre": None, "mood": None},
        "target": {"artist": None, "track": None},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries or ["lotr gimli aragorn mountains music",
                               "rohan theme running scene song"],
    }
    if source is not None:
        extract["reference"]["source"] = source
    return extract


def _rank_json(ranked=None, best_index=0, confidence="high",
               need_clarify=False, clarify_question=""):
    """LLM#2 rank — a fake parsed dict; defaults keep BOTH the fact-stop off
    (best 0.9 >= 0.4 with non-low confidence -> no verification call) and a
    confident delivery shape."""
    if ranked is None:
        ranked = [
            {"video_id": "f1", "score": 0.9, "rationale": "Похоже на тему."},
            {"video_id": "f2", "score": 0.5, "rationale": "Близкий вариант."},
            {"video_id": "f3", "score": 0.3, "rationale": "Запасной."},
        ]
    return {"ranked": ranked, "best_index": best_index, "confidence": confidence,
            "verdict": "Вот эта композиция.", "need_clarify": need_clarify,
            "clarify_question": clarify_question}


def _fan_pool():
    """The 16.11 incident pool: fan rips with film footage only — no
    Topic/VEVO/Official-title/label channel anywhere (scene words in titles
    are deliberate: they are what fan rips look like)."""
    return [
        {"video_id": "f1", "title": "CR - The Riders of Rohan (film scene)",
         "channel": "TheLotrTV", "duration": 250, "views": 900000},
        {"video_id": "f2", "title": "Aragorn and Gimli run music edit",
         "channel": "jediking12", "duration": 240, "views": 300000},
        {"video_id": "f3", "title": "Rohirrim theme edit",
         "channel": "OSTClips", "duration": 230, "views": 90000},
    ]


def _topic_led_pool():
    """One official YouTube Music upload plus the fan rips — the pool the
    probe/seed recovery is expected to produce."""
    return [
        {"video_id": "t1", "title": "The Riders of Rohan",
         "channel": "Howard Shore - Topic", "duration": 280, "views": 4000000},
        {"video_id": "f1", "title": "CR - The Riders of Rohan (film scene)",
         "channel": "TheLotrTV", "duration": 250, "views": 900000},
        {"video_id": "f2", "title": "Aragorn and Gimli run music edit",
         "channel": "jediking12", "duration": 240, "views": 300000},
    ]


def _label_pool():
    """The official OST cue on the soundtrack LABEL's channel (no Topic
    anywhere): the score-label shape the short list covers."""
    return [
        {"video_id": "l1",
         "title": "The Riders of Rohan (Original Motion Picture Soundtrack)",
         "channel": "WaterTower Music", "duration": 280, "views": 2000000},
        {"video_id": "f1", "title": "CR - The Riders of Rohan (film scene)",
         "channel": "TheLotrTV", "duration": 250, "views": 900000},
        {"video_id": "f2", "title": "Aragorn and Gimli run music edit",
         "channel": "jediking12", "duration": 240, "views": 300000},
    ]


class _StoreStub:
    """Source-fact cache stub counting reads/writes (the П3 contract,
    the test_from_source_pipeline.py pattern)."""

    def __init__(self, cached=None):
        self.cached = cached
        self.get_calls = []
        self.save_calls = []

    async def get_fact(self, artist, key, lang):
        self.get_calls.append((artist, key, lang))
        return self.cached

    async def save_fact(self, artist, key, lang, fact):
        self.save_calls.append((artist, key, lang, fact))


class _MetaSvc:
    """getSimilar for the analog chain, recording calls."""

    _lastfm_key = "k"

    def __init__(self, neighbors=None):
        self._neighbors = neighbors if neighbors is not None else []
        self.calls = []

    async def similar_artists(self, artist, limit=10):
        self.calls.append(("getSimilar", artist))
        return self._neighbors


async def _run(bothub_client, description, extract, entries, rank=None,
               ground=None, probe=None, probe_error=None, seed=None,
               factstop=None, store=None, excluded_ids=None,
               metadata_svc=None):
    """Run the pipeline DIRECTLY on a PipelineContext. captured holds the
    gen_name of every chat_json call, the sonar-ground/probe questions, the
    discovery queries and the fact-stop call count. `ground`/`probe`/`seed`
    are the canned parsed answers per sonar call (None = miss/unsure);
    probe_error makes the probe call raise (the timeout/outage path)."""
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault("calls", []).append(gen_name)
        if gen_name == "sonar-ground":
            captured["ground_q"] = user
            return ground
        if gen_name == "sonar-ground-probe":
            captured["probe_q"] = user
            captured["probe_system"] = system
            if probe_error is not None:
                raise probe_error
            return probe
        if gen_name == "sonar-seed":
            captured["seed_q"] = user
            return seed
        if gen_name == "sonar-factstop":
            captured["factstop_calls"] = captured.get("factstop_calls", 0) + 1
            return factstop
        if gen_name == "llm-rank":
            captured["rank_input"] = user
            return rank if rank is not None else _rank_json()
        raise AssertionError(f"unexpected LLM call: {gen_name}")

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat):
        with patch("services.llm_search.search_metadata") as mock_yt:
            async def cap(queries, per_query=20, on_progress=None):
                captured["queries"] = list(queries)
                return entries
            mock_yt.side_effect = cap
            ctx = PipelineContext(
                description=description, bothub=bothub_client, extract=extract,
                t_start=time.perf_counter(), t_extract=0.05, store=store,
                excluded_ids=excluded_ids, metadata_svc=metadata_svc)
            out = await FromSourcePipeline().run(ctx)
    return out, captured


_INCIDENT_DESC = "песня где гимли и арагон бегут по горам"
_PROBE_GUESS = {"artist": "Howard Shore", "track": "The Riders of Rohan",
                "sure": False}
_PROBE_PHRASE = "Howard Shore - The Riders of Rohan"


# ════ 1. the unsure ground probe ════

@pytest.mark.asyncio
async def test_probe_unsure_guess_leads_queries_and_stays_out_of_fact_cache(
        bothub_client, caplog):
    """The 16.11 incident: ground-concrete EMPTY (the sure-only filter
    dropped the unsure guess), no source name, no analog marker -> ONE
    tolerant probe call; the guess becomes queries[0] before the descriptive
    LLM queries, the delivery runs normally, and the source-fact store saw
    a READ (the ground path consulted it) but ZERO writes — the probe is
    strictly a ytsearch query and never pins a fact."""
    bothub_client._sonar_model = "sonar"
    store = _StoreStub(cached=None)
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, captured = await _run(
            bothub_client, _INCIDENT_DESC, _extract_json(), _topic_led_pool(),
            ground=None, probe=_PROBE_GUESS, store=store)

    # one probe call, between the missed ground and the rank (source-less:
    # the sure-only seed layer has nothing to name and never calls)
    assert captured["calls"] == ["sonar-ground", "sonar-ground-probe",
                                 "llm-rank"]
    assert "даже если не уверен" in captured["probe_system"]
    # the guessed phrase LEADS discovery, the descriptive queries follow
    assert captured["queries"][0] == _PROBE_PHRASE
    assert "lotr gimli aragorn mountains music" in captured["queries"]
    # the fact cache: consulted (the ground's own path), never written —
    # neither by the missed ground nor by the probe
    assert store.get_calls
    assert store.save_calls == []
    # the probe phrase is not a reference/ban key: nothing reference-shaped
    # leaked into the delivered out
    assert not (out.reference.get("artist") or "")
    assert not (out.reference.get("track") or "")
    # normal delivery (the Topic upload in the pool keeps the officials gate
    # asleep — that is the probe's whole point)
    assert out.candidates and not out.need_clarify
    assert any("unsure ground probe query: added" in r.message
               for r in caplog.records)


@pytest.mark.asyncio
async def test_probe_failure_leaves_queries_unchanged(bothub_client):
    """A raising probe call (timeout/outage shape) never breaks the search
    and never touches the query list — the auxiliary doctrine."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, _INCIDENT_DESC, _extract_json(), _topic_led_pool(),
        ground=None, probe_error=RuntimeError("sonar circuit open"))

    assert "sonar-ground-probe" in captured["calls"]   # the attempt was made
    assert captured["queries"] == ["lotr gimli aragorn mountains music",
                                   "rohan theme running scene song"]
    assert out.candidates and not out.need_clarify


@pytest.mark.asyncio
async def test_probe_sure_true_adds_nothing(bothub_client):
    """A sure=true probe answer is the regular ground's territory (its
    sure-only filter keeps such answers): the probe must NOT duplicate the
    phrase as a second query-leading path — nothing is added at all."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, _INCIDENT_DESC, _extract_json(), _topic_led_pool(),
        ground=None,
        probe={"artist": "Howard Shore", "track": "The Riders of Rohan",
               "sure": True})

    assert "sonar-ground-probe" in captured["calls"]
    assert _PROBE_PHRASE not in captured["queries"]
    assert captured["queries"] == ["lotr gimli aragorn mountains music",
                                   "rohan theme running scene song"]
    assert out.candidates and not out.need_clarify


@pytest.mark.asyncio
async def test_probe_off_with_source_or_analog_or_ground_hit(bothub_client):
    """Call economy, the OFF switches (2026-11-16 relaxed semantics): (a) a
    source name with a SURE seed owns the recovery — no probe call; (a2) the
    incident shape — source known but the seed came back unsure and added
    nothing — the probe DOES fire (that is the gimli/aragorn fix); (b) an
    analog marker never probes; (c) a ground hit leads already."""
    bothub_client._sonar_model = "sonar"
    # (a) source present + sure seed: the seed recovered, the probe never does
    _, captured = await _run(
        bothub_client, _INCIDENT_DESC, _extract_json(source="Властелин колец"),
        _topic_led_pool(), ground=None, probe=_PROBE_GUESS,
        seed={"songs": ["Howard Shore - Concerning Hobbits"], "sure": True})
    assert captured["calls"] == ["sonar-ground", "sonar-seed", "llm-rank"]
    assert captured["queries"][0] != _PROBE_PHRASE
    # (a2) source present + unsure seed (the 16.11 incident): probe fires
    _, captured = await _run(
        bothub_client, _INCIDENT_DESC, _extract_json(source="Властелин колец"),
        _topic_led_pool(), ground=None, probe=_PROBE_GUESS, seed=None)
    assert "sonar-ground-probe" in captured["calls"]
    assert captured["queries"][0] == _PROBE_PHRASE
    # (b) analog-marked description, still ground-less: no probe call
    _, captured = await _run(
        bothub_client, "Аналог песенки где гимли и арагон бегут по горам",
        _extract_json(), _topic_led_pool(), ground=None, probe=_PROBE_GUESS)
    assert "sonar-ground-probe" not in captured["calls"]
    # (c) ground hit: the concrete phrase leads, the probe never fires
    _, captured = await _run(
        bothub_client, _INCIDENT_DESC, _extract_json(), _topic_led_pool(),
        ground={"artist": "Howard Shore", "track": "The Riders of Rohan",
                "sure": True},
        probe=_PROBE_GUESS)
    assert "sonar-ground-probe" not in captured["calls"]
    assert captured["queries"][0] == _PROBE_PHRASE


@pytest.mark.parametrize("ans,phrase", [
    (_PROBE_GUESS, _PROBE_PHRASE),
    ({"artist": " Howard ", "track": " Shore\nRiders ", "sure": False},
     "Howard - Shore Riders"),           # control chars scrubbed, collapsed
    ({"artist": "Howard Shore", "track": "Riders", "sure": "false"},
     "Howard Shore - Riders"),           # string sure flag, tolerant parse
    ({"artist": "Howard Shore", "track": "Riders", "sure": True}, ""),
    ({"artist": "Howard Shore", "track": "Riders"}, ""),   # sure missing
    ({"artist": "Howard Shore", "track": "", "sure": False}, ""),
    ({"artist": "", "track": "Riders", "sure": False}, ""),
    ({"artist": None, "track": "Riders", "sure": False}, ""),
    ({}, ""),
    (None, ""),                          # chat_json garbage
    ("не JSON вообще", ""),
])
def test_probe_phrase_tolerant_parse(ans, phrase):
    """Only an explicitly-unsure complete {artist, track} guess parses to a
    phrase; everything else (sure=true, missing fields, garbage) is a silent
    miss — never an exception."""
    assert _probe_phrase(ans) == phrase


# ════ 2. the officials-only delivery gate ════

@pytest.mark.asyncio
async def test_fan_only_pool_honest_clarify_not_delivered(
        bothub_client, caplog):
    """The owner's 16.11 decision: a plain from_source pool of pure fan rips
    (TheLotrTV, jediking12 — no Topic/VEVO/Official/label anywhere) does NOT
    deliver. The honest clarify names the problem («фанатские рипы») and
    asks for a sharper scene or the artist; the pool never reaches the user;
    reference.source survives (the anchor lives for the fragment answers)."""
    bothub_client._sonar_model = "sonar"
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, captured = await _run(
            bothub_client, "песня из властелина колец где гимли и арагон "
                           "бегут по горам",
            _extract_json(source="Властелин колец"), _fan_pool(),
            seed=None)

    assert out.need_clarify and not out.candidates and out.best_index == -1
    q = out.clarify_question
    assert "Не нашёл официальную запись из этого источника" in q
    assert "фанатские рипы" in q
    assert "Опишешь сцену точнее или назовёшь исполнителя?" in q
    # the anchor lives: the clarify out keeps the source name
    assert out.reference.get("source") == "Властелин колец"
    # the fan pool was searched but is NOT in the delivery
    assert all(c.video_id not in {"f1", "f2", "f3"} for c in out.candidates)
    assert any("officials-only gate: 0 officials in pool of 3, honest clarify"
               in r.message for r in caplog.records)
    # the fact-stop never fired (high score + high confidence): the officials
    # gate alone stopped this delivery
    assert captured.get("factstop_calls", 0) == 0


@pytest.mark.asyncio
async def test_one_topic_upload_keeps_delivery(bothub_client, caplog):
    """One official (the YouTube Music Topic upload) among the fan rips is
    enough: the gate sleeps, the delivery is exactly the rank tail's output
    (candidates, best, no clarify) — the ordering guards already work."""
    bothub_client._sonar_model = "sonar"
    rank = _rank_json(ranked=[
        {"video_id": "f1", "score": 0.9, "rationale": "Фан-рип с кадрами."},
        {"video_id": "t1", "score": 0.85, "rationale": "Официальная тема."},
        {"video_id": "f2", "score": 0.4, "rationale": "Ещё рип."},
    ], best_index=0)
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, _ = await _run(
            bothub_client, "песня из властелина колец где гимли бегут",
            _extract_json(source="Властелин колец"), _topic_led_pool(),
            rank=rank, seed=None)

    assert out.candidates and not out.need_clarify
    assert out.candidates[out.best_index].video_id in {"f1", "t1"}
    assert not any("officials-only gate" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_soundtrack_label_channel_counts_as_official(bothub_client):
    """The OST cue on the label's own channel (WaterTower Music — the short
    known-label list) is an official upload: the gate sleeps and the pool
    delivers as today."""
    bothub_client._sonar_model = "sonar"
    rank = _rank_json(ranked=[
        {"video_id": "l1", "score": 0.9, "rationale": "Официальный саундтрек."},
        {"video_id": "f1", "score": 0.5, "rationale": "Фан-рип."},
        {"video_id": "f2", "score": 0.3, "rationale": "Ещё рип."},
    ], best_index=0)

    out, _ = await _run(
        bothub_client, "песня из властелина колец где гимли бегут",
        _extract_json(source="Властелин колец"), _label_pool(),
        rank=rank, seed=None)

    assert out.candidates and not out.need_clarify
    assert out.candidates[out.best_index].video_id == "l1"


@pytest.mark.asyncio
async def test_officials_clarify_english_variant(bothub_client):
    """signals.language is the pipeline's only lang notion: a clearly-EN
    query gets the EN wording of the honest-out message."""
    bothub_client._sonar_model = "sonar"
    out, _ = await _run(
        bothub_client, "song where gimli and aragorn run through the mountains",
        _extract_json(source="The Lord of the Rings", lang="en"), _fan_pool(),
        seed=None)

    assert out.need_clarify and not out.candidates
    q = out.clarify_question
    assert "I could not find an official recording from this source" in q
    assert "fan rips" in q
    assert "describe the scene more precisely, or name the artist" in q


@pytest.mark.asyncio
async def test_analog_chain_not_gated(bothub_client):
    """The analog chain ranks in SIMILAR mode — a reference hunt where a fan
    pool is fine: even a pure fan pool delivers as today, the officials gate
    must not fire (the same plain-flow scope as the scene-clip demotion and
    the fact-stop)."""
    bothub_client._sonar_model = "sonar"
    svc = _MetaSvc(neighbors=["Neighbor Band"])
    entries = [
        {"video_id": "n1", "title": "Neighbor Band - Close Cue",
         "channel": "Fan Cuts One", "duration": 210, "views": 500000},
        {"video_id": "n2", "title": "Neighbor Band - Other Cue",
         "channel": "Fan Cuts Two", "duration": 220, "views": 400000},
    ]
    rank = _rank_json(ranked=[
        {"video_id": "n1", "score": 0.8, "rationale": "Звучит похоже."},
        {"video_id": "n2", "score": 0.5, "rationale": "Тоже рядом."},
    ], best_index=0)

    out, captured = await _run(
        bothub_client,
        "Аналог песенки из властелина колец, где гимли и арагон бегут "
        "по горам",
        _extract_json(source="Властелин колец"), entries, rank=rank,
        ground={"artist": "Howard Shore", "track": "The Riders of Rohan",
                "sure": True},
        metadata_svc=svc)

    assert svc.calls == [("getSimilar", "Howard Shore")]   # the chain ran
    assert "sonar-ground-probe" not in captured["calls"]  # ground hit: no probe
    assert out.candidates and not out.need_clarify         # fan pool delivered
    assert out.candidates[out.best_index].video_id == "n1"


@pytest.mark.asyncio
async def test_empty_pool_keeps_previous_empty_path(bothub_client):
    """Hostile shape: the search found entries but every one was excluded
    (0 candidates in the delivery) — that is the PREVIOUS empty path, not
    this gate's clarify (an empty pool has no fan-rip delivery to stop)."""
    bothub_client._sonar_model = "sonar"
    out, _ = await _run(
        bothub_client, "песня где гимли и арагон бегут по горам",
        _extract_json(), _fan_pool(), excluded_ids={"f1", "f2", "f3"})

    assert out.candidates == [] and out.best_index == -1
    assert not out.need_clarify and not out.clarify_question


# ════ 3. the official-upload criteria (unit) ════

def _cand(channel, title="", canonical=""):
    from services.llm_search import Candidate
    return Candidate("x1", title or "Some Track", channel, 240, 100000,
                     canonical_artist=canonical)


@pytest.mark.parametrize("cand,official", [
    (_cand("Howard Shore - Topic"), True),          # YouTube Music auto-channel
    (_cand("Howard Shore – Topic"), True),          # en/em dash variants
    (_cand("Howard Shore — Topic"), True),
    (_cand("SomeArtistVEVO"), True),                # VEVO channel
    (_cand("Fan Channel", "Song (Official Video)"), True),   # Official title
    (_cand("Fan Channel", "Song OFFICIAL AUDIO"), True),
    (_cand("Howard Shore", canonical="Howard Shore"), True),  # canonical artist
    (_cand("WaterTower Music"), True),              # known soundtrack label
    (_cand("Sony Soundtracks"), True),
    (_cand("TheLotrTV", "CR - The Riders of Rohan"), False),  # the 16.11 fans
    (_cand("jediking12", "Aragorn and Gimli run music edit"), False),
    (_cand("Fan Orchestra", "Cool Soundtrack Theme"), False),
    (_cand("", ""), False),                         # shape-tolerant: not official
])
def test_is_official_upload_criteria(cand, official):
    """Any ONE of the wide criteria qualifies; a pure fan upload does not
    (the false-negative-averse design: false officials deliver as today,
    only true all-fan pools clarify)."""
    assert _is_official_upload(cand) is official


# ════ 4. variant-aware should_auto_download ════

def _detector(title, channel):
    """A stand-in for the llm_search contract (VARIANT_WORDS +
    is_variant_upload): the variant kind when the title carries one, else
    None (the original)."""
    return "slowed" if "slowed" in (title or "").lower() else None


def test_want_variant_non_variant_never_auto_downloads(monkeypatch):
    """want_variant set + the best candidate is NOT a variant upload ->
    False even at a high score: the original must not download silently,
    the user picks the wanted variant from the preview."""
    monkeypatch.setattr("services.llm_search.is_variant_upload", _detector,
                        raising=False)
    assert should_auto_download("this_track", 0.9, want_variant="slowed",
                                title="Artist - Song",
                                channel="Artist") is False
    assert should_auto_download("similar", 0.9, want_variant="slowed",
                                title="Artist - Song",
                                channel="Artist") is False


def test_want_variant_variant_upload_keeps_threshold(monkeypatch):
    """A variant candidate under a variant ask: the previous threshold
    logic untouched — auto above, preview below."""
    monkeypatch.setattr("services.llm_search.is_variant_upload", _detector,
                        raising=False)
    assert should_auto_download("this_track", 0.9, want_variant="slowed",
                                title="Artist - Song (Slowed)",
                                channel="Artist") is True
    assert should_auto_download("this_track", 0.5, want_variant="slowed",
                                title="Artist - Song (Slowed)",
                                channel="Artist") is False


def test_want_variant_empty_is_previous_behavior(monkeypatch):
    """want_variant unset (the handler does not pass it yet): the two-argument
    behavior exactly as before, variant detection never consulted."""
    monkeypatch.setattr("services.llm_search.is_variant_upload", _detector,
                        raising=False)
    assert should_auto_download("this_track", 0.9) is True
    assert should_auto_download("this_track", 0.9, want_variant="",
                                title="Artist - Song",
                                channel="Artist") is True
    assert should_auto_download("from_source", 0.99) is False


def test_want_variant_helper_not_landed_previous_behavior(monkeypatch):
    """The llm_search side (VARIANT_WORDS + is_variant_upload) has not landed
    yet: detection is unavailable (None) and the decision keeps the PREVIOUS
    behavior — a missing dependency must not silently flip auto-downloads
    off. Works whether or not the attribute currently exists."""
    monkeypatch.delattr("services.llm_search.is_variant_upload",
                        raising=False)
    assert should_auto_download("this_track", 0.9, want_variant="slowed",
                                title="Artist - Song",
                                channel="Artist") is True


# ════ (2026-11-16 review) известный артист = официальный канал ════

def test_official_known_artist_channel_unit():
    """LoTR-кейс: Topic-загрузок нет вовсе, официальный канал — сам
    композитор; артист известен из sonar (ground/probe/seed)."""
    from services.pipelines.from_source import _is_official_upload
    cand = type("C", (), {"channel": "Howard Shore", "title": "The Riders of Rohan"})()
    assert not _is_official_upload(cand)                       # без знания — фан
    assert _is_official_upload(cand, ("Howard Shore",))        # с знанием — официальный
    other = type("C", (), {"channel": "dogsrule3444", "title": "x"})()
    assert not _is_official_upload(other, ("Howard Shore",))


# ════ 5. scene-clip demotion (16.11, 2026-11-16) ════
#
# The 16.11 acceptance shape: officials WERE in the pool (the seed worked),
# yet the rank crowned the scene-clip montage — "The Lord of the Rings:
# The Two Towers | Warg Attack" from "ClipZone: Heroes & Villains" (llm
# 0.90) over Annie Lennox — because the prompt's "no video clips" rule
# loses to a strong title match. The deterministic guard (the plain flow's
# last word, after the officials-only gate) pushes every detected scene
# clip below the WEAKEST official; officials themselves are never clips.

def _clip_mixed_pool():
    """The acceptance pool: the ClipZone scene clip on top of the official
    OST upload, a plain (non-clip) fan upload and a fan-TV clip. Titles are
    deliberately free of scene words: the F4 title/category demotion stays
    inert, so the ordering below is THIS guard's work alone."""
    return [
        {"video_id": "c1",
         "title": "The Lord of the Rings: The Two Towers | Warg Attack",
         "channel": "ClipZone: Heroes & Villains", "duration": 240,
         "views": 800000},
        {"video_id": "o1", "title": "Annie Lennox - Into the West",
         "channel": "Annie Lennox", "duration": 270, "views": 5000000},
        {"video_id": "d1", "title": "lotr rohan run song",
         "channel": "dogsrule3444", "duration": 230, "views": 20000},
        {"video_id": "c2", "title": "The Riders of Rohan",
         "channel": "TheLotrTV", "duration": 250, "views": 900000},
    ]


@pytest.mark.asyncio
async def test_scene_clip_demotion_incident(bothub_client, caplog):
    """ИНЦИДЕНТ 16.11: the rank crowned the ClipZone clip (0.90) over the
    official Into the West (0.6); the guard sinks BOTH clips (ClipZone,
    TheLotrTV) below the official by the 0.05 gap, the delivery is led by
    the official, and the log names the demotion. The official here rides
    the known_artists tuple (the sonar ground named Annie Lennox — the
    same tuple the officials gate threads through)."""
    bothub_client._sonar_model = "sonar"
    rank = _rank_json(ranked=[
        {"video_id": "c1", "score": 0.90, "rationale": "Точное имя сцены."},
        {"video_id": "o1", "score": 0.60, "rationale": "Официальный трек."},
        {"video_id": "d1", "score": 0.50, "rationale": "Фан."},
        {"video_id": "c2", "score": 0.55, "rationale": "Фан-нарезка."},
    ], best_index=0)
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, captured = await _run(
            bothub_client, _INCIDENT_DESC,
            _extract_json(source="Властелин колец"), _clip_mixed_pool(),
            rank=rank,
            ground={"artist": "Annie Lennox", "track": "Into the West",
                    "sure": True})

    # the ground hit led discovery (that is how the official entered the
    # pool in the incident) and no probe/seed call follows it
    assert captured["calls"] == ["sonar-ground", "llm-rank"]
    assert captured["queries"][0] == "Annie Lennox - Into the West"
    # normal delivery, no clarify anywhere
    assert out.candidates and not out.need_clarify
    # the official leads; both clips sit below it; the plain fan keeps 0.5
    order = [c.video_id for c in out.candidates]
    assert order[0] == "o1"
    assert order.index("c1") > order.index("o1")
    assert order.index("c2") > order.index("o1")
    by_id = {c.video_id: c for c in out.candidates}
    assert by_id["c1"].score == pytest.approx(0.55)   # min(0.90, 0.60-0.05)
    assert by_id["c2"].score == pytest.approx(0.55)   # min(0.55, 0.60-0.05)
    assert by_id["d1"].score == pytest.approx(0.50)   # not a clip: untouched
    assert by_id["o1"].score == pytest.approx(0.60)   # official: untouched
    # the clip that was the LLM's best hands the slot to the official
    assert out.candidates[out.best_index].video_id == "o1"
    assert any("scene-clip demotion: 2 clip(s) below 1 official(s)" in r.message
               for r in caplog.records)


@pytest.mark.asyncio
async def test_clip_best_hands_slot_to_official(bothub_client, caplog):
    """The rank's best WAS the clip: after the guard the slot goes to the
    leader — the official by construction — and the clip sits right below
    it. Officialness here rides the Topic suffix (no sonar knowledge at
    all: the known_artists tuple is empty)."""
    bothub_client._sonar_model = "sonar"
    rank = _rank_json(ranked=[
        {"video_id": "c1", "score": 0.90, "rationale": "Точное имя сцены."},
        {"video_id": "o1", "score": 0.60, "rationale": "Официальный трек."},
    ], best_index=0)
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, _ = await _run(
            bothub_client, _INCIDENT_DESC, _extract_json(),
            [
                {"video_id": "c1",
                 "title": "The Two Towers | Warg Attack",
                 "channel": "ClipZone: Heroes & Villains",
                 "duration": 240, "views": 800000},
                {"video_id": "o1", "title": "Annie Lennox - Into the West",
                 "channel": "Annie Lennox - Topic", "duration": 270,
                 "views": 5000000},
            ], rank=rank, ground=None, probe=None)

    assert [c.video_id for c in out.candidates] == ["o1", "c1"]
    assert out.candidates[out.best_index].video_id == "o1"
    assert out.candidates[1].score == pytest.approx(0.55)
    assert any("scene-clip demotion: 1 clip(s) below 1 official(s)" in r.message
               for r in caplog.records)


@pytest.mark.asyncio
async def test_scene_clip_demotion_skipped_without_officials(
        bothub_client, caplog):
    """No official anywhere in the pool -> the officials-only gate's honest
    clarify intercepts BEFORE the demotion (ordering fan rips is moot when
    nothing should deliver at all): no candidate is demoted, no demotion
    log, the clarify shape returns."""
    bothub_client._sonar_model = "sonar"
    rank = _rank_json(ranked=[
        {"video_id": "c1", "score": 0.90, "rationale": "Клип."},
        {"video_id": "c2", "score": 0.55, "rationale": "Фан-ТВ."},
    ], best_index=0)
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, _ = await _run(
            bothub_client, "песня из властелина колец где гимли бегут",
            _extract_json(source="Властелин колец"),
            [
                {"video_id": "c1",
                 "title": "The Two Towers | Warg Attack",
                 "channel": "ClipZone: Heroes & Villains",
                 "duration": 240, "views": 800000},
                {"video_id": "c2", "title": "The Riders of Rohan",
                 "channel": "TheLotrTV", "duration": 250, "views": 900000},
            ], rank=rank, seed=None)

    assert out.need_clarify and not out.candidates and out.best_index == -1
    assert not any("clip(s) below" in r.message for r in caplog.records)


@pytest.mark.parametrize("cand,clip", [
    (_cand("ClipZone: Heroes & Villains"), True),   # clip (glued, the incident)
    (_cand("Epic Clips"), True),                    # clips, the plain word
    (_cand("TheLordoftheRings Fans"), True),        # fans
    (_cand("TheLotrTV"), True),                     # TV suffix
    (_cand("SG LOTR TV"), True),                    # TV suffix, case-tolerant
    (_cand("jediking12", "The Two Towers CR - 08. The Wolves"), True),  # CR rip
    (_cand("jediking12", "CR - 08. The Wolves"), True),  # CR at the start
])
def test_scene_clip_detector_matches(cand, clip):
    """The 16.11 clip shapes all match: the clip/fans channel markers
    ("clip" keeps only its leading boundary: ClipZone/Epic Clips match,
    mid-word glues like "eclipse" never do — and the movieclips shape is
    the studio-channel filter's territory upstream anyway), the fan-TV
    suffix and the Complete Recordings "CR - " title pattern."""
    assert _is_scene_clip(cand) is clip


@pytest.mark.parametrize("cand,clip", [
    (_cand("Howard Shore", "The Riders of Rohan"), False),  # official channel
    (_cand("Interscope Records", "Lose My Mind"), False),   # a real label
    (_cand("Don Toliver", "Lose My Mind (From F1® The Movie)"), False),
    (_cand("Official Top Videos", "Best Hits"), False),  # "videos" NOT a marker
    (_cand("Eclipse Records", "Some Song"), False),  # "clip" inside "eclipse" — no
    (_cand("", ""), False),                        # shape-tolerant: not a clip
])
def test_scene_clip_detector_false_positives(cand, clip):
    """Ложные срабатывания хуже пропусков: the composer's own channel, a
    real label, an OST-shaped title on the artist's channel, a "videos"
    channel (not in the marker list) and the inside-a-word "eclipse" shape
    all stay NOT clips — officials are closed by _is_official_upload, which
    the guard checks first."""
    assert _is_scene_clip(cand) is clip


def test_scene_clip_bluray_rip_marker_unit():
    """(16.11 acceptance S10) «Rohirrim Charge HD (Blu-ray) 1080p» на фан-канале —
    клип по rip-маркеру заголовка; официальный трек маркеров не несёт."""
    from services.pipelines.from_source import _is_scene_clip
    rip = type("C", (), {"channel": "RokujyuuyonShou",
                         "title": "Rohirrim Charge HD (Blu-ray) 1080p"})()
    assert _is_scene_clip(rip)
    clean = type("C", (), {"channel": "Howard Shore",
                           "title": "The Ride of the Rohirrim"})()
    assert not _is_scene_clip(clean)


# ════ 6. confirmed signals + filter wanted (17.11, 2026-11-17) ════
#
# Incidents: (Б/В) a SURE sonar ground/seed was zeroed or outranked by the
# ranker (the false "not from the source"; the fan video over the official
# clip) — signals["confirmed"] is the frozen transport handing llm_search
# the certainty (the rank-side use there is ANOTHER AGENT's work; these
# tests pin the WRITE side only). (А) the coffee search: 8 copies of the
# exact target in raw, the views-sorted top-20 kept 1 — wanted threads the
# target's title keys into the filter so its copies survive AND lead.

_COFFEE_DESC = "песня про чёрный кофе и сигареты как в том фильме"
_COFFEE_GROUND = {"artist": "Mo Rodgers",
                  "track": "Black Coffee and Cigarettes", "sure": True}
_COFFEE_CONCRETE = "Mo Rodgers - Black Coffee and Cigarettes"


def _coffee_copies():
    """The exact target's uploads: distinct titles (the filter's dedup keeps
    each), small views (the incident's point), one Topic channel (an
    official — the officials-only gate sleeps on this pool)."""
    return [
        {"video_id": "c1", "title": "Mo Rodgers - Black Coffee and Cigarettes",
         "channel": "Mo Rodgers - Topic", "duration": 250, "views": 15000},
        {"video_id": "c2", "title": "Black Coffee and Cigarettes",
         "channel": "Blues Vault", "duration": 245, "views": 9000},
        {"video_id": "c3",
         "title": "Mo Rodgers - Black Coffee and Cigarettes (Official Audio)",
         "channel": "Mo Rodgers", "duration": 250, "views": 5000},
    ]


def _giant_neighbors(n):
    """300M-view neighbors crowding the copies out of any views-sorted top
    slice. Titles/channels deliberately free of clip/TV/trailer and
    non-music markers (the guards stay asleep — the ordering under test is
    the wanted-survival work alone)."""
    return [{"video_id": f"g{i}", "title": f"Neighbor Hit {i}",
             "channel": f"Hits Channel {i}", "duration": 210,
             "views": 300_000_000 - i * 1_000_000}
            for i in range(1, n + 1)]


@pytest.mark.asyncio
async def test_confirmed_track_from_sure_ground(bothub_client):
    """(17.11 incident В transport) A SURE ground concrete rides
    signals["confirmed"]["track"] verbatim; songs stay empty (no seed ran —
    the ground hit owns the lead). The rank input carries the «подтверждён»
    legend — the llm_read side (another agent's frozen-contract work, already
    landed in llm_search) rendering THIS pipeline's signal: end-to-end proof
    the transport reached the ranker; the write-side guarantee under test is
    the signals dict itself."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, "песня из фильма где степ ап танцуют",
        _extract_json(), _topic_led_pool(),
        ground={"artist": "Chromeo", "track": "Fancy Footwork", "sure": True})

    assert out.signals["confirmed"] == {"track": "Chromeo - Fancy Footwork",
                                        "songs": []}
    assert "подтвержд" in captured["rank_input"].lower()
    assert out.candidates and not out.need_clarify


@pytest.mark.asyncio
async def test_confirmed_songs_from_sure_seed(bothub_client):
    """(17.11 incident Б transport) A sure seed of 3 soundtrack songs rides
    signals["confirmed"]["songs"] (the exact seeded lines); track stays
    empty — the ground missed, and the seed is the only certainty."""
    bothub_client._sonar_model = "sonar"
    songs = ["T-Pain - Take Your Shirt Off",
             "Sophia Fresh - This Instant",
             "Enur - Calabria 2010"]
    out, captured = await _run(
        bothub_client, "песня из фильма шаг вперёд 3д финальный танец",
        _extract_json(source="Шаг вперёд 3D"), _topic_led_pool(),
        ground=None, seed={"songs": songs, "sure": True})

    assert out.signals["confirmed"] == {"track": "", "songs": songs}
    assert captured["calls"] == ["sonar-ground", "sonar-seed", "llm-rank"]
    assert out.candidates and not out.need_clarify


@pytest.mark.asyncio
async def test_probe_phrase_never_confirmed(bothub_client):
    """The unsure probe DID lead discovery (the guess became queries[0])
    but signals["confirmed"] stays strictly empty — a guess is not a
    confirmation (the whole point of the 17.11 contract)."""
    bothub_client._sonar_model = "sonar"
    out, captured = await _run(
        bothub_client, _INCIDENT_DESC, _extract_json(), _topic_led_pool(),
        ground=None, probe=_PROBE_GUESS)

    assert captured["queries"][0] == _PROBE_PHRASE
    assert out.signals["confirmed"] == {"track": "", "songs": []}
    # nothing confirmed -> the ranker sees no confirmation legend either
    # (the empty-but-strict signal must not light llm_search's renderer)
    assert "подтвержд" not in captured["rank_input"].lower()


@pytest.mark.asyncio
async def test_confirmed_track_from_analog_reference(bothub_client):
    """The analog chain's sonar-resolved reference is a confirmation:
    signals["confirmed"]["track"] == the reference concrete phrase; songs
    stay empty (the chain never seeds — its picks are deliberately NOT the
    soundtrack)."""
    bothub_client._sonar_model = "sonar"
    svc = _MetaSvc(neighbors=["Neighbor Band"])
    entries = [
        {"video_id": "n1", "title": "Neighbor Band - Close Cue",
         "channel": "Fan Cuts One", "duration": 210, "views": 500000},
        {"video_id": "n2", "title": "Neighbor Band - Other Cue",
         "channel": "Fan Cuts Two", "duration": 220, "views": 400000},
    ]
    rank = _rank_json(ranked=[
        {"video_id": "n1", "score": 0.8, "rationale": "Звучит похоже."},
        {"video_id": "n2", "score": 0.5, "rationale": "Тоже рядом."},
    ], best_index=0)
    out, _ = await _run(
        bothub_client,
        "Аналог песенки из властелина колец, где гимли и арагон бегут "
        "по горам",
        _extract_json(source="Властелин колец"), entries, rank=rank,
        ground={"artist": "Howard Shore", "track": "The Riders of Rohan",
                "sure": True},
        metadata_svc=svc)

    assert out.signals["confirmed"] == {
        "track": "Howard Shore - The Riders of Rohan", "songs": []}


@pytest.mark.asyncio
async def test_wanted_survival_incident_a(bothub_client):
    """(17.11 incident А) 12-candidate pool: the target's 3 small-view
    copies + 9 300M-view neighbors. The ground-SURE concrete threads wanted
    into the filter; the copies survive AND take the head — all three are
    in the delivery AND in the rank input (the enrich pass and the rank
    input read the top slice; tail slots are invisible to the ranker, which
    is how the incident lost 7 of its 8 copies)."""
    bothub_client._sonar_model = "sonar"
    entries = _coffee_copies() + _giant_neighbors(9)
    out, captured = await _run(
        bothub_client, _COFFEE_DESC, _extract_json(), entries,
        ground=_COFFEE_GROUND)

    ids = [c.video_id for c in out.candidates]
    assert {"c1", "c2", "c3"} <= set(ids)
    assert ids[:3] == ["c1", "c2", "c3"]          # copies lead (views order)
    for vid in ("c1", "c2", "c3"):
        assert f"video_id={vid} " in captured["rank_input"]
    # the confirmed track rode the signals transport alongside the rescue
    assert out.signals["confirmed"]["track"] == _COFFEE_CONCRETE


@pytest.mark.asyncio
async def test_wanted_survival_rescues_dropped_copy(bothub_client, caplog):
    """The overflow shape: 21 candidates, the views cut keeps the top-20
    and drops the smallest-view copy (c3, 5k views) — the raw-rescue path
    pulls it back so the 3-slot floor holds (the 8-copies incident: at
    least 3 copies survive whatever the neighbors' views)."""
    bothub_client._sonar_model = "sonar"
    entries = _coffee_copies() + _giant_neighbors(18)
    with caplog.at_level(logging.INFO, logger="bot.llm_search"):
        out, captured = await _run(
            bothub_client, _COFFEE_DESC, _extract_json(), entries,
            ground=_COFFEE_GROUND)

    ids = [c.video_id for c in out.candidates]
    assert {"c1", "c2", "c3"} <= set(ids)
    assert ids[:3] == ["c1", "c2", "c3"]
    for vid in ("c1", "c2", "c3"):
        assert f"video_id={vid} " in captured["rank_input"]
    assert any("wanted survival: rescued 1 exact-target copy" in r.message
               for r in caplog.records)


@pytest.mark.asyncio
async def test_wanted_from_target_track_without_ground(bothub_client):
    """target.track alone (the extract knew the exact song; sonar added
    nothing — no ground, no seed, no probe answer) still threads wanted:
    the copies survive on the extract's own precision, while
    signals["confirmed"] stays strictly empty (nothing was CONFIRMED)."""
    bothub_client._sonar_model = "sonar"
    extract = _extract_json()
    extract["target"] = {"artist": "Mo Rodgers",
                         "track": "Black Coffee and Cigarettes"}
    out, captured = await _run(
        bothub_client, _COFFEE_DESC, extract,
        _coffee_copies() + _giant_neighbors(18), ground=None, probe=None)

    ids = [c.video_id for c in out.candidates]
    assert {"c1", "c2", "c3"} <= set(ids)
    assert ids[:3] == ["c1", "c2", "c3"]
    assert out.signals["confirmed"] == {"track": "", "songs": []}
