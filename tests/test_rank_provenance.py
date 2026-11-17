"""Rank provenance (2026-11-17): the ranker must trust CONFIRMED facts.

The 17.11 disease, three incidents of one bug — «the ranking does not
trust verified facts»:
  А (КОФЕ): «black coffee and cifgarute» (this_track, the official Mighty
    Mo Rodgers sat in the top-20) — the ranker zeroed everything -> the
    circular clarify;
  Б (ШАГ-ВПЕРЁД site): from_source, a confident sonar seed of 3 Step Up 3D
    songs, the filter kept 14 of their copies in the top-20 — the ranker
    zeroed ALL of them with a false «not from Step Up 3D»;
  В (ШАГ-ВПЕРЁД bot): sonar-ground was SURE «Chromeo - Fancy Footwork»,
    the official Chromeo video was in the pool — the ranker gave it 0.0
    and the fan «Fancy Footwork : by Step up 3 (HD)» (uploader 吳昇諳) 1.0
    for the words «Step up» in the title; the official-first guard failed
    to GROUP the pair (the by/step/up/3/hd garbage tokens diluted the
    intersection) and the card carried the ytdlp garbage name.

Fixes covered here (services/llm_search.py):
  1. _confirmed_keys/_matches_confirmed — deterministic handles on the
     frozen signals["confirmed"] contract ({"track": "Artist - Track",
     "songs": ["Artist - Song", ...]}; both keys optional, unsure probes
     never land there);
  2. the [подтверждён] marker + rule line in the LLM#2 rank input;
  3. _group_title_tokens cuts the fan-edit garbage tails before
     tokenization, so the official-first grouping catches the incident-В
     pair;
  4. the sec4 zeroed-rank rescue: confirmed candidates keep the pool and
     the circular clarify is dropped;
  5. the confirmed-track card override (canonical_artist/canonical_track
     from the sonar fact, over the ytdlp garbage).

Conventions follow tests/test_variant_modifiers.py: BothubClient.chat_json
is mocked with a side_effect returning ALREADY parsed dicts,
services.llm_search.search_metadata is patched, the conftest autouse
fixtures keep yt-dlp/suggest/meta-cache offline.
"""
import logging
from unittest.mock import patch

import pytest

from services.llm_search import (
    _confirmed_keys,
    _group_title_tokens,
    _matches_confirmed,
    search_candidates,
)


@pytest.fixture(autouse=True)
def _no_ytdlp_network():
    """Enrich step offline (same as tests/test_variant_modifiers.py): the
    top-5 enrich would be a real yt-dlp call; the stub returns None. The
    canonical-override test re-patches it with its own side_effect."""
    with patch("services.search._extract_info_no_dl", return_value=None):
        yield


# ════ helpers ════

def _extract_json(intent="this_track", target=None, queries=None,
                  signals=None, reference=None):
    """LLM#1 extract — a fake already-parsed answer. `signals` REPLACES the
    default dict so a test can carry signals['confirmed'] (the frozen
    contract the pipelines write)."""
    return {
        "intro": "Ищу трек...",
        "intent": intent,
        "signals": signals if signals is not None else {
            "language": "en", "era": None, "vocal": None,
            "genre": None, "mood": None},
        "target": target or {"artist": None, "track": None, "variant": None},
        "reference": reference or {"artist": None, "track": None},
        "association_candidates": [],
        "queries": queries or ["black coffee and cigarette"],
    }


def _rank_json(scores, best_index=0, verdict="Есть совпадение.",
               need_clarify=False, clarify=""):
    """LLM#2 rank — scores is a list of (video_id, score) in ranked order."""
    return {
        "ranked": [{"video_id": vid, "score": sc, "rationale": "ok"}
                   for vid, sc in scores],
        "best_index": best_index,
        "confidence": "high",
        "verdict": verdict,
        "need_clarify": need_clarify,
        "clarify_question": clarify,
    }


def _zeroed_rank(vids, verdict="Точного совпадения нет."):
    """LLM#2 rank of the incidents: every score zeroed but a verdict IS
    given — the sec4 «rank zeroed all candidates» shape."""
    return _rank_json([(v, 0) for v in vids], verdict=verdict)


# ════ 1. _group_title_tokens: the garbage tails ════

def test_group_title_tokens_incident_b_fan_tail_cut():
    """«Fancy Footwork : by Step up 3 (HD)» cleans to the tokens of
    «Fancy Footwork» — the uploader's «featured in» credit tail and the
    resolution stamp describe the UPLOAD, not the song."""
    assert _group_title_tokens("Fancy Footwork : by Step up 3 (HD)") == \
        _group_title_tokens("Fancy Footwork") == frozenset(
            {"fancy", "footwork"})


@pytest.mark.parametrize("title", [
    "Fancy Footwork (HD)",
    "Fancy Footwork [4K]",
    "Fancy Footwork (HD 1080)",
    "Fancy Footwork [HD]",
])
def test_group_title_tokens_resolution_stamps_cut(title):
    """Trailing (HD)/(4K)/(HD 1080)/[HD] stamps are cut."""
    assert _group_title_tokens(title) == frozenset({"fancy", "footwork"})


def test_group_title_tokens_official_video_bracket_kept():
    """«(Official Video)» is NOT one of the garbage tails — fuzzy's own
    bracket-junk cleaning handles it; nothing extra is stripped here."""
    assert _group_title_tokens("Bohemian Rhapsody (Official Video)") == \
        frozenset({"bohemian", "rhapsody"})


# ════ 2. the confirmed-key helpers ════

def test_confirmed_keys_and_matching():
    """Keys from confirmed.track AND every confirmed.songs entry; a title
    matches through the same _same_record comparator the official-first
    grouping uses (full-title or track-part key)."""
    signals = {"confirmed": {
        "track": "Mighty Mo Rodgers - Black Coffee and Cigarette",
        "songs": ["Chromeo - Fancy Footwork", "Flo Rida - Club Can't Handle Me"],
    }}
    keys = _confirmed_keys(signals)
    assert "black coffee and cigarette" in keys      # track part of the dash split
    assert "mighty mo rodgers black coffee and cigarette" in keys
    assert "fancy footwork" in keys
    # the official upload (decorated title, Topic layout) matches
    assert _matches_confirmed(
        "Mighty Mo Rodgers - Black Coffee and Cigarette (Official Video)", keys)
    # the fan edit matches through the cut garbage tail (incident В)
    assert _matches_confirmed("Fancy Footwork : by Step up 3 (HD)", keys)
    # a Step Up 3D seed copy matches (incident Б)
    assert _matches_confirmed("Club Can't Handle Me (From Step Up 3D)", keys)
    # an unrelated record does not
    assert not _matches_confirmed("Some Other Song", keys)


def test_confirmed_keys_empty_contracts():
    """No 'confirmed' key, a non-dict, empty strings — everything folds to
    an empty key set and _matches_confirmed is False (the no-op gate)."""
    assert _confirmed_keys({}) == set()
    assert _confirmed_keys({"confirmed": None}) == set()
    assert _confirmed_keys({"confirmed": "Chromeo - Fancy Footwork"}) == set()
    assert _confirmed_keys({"confirmed": {"track": "", "songs": []}}) == set()
    assert not _matches_confirmed("Anything", set())


def test_confirmed_match_short_title_strictness():
    """A one-word confirmed key cannot subset-match a longer title:
    _same_record requires the smaller side to hold >= 2 tokens, so «Hello»
    never marks «Adele - Hello». Only a title that cleans to exactly the
    one word matches."""
    keys = _confirmed_keys({"confirmed": {"track": "Hello"}})
    assert keys == {"hello"}
    assert not _matches_confirmed("Adele - Hello", keys)
    assert _matches_confirmed("Hello", keys)
    assert _matches_confirmed("Hello (HD)", keys)


# ════ 3. incident В: the official-first grouping end-to-end ════

async def test_incident_b_official_first_now_groups_fan_edit(bothub_client):
    """INCIDENT В (without any confirmed facts — the grouping is a data
    fix, not a provenance one): the ranker scores the fan edit
    «Fancy Footwork : by Step up 3 (HD)» 1.0 over the official Chromeo
    upload at 0.5. With the garbage tail cut, both clean into one record
    group and the official-first guard clamps the fan to 0.45 — the
    official heads the list and takes the best slot."""
    entries = [
        {"video_id": "fan1", "title": "Fancy Footwork : by Step up 3 (HD)",
         "channel": "吳昇諳", "duration": 210, "views": 900_000},
        {"video_id": "off1", "title": "Chromeo - Fancy Footwork",
         "channel": "Chromeo - Topic", "duration": 190, "views": 5_000_000},
    ]

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(
                target={"artist": "Chromeo", "track": "Fancy Footwork",
                        "variant": None},
                queries=["chromeo fancy footwork step up 3d"])
        return _rank_json([("fan1", 1.0), ("off1", 0.5)], best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = entries
        out = await search_candidates("chromeo fancy footwork step up",
                                      bothub_client)

    assert [c.video_id for c in out.candidates] == ["off1", "fan1"]
    off = next(c for c in out.candidates if c.video_id == "off1")
    fan = next(c for c in out.candidates if c.video_id == "fan1")
    assert off.score == pytest.approx(0.5)          # the official is untouched
    assert fan.score == pytest.approx(0.45)         # 0.5 - 0.05, demoted
    assert out.candidates[out.best_index].video_id == "off1"


# ════ 4. the [подтверждён] marker in the rank input ════

async def test_confirmed_marker_in_rank_input(bothub_client):
    """(incidents А/Б) A confirmed.songs match carries « [подтверждён]» in
    its LLM#2 candidate line, the rule line teaches the ranker what the
    mark means, an unrelated candidate carries nothing — and the
    signals['confirmed'] dict itself never leaks into the Сигналы join."""
    captured = {}
    entries = [
        {"video_id": "s1", "title": "Club Can't Handle Me (From Step Up 3D)",
         "channel": "FloRidaVEVO", "duration": 230, "views": 40_000_000},
        {"video_id": "x1", "title": "Unrelated Dance Record",
         "channel": "Someone", "duration": 220, "views": 1_000},
    ]

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault(gen_name, []).append(user)
        if gen_name == "llm-extract":
            return _extract_json(
                queries=["step up 3d soundtrack"],
                signals={"language": "en", "era": None, "vocal": None,
                         "genre": None, "mood": None,
                         "confirmed": {"track": "",
                                       "songs": ["Flo Rida - Club Can't Handle Me"]}})
        return _rank_json([("s1", 0.9), ("x1", 0.4)], best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = entries
        out = await search_candidates("песня из фильма step up 3d",
                                      bothub_client)

    rank_msg = captured["llm-rank"][0]
    # the rule line, exactly once
    assert rank_msg.count(
        "Подтверждённые (метка [подтверждён]) — принадлежность источнику/"
        "запросу проверена факт-поиском") == 1
    # the matching candidate line carries the marker; the other does not
    s1_line = next(ln for ln in rank_msg.splitlines() if "video_id=s1" in ln)
    x1_line = next(ln for ln in rank_msg.splitlines() if "video_id=x1" in ln)
    assert "[подтверждён]" in s1_line
    assert "[подтверждён]" not in x1_line
    # the contract dict stays out of the generic Сигналы join
    assert "confirmed=" not in rank_msg
    assert out.candidates[out.best_index].video_id == "s1"


async def test_no_confirmed_no_marker_no_rule_line(bothub_client):
    """Without confirmed facts the rank input is byte-for-bit as before:
    no marker, no rule line."""
    captured = {}

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        captured.setdefault(gen_name, []).append(user)
        if gen_name == "llm-extract":
            return _extract_json(queries=["chromeo fancy footwork"])
        return _rank_json([("off1", 0.9)], best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = [
            {"video_id": "off1", "title": "Chromeo - Fancy Footwork",
             "channel": "Chromeo - Topic", "duration": 190, "views": 5_000_000}]
        await search_candidates("chromeo fancy footwork", bothub_client)

    assert "подтверждён" not in captured["llm-rank"][0]


# ════ 5. incident А: the sec4 rescue on a zeroed rank ════

async def test_incident_a_zeroed_rank_confirmed_rescue(bothub_client, caplog):
    """INCIDENT А: «black coffee and cifgarute» — the ranker zeroed every
    candidate (verdict given, no clarify of its own) while the official
    Mighty Mo Rodgers upload sat in the pool and sonar had confirmed the
    record. The rescue keeps the pool, drops the circular clarify and the
    delivery stands; the best card even carries the confirmed names."""
    entries = [
        {"video_id": "c1", "title": "Black Coffee and Cigarette",
         "channel": "Mighty Mo Rodgers - Topic", "duration": 280,
         "views": 300_000},
        {"video_id": "c2",
         "title": "Mighty Mo Rodgers - Black Coffee and Cigarette (Official Video)",
         "channel": "MightyMoRodgersVEVO", "duration": 285, "views": 90_000},
    ]

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(
                target={"artist": None,
                        "track": "Black Coffee and Cigarette",
                        "variant": None},
                queries=["black coffee and cifgarute"],
                signals={"language": "en", "era": None, "vocal": None,
                         "genre": None, "mood": None,
                         "confirmed": {
                             "track": "Mighty Mo Rodgers - Black Coffee and Cigarette"}})
        return _zeroed_rank(["c1", "c2"], verdict="Нет такого трека.")

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = entries
        with caplog.at_level(logging.INFO, logger="bot.llm_search"):
            out = await search_candidates("black coffee and cifgarute",
                                          bothub_client)

    # the pool survived, the clarify was dropped — a delivery
    assert out.candidates, "confirmed candidates must keep the pool"
    assert out.need_clarify is False
    assert out.clarify_question == ""
    assert 0 <= out.best_index < len(out.candidates)
    assert any("rank zeroed, but 2 confirmed candidate(s) keep the pool"
               in r.getMessage() for r in caplog.records)
    # task 5 rides along: the best card carries the confirmed names
    best = out.candidates[out.best_index]
    assert best.canonical_artist == "Mighty Mo Rodgers"
    assert best.canonical_track == "Black Coffee and Cigarette"


# ════ 6. incident Б: from_source soundtrack seed, zeroed rank ════
#
# The from_source pipeline OWNS the confirmed-contract writer
# (f.signals["confirmed"] = {"track": <sure ground>, "songs": <sure seed>},
# 17.11): the test drives it the way the incident really ran — sonar
# enabled, the ground MISSes (no concrete track), the seed answers SURE
# with the Step Up 3D songs, the ranker still zeroes everything with the
# false «не из Step Up 3D». The fact-stop sonar answers YES (the songs ARE
# on the soundtrack) so the tail gates pass and the sec4 rescue is what
# saves the pool.

async def test_incident_b_from_source_zeroed_rank_rescue():
    """INCIDENT Б: the confident sonar seed left the Step Up 3D song copies
    in the pool and the ranker zeroed ALL of them with a false «не из
    Step Up 3D». signals.confirmed.songs (written by the from_source
    writer from the sure seed) keeps the pool through the sec4 zeroing."""
    from services.llm_search import BothubClient
    # A sonar-enabled client: the ground/seed/fact-stop layers must run
    # (they are what writes/verifies the confirmed contract).
    bothub = BothubClient("http://fake.local", "k", llm_model="m",
                          sonar_model="sonar-pro")
    entries = [
        {"video_id": "s1", "title": "Flo Rida - Club Can't Handle Me",
         "channel": "FloRidaVEVO", "duration": 230, "views": 40_000_000},
        {"video_id": "s2", "title": "Club Can't Handle Me (From Step Up 3D)",
         "channel": "StepUp3DMusic", "duration": 228, "views": 2_000_000},
        {"video_id": "s3", "title": "Roscoe Dash - Show Out",
         "channel": "RoscoeDashVEVO", "duration": 240, "views": 3_000_000},
    ]

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(
                intent="from_source",
                queries=["step up 3d soundtrack"],
                reference={"artist": None, "track": None,
                           "source": "Step Up 3D"})
        if gen_name == "sonar-ground":
            return {"artist": "", "track": "", "sure": False}   # a miss
        if gen_name == "sonar-seed":
            # the confident seed of the official soundtrack (incident Б)
            return {"songs": ["Flo Rida - Club Can't Handle Me",
                              "Roscoe Dash - Show Out"],
                    "sure": True}
        if gen_name == "sonar-factstop":
            return {"in_soundtrack": True, "sure": True,
                    "soundtrack_songs": []}
        return _zeroed_rank(["s1", "s2", "s3"],
                            verdict="Ни один трек не из Step Up 3D.")

    with patch.object(bothub, "chat_json", side_effect=fake_chat), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = entries
        out = await search_candidates("песня из фильма step up 3d",
                                      bothub)

    assert len(out.candidates) == 3, "the seed copies keep the pool"
    assert out.need_clarify is False
    assert out.clarify_question == ""


# ════ 7. no confirmed facts: bit-for-bit previous behavior ════

async def test_zeroed_rank_without_confirmed_still_clarifies(bothub_client):
    """The rescue is strictly provenance-gated: a zeroed rank WITHOUT
    confirmed facts behaves exactly as today — the pool is emptied and the
    shared empty-result clarify is asked (the pre-17.11 sec4 path)."""
    entries = [
        {"video_id": "c1", "title": "Black Coffee and Cigarette",
         "channel": "Mighty Mo Rodgers - Topic", "duration": 280,
         "views": 300_000},
        {"video_id": "z9", "title": "Some Other Record",
         "channel": "Someone", "duration": 220, "views": 1_000},
    ]

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(
                target={"artist": None,
                        "track": "Black Coffee and Cigarette",
                        "variant": None},
                queries=["black coffee and cifgarute"])
        return _zeroed_rank(["c1", "z9"], verdict="Нет такого трека.")

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.pipelines.this_track.FUZZY_SHORTCUT_MIN", 1.1), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = entries
        out = await search_candidates("black coffee and cifgarute",
                                      bothub_client)

    assert out.candidates == []
    assert out.best_index == -1
    assert out.need_clarify is True
    assert out.clarify_question  # the shared empty-result clarify


# ════ 8. incident В tail: the confirmed-track card override ════

async def test_confirmed_track_overrides_ytdlp_garbage_card(bothub_client):
    """INCIDENT В tail: the delivered best is the fan edit whose ytdlp
    enrich resolved the UPLOADER (吳昇諳) as the artist. confirmed.track
    matches the record — the override rewrites the card's canonical
    artist/track from the sonar fact, so the garbage name never reaches
    the user."""
    entries = [
        {"video_id": "ffan", "title": "Fancy Footwork : by Step up 3 (HD)",
         "channel": "吳昇諳", "duration": 210, "views": 900_000},
        {"video_id": "r1", "title": "Round and Round",
         "channel": "Some Fan Channel", "duration": 220, "views": 50_000},
    ]

    def fake_extract_info(url):
        # the ytdlp enrich of the fan upload: the uploader as the "artist"
        if "ffan" in (url or ""):
            return {"artist": "吳昇諳",
                    "track": "Fancy Footwork : by Step up 3",
                    "album": "", "categories": []}
        return None

    async def fake_chat(system, user, max_tokens=None, trace=None,
                        gen_name="llm", model=None):
        if gen_name == "llm-extract":
            return _extract_json(
                target={"artist": None, "track": "Fancy Footwork",
                        "variant": None},
                queries=["fancy footwork step up 3d"],
                signals={"language": "en", "era": None, "vocal": None,
                         "genre": None, "mood": None,
                         "confirmed": {"track": "Chromeo - Fancy Footwork"}})
        return _rank_json([("ffan", 0.9), ("r1", 0.3)], best_index=0)

    with patch.object(bothub_client, "chat_json", side_effect=fake_chat), \
         patch("services.search._extract_info_no_dl",
               side_effect=fake_extract_info), \
         patch("services.llm_search.search_metadata") as mock_yt:
        mock_yt.return_value = entries
        out = await search_candidates("fancy footwork step up", bothub_client)

    best = out.candidates[out.best_index]
    assert best.video_id == "ffan"
    # the confirmed fact replaced the ytdlp garbage on the card
    assert best.canonical_artist == "Chromeo"
    assert best.canonical_track == "Fancy Footwork"
    # sanity: the override only touched the confirmed record's best
    other = next(c for c in out.candidates if c.video_id == "r1")
    assert other.canonical_artist == ""
