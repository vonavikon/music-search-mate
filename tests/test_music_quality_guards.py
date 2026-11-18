"""Quality guards for the F1 incident fixes (2026-11-14).

Three prod defects (13-14.11, «F1 The Movie» search) and their guards:
  A. the official «F1® The Movie | Main Trailer» (Warner Bros.) reached the
     top-3 preview -> trailer/studio guard in _filter_candidates;
  B. the fan cut «Sonny Hayes - Lose My Mind» (character name, channel
     F4STR) outranked the official Don Toliver upload -> official-first
     guard at the search_candidates choke point;
  C. the circular-clarify sanitizer repeated the SAME scene question three
     times -> restart_streak-varied wording.

Conventions mirror tests/test_llm_search.py: BothubClient.chat_json mocked
with side_effect (extract + rank dicts, already parsed), search_metadata
mocked with patch, the conftest autouse fixtures keep yt-dlp/suggest/meta
cache offline. Only this file's tests run hermetically.
"""
import pytest
from unittest.mock import patch

from services.llm_search import (
    Candidate,
    _filter_candidates,
    _is_official_upload,
    _is_trailer_or_studio,
    search_candidates,
)


# ════ helpers ════

def _c(vid, title, channel, duration=240, views=100000):
    """A raw (pre-filter) candidate dict, the yt-dlp shape."""
    return {"video_id": vid, "title": title, "channel": channel,
            "duration": duration, "views": views}


def _extract(intent="from_source", language="ru"):
    """LLM#1 extract — an already parsed dict (chat_json contract)."""
    return {
        "intro": "Ищем...",
        "intent": intent,
        "signals": {"language": language, "era": None, "vocal": None,
                    "genre": "rock", "mood": None},
        "target": {"artist": None, "track": None},
        "reference": {"artist": None, "track": None},
        "association_candidates": [],
        "queries": ["lose my mind f1 movie"],
    }


def _rank(scored, best_index=0, need_clarify=False, clarify=""):
    """LLM#2 rank — scored is [(video_id, score), ...]."""
    return {
        "ranked": [{"video_id": vid, "score": sc, "rationale": "ok"}
                   for vid, sc in scored],
        "best_index": best_index,
        "confidence": "high",
        "verdict": "Подходит.",
        "need_clarify": need_clarify,
        "clarify_question": clarify,
    }


_CIRCULAR_ASK = "Уточните исполнителя или название трека"

# A minimal non-empty pool: with an empty one the pipeline never reaches
# LLM#2 and the choke point's f3 branch swaps in its own clarify, bypassing
# the sanitizer under test.
_POOL = [_c("v1", "Lose My Mind", "Don Toliver - Topic", 210, 50_000_000)]

# The sanitizer's two wordings (part C): streak 0 keeps the original scene
# question, streak >= 1 escalates to the detailed scene + lyrics ask.
_A_RU = "В какой момент звучала музыка?"
_B_RU = "Опиши сцену подробнее"
_B_EN = "Describe the scene in more detail"


# ════ part A: trailer / film-studio guard ════

def test_trailer_guard_drops_incident_trailers():
    """(2026-11-14, F1 incident A) The hostile battery: every trailer
    titling and studio upload is dropped from the raw pool."""
    hostile = [
        _c("t1", "F1® The Movie | Main Trailer", "Warner Bros."),
        _c("t2", "F1 | Official Trailer #2", "F1 Movie"),
        _c("t3", "Формула 1 | трейлер", "Кинопоиск"),
        _c("t4", "F1 | OfficiaI Trailer", "F1 Movie"),   # homoglyph: capital-I for l
        _c("t5", "F1 OfficiaI Trailer", "F1 Movie"),     # homoglyph, no pipe at all
        _c("t6", "F1 The Movie | Final TraiIer", "F1 Movie"),
        _c("t7", "Dune full movie", "Some Guy"),
        _c("t8", "Интерстеллар | полный фильм", "RU Movies"),
        _c("t9", "Movie Preview: F1", "F1 Movie"),
        _c("t10", "Mad Max | Preview", "Warner Bros. Pictures"),
        _c("t11", "Some Ballad", "Sony Pictures"),        # studio channel, clean title
    ]
    kept = {c["video_id"] for c in _filter_candidates(hostile)}
    assert kept == set(), f"trailers survived the guard: {sorted(kept)}"


def test_trailer_guard_keeps_music_false_positives():
    """(2026-11-14, F1 incident A) The false-positive battery: a lone
    «trailer»/«teaser» in the left part or a song tag is NOT trailer
    titling; «Official» in a music title is not a trailer marker."""
    legit = [
        _c("f1", "Trailer Choirs - Rockin' The Beer Gut", "Trailer Choirs"),
        _c("f2", "Foo Fighters - The Sky Is A Neighborhood (Official Music Video)",
           "Foo Fighters"),
        _c("f3", "Nirvana - Something in the Way (Official Remastered)", "Nirvana"),
        _c("f4", "Rockin' The Beer Gut (Trailer)", "Trailer Choirs"),
        _c("f5", "Teaser - Song Name", "Some Band"),
    ]
    kept = {c["video_id"] for c in _filter_candidates(legit)}
    assert kept == {"f1", "f2", "f3", "f4", "f5"}


@pytest.mark.parametrize("channel", [
    "Warner Bros.", "Warner Bros. Pictures", "Sony Pictures",
    "Universal Pictures", "Paramount", "Paramount Pictures",
    "20th Century Studios", "Walt Disney Studios", "Movieclips",
    "Searchlight", "A24 Films",
])
def test_studio_channel_pattern(channel):
    """Every listed studio surface drops any title it uploaded."""
    assert _is_trailer_or_studio(_c("s", "Some Ballad", channel))


def test_bare_a24_channel_is_not_a_studio_drop():
    """Bare «A24» is deliberately too wide — only «A24 Films» counts."""
    assert not _is_trailer_or_studio(_c("s", "Some Ballad", "A24"))


def test_trailer_guard_drops_but_keeps_neighbours_in_pool():
    """The guard cuts only the trailers; the music around them survives
    (the incident pool must still deliver the actual song)."""
    pool = [
        _c("t", "F1® The Movie | Main Trailer", "Warner Bros."),
        _c("m", "Sonny Hayes - Lose My Mind", "F4STR"),
    ]
    kept = [c["video_id"] for c in _filter_candidates(pool)]
    assert kept == ["m"]


# ════ part B: official-first guard ════

def test_is_official_upload_criteria():
    """The four officialness evidences, one by one."""
    assert _is_official_upload(Candidate("a", "T", "KatyPerryVEVO", 200, 10))
    assert _is_official_upload(Candidate("a", "T", "Katy Perry - Topic", 200, 10))
    assert _is_official_upload(Candidate("a", "T (Official Music Video)", "Fan", 200, 10))
    c = Candidate("a", "Lose My Mind (From F1® The Movie)", "Don Toliver", 200, 10)
    c.canonical_artist = "Don Toliver"   # enriched by the metadata layer
    assert _is_official_upload(c)
    fan = Candidate("a", "Sonny Hayes - Lose My Mind", "F4STR", 200, 10)
    assert not _is_official_upload(fan)


async def test_official_first_demotes_fan_cut_below_official(bothub_client):
    """(2026-11-14, F1 incident B) The exact prod pair: the fan cut
    «Sonny Hayes - Lose My Mind» at 0.9 over the official Don Toliver
    upload at 0.5 -> the official heads the list, the fan is clamped to
    0.45, best_index follows the official. The official side is detected
    via the enriched canonical_artist matching the channel."""
    entries = [
        _c("fan1", "Sonny Hayes - Lose My Mind", "F4STR", 230, 900000),
        _c("off1", "Lose My Mind (feat. Doja Cat) (From F1® The Movie)",
           "Don Toliver", 210, 50_000_000),
    ]
    # Simulate the enriched top-5: the pipeline's Candidate build keeps the
    # canonical artist (the field the guard reads via getattr).
    canon = {"off1": "Don Toliver"}
    _real_candidate = Candidate

    def _cand_with_canonical(video_id, *a, **kw):
        c = _real_candidate(video_id, *a, **kw)
        if video_id in canon:
            c.canonical_artist = canon[video_id]
        return c

    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract(intent="from_source"),
            _rank([("fan1", 0.9), ("off1", 0.5)], best_index=0),
        ]
        with patch("services.llm_search.search_metadata", return_value=entries):
            with patch("services.llm_search.Candidate",
                       side_effect=_cand_with_canonical):
                out = await search_candidates("рок из фильма f1", bothub_client)

    assert out.candidates[0].video_id == "off1"       # official heads the list
    fan = next(c for c in out.candidates if c.video_id == "fan1")
    off = next(c for c in out.candidates if c.video_id == "off1")
    assert fan.score == pytest.approx(0.45)           # official 0.5 - 0.05
    assert off.score == pytest.approx(0.5)            # the official is untouched
    assert out.candidates[out.best_index].video_id == "off1"


async def test_official_first_topic_channel_end_to_end(bothub_client):
    """The '- Topic' officialness evidence, no test doubles beyond the
    standard chat/search mocks: a fan cut above a YouTube Music upload of
    the same record is demoted below it."""
    entries = [
        _c("fan1", "Lose My Mind (From F1® The Movie)", "F1 Edits", 230, 900000),
        _c("off1", "Lose My Mind", "Don Toliver - Topic", 210, 50_000_000),
    ]
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract(intent="from_source"),
            _rank([("fan1", 0.85), ("off1", 0.6)], best_index=0),
        ]
        with patch("services.llm_search.search_metadata", return_value=entries):
            out = await search_candidates("рок из фильма f1", bothub_client)

    assert out.candidates[0].video_id == "off1"
    fan = next(c for c in out.candidates if c.video_id == "fan1")
    assert fan.score == pytest.approx(0.55)           # 0.6 - 0.05
    assert out.candidates[out.best_index].video_id == "off1"


async def test_official_first_fan_only_pool_unchanged(bothub_client):
    """No official upload in the RECORD GROUP -> the guard is a no-op.
    (2026-11-16) an official of ANOTHER record rides along: the
    officials-only delivery gate (from_source) needs one official in the
    pool to let the fan-only group stand - exactly this no-op case."""
    entries = [
        _c("fan1", "Sonny Hayes - Lose My Mind", "F4STR", 230, 900000),
        _c("fan2", "Lose My Mind (From F1® The Movie)", "F1 Soundtracks", 210, 5000),
        _c("off0", "Hans Zimmer - F1 (From F1® The Movie)", "Hans Zimmer - Topic",
           200, 5_000_000),
    ]
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract(intent="from_source"),
            _rank([("fan1", 0.9), ("fan2", 0.7), ("off0", 0.1)], best_index=0),
        ]
        with patch("services.llm_search.search_metadata", return_value=entries):
            out = await search_candidates("рок из фильма f1", bothub_client)

    assert [c.video_id for c in out.candidates][:2] == ["fan1", "fan2"]
    assert [c.score for c in out.candidates][:2] == [0.9, 0.7]
    assert out.best_index == 0


async def test_official_first_official_only_pool_unchanged(bothub_client):
    """Only official uploads of one record -> no fan to demote, no-op."""
    entries = [
        _c("off1", "Lose My Mind", "Don Toliver - Topic", 210, 50_000_000),
        _c("off2", "Lose My Mind (Official Audio)", "Don Toliver - Topic", 220, 10_000_000),
    ]
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract(intent="from_source"),
            _rank([("off1", 0.5), ("off2", 0.9)], best_index=1),
        ]
        with patch("services.llm_search.search_metadata", return_value=entries):
            out = await search_candidates("рок из фильма f1", bothub_client)

    assert [c.video_id for c in out.candidates] == ["off2", "off1"]
    assert [c.score for c in out.candidates] == [0.9, 0.5]
    assert out.best_index == 0                       # still the 0.9 official


# ════ part C: clarify wording varies with restart_streak ════

async def test_clarify_sanitizer_streak0_keeps_scene_question(bothub_client):
    """Streak 0: the circular artist/title ask is replaced with the
    ORIGINAL scene question (unchanged behavior)."""
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract(intent="from_source"),
            _rank([("v1", 0.5)], need_clarify=True, clarify=_CIRCULAR_ASK),
        ]
        with patch("services.llm_search.search_metadata", return_value=list(_POOL)):
            out = await search_candidates("песня из фильма f1", bothub_client)

    assert out.need_clarify
    assert _A_RU in out.clarify_question
    assert _B_RU not in out.clarify_question


async def test_clarify_sanitizer_streak1_switches_to_detailed_ru(bothub_client):
    """(2026-11-14, F1 incident C) Streak 1: the SAME ask must not repeat —
    the wording escalates to the detailed scene + lyrics question (RU)."""
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract(intent="from_source"),
            _rank([("v1", 0.5)], need_clarify=True, clarify=_CIRCULAR_ASK),
        ]
        with patch("services.llm_search.search_metadata", return_value=list(_POOL)):
            out = await search_candidates("песня из фильма f1", bothub_client,
                                          restart_streak=1)

    assert out.need_clarify
    q = out.clarify_question
    assert q.startswith(_B_RU)
    assert "что происходило на экране" in q
    assert "пару слов из песни" in q
    assert _A_RU not in q


async def test_clarify_sanitizer_streak1_english_variant(bothub_client):
    """Streak 1 + signals language=en -> the EN escalation wording."""
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract(intent="from_source", language="en"),
            _rank([("v1", 0.5)], need_clarify=True, clarify=_CIRCULAR_ASK),
        ]
        with patch("services.llm_search.search_metadata", return_value=list(_POOL)):
            out = await search_candidates("rock song from the f1 movie",
                                          bothub_client, restart_streak=1)

    assert out.need_clarify
    q = out.clarify_question
    assert q.startswith(_B_EN)
    assert "few words of the lyrics" in q
    assert "В какой" not in q


async def test_clarify_sanitizer_this_track_not_replaced(bothub_client):
    """A non-from_source/mood intent keeps its clarify verbatim: the
    sanitizer only owns the scene-search intents."""
    with patch.object(bothub_client, "chat_json") as mock_chat:
        mock_chat.side_effect = [
            _extract(intent="this_track"),
            _rank([("v1", 0.5)], need_clarify=True, clarify=_CIRCULAR_ASK),
        ]
        with patch("services.llm_search.search_metadata", return_value=list(_POOL)):
            out = await search_candidates("тот трек про гонки", bothub_client)

    assert out.need_clarify
    assert out.clarify_question == _CIRCULAR_ASK


# ════ part D: deterministic tie-break in the guard re-sorts (18.11) ════

async def test_official_first_equal_demoted_scores_tiebreak(bothub_client):
    """(2026-11-18, 18.11 acceptance) Two fan cuts of one record BOTH score
    above the official and clamp to the SAME 0.45: the guard's re-sort
    orders the equal-score tie by video_id, so the final order is identical
    whichever fan YouTube listed first (Python's stable sort used to keep
    the input order, making the pick depend on the result order)."""
    entries = [
        _c("fanB", "Sonny Hayes - Lose My Mind", "F4STR", 230, 900000),
        _c("fanA", "Sonny Hayes - Lose My Mind (Fan Cut)", "F1 Edits",
           220, 800000),
        _c("off1", "Lose My Mind (feat. Doja Cat) (From F1® The Movie)",
           "Don Toliver - Topic", 210, 50_000_000),
    ]

    async def _search(pool):
        with patch.object(bothub_client, "chat_json") as mock_chat:
            mock_chat.side_effect = [
                _extract(intent="from_source"),
                _rank([("fanB", 0.9), ("fanA", 0.7), ("off1", 0.5)],
                      best_index=0),
            ]
            with patch("services.llm_search.search_metadata",
                       return_value=list(pool)):
                return await search_candidates("рок из фильма f1", bothub_client)

    out_b_first = await _search(entries)                    # fanB listed first
    out_a_first = await _search([entries[1], entries[0], entries[2]])

    expected = ["off1", "fanA", "fanB"]   # official, then the tie by video_id
    assert [c.video_id for c in out_b_first.candidates] == expected
    assert [c.video_id for c in out_a_first.candidates] == expected
    assert [c.score for c in out_b_first.candidates] == [0.5, 0.45, 0.45]
    assert out_b_first.candidates[out_b_first.best_index].video_id == "off1"
    assert out_a_first.candidates[out_a_first.best_index].video_id == "off1"
