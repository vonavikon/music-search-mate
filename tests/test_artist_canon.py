"""Unit tests for services/artist_canon.py: a motion-picture leftover in the
artist slot is replaced by the real performer via Last.fm track.search, with
NO network call for plausible band names. All offline: a fake metadata service
with canned Last.fm-like results."""
from services.artist_canon import FILM_TITLE_ARTISTS, _norm, canonize_artist


class FakeMeta:
    """MetadataService stand-in: records track_search calls, returns canned
    Last.fm-like rows [{artist, name, listeners}]."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    async def track_search(self, track, limit=8):
        self.calls.append((track, limit))
        return self.results


class ExplodingMeta:
    async def track_search(self, track, limit=8):
        raise RuntimeError("lastfm down")


# --- film-marker fast path: canonize via best token overlap ---

async def test_film_marker_canonizes_to_best_overlap_not_first_result():
    """artist carries an OST marker -> Last.fm is asked; the BEST title overlap
    wins even when a higher-listeners namesake sits first in the list."""
    fake = FakeMeta([
        {"artist": "Talk Talk", "name": "My Life", "listeners": 5000},      # 1/4 overlap
        {"artist": "The Dust Brothers", "name": "This Is Your Life", "listeners": 900},
    ])
    out = await canonize_artist("Fight Club OST", "This Is Your Life", fake)
    assert out == "The Dust Brothers"
    assert fake.calls == [("This Is Your Life", 5)]  # limit=5 as specified


async def test_live_case_where_is_my_mind_canonizes_to_pixies():
    """The second live example: "Fight Club - Where Is My Mind?" -> Pixies.
    Punctuation ("?") is stripped before the token comparison."""
    fake = FakeMeta([
        {"artist": "Tribute Covers", "name": "Where Is Your Mind", "listeners": 800},  # 3/4
        {"artist": "Pixies", "name": "Where Is My Mind?", "listeners": 3000},          # 4/4
    ])
    out = await canonize_artist("Fight Club (1999)", "Where Is My Mind?", fake)
    assert out == "Pixies"


async def test_year_in_parentheses_triggers_canonization():
    fake = FakeMeta([{"artist": "The Dust Brothers", "name": "This Is Your Life",
                      "listeners": 100}])
    out = await canonize_artist("Fight Club (1999)", "This Is Your Life", fake)
    assert out == "The Dust Brothers"


async def test_all_caps_with_digits_triggers_canonization():
    """No listed marker word here — the all-caps >3 words + digits pattern
    ("TERMINATOR 2 JUDGMENT DAY THEME") is the trigger."""
    fake = FakeMeta([{"artist": "Brad Fiedel", "name": "Terminator 2 Theme",
                      "listeners": 40}])
    out = await canonize_artist("TERMINATOR 2 JUDGMENT DAY THEME",
                                "Terminator 2 Theme", fake)
    assert out == "Brad Fiedel"


async def test_russian_soundtrack_marker_triggers():
    fake = FakeMeta([{"artist": "Hans Zimmer", "name": "Dune", "listeners": 60}])
    out = await canonize_artist("Дюна Саундтрек", "Dune", fake)
    assert out == "Hans Zimmer"


# --- curated film-title seed: the bare flagship case ---

async def test_bare_seeded_film_title_canonizes():
    """The flagship live case: bare "Fight Club" (no markers, plausible word
    count) is in the curated seed -> the performer fast path is skipped and
    Last.fm canonizes to the real performer."""
    fake = FakeMeta([
        {"artist": "Talk Talk", "name": "My Life", "listeners": 5000},  # 1/4 overlap
        {"artist": "The Dust Brothers", "name": "This Is Your Life", "listeners": 900},
    ])
    out = await canonize_artist("Fight Club", "This Is Your Life", fake)
    assert out == "The Dust Brothers"
    assert fake.calls == [("This Is Your Life", 5)]


async def test_bare_seeded_ru_title_matches_normalized():
    """Hyphens/case fall away on both sides: "Кин-дза-дза" hits its
    normalized seed entry and canonizes."""
    fake = FakeMeta([{"artist": "Giya Kancheli", "name": "Melody", "listeners": 30}])
    out = await canonize_artist("Кин-дза-дза", "Melody", fake)
    assert out == "Giya Kancheli"
    assert fake.calls == [("Melody", 5)]


async def test_seed_entries_are_stored_normalized():
    """Every frozenset entry is a fixed point of _norm (casefold, punctuation
    out) — lookups against raw artist strings never miss on formatting."""
    assert FILM_TITLE_ARTISTS
    assert all(_norm(t) == t for t in FILM_TITLE_ARTISTS)


# --- plausible-band fast path: NO network ---

async def test_plausible_band_makes_no_network_call():
    """1-4 words, no digits, no film markers -> trusted as-is even though the
    canned answer would have replaced it."""
    fake = FakeMeta([{"artist": "The Dust Brothers", "name": "This Is Your Life",
                      "listeners": 100}])
    assert await canonize_artist("Dust Brothers", "This Is Your Life", fake) == ""
    assert fake.calls == []


async def test_caps_band_name_without_digits_is_plausible():
    fake = FakeMeta([])
    assert await canonize_artist("NINE INCH NAILS", "Hurt", fake) == ""
    assert fake.calls == []


async def test_bare_band_name_not_in_seed_fast_paths():
    """The mirror of the seed test: a real band name that is NOT a film title
    stays on the no-network fast path even when the canned answer would have
    replaced it."""
    fake = FakeMeta([{"artist": "The Dust Brothers", "name": "Smells Like Teen Spirit",
                      "listeners": 100}])
    assert await canonize_artist("Nirvana", "Smells Like Teen Spirit", fake) == ""
    assert fake.calls == []


async def test_digits_but_no_film_evidence_no_network():
    """"U2" / "50 Cent" fail the performer check but carry no motion-picture
    evidence either -> "" without touching Last.fm."""
    fake = FakeMeta([])
    assert await canonize_artist("U2", "With or Without You", fake) == ""
    assert fake.calls == []
    assert await canonize_artist("50 Cent", "In Da Club", fake) == ""
    assert fake.calls == []


# --- no improvement -> "" ---

async def test_weak_overlap_returns_empty():
    """Best candidate covers only 1/4 of the track tokens (0.25 < 0.5)."""
    fake = FakeMeta([{"artist": "Talk Talk", "name": "My Life", "listeners": 10}])
    assert await canonize_artist("Fight Club OST", "This Is Your Life", fake) == ""


async def test_threshold_half_is_inclusive():
    """Exactly 2/4 overlapping tokens (0.5) passes the >= 0.5 gate."""
    fake = FakeMeta([{"artist": "Moby", "name": "Where Is", "listeners": 5}])
    out = await canonize_artist("Fight Club OST", "Where Is My Mind", fake)
    assert out == "Moby"


async def test_original_artist_already_correct_returns_empty():
    """Last.fm confirms the same artist (case/punctuation-insensitive) — no
    substitution."""
    fake = FakeMeta([{"artist": "fight club soundtrack",
                      "name": "This Is Your Life", "listeners": 20}])
    assert await canonize_artist("Fight Club Soundtrack", "This Is Your Life", fake) == ""


async def test_track_search_failure_returns_empty():
    assert await canonize_artist("Fight Club OST", "This Is Your Life",
                                 ExplodingMeta()) == ""


async def test_track_search_empty_results_returns_empty():
    fake = FakeMeta([])
    assert await canonize_artist("Fight Club OST", "This Is Your Life", fake) == ""
    assert fake.calls == [("This Is Your Life", 5)]


# --- HEAD track_search shape: rows without a title ---

async def test_untitled_rows_fall_back_to_top_listeners():
    """MetadataService.track_search (HEAD) returns [{artist, listeners}] with no
    title field: the query is the title match, so the top hit is trusted."""
    fake = FakeMeta([
        {"artist": "The Dust Brothers", "listeners": 120},
        {"artist": "Some Cover Act", "listeners": 3},
    ])
    out = await canonize_artist("Fight Club OST", "This Is Your Life", fake)
    assert out == "The Dust Brothers"


async def test_untitled_top_row_equal_to_original_returns_empty():
    fake = FakeMeta([{"artist": "FIGHT CLUB OST", "listeners": 10}])
    assert await canonize_artist("Fight Club OST", "This Is Your Life", fake) == ""


# --- degenerate inputs ---

async def test_empty_artist_or_track_returns_empty():
    assert await canonize_artist("", "Track", FakeMeta([])) == ""
    assert await canonize_artist("Fight Club OST", "", FakeMeta([])) == ""


async def test_missing_metadata_service_returns_empty():
    assert await canonize_artist("Fight Club OST", "This Is Your Life", None) == ""
