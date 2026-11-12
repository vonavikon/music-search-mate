"""Soundtrack-artist canonization: replace a film/series title that the
YouTube title-split planted into the artist slot with the REAL performer.

Product case (live 12.11): titles like "Fight Club - This Is Your Life" or
"Fight Club - Where Is My Mind?" are split by services/metadata.py into
artist="Fight Club" + track="This Is Your Life" / "Where Is My Mind?" — the
card then shows the MOVIE as the artist while the real performers are
The Dust Brothers ("This Is Your Life") and Pixies ("Where Is My Mind?").
canonize_artist() recognizes motion-picture leftovers in the artist slot (a
year in parentheses, OST/soundtrack/score/film/... markers, all-caps-with-
digits junk), asks Last.fm track.search for the track and returns the artist
of the result whose title tokens best overlap the track tokens. "" when the
artist already looks like a person/band or nothing improves; never raises.

The gate is deliberately one-sided: a PLAUSIBLE name (1-4 words, no digits,
no film markers) is never checked against Last.fm — no extra load on every
card. The flagship bare case ("Fight Club" — no marker, plausible word count)
is covered by FILM_TITLE_ARTISTS: a curated seed of well-known film/series
titles that show up as split-artists on soundtrack uploads. A seeded title
skips the plausible-performer fast path and goes straight to the Last.fm step
(zero network for everything not in the seed). An unseeded obscure film title
without markers still stays as-is — cheap heuristics cannot tell it from a
band name.
"""
import logging
import re

_log = logging.getLogger("bot.artist_canon")

# Motion-picture markers in the ARTIST slot. A false positive on a real band
# name ("The Score") is safe: the Last.fm step only replaces the artist when
# it finds a DIFFERENT one for the same track, so the cost is one cached call.
# Russian markers mirror the tail-strip list in services/metadata.py.
_FILM_MARKER_RE = re.compile(
    r"\b(?:o\.?s\.?t\.?|soundtracks?|scores?|motion\s+picture|films?|movies?|"
    r"series|trailers?|"
    r"саундтрек\w*|фильм\w*|сериал\w*|мультфильм\w*|кино|трейлер\w*|к/ф|т/с)\b",
    re.I,
)
_YEAR_PAREN_RE = re.compile(r"\(\s*(?:19|20)\d{2}\s*\)")  # "Fight Club (1999)"
_MIN_OVERLAP = 0.5  # >= this share of the track's tokens must appear in the match title


def _norm(s: str) -> str:
    """Compare key: casefold, punctuation -> space, whitespace collapsed."""
    s = re.sub(r"[^\w\s]", " ", (s or "").casefold())
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if t}


# Well-known film/series titles that the YouTube title-split plants into the
# artist slot on soundtrack uploads — BARE, without any OST marker and with a
# plausible word count ("Fight Club - This Is Your Life"). Curated seed: a
# match skips the plausible-performer fast path and goes straight to the
# Last.fm step; extend when a live case appears. Stored normalized via _norm
# (casefold, punctuation out), so the lookup is one dict hit, zero network
# for everything else. NOTE: entries are still gated by the Last.fm
# artist-inequality check — a band genuinely named "Joker" is not renamed.
_FILM_TITLE_SEED = (
    # EN
    "fight club", "the matrix", "interstellar", "blade runner", "blade runner 2049",
    "tron legacy", "the dark knight", "inception", "dune", "dune part two",
    "pulp fiction", "the godfather", "taxi driver", "the shining", "joker",
    "mad max fury road", "stranger things", "the witcher", "game of thrones",
    "pirates of the caribbean", "the lion king", "frozen", "moana",
    "la la land", "whiplash",
    # RU
    "бременские музыканты", "ирония судьбы", "кин-дза-дза", "брат", "брат 2",
    "дневник памяти",
)
FILM_TITLE_ARTISTS = frozenset(_norm(t) for t in _FILM_TITLE_SEED)


def _looks_like_performer(artist: str) -> bool:
    """Plausible person/band: 1-4 words, no digits, no film markers. Such an
    artist is trusted as-is — no Last.fm call (the canonization must not add
    load to every card, only to the suspicious ones)."""
    words = artist.split()
    return (1 <= len(words) <= 4
            and not any(ch.isdigit() for ch in artist)
            and not _FILM_MARKER_RE.search(artist))


def _looks_like_film(artist: str) -> bool:
    """The artist slot carries motion-picture evidence: a year in parentheses,
    a soundtrack/film marker word, or all-caps multi-word junk with digits
    ("TERMINATOR 2 JUDGMENT DAY THEME")."""
    if _YEAR_PAREN_RE.search(artist):
        return True
    if _FILM_MARKER_RE.search(artist):
        return True
    words = artist.split()
    return len(words) > 3 and artist.isupper() and any(ch.isdigit() for ch in artist)


async def canonize_artist(artist: str, track: str, metadata_svc) -> str:
    """artist/track from the metadata cascade (+ a MetadataService) -> the REAL
    artist when the split planted a film title into the artist slot, else "".
    Never raises; every decision is logged at info."""
    artist = (artist or "").strip()
    track = (track or "").strip()
    if not artist or not track:
        return ""
    known_film = _norm(artist) in FILM_TITLE_ARTISTS
    if known_film:
        # seeded bare film title ("Fight Club", "Кин-дза-дза"): plausible-word-
        # count but a known movie — skip the performer fast path, ask Last.fm
        _log.info("canonize: %r is a seeded film title, asking last.fm", artist)
    elif _looks_like_performer(artist):
        _log.debug("canonize: %r looks like a real performer, no lookup", artist)
        return ""
    elif not _looks_like_film(artist):
        # neither film-like nor performer-like ("U2", a 6-word label) — nothing
        # cheap says the artist is wrong; don't spend a Last.fm call
        _log.debug("canonize: %r carries no motion-picture evidence, no lookup", artist)
        return ""
    track_tokens = _tokens(track)
    if not track_tokens:
        return ""
    try:
        searcher = getattr(metadata_svc, "track_search", None)
        if not callable(searcher):
            return ""
        results = await searcher(track, limit=5) or []
    except Exception as e:
        _log.debug("canonize: track_search(%r) failed: %s", track, str(e)[:80])
        return ""
    best_artist, best_score, saw_titles = "", 0.0, False
    try:
        for r in results:
            if not isinstance(r, dict):
                continue
            cand = (r.get("artist") or "").strip()
            title = (r.get("name") or r.get("title") or "").strip()
            if not cand or not title:
                continue
            saw_titles = True
            # track-token recall: how much of OUR track the candidate title covers
            score = len(track_tokens & _tokens(title)) / len(track_tokens)
            if score > best_score:  # ties keep the earlier (higher-listeners) row
                best_artist, best_score = cand, score
        if not saw_titles and results:
            # MetadataService.track_search returns [{artist, listeners}] without
            # titles: the query itself is the title match, so trust the top hit
            for r in results:
                if isinstance(r, dict) and (r.get("artist") or "").strip():
                    best_artist, best_score = r["artist"].strip(), 1.0
                    break
    except Exception as e:
        _log.info("canonize: scoring failed: %s", str(e)[:80])
        return ""
    if not best_artist or best_score < _MIN_OVERLAP:
        _log.info("canonize: %r + %r -> no confident match (best overlap %.2f)",
                  artist, track, best_score)
        return ""
    if _norm(best_artist) == _norm(artist):
        _log.info("canonize: %r already the real artist for %r", artist, track)
        return ""
    _log.info("canonize: %r -> %r for %r (overlap %.2f)", artist, best_artist, track, best_score)
    return best_artist
