"""Fuzzy matching of a candidate's artist/title against the search target.

Pure stdlib module (difflib + re only): no project imports, no I/O, no
transliteration. Converting a Cyrillic query ("депеш мод стренджлов") into a
Latin target is LLM#1's job upstream; this module only has to keep near-misses
from scoring zero once the right target string is in hand:

  typos            "Deoche mode"            vs "Depeche Mode"
  split/merged     "Strange Love"           vs "Strangelove"
                   "Freestate"              vs "Free State"
  reordered tokens "King of Deep Purple Dreams" vs "Deep Purple - King of Dreams"
  YouTube junk     "(Official Video)", "[Remaster]", "(HD)",
                   "(Full Official Track)", "- Topic" channels

Public API (the search pipeline builds against exactly this):
  normalize(s) -> str
  clean_title(s) -> str
  similarity(a, b) -> float          0..1
  artist_match(candidate, target) -> float
  track_match(candidate, target) -> float
  pair_match(candidate_artist, candidate_title,
             target_artist, target_track) -> float
"""
import re
from difflib import SequenceMatcher

__all__ = [
    "normalize", "clean_title", "similarity",
    "artist_match", "track_match", "pair_match",
]

# Blend weights and guards for pair_match. A right track by a right artist
# must reach >= 0.85 (auto-download band); a right track by a clearly WRONG
# artist (covers/tributes/substitute lineups) must stay below 0.6 — a plain
# 0.6*track + 0.4*artist blend already gives an exact-title match 0.6, so a
# low artist score additionally caps the pair.
_TRACK_WEIGHT = 0.6
_ARTIST_WEIGHT = 0.4
ARTIST_GATE = 0.5     # artist_match below this = wrong artist
WRONG_ARTIST_CAP = 0.55
_CONTAINMENT_SCORE = 0.95  # candidate artist contains target as a token-run

# --- title junk cleaning ----------------------------------------------------

# YouTube auto-generated artist channels: "Depeche Mode - Topic".
_TOPIC_SUFFIX = re.compile(r"\s*[-–—]\s*topic\s*$", re.I)

# Unbracketed junk phrases, removed wherever they appear (case-insensitive):
# "official music video", "lyric video", "full official track", ...
_JUNK_PHRASES = re.compile(
    r"\b(?:"
    r"(?:full\s+)?official(?:\s+music)?(?:\s+(?:video|audio|track|song|visuali[sz]er|lyric\w*))"
    r"|music\s+video"
    r"|lyric\w*\s+video"
    r"|full\s+(?:track|song|video|audio|album|version)"
    r"|(?:hd|hq|4k)\s+video"
    r")\b",
    re.I,
)

# A (...) / [...] group is junk when every word in it is one of these words
# or a number: "(Official Video)", "[Remaster]", "(Remastered 2020)", "(HD)",
# "(Full Official Track)". Meaningful parentheticals — "(Are Made of This)",
# "(Live at Wembley)" — survive because their words are not all junk.
_JUNK_BRACKET_WORDS = {
    "official", "video", "audio", "music", "lyrics", "lyric", "lyrical",
    "hd", "hq", "4k", "uhd", "mv", "visualizer", "visualiser",
    "remaster", "remastered", "full", "track", "version", "bonus",
    "ost", "soundtrack", "single", "album", "mono", "stereo",
}
_FEAT_PREFIX = re.compile(r"^(?:feat|ft|featuring)\.?$", re.I)
_BRACKET_GROUP = re.compile(r"[\(\[\{][^\(\)\[\]\}]*[\)\]\}]")
_WORD_PUNCT = ".,:;!?\"'`"
_NUM_LIKE = re.compile(r"^\d{3,4}[pk]?$")


def _bracket_is_junk(inner: str) -> bool:
    """True when a bracket group carries no title information."""
    words = [w for w in re.split(r"[\s/]+", inner.strip()) if w]
    if not words:
        return True
    if _FEAT_PREFIX.match(words[0].strip(_WORD_PUNCT)):
        return True  # "(feat. Someone)" — guest list, not the title
    for w in words:
        w = w.lower().strip(_WORD_PUNCT)
        if w in _JUNK_BRACKET_WORDS or _NUM_LIKE.match(w):
            continue
        return False
    return True


# Standalone junk tokens dropped after phrase/bracket cleaning. Symmetric —
# both sides of every comparison pass through clean_title — so a real
# one-word title ("Audio", "Video") still self-matches at 1.0.
_JUNK_TOKENS = {
    "official", "video", "audio", "lyrics", "lyric", "lyrical",
    "hd", "hq", "4k", "uhd", "mv", "visualizer", "visualiser",
    "remaster", "remastered",
}

_PUNCT_TO_SPACE = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Lowercase, punctuation -> space (word boundaries survive), collapse
    whitespace. Deliberately no diacritic folding and no transliteration —
    out of scope for this module."""
    s = (s or "").lower()
    s = _PUNCT_TO_SPACE.sub(" ", s)
    return _WHITESPACE.sub(" ", s).strip()


def clean_title(s: str) -> str:
    """Strip YouTube title/channel junk, then normalize():
    bracketed junk groups, junk phrases, "- Topic" suffix, standalone junk
    tokens. An all-junk title cleans to "" (comparison then scores 0)."""
    s = s or ""
    s = _TOPIC_SUFFIX.sub("", s)
    s = _JUNK_PHRASES.sub(" ", s)
    s = _BRACKET_GROUP.sub(
        lambda m: " " if _bracket_is_junk(m.group(0)[1:-1]) else m.group(0), s)
    s = normalize(s)
    return " ".join(t for t in s.split() if t not in _JUNK_TOKENS)


# --- similarity -------------------------------------------------------------

def similarity(a: str, b: str) -> float:
    """Best of full-string, token and squashed comparison, 0..1.

    - full ratio: plain SequenceMatcher on the normalized strings;
    - token ratio: order-insensitive (sorted unique tokens), so reordered
      words ("King of Deep Purple Dreams" vs "Deep Purple King of Dreams")
      do not lose points;
    - subset coverage: when the shorter side has >= 2 tokens and they all
      appear on the longer side, score 1.0 — candidate titles that wrap the
      track with extra entities ("Deep Purple - King of Dreams" vs track
      "King of Dreams") are rewarded, per the token-subset requirement;
    - squashed ratio: spaces removed on both sides, so "Strange Love" vs
      "Strangelove" and "Free State" vs "Freestate" compare clean.

    Empty (or all-punctuation) side -> 0.0.
    """
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    best = SequenceMatcher(None, na, nb).ratio()

    ta, tb = na.split(), nb.split()
    if len(ta) > len(tb):
        ta, tb = tb, ta
    joined = SequenceMatcher(
        None, " ".join(sorted(set(ta))), " ".join(sorted(set(tb)))).ratio()
    best = max(best, joined)

    if len(ta) >= 2:
        longer = set(tb)
        hit = sum(len(t) for t in ta if t in longer)
        total = sum(len(t) for t in ta)
        if total:
            best = max(best, hit / total)

    sa, sb = na.replace(" ", ""), nb.replace(" ", "")
    if sa and sb:
        best = max(best, SequenceMatcher(None, sa, sb).ratio())
    return best


def _contains_token_run(haystack: list, needle: list) -> bool:
    """True when needle appears in haystack as consecutive tokens."""
    n = len(needle)
    if not n or n > len(haystack):
        return False
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


# --- matchers ---------------------------------------------------------------

def artist_match(candidate_artist: str, target_artist: str) -> float:
    """Channel/artist title vs target artist, 0..1.

    Channel titles arrive as "Depeche Mode - Topic" (cleaned) and often
    append promo words ("Depeche Mode Official"): a candidate that fully
    contains the target artist as a token-run scores >= 0.9."""
    ca = clean_title(candidate_artist)
    ta = clean_title(target_artist)
    if not ca or not ta:
        return 0.0
    if ca == ta:
        return 1.0
    if _contains_token_run(ca.split(), ta.split()):
        return _CONTAINMENT_SCORE
    return similarity(ca, ta)


def track_match(candidate_title: str, target_track: str) -> float:
    """Candidate title vs target track, 0..1. clean_title on both sides
    first, so "(Full Official Track)"-style junk cannot zero the score."""
    return similarity(clean_title(candidate_title), clean_title(target_track))


def pair_match(candidate_artist: str, candidate_title: str,
               target_artist: str | None, target_track: str | None) -> float:
    """Combined artist+track score, 0..1.

    target_artist None -> track_match alone; target_track empty ->
    artist_match alone; otherwise 0.6*track + 0.4*artist. A clearly wrong
    artist (artist_match < ARTIST_GATE) caps the pair at WRONG_ARTIST_CAP:
    the right title by somebody else (cover, tribute, mashup) must not
    reach the 0.6 auto band on title strength alone."""
    if target_artist is None:
        return track_match(candidate_title, target_track or "")
    if not target_track:
        return artist_match(candidate_artist, target_artist)
    art = artist_match(candidate_artist, target_artist)
    trk = track_match(candidate_title, target_track)
    score = _TRACK_WEIGHT * trk + _ARTIST_WEIGHT * art
    if art < ARTIST_GATE:
        score = min(score, WRONG_ARTIST_CAP)
    return score
