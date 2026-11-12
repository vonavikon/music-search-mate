"""Channel-name cleaning for artist lookups and cache keys.

Backport of the web wave-18 channel fix: the channel «Deep Purple Official»
went to Last.fm verbatim -> an artist.getinfo miss -> cards without bio,
album or year. Channel-title litter («- Topic», «- Official», «Official»,
«VEVO», «(Official)», «(Official Video)», any case) is stripped BEFORE any
metadata lookup or cache key is built (bot/handler feeds resolve() and the
meta_cache keys with the cleaned name).

Case-preserving: the display name survives, only the channel litter goes
(same contract as the web-study _clean_channel tests).
"""
import re

# «Artist - Topic» (YouTube auto-artist channels), any dash form
_TOPIC_RE = re.compile(r"\s*[-\u2013\u2014]\s*topic\s*$", re.I)
# «Artist (Official)» / «Artist (Official Video)» / «Artist (VEVO)»: any
# bracketed tail starting with official/vevo
_BRACKET_RE = re.compile(r"\(\s*(?:official|vevo)[^)]*\)\s*$", re.I)
# «Artist - Official» / «Artist Official» / «ArtistVEVO»: a bare suffix,
# the dash optional so both spaced and glued forms match
_WORD_RE = re.compile(r"\s*[-\u2013\u2014]?\s*(?:official|vevo)\s*$", re.I)


def clean_channel_name(name: str) -> str:
    """Raw YouTube channel -> a clean artist name.

    Strips channel-title suffix litter ('- Topic', '- Official', 'Official',
    'VEVO', '(Official)', '(Official Video)', case-insensitive, dash forms
    -, en/em), then trims separator crumbs. Blank/emptied input ->
    'Unknown Artist' (the metadata cascade's fallback marker). A clean name
    passes through unchanged.
    """
    ch = (name or "").strip()
    if not ch:
        return "Unknown Artist"
    ch = _TOPIC_RE.sub("", ch)
    ch = _BRACKET_RE.sub("", ch)
    ch = _WORD_RE.sub("", ch)
    ch = ch.strip(" -\u2013\u2014|")
    return ch or "Unknown Artist"
