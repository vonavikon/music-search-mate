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


# ---------- bio disambiguation boilerplate (Last.fm wiki opener) ----------
# Real shapes observed in the production caches (12.11, Nirvana):
#   RU: «На Last.fm отслеживается несколько исполнителей с именем "Nirvana".
#        Вот шесть из них, перечисленные в порядке известности:\n\n1) Nirvana...»
#   EN: "There are multiple artists tracked as \"Nirvana\" on Last.fm.
#        Following are six, listed in order of prominence:\n\n1) Nirvana..."
# The block is TWO sentences flowing into a numbered list. A surgical
# sentence cut leaves fragments («Вот шесть из них...» survived the first
# fix); the reliable anchor is the first numbered item.
import re as _re

_DISAMBIG_MARK_RE = _re.compile(
    r"(multiple\s+artists"
    r"|tracked\s+as\s+\""
    r"|with\s+this\s+name"
    r"|на\s+last\.fm\s+отслежива"
    r"|отслежива\w*\s+(несколько|\d+|six|шесть)\s+исполнител"
    r"|несколько\s+исполнител\w*\s+с\s+(этим|тем)\s+именем"
    r"|есть\s+несколько\s+исполнител)", _re.I)
_LIST_ITEM_RE = _re.compile(r"\n\s*\d{1,2}\)\s")
_SENT_END_RE = _re.compile(r"[.!?](?:\s|$)")
_INTRO_TAIL_RE = _re.compile(r"^[^\n]{10,160}:\s*(?:\n|$)")


def strip_bio_disambiguation(bio: str) -> str:
    """Drop the Last.fm 'multiple artists with this name' opener.

    The real bio starts either at the first numbered item (the common
    list shape) or right after the boilerplate sentence. A bio that is
    nothing but boilerplate becomes '' (no info line at all). A clean bio
    passes through unchanged.
    """
    b = (bio or "").strip()
    if not b:
        return ""
    mark = _DISAMBIG_MARK_RE.search(b[:600])
    if not mark:
        return b
    # Shape 1 (both production shapes): opener -> intro sentence -> list
    item = _LIST_ITEM_RE.search(b[:900])
    if item and item.start() >= mark.start():
        return b[item.end():].lstrip()
    # Shape 2: a single boilerplate sentence, no list
    sent = _SENT_END_RE.search(b, mark.end())
    rest = b[sent.end():].lstrip() if sent else ""
    # an intro-to-list tail may still lead («Вот шесть из них...:»)
    for _ in range(2):
        tail = _INTRO_TAIL_RE.match(rest)
        if not tail:
            break
        rest = rest[tail.end():].lstrip()
    return rest
