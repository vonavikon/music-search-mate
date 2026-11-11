"""Shared artist diversity guard - the «не три трека одного исполнителя»
rule (RU: "no three tracks by one artist"), enforced in code before
ranking.

Extracted from MoodPipeline (which introduced it, prod m1: 4/5 slots one
artist) so the similar pipeline enforces the same guarantee (acceptance
rerun s3: the «песня как Nights in White Satin» query, RU for "a song
like Nights in White Satin", returned four Jethro Tull
tracks in the top-5). One implementation, two callers: mood.py keeps a
delegating MoodPipeline.diversity_guard (same signature, log label
"mood"), similar.py calls artist_diversity_guard directly (label
"similar").

The guard is DETERMINISTIC and order-preserving: it reads the filter's
raw ordering (non-noise by views first) and never reorders anything —
rank-time signals (the similar [neighbor?] marker, LLM#2 scores) all come
after it. Keyed by canonical ARTIST, not channel, so one artist cannot
fill the slots under two upload channels.

SAME-RECORD INVARIANT (live F3, 2026-11-05: the «музыка для бега» query,
RU for "music for running", served
Hilary Duff «Come Clean» four times in one card under three different
diversity keys): candidates that are the same RECORD — the same song,
however the uploader ordered, decorated or re-uploaded the title — must
share one diversity outcome: a same-record group keeps only its best 1
candidate. An artist bucket keeps 2, because two DIFFERENT songs by one
artist are legitimate neighbours while two copies of one song are not.
Enforced in two passes: (1) same-record collapse groups the top window by
its cleaned, order-insensitive title key (equal token sets, or a >=
2-token subset — 'come clean' is 'hilary duff come clean' minus the
artist credit); (2) artist crowding on the survivors, the original
keep-2 rule. artist_key validates the title split (a reversed 'Come
Clean - Hilary Duff' split, a mirrored split, a username-like artist
side) and degrades to the channel, then to a title-hash bucket, so a bad
pool entry cannot smuggle one record into three different buckets.

normalized_title_key(title) is the exported record identity — the bot's
miss chain (bot/handler.py) reuses it so a miss never moves to rank+1
when that is the same song again.
"""
import logging
import re

# Same logger name as the pipelines: the deployed log filters and
# dashboards key on "bot.llm_search".
_log = logging.getLogger("bot.llm_search")

# 3+ same-ARTIST candidates among the top-6 trigger the guard; each
# crowded artist keeps only this many candidates (the best by the
# filter's raw ordering — the list is already sorted non-noise/views
# first). A same-record group (the exact same song) keeps 1, not 2.
DIVERSITY_WINDOW = 6
DIVERSITY_TRIGGER = 3
DIVERSITY_KEEP = 2
DIVERSITY_KEEP_RECORD = 1

# The "Artist - Track" split (same dash characters as before the hardening).
_SPLIT_RE = re.compile(r"^(.+?)\s+[-–—]\s+(.+)$")

# A single-token key that looks like a username/channel handle rather than
# an artist name: a letter+digit mix ("agnusia89", "fan2010") or a long
# random-ish run with no spaces. Live F3 keyed one Come Clean copy as
# {agnusia89} — a channel, not an artist.
_JUNK_HANDLE_MIN_LEN = 12
_LETTER_DIGIT_MIX = re.compile(r"[a-z]\d|\d[a-z]")

# The split counts as REVERSED (track first: "Come Clean - Hilary Duff")
# when the channel names the RIGHT side of the dash this much better than
# the left side. Only a clear winner fires, and a junk-handle channel
# never votes (HilaryDuffVEVO would match the artist by squashed
# similarity while being useless as a key).
_REVERSAL_MARGIN = 0.25


def normalized_title_key(title: str) -> str:
    """Order-insensitive, junk-stripped identity of a title: YouTube junk
    (bracket groups, "official video" phrases, 4k/hd tokens, "- Topic")
    removed via fuzzy.clean_title, then the remaining tokens sorted. Any
    two uploads of the same record — "Hilary Duff - Come Clean (Official
    Video) [HD]", "Come Clean - Hilary Duff (Official Music Video) 4K" —
    produce the SAME key ("clean come duff hilary"), whatever the word
    order or decoration. Exported for the miss chain in bot/handler.py:
    rank+1 with the same key is the same song, not the next candidate."""
    from services import fuzzy
    return " ".join(sorted(fuzzy.clean_title(title or "").split()))


def _title_tokens(title: str) -> frozenset:
    """The token set behind normalized_title_key (grouping works on sets,
    logging/debugging on the joined string)."""
    from services import fuzzy
    return frozenset(fuzzy.clean_title(title or "").split())


def _is_junk_handle(tokens: frozenset) -> bool:
    """True for a single-token key that cannot be an artist name: a
    username-style letter+digit mix or a random-ish run of >= 12 chars.
    Multi-token keys and short plain words ("adele", "daft" "punk") stay
    artists."""
    if len(tokens) != 1:
        return False
    w = next(iter(tokens))
    return bool(_LETTER_DIGIT_MIX.search(w)) or len(w) >= _JUNK_HANDLE_MIN_LEN


def _valid_artist_side(artist_raw: str, track_raw: str,
                       artist_tokens: frozenset, track_tokens: frozenset,
                       channel_raw: str, channel_tokens: frozenset) -> bool:
    """Title-split validation (live F3): the LEFT side of the dash may key
    the artist bucket only when it plausibly IS the artist —
      - not a mirrored split (artist side == track side token set: junk);
      - at least one artist token of length >= 3 (drops "a - Song" splits);
      - not a username-like single word ("agnusia89 - Come Clean");
      - not REVERSED: a channel that clearly names the RIGHT side ("Come
        Clean - Hilary Duff" on a Hilary Duff channel) invalidates it."""
    if artist_tokens == track_tokens:
        return False
    if not any(len(t) >= 3 for t in artist_tokens):
        return False
    if _is_junk_handle(artist_tokens):
        return False
    if channel_tokens and not _is_junk_handle(channel_tokens):
        from services import fuzzy
        if (fuzzy.artist_match(track_raw, channel_raw)
                > fuzzy.artist_match(artist_raw, channel_raw) + _REVERSAL_MARGIN):
            return False
    return True


def artist_key(candidate: dict) -> tuple[str, frozenset]:
    """Canonical diversity key of a candidate as a (kind, token-set) pair:

      ("artist", tokens) — the artist parsed from a VALIDATED "Artist -
      Track" title split, falling back to the channel minus the YouTube
      Music "- Topic" suffix;
      ("title", tokens) — the title-hash bucket, the last resort when the
      split fails validation and the channel is a junk handle: the key is
      the cleaned full-title token set, so every copy of the same record
      lands in ONE bucket.

    Keyed by ARTIST, not channel, so one artist cannot fill the slots
    under two upload channels (prod m1: 4/5 slots under two channel
    names). Honest limits: an "artistvevo" channel key does not collapse
    with the title-parsed "artist" (the title split is the primary signal
    and tag-pool queries always carry it), and a reversed split on a
    junk-handle channel keeps its left side as the artist key — in both
    cases the guard's same-record pass still collapses the copies by
    title."""
    from services import fuzzy
    title = (candidate.get("title") or "").strip()
    channel_raw = (candidate.get("channel") or "").removesuffix(" - Topic").strip()
    channel_tokens = frozenset(fuzzy.normalize(channel_raw).split())

    m = _SPLIT_RE.search(title)
    if m:
        artist_raw, track_raw = m.group(1).strip(), m.group(2).strip()
        artist_tokens = frozenset(fuzzy.normalize(artist_raw).split())
        track_tokens = frozenset(fuzzy.clean_title(track_raw).split())
        if _valid_artist_side(artist_raw, track_raw, artist_tokens,
                              track_tokens, channel_raw, channel_tokens):
            return "artist", artist_tokens

    # Fallback 1: the upload channel (minus "- Topic") when it can pass
    # for an artist name.
    if channel_tokens and not _is_junk_handle(channel_tokens):
        return "artist", channel_tokens

    # Fallback 2: the title-hash bucket — same record, one bucket.
    return "title", _title_tokens(title)


def _same_record(a: frozenset, b: frozenset) -> bool:
    """Two cleaned-title token sets identify the same record when EQUAL,
    or when the smaller side (>= 2 tokens — one-word titles are too
    ambiguous to subset-match) is a strict subset of the larger:
    'come clean' is 'hilary duff come clean' minus the artist credit, and
    a reordered or reversed upload cleans to the same set either way.
    Two different songs by one artist ('hilary duff come clean' vs
    'hilary duff fly') never subset-match: each keeps its own words."""
    if not a or not b:
        return False
    if a == b:
        return True
    return (len(a) >= 2 and a < b) or (len(b) >= 2 and b < a)


def _same_record_collapse(filtered: list, enriched: list,
                          label: str) -> tuple[list, list]:
    """Guard pass 1 (live F3): group the top-DIVERSITY_WINDOW candidates
    by record identity (union-find over the _same_record relation) and,
    for every group of DIVERSITY_TRIGGER+ copies, keep only the best
    DIVERSITY_KEEP_RECORD candidate across the WHOLE pool (beyond-window
    copies of a crowded record drop too, mirroring the artist rule).
    Deterministic and order-preserving: 'best' is the filter's raw
    ordering. Returns the (possibly unchanged) lists plus one log line
    per run that dropped something."""
    window = filtered[:DIVERSITY_WINDOW]
    keys = [_title_tokens(c.get("title") or "") for c in window]
    n = len(window)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _same_record(keys[i], keys[j]):
                a, b = _find(i), _find(j)
                if a != b:
                    parent[b] = a

    groups: dict[int, list[int]] = {}
    for i in range(n):
        if keys[i]:
            groups.setdefault(_find(i), []).append(i)
    # One representative token set per crowded group (its first member).
    group_keys = [keys[idxs[0]] for idxs in groups.values()
                  if len(idxs) >= DIVERSITY_TRIGGER]
    if not group_keys:
        return filtered, enriched

    kept, drop, seen = [], set(), [0] * len(group_keys)
    for c in filtered:
        t = _title_tokens(c.get("title") or "")
        g = next((gi for gi, gk in enumerate(group_keys)
                  if _same_record(t, gk)), None)
        if g is None:
            kept.append(c)
            continue
        seen[g] += 1
        if seen[g] > DIVERSITY_KEEP_RECORD:
            drop.add(c["video_id"])
        else:
            kept.append(c)
    if not drop:
        return filtered, enriched
    enriched = [c for c in enriched if c["video_id"] not in drop]
    _log.info("%s diversity: dropped %d duplicate(s) of the same record(s) '%s', "
              "kept %d (same song)",
              label, len(drop),
              ", ".join(sorted(" ".join(sorted(g)) for g in group_keys)),
              DIVERSITY_KEEP_RECORD)
    return kept, enriched


def _artist_crowding(filtered: list, enriched: list,
                     label: str) -> tuple[list, list]:
    """Guard pass 2 - the original «не три трека одного исполнителя» rule
    (RU: "no three tracks by one artist").

    If the top-DIVERSITY_WINDOW filtered candidates hold DIVERSITY_TRIGGER+
    with the same canonical artist key (the (kind, token-set) from
    artist_key), that bucket keeps only its best candidates —
    DIVERSITY_KEEP for artist buckets, DIVERSITY_KEEP_RECORD for
    title-hash buckets (the exact same song) — and the rest drop from the
    whole pool, `enriched` included."""
    counts: dict[tuple[str, frozenset], int] = {}
    for c in filtered[:DIVERSITY_WINDOW]:
        kind, toks = artist_key(c)
        if toks:
            counts[(kind, toks)] = counts.get((kind, toks), 0) + 1
    crowded = {k for k, n in counts.items() if n >= DIVERSITY_TRIGGER}
    if not crowded:
        return filtered, enriched
    keep = {k: (DIVERSITY_KEEP_RECORD if k[0] == "title" else DIVERSITY_KEEP)
            for k in crowded}
    kept, seen, dropped = [], {}, set()
    for c in filtered:
        key = artist_key(c)
        if key in crowded:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > keep[key]:
                dropped.add(c["video_id"])
                continue
        kept.append(c)
    if dropped:
        enriched = [c for c in enriched if c["video_id"] not in dropped]
        if all(kind == "artist" for kind, _ in crowded):
            # The classic line (dashboards/tests key on it): crowded
            # ARTISTS only, keep-2 semantics.
            _log.info("%s diversity: dropped %d candidate(s) of crowded artist(s) '%s', "
                      "kept %d each (by artist)",
                      label, len(dropped),
                      ", ".join(sorted(" ".join(sorted(t)) for _, t in crowded)),
                      DIVERSITY_KEEP)
        else:
            _log.info("%s diversity: dropped %d candidate(s) of crowded bucket(s) '%s', "
                      "kept %d per artist / %d per same record",
                      label, len(dropped),
                      ", ".join(sorted(" ".join(sorted(t)) for _, t in crowded)),
                      DIVERSITY_KEEP, DIVERSITY_KEEP_RECORD)
    return kept, enriched


def artist_diversity_guard(filtered: list, enriched: list,
                           label: str) -> tuple[list, list]:
    """The «не три трека одного исполнителя» rule (RU: "no three tracks by
    one artist") plus same-record collapse, enforced in code.

    Pass 1 (same-record, live F3): 3+ of the top window that are the same
    RECORD — equal or >= 2-token-subset cleaned title keys, i.e. the same
    song under any title ordering, decoration or channel — keep only the
    best 1.

    Pass 2 (artist crowding): 3+ of the top window with the same canonical
    artist key keep only the best 2 (title-hash buckets keep 1), dropping
    the rest from the whole pool, `enriched` included. Returns the
    (possibly unchanged) lists; one log line per pass that dropped.
    `label` is the calling pipeline's name, rendered into the log lines
    ("mood diversity: ..." / "similar diversity: ...")."""
    filtered, enriched = _same_record_collapse(filtered, enriched, label)
    return _artist_crowding(filtered, enriched, label)
