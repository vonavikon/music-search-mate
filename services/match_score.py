"""Rule-based match score of a candidate against the request.

Replaces uncalibrated LLM confidence with a numeric score built from
features. Goals (cross-checked against gpt-5.2): calibrated auto-download,
explainability (score = weighted feature sum with a reason log), and
protection against metadata substitution. Deterministic: it does not depend
on the LLM being obedient. Soft constraints (prompts) cover typical cases;
hard ones (this module) guard the critical invariants of automatic delivery.

Score range 0..1. Features (weights tuned so that auto-download requires a
strong artist+title match for this_track):
  artist_match  0..0.35  candidate channel/artist vs target.artist
  title_match   0..0.35  cleaned title vs target.track
  official      0..0.15  topic channel (YT Music) / VEVO / Official
  fidelity      0..0.10  duration within the typical range for the track
  views         0..0.05  popularity (weak signal, views can be inflated)
  noise        -0.20     remix/cover/live/tribute (is_noise from the filter)

For similar/from_source (target is None) there is no comparison object:
base = LLM rank score * 0.7 (relevance is the LLM's call, discounted for
overconfidence) plus the same quality features. Thresholds: >= AUTO_THRESHOLD
triggers auto-download, otherwise a top-3 preview. The initial thresholds are
meant to be retuned from production hit/miss feedback once match_score labels
accumulate.
"""
import re
from dataclasses import dataclass, field


# Auto-download thresholds per intent (cross-checked against gpt-5.2): this_track
# is softer (exact comparison target + artist+title+fidelity bonus), similar is
# stricter (relevance is the LLM's call, discounted for overconfidence).
# from_source is always preview (1/8 hit rate, uncalibrated).
AUTO_THIS_TRACK = 0.65
# Preview cards with covers made visual picking the main UX for similar —
# auto-download remains only for very confident matches.
AUTO_SIMILAR = 0.85
LOW_THRESHOLD = 0.4    # below this the card gets an "unsure" marker
AMBIGUITY_GAP = 0.08   # top1-top2 match_score gap below this = ambiguous


def should_auto_download(intent: str, match_score: float) -> bool:
    """Auto-download decision from the rule-based match_score and intent.
    from_source is always preview. this_track/similar auto-download on a strong
    match above the intent threshold. Single decision point — the handler calls
    this, never the threshold directly."""
    if intent == "from_source":
        return False
    threshold = AUTO_THIS_TRACK if intent == "this_track" else AUTO_SIMILAR
    return match_score >= threshold


def is_ambiguous(candidates: list, best_index: int, confidence: str) -> bool:
    """Ambiguous query ("Yesterday", "Hallelujah" — many versions).

    Even when the best candidate's match_score is high, if the runner-up is nearly
    equal (gap < AMBIGUITY_GAP) or the model itself is unsure (confidence low),
    don't download one version — show top-3 for the user to pick. Catches the case
    where several versions of the same title are equally likely and a guess would
    serve the wrong one.

    candidates is the ranked list (by LLM score); best_index is the LLM's pick.
    The runner-up only counts when it is a DIFFERENT track (track_key): duplicate
    uploads of the same song (Official/Topic/VEVO of the same recording) all
    serve the user equally well, so equal scores among them are confidence, not
    ambiguity. A higher-scoring different track is still a disagreement
    signal → preview (negative gap < AMBIGUITY_GAP)."""
    if confidence == "low":
        return True
    if not candidates or not (0 <= best_index < len(candidates)):
        return False
    best = candidates[best_index]
    key = track_key(best)
    rest = [c for i, c in enumerate(candidates) if i != best_index]
    if not rest:
        return False
    others = [c for c in rest if track_key(c) != key]
    if not others:
        return False
    runner_up = max(others, key=lambda c: getattr(c, "match_score", 0.0))
    return (best.match_score - runner_up.match_score) < AMBIGUITY_GAP


@dataclass
class Scored:
    score: float
    reasons: list = field(default_factory=list)


def _norm(s: str) -> str:
    """Normalization for comparison: casefold, ё=е, punctuation->space, &->and, collapse spaces.
    &: replace BEFORE stripping punctuation — after the sub nothing remains of '&'
    (review caught the replacement was dead code: "Simon & Garfunkel" != "Simon and…")."""
    s = (s or "").casefold().strip()
    s = s.replace("ё", "е").replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Metadata tails in candidate titles that hurt comparison. Local copy of the
# cleanup (like metadata._clean_raw_title) — keeps the module self-contained,
# with no dependency on config.
_NOISE_TAIL = re.compile(
    r"\s*[\(\[\|]\s*(?:official\s*(?:video|audio|music\s*video|lyric\s*video)|"
    r"music\s*video|hd|4k|hq|lyrics?|audio|video|remastered(?:\s*\d{4})?|"
    r"ost|soundtrack|feat\.?\s+[^)\]\|]+|ft\.?\s+[^)\]\|]+|"
    r"cover|karaoke|tribute|live)\s*[\)\]\|]",
    re.I,
)
_EMPTY_PARENS = re.compile(r"\s*[\(\[\|]\s*[\)\]\|]\s*$")


def _clean_title(t: str) -> str:
    t = _NOISE_TAIL.sub("", t or "")
    t = _EMPTY_PARENS.sub("", t)
    return t.strip(" -–—|")


def track_key(candidate) -> str:
    """Identity of the underlying track for duplicate detection.

    YouTube carries several uploads of the same song (Official video, Topic
    auto-channel, VEVO, lyric videos): they all answer the query equally, so
    they must not look like ambiguity. Key = normalized cleaned title; when
    the title carries no "artist - track" separator (Topic uploads often put
    the artist only in the channel), the channel (minus "- Topic") is folded
    in. Imperfect by design: a rare false "different" only falls back to a
    top-3 preview, which is safe.
    """
    title = _clean_title(getattr(candidate, "title", "") or "")
    key = _norm(title)
    if " - " not in (title or ""):
        ch = re.sub(r"\s*[-–—]\s*topic\s*$", "", getattr(candidate, "channel", "") or "", flags=re.I)
        key = _norm(f"{ch} {title}")
    return key


def _artist_score(channel: str, title: str, target_artist: str) -> tuple[float, str]:
    """Artist match: exact by channel (0.35) or token-match in channel/title (0.20).
    Token-match (word boundaries), not substring: "M83" won't match "m8300"."""
    ta = _norm(target_artist)
    if not ta:
        return 0.0, ""
    ch = re.sub(r"\s*[-–—]\s*topic\s*$", "", channel or "", flags=re.I)
    ch_key = _norm(ch)
    if ch_key == ta:
        return 0.35, "artist:channel exact"
    pat = r"\b" + re.escape(ta).replace(r"\ ", r"\s+") + r"\b"
    if ch_key and re.search(pat, ch_key):
        return 0.20, "artist:channel token"
    if re.search(pat, _norm(title)):
        return 0.20, "artist:title token"
    return 0.0, ""


def _title_score(title: str, target_track: str) -> tuple[float, str]:
    """Title match: exact (0.35), token overlap >=60% (0.25), partial (0.10)."""
    tt = _norm(target_track)
    if not tt:
        return 0.0, ""
    ct = _norm(_clean_title(title))
    if not ct:
        return 0.0, ""
    if ct == tt:
        return 0.35, "title exact"
    tt_toks = set(tt.split())
    ct_toks = set(ct.split())
    if tt_toks and len(tt_toks & ct_toks) / len(tt_toks) >= 0.6:
        return 0.25, "title token overlap"
    if tt in ct or ct in tt:
        return 0.10, "title partial"
    return 0.0, ""


def _official_score(channel: str) -> tuple[float, str]:
    ch = (channel or "").lower()
    if ch.endswith("- topic") or "- topic" in ch:
        return 0.15, "topic-channel"
    if "vevo" in ch or "official" in ch:
        return 0.10, "vevo/official"
    return 0.0, ""


def _fidelity_score(duration: int) -> tuple[float, str]:
    if 120 <= duration <= 480:
        return 0.10, "dur typical"
    if 60 <= duration <= 900:
        return 0.05, "dur plausible"
    return 0.0, ""


def _views_score(views: int) -> tuple[float, str]:
    if views and views > 100000:
        return 0.05, "views>100k"
    if views and views > 10000:
        return 0.02, "views>10k"
    return 0.0, ""


def score_candidate(c, target: dict, intent: str, llm_score: float) -> Scored:
    """Score a candidate. target={artist,track} for this_track, else {}.
    c is a Candidate object (title/channel/duration/views/is_noise). llm_score is the LLM#2 rank-score."""
    reasons = []
    tgt_artist = (target or {}).get("artist")
    tgt_track = (target or {}).get("track")
    has_target = bool(tgt_artist or tgt_track)

    fid_v, fid_r = _fidelity_score(getattr(c, "duration", 0))

    if has_target:
        # this_track: deterministic matching against the target.
        base = 0.0
        art_v = 0.0
        # A title with extra entities beyond the target (case: searching
        # "AC/DC Back in Black" surfaced "AC/DC and Axl Rose BACK IN BLACK HD")
        # signals a guest/substitute lineup or someone else's recording.
        # Penalize deterministically; the LLM ranker missed it.
        if tgt_track:
            _stop = {"hd", "4k", "hq", "official", "video", "audio", "lyrics",
                     "clip", "remastered", "remaster", "live", "sub", "es"}
            tgt_tokens = {t for t in re.split(r"\W+", (tgt_artist or "").lower() + " " + tgt_track.lower()) if t and t not in _stop}
            ttl_tokens = [t for t in re.split(r"\W+", c.title.lower()) if t and t not in _stop]
            extra = [t for t in ttl_tokens if t not in tgt_tokens]
            # guest-performance connectors are always extra here
            guest = sum(1 for t in extra if t in ("and", "&", "feat", "ft", "with", "vs"))
            if extra and (len(extra) >= 3 or guest):
                base -= 0.25
                reasons.append(f"title-extra(-{len(extra)}t)")

        if tgt_artist:
            art_v, r = _artist_score(c.channel, c.title, tgt_artist)
            base += art_v
            if r:
                reasons.append(r)
        ttl_v = 0.0
        if tgt_track:
            ttl_v, r = _title_score(c.title, tgt_track)
            base += ttl_v
            if r:
                reasons.append(r)
        if fid_v:
            base += fid_v
            reasons.append(fid_r)
        # both+fid bonus (cross-checked against gpt-5.2): for this_track, matching
        # artist AND title AND typical duration is a strong invariant of the exact
        # track. Small (+0.10), doesn't inflate the individual weights: doesn't
        # lift partial/feat/mashup/noise versions.
        if art_v > 0 and ttl_v > 0 and fid_v > 0:
            base += 0.10
            reasons.append("both+fid match")
    else:
        # similar/from_source: relevance is the LLM's call, discounted for overconfidence.
        base = (llm_score or 0) * 0.7
        reasons.append(f"llm-rank {(llm_score or 0):.2f}")
        if fid_v:
            base += fid_v
            reasons.append(fid_r)

    v, r = _official_score(getattr(c, "channel", ""))
    base += v
    if r:
        reasons.append(r)
    v, r = _views_score(getattr(c, "views", 0))
    base += v
    if r:
        reasons.append(r)
    if getattr(c, "is_noise", False):
        base -= 0.20
        reasons.append("noise -0.20")

    return Scored(max(0.0, min(1.0, base)), reasons)
