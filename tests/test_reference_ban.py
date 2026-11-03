"""Unit tests for the reference ban: detecting candidate matches against reference.artist."""
import pytest
from services.llm_search import Candidate


def apply_reference_ban(
    ranked: list[dict],
    candidates: list[Candidate],
    reference: dict,
) -> tuple[list[dict], int]:
    """Return (reordered ranked list, number of banned candidates)."""
    ref_artist = (reference or {}).get("artist", "").strip() if reference else ""
    if not ref_artist:
        return ranked, 0

    ra = ref_artist.casefold()
    banned = []
    kept = []
    for i, item in enumerate(ranked):
        c = candidates[i] if i < len(candidates) else None
        channel = (c.channel or "").casefold() if c else ""
        title = (c.title or "").casefold() if c else ""
        if ra in channel or ra in title:
            banned.append(item)
        else:
            kept.append(item)

    return kept + banned, len(banned)


def _d(video_id="vid") -> dict:
    return {"video_id": video_id}


def _c(channel="C", title="T") -> Candidate:
    return Candidate(video_id="vid", title=title, channel=channel, duration=200, views=1000)


class TestReferenceBan:
    def test_no_reference_artist(self):
        ranked = [_d("a"), _d("b"), _d("c")]
        cand = [_c("A"), _c("B"), _c("C")]
        result, n = apply_reference_ban(ranked, cand, {"artist": ""})
        assert result == ranked
        assert n == 0

    def test_ban_single_match(self):
        ranked = [_d("a"), _d("b"), _d("c")]
        cand = [_c("A"), _c("B"), _c("C")]
        result, n = apply_reference_ban(ranked, cand, {"artist": "A"})
        assert n == 1
        assert result[0]["video_id"] == "b"

    def test_ban_all_candidates(self):
        ranked = [_d("a"), _d("b")]
        cand = [_c("Depeche Mode"), _c("Depeche Mode Official")]
        result, n = apply_reference_ban(ranked, cand, {"artist": "Depeche Mode"})
        assert n == 2
        assert len(result) == 2

    def test_ban_artist_in_title(self):
        ranked = [_d("a")]
        cand = [_c("Random", "Depeche Mode Tribute Mix")]
        result, n = apply_reference_ban(ranked, cand, {"artist": "Depeche Mode"})
        assert n == 1
        assert len(result) == 1

    def test_case_insensitive(self):
        ranked = [_d("a")]
        cand = [_c("DEPECHE MODE")]
        result, n = apply_reference_ban(ranked, cand, {"artist": "depeche mode"})
        assert n == 1
