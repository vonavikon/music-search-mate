"""Unit tests for match_score: pure logic, no mocks, no network."""
import pytest
from services.match_score import (
    _norm, _clean_title, _artist_score, _title_score,
    _official_score, _fidelity_score, _views_score,
    score_candidate, should_auto_download, is_ambiguous,
)
from services.llm_search import Candidate


class TestNorm:
    def test_casefold(self):
        assert _norm("AC/DC") == _norm("ac/dc")

    def test_yo_to_ye(self):
        assert _norm("ёлка") == _norm("елка")

    def test_punctuation_stripped(self):
        assert _norm("Hello, World!") == "hello world"

    def test_and_expansion(self):
        # The & -> and expansion runs BEFORE punctuation stripping — previously the regex
        # consumed "&" first and the expansion never fired (this test pins the bug)
        assert _norm("Rock & Roll") == "rock and roll"
        assert _norm("Simon & Garfunkel") == _norm("Simon and Garfunkel")

    def test_extra_spaces_collapsed(self):
        assert _norm("a   b") == "a b"


class TestCleanTitle:
    def test_removes_official_video(self):
        assert _clean_title("Hello (Official Video)") == "Hello"

    def test_removes_lyrics_tag(self):
        assert _clean_title("Yesterday (Lyrics)") == "Yesterday"

    def test_removes_hd_tag(self):
        assert _clean_title("Bohemian Rhapsody (HD)") == "Bohemian Rhapsody"

    def test_removes_remastered_year(self):
        assert _clean_title("Imagine [Remastered 2020]") == "Imagine"

    def test_keeps_normal_title(self):
        assert _clean_title("Hotel California") == "Hotel California"

    def test_empty(self):
        assert _clean_title("") == ""

    def test_removes_live_tag(self):
        # the regex strips only a bare "(Live)" in parentheses, with no year inside
        assert _clean_title("Stairway to Heaven (Live)") == "Stairway to Heaven"


class TestArtistScore:
    def test_channel_exact_match(self):
        score, reason = _artist_score("Michael Jackson", "Billie Jean", "Michael Jackson")
        assert score == 0.35
        assert "channel exact" in reason

    def test_channel_with_topic_suffix(self):
        score, reason = _artist_score("Queen - Topic", "Bohemian Rhapsody", "Queen")
        assert score == 0.35
        assert "channel exact" in reason

    def test_token_match_in_channel(self):
        score, reason = _artist_score("Queen Official Channel", "Some Title", "Queen")
        assert score == 0.20
        assert "channel token" in reason

    def test_token_match_in_title(self):
        score, reason = _artist_score("Random Channel", "Queen Tribute Mix", "Queen")
        assert score == 0.20
        assert "title token" in reason

    def test_no_match(self):
        score, reason = _artist_score("AC/DC", "Back in Black", "Queen")
        assert score == 0.0
        assert reason == ""

    def test_empty_target(self):
        score, reason = _artist_score("AC/DC", "Back in Black", "")
        assert score == 0.0

    def test_substring_not_token(self):
        score, reason = _artist_score("m8300", "track", "M83")
        assert score == 0.0


class TestTitleScore:
    def test_exact_match(self):
        score, reason = _title_score("Billie Jean", "Billie Jean")
        assert score == 0.35
        assert reason == "title exact"

    def test_exact_after_clean(self):
        score, reason = _title_score("Billie Jean (Official Video)", "Billie Jean")
        assert score == 0.35

    def test_token_overlap_60_percent(self):
        score, reason = _title_score("Bohemian Rhapsody Live", "Bohemian Rhapsody")
        assert score >= 0.25
        assert "token overlap" in reason

    def test_partial(self):
        score, reason = _title_score("Old Town Road Remix", "Old Town Road")
        assert score >= 0.10

    def test_no_match(self):
        score, reason = _title_score("Hello", "Goodbye")
        assert score == 0.0

    def test_empty_target(self):
        score, reason = _title_score("Hello", "")
        assert score == 0.0


def _c(title="T", channel="C", dur=200, views=1000000, noise=False) -> Candidate:
    return Candidate(
        video_id="fake_id",
        title=title, channel=channel,
        duration=dur, views=views,
        is_noise=noise,
    )


class TestScoreCandidate:
    def test_this_track_full_match_auto(self):
        c = _c("Billie Jean", "Michael Jackson", dur=294)
        result = score_candidate(c, {"artist": "Michael Jackson", "track": "Billie Jean"}, "this_track", 0.0)
        assert result.score >= 0.65, f"score={result.score} should auto-download"
        assert any("artist:channel exact" in r for r in result.reasons)
        assert any("title exact" in r for r in result.reasons)

    def test_this_track_partial_no_auto(self):
        c = _c("Random Title", "Random Channel", dur=200)
        result = score_candidate(c, {"artist": "Michael Jackson", "track": "Billie Jean"}, "this_track", 0.0)
        assert result.score < 0.65

    def test_similar_relies_on_llm_rank(self):
        c = _c("Krokus Screaming in the Night", "Krokus", dur=240)
        result = score_candidate(c, {}, "similar", llm_score=0.95)
        assert result.score >= 0.6
        assert any("llm-rank" in r for r in result.reasons)

    def test_noise_penalty(self):
        c = _c("Song Title (Nightcore Remix)", "Random", dur=180, noise=True)
        result = score_candidate(c, {}, "similar", llm_score=0.90)
        assert any("noise" in r for r in result.reasons)
        # the noise penalty pushes the score below AUTO_SIMILAR (0.70), so auto-download will not trigger
        assert result.score < 0.70

    def test_short_duration_no_fidelity(self):
        c = _c("Short Clip", "Random", dur=30)
        result = score_candidate(c, {"artist": "X", "track": "Y"}, "this_track", llm_score=0.0)
        assert not any("dur" in r for r in result.reasons)


class TestShouldAutoDownload:
    def test_this_track_above_threshold(self):
        assert should_auto_download("this_track", 0.70) is True

    def test_this_track_below_threshold(self):
        assert should_auto_download("this_track", 0.60) is False

    def test_similar_above_threshold(self):
        # threshold raised from 0.70 to 0.85 (preview cards are the primary similar-intent UX)
        assert should_auto_download("similar", 0.72) is False
        assert should_auto_download("similar", 0.86) is True

    def test_similar_below_threshold(self):
        assert should_auto_download("similar", 0.65) is False

    def test_from_source_never_auto(self):
        assert should_auto_download("from_source", 0.99) is False

    def test_unknown_intent(self):
        assert should_auto_download("unknown", 0.86) is True  # similar threshold of 0.85 applies


class TestIsAmbiguous:
    def _c(self, score) -> Candidate:
        c = _c()
        c.match_score = score
        return c

    def test_confidence_low_always_ambiguous(self):
        cands = [self._c(0.90), self._c(0.20)]
        assert is_ambiguous(cands, 0, "low") is True

    def test_clear_gap_not_ambiguous(self):
        cands = [self._c(0.95), self._c(0.50)]
        assert is_ambiguous(cands, 0, "high") is False

    def test_tight_gap_ambiguous(self):
        # "Yesterday" without an artist: top1 and top2 are nearly equal -> offer a choice of versions instead of guessing
        cands = [self._c(0.72), self._c(0.70)]
        assert is_ambiguous(cands, 0, "medium") is True

    def test_gap_exactly_threshold(self):
        cands = [self._c(0.80), self._c(0.72)]  # gap == 0.08 -> NOT ambiguous (>= threshold)
        assert is_ambiguous(cands, 0, "high") is False

    def test_single_candidate_not_ambiguous(self):
        assert is_ambiguous([self._c(0.99)], 0, "high") is False

    def test_empty_not_ambiguous(self):
        assert is_ambiguous([], -1, "high") is False

    def test_runner_up_is_higher_than_best(self):
        # the LLM pick (index 0) disagrees with rule-based scoring: another candidate has a higher match_score -> preview
        cands = [self._c(0.55), self._c(0.80)]
        assert is_ambiguous(cands, 0, "high") is True


def test_guest_vocalist_version_penalized():
    """Extra entities in the title versus the target (guest-vocalist cover)
    trigger a -0.25 penalty, so the original wins deterministically."""
    from services.match_score import score_candidate

    class Axl:
        title = "AC/DC and Axl Rose BACK IN BLACK HD Ceres 1"
        channel = "Rock Island Warehouse"; duration = 281; views = 10**6; is_noise = False
    class Orig:
        title = "AC/DC - Back In Black (Official Video)"
        channel = "AC/DC"; duration = 255; views = 10**8; is_noise = False
    t = {"artist": "AC/DC", "track": "Back in Black"}
    axl = score_candidate(Axl(), t, "this_track", 0.95)
    orig = score_candidate(Orig(), t, "this_track", 0.80)
    assert any("title-extra" in x for x in axl.reasons)
    assert orig.score > axl.score + 0.2
