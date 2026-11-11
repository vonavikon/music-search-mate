"""Unit tests for fuzzy: pure stdlib logic, no mocks, no network."""
import pytest

from services.fuzzy import (
    normalize, clean_title, similarity, artist_match, track_match, pair_match,
)


class TestNormalize:
    @pytest.mark.parametrize("raw,expected", [
        ("Depeche Mode", "depeche mode"),
        ("Hello, World!", "hello world"),
        # punctuation becomes a space, so word boundaries survive
        ("AC/DC", "ac dc"),
        ("Rock-N-Roll", "rock n roll"),
        ("  Multiple   Spaces  ", "multiple spaces"),
        ("", ""),
        ("...", ""),
    ])
    def test_table(self, raw, expected):
        assert normalize(raw) == expected

    def test_diacritics_not_folded(self):
        # diacritic folding is explicitly out of scope for this module
        assert normalize("Dîäcritics") == "dîäcritics"

    def test_cyrillic_kept_as_is(self):
        # no transliteration here — that is LLM#1's job
        assert normalize("Депеш Мод") == "депеш мод"


class TestCleanTitle:
    @pytest.mark.parametrize("raw,expected", [
        ("Hotel California", "hotel california"),
        # bracketed junk
        ("Hello (Official Video)", "hello"),
        ("Imagine [Remastered 2020]", "imagine"),
        ("Bohemian Rhapsody (HD)", "bohemian rhapsody"),
        ("Yesterday (Lyrics)", "yesterday"),
        ("Track Name (Audio)", "track name"),
        ("Dior wild man (Full Official Track)", "dior wild man"),
        ("Song (HD 1080p)", "song"),
        ("(2011)", ""),
        # unbracketed junk phrases and standalone junk tokens
        ("Song official music video", "song"),
        ("Song Lyric Video", "song"),
        # channel suffix
        ("Depeche Mode - Topic", "depeche mode"),
        ("Enrique Iglesias — Topic", "enrique iglesias"),
        # guest lists are junk, meaningful parentheticals are kept
        ("Song (feat. Taylor Swift)", "song"),
        ("Sweet Dreams (Are Made of This)", "sweet dreams are made of this"),
        ("Live and Let Die (Live at Wembley)", "live and let die live at wembley"),
        # degenerate inputs
        ("", ""),
        ("(Official Video)", ""),
    ])
    def test_table(self, raw, expected):
        assert clean_title(raw) == expected

    def test_all_junk_title_empties(self):
        assert clean_title("[Remaster] (HD) official video lyrics") == ""


class TestSimilarity:
    @pytest.mark.parametrize("a,b,expected", [
        ("hello", "hello", 1.0),
        ("ENJOY THE SILENCE", "enjoy the silence", 1.0),
        ("Hello!", "hello", 1.0),
    ])
    def test_exact_after_normalize(self, a, b, expected):
        assert similarity(a, b) == pytest.approx(expected)

    @pytest.mark.parametrize("a,b", [
        ("", "hello"),
        ("hello", ""),
        ("", ""),
        ("!!!", "hello"),  # punctuation-only side normalizes to empty
    ])
    def test_empty_side_is_zero(self, a, b):
        assert similarity(a, b) == 0.0

    @pytest.mark.parametrize("a,b,minimum", [
        # split/merged words
        ("Strange Love", "Strangelove", 0.85),
        ("Freestate", "Free State", 0.85),
        # near-miss spelling
        ("King of Dreams", "King of Dream", 0.85),
        # reordered tokens compare order-insensitively
        ("Deep Purple – King of Dreams", "King of Deep Purple Dreams", 0.9),
        # typo
        ("depeche mode", "deoche mode", 0.8),
    ])
    def test_near_misses_score_high(self, a, b, minimum):
        assert similarity(a, b) >= minimum

    @pytest.mark.parametrize("a,b,maximum", [
        # no transliteration in this module: Cyrillic vs Latin stays low
        ("depeche mode", "депеш мод", 0.3),
        ("metallica", "depeche mode", 0.5),
    ])
    def test_unrelated_stay_low(self, a, b, maximum):
        assert similarity(a, b) <= maximum

    @pytest.mark.parametrize("a,b", [
        ("Enjoy the Silence", "The Sound of Silence"),
        ("Personal Jesus", "Enjoy the Silence"),
        ("Queen", "Metallica"),
    ])
    def test_result_stays_in_unit_range(self, a, b):
        score = similarity(a, b)
        assert 0.0 <= score <= 1.0


class TestArtistMatch:
    @pytest.mark.parametrize("candidate,target,expected", [
        ("Depeche Mode", "Depeche Mode", 1.0),
        # "- Topic" channel suffix is cleaned away
        ("Depeche Mode - Topic", "Depeche Mode", 1.0),
    ])
    def test_exact(self, candidate, target, expected):
        assert artist_match(candidate, target) == pytest.approx(expected)

    @pytest.mark.parametrize("candidate,target,minimum", [
        # candidate contains the target artist as a token-run -> >= 0.9
        ("Depeche Mode Official", "Depeche Mode", 0.9),
        ("Queen Official Channel", "Queen", 0.9),
        # typo target
        ("Depeche Mode", "Deoche mode", 0.8),
    ])
    def test_strong_matches(self, candidate, target, minimum):
        assert artist_match(candidate, target) >= minimum

    @pytest.mark.parametrize("candidate,target,maximum", [
        ("Metallica", "Depeche Mode", 0.5),
        # Cyrillic target is LLM#1's problem, not this module's
        ("Depeche Mode", "депеш мод", 0.55),
    ])
    def test_wrong_artist_stays_low(self, candidate, target, maximum):
        assert artist_match(candidate, target) <= maximum

    @pytest.mark.parametrize("candidate,target", [
        ("", "Depeche Mode"),
        ("Depeche Mode", ""),
        ("(Official Video)", "Depeche Mode"),  # all-junk candidate
    ])
    def test_empty_sides(self, candidate, target):
        assert artist_match(candidate, target) == 0.0


class TestTrackMatch:
    @pytest.mark.parametrize("candidate,target,expected", [
        ("Enjoy the Silence", "Enjoy the Silence", 1.0),
        ("Enjoy the Silence (Official Video)", "Enjoy the Silence", 1.0),
        ("O", "O", 1.0),  # one-token title
    ])
    def test_exact(self, candidate, target, expected):
        assert track_match(candidate, target) == pytest.approx(expected)

    @pytest.mark.parametrize("candidate,target,minimum", [
        # prod Ry Cooder case: bracket junk must not zero the score
        ("Dior wild man (Full Official Track)", "Dior wild man", 0.9),
        ("Dior wild man (Full Official Track) Johnny Depp DIOR", "Dior wild man", 0.9),
        ("Strange Love", "Strangelove", 0.85),
        ("Freestate", "Free State", 0.85),
        ("King of Dreams", "King of Dream", 0.85),
        # artist - track candidates wrap the target track with extra tokens
        ("Deep Purple - King of Dreams", "King of Dreams", 0.85),
    ])
    def test_near_misses_score_high(self, candidate, target, minimum):
        assert track_match(candidate, target) >= minimum

    @pytest.mark.parametrize("candidate,target", [
        ("", "Song"),
        ("Song", ""),
        ("(Official Video)", "Song"),  # all-junk title
        ("!!!", "Song"),
    ])
    def test_empty_sides(self, candidate, target):
        assert track_match(candidate, target) == 0.0


class TestPairMatch:
    @pytest.mark.parametrize("candidate_artist,candidate_title,target_artist,target_track,minimum", [
        # right track by the right (typo'd) artist reaches the auto band
        ("Depeche Mode", "Enjoy the Silence", "Deoche mode", "enjoy the silence", 0.8),
        ("Depeche Mode - Topic", "Enjoy the Silence (Official Video)",
         "Depeche Mode", "Enjoy the Silence", 0.85),
    ])
    def test_right_track_right_artist(self, candidate_artist, candidate_title,
                                      target_artist, target_track, minimum):
        assert pair_match(candidate_artist, candidate_title,
                          target_artist, target_track) >= minimum

    @pytest.mark.parametrize("candidate_artist,candidate_title,target_artist,target_track,maximum", [
        # right title, wrong artist: must not leak into the 0.6 band
        ("Metallica", "Enjoy the Silence", "Depeche Mode", "Enjoy the Silence", 0.55),
        # Cyrillic target without transliteration: artist component ~0, capped
        ("Depeche Mode", "Enjoy the Silence", "депеш мод", "enjoy the silence", 0.55),
        # right artist, wrong track stays below the auto band
        ("Depeche Mode", "Personal Jesus", "Depeche Mode", "Enjoy the Silence", 0.6),
    ])
    def test_wrong_side_cannot_leak(self, candidate_artist, candidate_title,
                                    target_artist, target_track, maximum):
        assert pair_match(candidate_artist, candidate_title,
                          target_artist, target_track) <= maximum

    def test_no_target_artist_uses_track_alone(self):
        assert pair_match("Metallica", "Enjoy the Silence",
                          None, "Enjoy the Silence") == pytest.approx(
            track_match("Enjoy the Silence", "Enjoy the Silence"))

    def test_empty_target_track_uses_artist_alone(self):
        assert pair_match("Depeche Mode", "Whatever", "Depeche Mode", "") == \
            pytest.approx(artist_match("Depeche Mode", "Depeche Mode"))

    def test_no_target_at_all(self):
        assert pair_match("Anyone", "Anything", None, "") == 0.0


class TestTuningTable:
    """The production cases this module was tuned for (see module docstring):
    each row is a real query that dictionary-exact comparison scored 0.0-0.25."""

    @pytest.mark.parametrize("candidate_artist,candidate_title,target_artist,target_track,bound,kind", [
        # "Deoche mode" typo against the real "Depeche Mode" channel
        ("Depeche Mode", "Enjoy the Silence", "Deoche mode", "enjoy the silence", 0.8, "min"),
        # Ry Cooder / DIOR ad track buried under title junk
        ("Ry Cooder", "Dior wild man (Full Official Track)", "Ry Cooder", "Dior wild man", 0.9, "min"),
        # near-miss spellings must not zero the score
        ("Depeche Mode", "Strange Love", "Depeche Mode", "Strangelove", 0.85, "min"),
        ("Deep Purple", "King of Dreams", "Deep Purple", "King of Dream", 0.85, "min"),
        ("Depeche Mode", "Freestate", "Depeche Mode", "Free State", 0.85, "min"),
        # wrong artist must not leak into the auto band
        ("Metallica", "Enjoy the Silence", "Depeche Mode", "Enjoy the Silence", 0.55, "max"),
        # Cyrillic target: translit is LLM#1's job, the artist component stays low
        ("Depeche Mode", "Enjoy the Silence", "депеш мод", "enjoy the silence", 0.55, "max"),
    ])
    def test_prod_cases(self, candidate_artist, candidate_title,
                        target_artist, target_track, bound, kind):
        score = pair_match(candidate_artist, candidate_title,
                           target_artist, target_track)
        if kind == "min":
            assert score >= bound, f"{score:.3f} should be >= {bound}"
        else:
            assert score <= bound, f"{score:.3f} should be <= {bound}"
