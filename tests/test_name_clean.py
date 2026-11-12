"""Regression for the prod case behind services/name_clean.py: the channel
«Deep Purple Official» went to Last.fm verbatim -> an artist.getinfo miss ->
cards without bio/album/year. Port of the web wave-18 test
(test_clean_channel_official_vebo), adapted to clean_channel_name."""
from services.name_clean import clean_channel_name


def test_clean_channel_official_vevo():
    """The exact contract of the web-study regression: suffix litter is
    stripped, the real name survives case-preserving."""
    assert clean_channel_name("Deep Purple Official") == "Deep Purple"
    assert clean_channel_name("Deep Purple - Topic") == "Deep Purple"
    assert clean_channel_name("DaftPunkVEVO") == "DaftPunk"
    assert clean_channel_name("Artist - Official") == "Artist"
    assert clean_channel_name("Björk (Official)") == "Björk"
    assert clean_channel_name("Katy Perry (Official Video)") == "Katy Perry"


def test_clean_channel_case_variations():
    """The suffixes match in any case and dash form; a glued VEVO tail is
    stripped too."""
    assert clean_channel_name("deep purple OFFICIAL") == "deep purple"
    assert clean_channel_name("Artist – Official") == "Artist"
    assert clean_channel_name("ArtistVevo") == "Artist"
    assert clean_channel_name("artist (official video)") == "artist"


def test_clean_channel_passthrough():
    """Clean names (slashes, diacritics, numbers) pass through unchanged;
    blank input falls back to the cascade's unknown-artist marker."""
    assert clean_channel_name("Radiohead") == "Radiohead"
    assert clean_channel_name("AC/DC") == "AC/DC"
    assert clean_channel_name("P!nk") == "P!nk"
    assert clean_channel_name("") == "Unknown Artist"
    assert clean_channel_name("   ") == "Unknown Artist"
