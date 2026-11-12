"""Bio disambiguation boilerplate stripper (the 'multiple artists' artifact)."""
from bot.handler import _strip_bio_disambiguation


def test_ru_lastfm_boilerplate_stripped():
    bio = ("На Last.fm отслеживается несколько исполнителей с именем Knut. "
           "Knut — бельгийская сладж-метал группа, образованная в 1994 году "
           "в Брюссле и известная своими интенсивными концертами.")
    out = _strip_bio_disambiguation(bio)
    assert out.startswith("Knut —")
    assert "отслеживается" not in out


def test_en_multiple_artists_stripped():
    bio = ("There are multiple artists with this name. "
           "Nirvana was an American rock band formed in Aberdeen, "
           "Washington, in 1987.")
    out = _strip_bio_disambiguation(bio)
    assert out.startswith("Nirvana was")
    assert "multiple artists" not in out


def test_en_numbered_form_stripped():
    bio = ("There are at least three artists with this name: "
           "1) an American rock band. Founded in 1987, they released "
           "three studio albums.")
    out = _strip_bio_disambiguation(bio)
    assert "at least three" not in out


def test_boilerplate_only_yields_empty():
    bio = "На Last.fm отслечивается несколько исполнителей с этим именем."
    assert _strip_bio_disambiguation("There are multiple artists with this name.") == ""


def test_clean_bio_passthrough():
    bio = "Nirvana was an American rock band formed in 1987."
    assert _strip_bio_disambiguation(bio) == bio


def test_user_reported_wording_stripped():
    # the exact shape the owner saw on a card (12.11): «...на Last.fm
    # отслеживает несколько исполнителей с именем...»
    bio = ("Группа с этим именем не единственная: на Last.fm отслеживает "
           "несколько исполнителей с именем Chine. Настоящая биография "
           "начинается здесь.")
    out = _strip_bio_disambiguation(bio)
    assert "Last.fm" not in out


def test_record_class_live_vs_studio():
    from services.pipelines.legacy import record_class
    assert record_class("Nirvana - Come As You Are (Live at Reading 1992)") == "lv"
    assert record_class("Концерт в Лужниках, песня про лето") == "lv"
    assert record_class("вживую на радио") == "lv"
    assert record_class("Nirvana - Come As You Are (Official Music Video)") == "st"
    assert record_class("") == "st"


def test_fact_button_callback_carries_record_class():
    from bot.handler import _reply_keyboard
    kb = _reply_keyboard(1, 0, "search", vid="vAb1", cls="lv")
    btns = [b.text for row in kb.inline_keyboard for b in row]
    datas = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert any(d == "fact:vAb1|lv" for d in datas)
    kb2 = _reply_keyboard(1, 0, "search", vid="vAb1")
    datas2 = [b.callback_data for row in kb2.inline_keyboard for b in row]
    assert any(d == "fact:vAb1|st" for d in datas2)
    assert btns  # keyboard built
