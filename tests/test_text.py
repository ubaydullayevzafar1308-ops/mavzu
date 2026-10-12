import pytest

from mavzu.text import (
    OKINA,
    TUTUQ,
    cyrillic_to_latin,
    detect_script,
    fingerprint,
    latin_to_cyrillic,
    normalize,
    normalize_apostrophes,
)


@pytest.mark.parametrize(
    "cyr, lat",
    [
        ("Ўзбекистон", f"O{OKINA}zbekiston"),
        ("ғалаба қозонди", f"g{OKINA}alaba qozondi"),
        ("Тошкент шаҳрида", "Toshkent shahrida"),
        ("янги мактаб", "yangi maktab"),
        ("ер юзида", "yer yuzida"),
        ("маълумот", f"ma{TUTUQ}lumot"),
        ("Ёшлар сиёсати", "Yoshlar siyosati"),
        ("ЧИЛОНЗОР", "CHILONZOR"),
        ("Чилонзор", "Chilonzor"),
        ("Франция", "Fransiya"),
        ("милиция", "militsiya"),
        ("эртага", "ertaga"),
        ("оила", "oila"),
        ("поезд", "poyezd"),
    ],
)
def test_cyrillic_to_latin(cyr, lat):
    assert cyrillic_to_latin(cyr) == lat


@pytest.mark.parametrize(
    "variant",
    ["O'zbekiston", "O`zbekiston", "Oʻzbekiston", "O’zbekiston", "O‘zbekiston", "Oʼzbekiston"],
)
def test_apostrophe_variants_collapse(variant):
    assert normalize_apostrophes(variant) == f"O{OKINA}zbekiston"


def test_tutuq_is_distinct_from_okina():
    assert normalize_apostrophes("ma'lumot va g'alaba") == f"ma{TUTUQ}lumot va g{OKINA}alaba"


@pytest.mark.parametrize(
    "text",
    [
        "Toshkent shahrida yangi choyxona ochildi",
        "O'zbekiston terma jamoasi g'alaba qozondi",
        "Siyosat va hayot, ertaga yer yuzida",
        "Ma'lumotlarga ko'ra, yoshlar soni oshdi",
    ],
)
def test_roundtrip_latin_cyrillic_latin(text):
    latin = normalize(text)
    assert normalize(latin_to_cyrillic(latin)) == latin


def test_normalize_handles_mixed_script_and_urls():
    out = normalize("Янгилик:   O'zbekiston https://kun.uz/news/1  ҳақида")
    assert out == f"Yangilik: O{OKINA}zbekiston haqida"


def test_normalize_empty():
    assert normalize("") == ""


@pytest.mark.parametrize(
    "text, script",
    [("Салом дунё", "cyrillic"), ("Salom dunyo", "latin"), ("Салом dunyo", "mixed"), ("123 !!", "unknown")],
)
def test_detect_script(text, script):
    assert detect_script(text) == script


def test_fingerprint_ignores_script_and_punctuation():
    assert fingerprint("Ўзбекистон, ғалаба!") == fingerprint("O'zbekiston g'alaba")
