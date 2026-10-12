"""Uzbek text normalization: apostrophe unification and Cyrillic <-> Latin transliteration.

Uzbek is written in two scripts. The training corpus and TahrirchiBERT are Latin-only,
while a large share of real-world text (Telegram, government sites, older media) is
Cyrillic. Without normalization a Latin-trained model sees Cyrillic input as noise.

Conventions used here (they match the training corpus):
  * o‘ / g‘ / O‘ / G‘  use U+2018 LEFT SINGLE QUOTATION MARK
  * tutuq belgisi (ma’lumot, Qur’on) uses U+2019 RIGHT SINGLE QUOTATION MARK
"""

from __future__ import annotations

import re
import unicodedata

OKINA = "‘"  # ‘  used in o‘, g‘
TUTUQ = "’"  # ’  glottal stop / apostrophe

# Every apostrophe-like character people actually type.
_APOSTROPHES = "'`´ʻʼʹʽ‘’ʻʼ"
_APOS_CLASS = "[" + re.escape(_APOSTROPHES) + "]"

_OG_RE = re.compile(r"([oOgG])" + _APOS_CLASS)
# Apostrophe between letters that is not part of o‘/g‘ is a tutuq belgisi.
_TUTUQ_RE = re.compile(r"(?<=\w)(?<![oOgG])" + _APOS_CLASS + r"(?=\w)")
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")

_CYR_RE = re.compile(r"[Ѐ-ӿ]")
_LAT_RE = re.compile(r"[A-Za-z]")

_VOWELS_CYR = set("аеёиоуўэюяАЕЁИОУЎЭЮЯ")
_SEPARATORS_CYR = set("ъьЪЬ")

# Cyrillic -> Latin (lowercase). "е" and "ц" are context dependent and handled in code.
_CYR2LAT: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ё": "yo",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "ч": "ch",
    "ш": "sh",
    "щ": "sh",
    "ъ": TUTUQ,
    "ы": "i",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
    "ў": "o" + OKINA,
    "қ": "q",
    "ғ": "g" + OKINA,
    "ҳ": "h",
}

# Latin -> Cyrillic, longest match first.
_LAT2CYR_MULTI: list[tuple[str, str]] = [
    ("o" + OKINA, "ў"),
    ("g" + OKINA, "ғ"),
    ("sh", "ш"),
    ("ch", "ч"),
    ("yo", "ё"),
    ("yu", "ю"),
    ("ya", "я"),
    ("ye", "е"),
]
_LAT2CYR_SINGLE: dict[str, str] = {
    "a": "а",
    "b": "б",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "ҳ",
    "i": "и",
    "j": "ж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "қ",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "x": "х",
    "y": "й",
    "z": "з",
    TUTUQ: "ъ",
}


def normalize_apostrophes(text: str) -> str:
    """Unify the many apostrophe variants into the two canonical characters."""
    text = _OG_RE.sub(lambda m: m.group(1) + OKINA, text)
    return _TUTUQ_RE.sub(TUTUQ, text)


def detect_script(text: str) -> str:
    """Return 'cyrillic', 'latin', 'mixed' or 'unknown' based on letter counts."""
    cyr = len(_CYR_RE.findall(text))
    lat = len(_LAT_RE.findall(text))
    total = cyr + lat
    if total == 0:
        return "unknown"
    share = cyr / total
    if share >= 0.8:
        return "cyrillic"
    if share <= 0.2:
        return "latin"
    return "mixed"


def _apply_case(src: str, out: str, next_char: str) -> str:
    """Carry the case of a source letter over to its (possibly multi-letter) translation."""
    if not out or not src.isupper():
        return out
    # Whole word in caps (ШАҲАР -> SHAHAR) vs. capitalised word (Шаҳар -> Shahar).
    if next_char.isupper():
        return out.upper()
    return out[0].upper() + out[1:]


def cyrillic_to_latin(text: str) -> str:
    """Transliterate Uzbek Cyrillic to the official Latin alphabet."""
    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        low = ch.lower()
        prev = text[i - 1] if i > 0 else ""
        nxt = text[i + 1] if i + 1 < n else ""
        if low == "е":
            word_start = not prev.isalpha()
            after_vowel = prev in _VOWELS_CYR or prev in _SEPARATORS_CYR
            lat = "ye" if (word_start or after_vowel) else "e"
        elif low == "ц":
            # Loanwords only. Official rule: "ts" after a vowel, otherwise "s"
            # (Франция -> Fransiya, цирк -> sirk, милиция -> militsiya).
            lat = "ts" if prev in _VOWELS_CYR else "s"
        elif low in _CYR2LAT:
            lat = _CYR2LAT[low]
        else:
            out.append(ch)
            continue
        out.append(_apply_case(ch, lat, nxt))
    return "".join(out)


def latin_to_cyrillic(text: str) -> str:
    """Transliterate Uzbek Latin to Cyrillic.

    Used to build a synthetic Cyrillic test set for robustness evaluation, so it aims
    to be faithful on common words rather than perfect on every loanword.
    """
    text = normalize_apostrophes(text)
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunk2 = text[i : i + 2]
        low2 = chunk2.lower()
        matched = False
        for lat, cyr in _LAT2CYR_MULTI:
            if low2 == lat:
                upper = chunk2[0].isupper()
                out.append(cyr.upper() if upper else cyr)
                i += 2
                matched = True
                break
        if matched:
            continue
        ch = text[i]
        low = ch.lower()
        if low == "e":
            prev = text[i - 1] if i > 0 else ""
            cyr = "э" if not prev.isalpha() else "е"
            out.append(cyr.upper() if ch.isupper() else cyr)
        elif low in _LAT2CYR_SINGLE:
            cyr = _LAT2CYR_SINGLE[low]
            out.append(cyr.upper() if ch.isupper() else cyr)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def normalize(text: str, *, to_latin: bool = True, strip_urls: bool = True) -> str:
    """Full normalization pipeline applied before training and inference."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    if strip_urls:
        text = _URL_RE.sub(" ", text)
    if to_latin and _CYR_RE.search(text):
        text = cyrillic_to_latin(text)
    text = normalize_apostrophes(text)
    return _WS_RE.sub(" ", text).strip()


def fingerprint(text: str, n_chars: int = 400) -> str:
    """Cheap near-duplicate key: lowercase letters/digits of the first n characters."""
    key = re.sub(r"[^\w]", "", normalize(text).lower())[:n_chars]
    return key
