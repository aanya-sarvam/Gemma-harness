"""Lenient matcher: forgives formatting, honorifics and script (Odia vs Latin) differences that the strict scorer counts as errors.

Use via score2.py --lenient (monkeypatches score.match). Strict scoring is unchanged elsewhere.
"""
import re
from difflib import SequenceMatcher

import score as S

_strict = S.match

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

STOP = {"sub", "registrar", "registrars", "registry", "register", "registrar's", "office", "district", "dsr", "of", "the",
        "sro", "d", "s", "r", "dist", "ps", "po", "at", "ଉପ", "ନିବନ୍ଧକ", "କାର୍ଯ୍ୟାଳୟ", "ରେଜିଷ୍ଟ୍ରାର", "ଅଫିସ", "ଜିଲ୍ଲା", "ଉପନିବନ୍ଧକ"}

ODIA_MAP = {}
for chars, lat in [("କଖଗଘ", "k"), ("ଙଞଣନ", "n"), ("ଚଛଜଝଯ", "j"), ("ଟଠଡଢତଥଦଧ", "t"), ("ପଫବଭୱଵ", "p"), ("ମ", "m"),
                   ("ୟ", "j"), ("ର", "r"), ("ଲଳ", "l"), ("ଶଷସ", "s"), ("ଡ଼ଢ଼", "r")]:
    for ch in chars:
        ODIA_MAP[ch] = lat


def skeleton(text):
    """Consonant skeleton comparable across Odia and Latin script (very approximate, for names of places)."""
    t = S.unicodedata.normalize("NFC", str(text)).lower()
    out = []
    for ch in t:
        if ch in ODIA_MAP:
            out.append(ODIA_MAP[ch])
        elif "a" <= ch <= "z":
            out.append(ch)
    s = "".join(out)
    for a, b in [("ch", "j"), ("sh", "s"), ("kh", "k"), ("gh", "k"), ("th", "t"), ("dh", "t"), ("ph", "p"), ("bh", "p"),
                 ("jh", "j"), ("nj", "j")]:
        s = s.replace(a, b)
    s = s.translate(str.maketrans("gdbvwczfqxy", "ktppp" + "jjpkkj"))
    s = re.sub(r"[aeiouh]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def parse_date(x):
    t = str(x).translate(S.ODIA_DIGITS).lower()
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t)
    m = re.search(r"(\d{1,2})\s*[-/. ]\s*(\d{1,2}|[a-z]{3,9})\s*[-/. ,]\s*(\d{2,4})", t)
    if not m:
        return None
    d, mo, y = m.groups()
    mo = int(mo) if mo.isdigit() else MONTHS.get(mo[:3])
    if not mo:
        return None
    y = int(y)
    if y < 100:
        y += 2000 if y < 40 else 1900
    return (int(d), mo, y)


def number(x):
    t = str(x).translate(S.ODIA_DIGITS)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = re.sub(r"\.0+\b", "", t)
    m = re.search(r"\d+", t)
    return m.group(0).lstrip("0") or "0" if m else None


def strip_stop(s):
    return " ".join(w for w in S.norm(s).split() if w not in STOP)


def lenient_match(field, pred, gt):
    if _strict(field, pred, gt):
        return True
    if "date" in field:
        a, b = parse_date(pred), parse_date(gt)
        return a is not None and a == b
    if field.endswith(("consideration_amount", "old_reg_no", "khata", "plot")):
        a, b = number(pred), number(gt)
        if a and a == b and re.sub(r"[\d\s,./\-]", "", str(pred).translate(S.ODIA_DIGITS) + str(gt).translate(S.ODIA_DIGITS)) == "":
            return True
        if field.endswith(("khata", "plot")):
            return False
    a, b = strip_stop(pred), strip_stop(gt)
    if a and b:
        if a == b or (len(min(a, b, key=len)) >= 4 and (a in b or b in a)):
            return True
        if S._tok_f1(a, b) >= 0.6:
            return True
    ka, kb = skeleton(a or pred), skeleton(b or gt)
    return len(ka) >= 3 and len(kb) >= 3 and SequenceMatcher(None, ka, kb).ratio() >= 0.75


def enable():
    S.match = lenient_match
