"""How much of the GT is even present in the Gemini OCR text? (ceiling for any OCR-based extraction)

Per deed / GT field: is the value found in the OCR of any page, and on which page(s).
Dates: any date in the OCR equal to the GT date. Others: token recall (fuzzy token match) >= 0.7.
"""
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from score import ODIA_DIGITS, empty, flatten, norm

DATE_RE = re.compile(r"(\d{1,2})\s*[-/.|]\s*(\d{1,2})\s*[-/.|]\s*(\d{2,4})")


def dates_in(text):
    out = set()
    for d, m, y in DATE_RE.findall(text.translate(ODIA_DIGITS)):
        y = int(y) + (1900 if len(y) == 2 else 0)
        out.add((int(d), int(m), y))
    return out


def nums_in(text):
    """Numbers as digit strings, thousands commas removed, Odia digits -> Arabic."""
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*", text.translate(ODIA_DIGITS))}


def tok_recall(value, text_toks):
    vt = value.split()
    if not vt:
        return 0.0
    hit = 0
    for t in vt:
        if t in text_toks or any(SequenceMatcher(None, t, o).ratio() >= 0.8 for o in text_toks if abs(len(o) - len(t)) <= 2):
            hit += 1
    return hit / len(vt)


def item_text(x):
    return " ".join(str(v) for v in x.values() if not empty(v)) if isinstance(x, dict) else str(x)


def main(gt_dir="gt", ocr_dir="ocr", deeds="deeds.txt", out_dir="ocr_coverage_out"):
    rows = defaultdict(list)  # field -> [found bool]
    detail = []
    for reg in Path(deeds).read_text().split():
        gt = json.loads(Path(gt_dir, f"{reg}.json").read_text(encoding="utf-8"))
        pages = [p.read_text(encoding="utf-8") for p in sorted(Path(ocr_dir, reg).glob("page_*.txt"))]
        all_toks = set(norm(" ".join(pages)).split())
        page_toks = [set(norm(p).split()) for p in pages]
        page_nums = [nums_in(p) for p in pages]
        nums_all = set().union(*page_nums)
        page_dates = [dates_in(p) for p in pages]
        alld = set().union(*page_dates)
        for f, v in gt.items():
            if empty(v):
                continue
            if f in ("registration_date", "presentation_date"):
                mm = re.match(r"(\d{1,2})-(\d{1,2})-(\d{4})", v)
                tgt = (int(mm.group(1)), int(mm.group(2)), int(mm.group(3))) if mm else None
                where = [i + 1 for i, s in enumerate(page_dates) if tgt in s]
                rows[f].append(bool(where))
                detail.append((reg, f, bool(where), where))
            elif isinstance(v, list):
                for k, it in enumerate(v):
                    if f == "property_details":
                        # property rows: every numeric plot / khata value must appear as a number in the OCR
                        keys = [k_ for a in ("plot", "khata") if not empty(it.get(a))
                                for k_ in [it[a].translate(ODIA_DIGITS).replace(",", "").strip()] if k_.isdigit()]
                        ok = bool(keys) and all(k_ in nums_all for k_ in keys)
                        where = [i + 1 for i, s in enumerate(page_nums) if keys and all(k_ in s for k_ in keys)]
                    else:
                        nm = norm(it.get("name", ""))
                        r = tok_recall(nm, all_toks)
                        ok = r >= 0.7
                        where = [i + 1 for i, s in enumerate(page_toks) if tok_recall(nm, s) >= 0.7]
                    rows[f].append(ok)
                    detail.append((reg, f"{f}[{k}]", ok, where))
            else:
                nv = norm(v)
                if f in ("consideration_amount", "old_reg_no"):
                    nd = str(v).translate(ODIA_DIGITS).replace(",", "").strip()
                    ok = nd in nums_all
                    where = [i + 1 for i, s in enumerate(page_nums) if nd in s]
                else:
                    r = tok_recall(nv, all_toks)
                    ok = r >= 0.7
                    where = [i + 1 for i, s in enumerate(page_toks) if tok_recall(nv, s) >= 0.7]
                rows[f].append(ok)
                detail.append((reg, f, ok, where))
    print("GT value present in OCR text (any page):")
    for f, v in rows.items():
        print(f"  {f:22s} {sum(v):3d}/{len(v):3d} = {sum(v)/len(v):.2f}")
    tot = [x for v in rows.values() for x in v]
    print(f"  {'ALL':22s} {sum(tot):3d}/{len(tot):3d} = {sum(tot)/len(tot):.2f}")
    Path(out_dir).mkdir(exist_ok=True)
    with open(Path(out_dir, "ocr_coverage_detail.tsv"), "w", encoding="utf-8") as fh:
        for r in detail:
            fh.write("\t".join(map(str, r)) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
