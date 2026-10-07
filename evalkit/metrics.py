"""Field-level grading of one prediction against GT, with hallucination and error-category tagging.

Item outcomes
  scalar field : correct | wrong | missing | spurious (pred present, GT empty)
  list rows    : <role>_name / plot rows: correct (row matched) | missing (GT row unmatched) | spurious_row | duplicate
  attributes   : relation / address / village / khata / area on matched rows: correct | wrong | missing
Overall P/R/F1 use scalar fields + list-row items only (attributes are reported per field).
  TP = correct;  FP = wrong + spurious + spurious_row + duplicate;  FN = wrong + missing
Hallucination = a predicted item that is wrong/spurious AND (its value is not found in the provided OCR OR the GT field is empty).
"""
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import lenient as L
import ocr_coverage as C
import score as S

SCALARS = ["deed_type", "district", "office", "registration_date", "presentation_date", "consideration_amount", "old_reg_no"]
ROLES = {"seller_details": "seller", "buyer_details": "buyer"}
EMPTY_STR = {"", "null", "none", "na", "n/a", "nil", "-"}


def E(v):
    return v is None or v == [] or v == {} or (isinstance(v, str) and v.strip().lower() in EMPTY_STR)


def sv(v):
    return "" if E(v) else str(v).strip()


def ocr_ctx(text):
    nums = {n.lstrip("0") or "0" for n in C.nums_in(text)}
    return dict(toks=set(S.norm(text).split()), nums=nums, dates=C.dates_in(text))


def present(field, val, o):
    """Is this value found in the OCR text (fuzzy)?"""
    v = sv(val)
    if not v:
        return False
    if "date" in field:
        d = L.parse_date(v)
        return d in o["dates"] if d else False
    if field in ("consideration_amount", "old_reg_no", "khata", "plot"):
        n = L.number(v)
        return bool(n) and n in o["nums"]
    return C.tok_recall(S.norm(v), o["toks"]) >= 0.5


def row(r, attrs):
    return {a: sv(r.get(a)) for a in attrs} if isinstance(r, dict) else {}


def rows_of(v, attrs):
    v = [v] if isinstance(v, dict) else (v if isinstance(v, list) else [])
    out = [row(r, attrs) for r in v]
    return [r for r in out if any(r.values())]


def item_text(r):
    return " ".join(x for x in r.values() if x)


def match_party(m, p, g):
    if p.get("name") and g.get("name"):
        return m("name", p["name"], g["name"])
    return m("address", item_text(p), item_text(g)) if item_text(p) and item_text(g) else False


def match_prop(m, p, g):
    if p.get("plot") and g.get("plot"):
        return m("plot", p["plot"], g["plot"])
    if not g.get("plot") and p.get("village") and g.get("village"):
        return m("village", p["village"], g["village"])
    return False


def _match_rows(pred_rows, gt_rows, same):
    used, pairs, dup, spurious = set(), [], [], []
    for gi, g in enumerate(gt_rows):
        for j, p in enumerate(pred_rows):
            if j not in used and same(p, g):
                used.add(j)
                pairs.append((j, gi))
                break
    matched_gt = {gi for _, gi in pairs}
    for j, p in enumerate(pred_rows):
        if j in used:
            continue
        if any(same(p, gt_rows[gi]) for gi in matched_gt):
            dup.append(j)
        else:
            spurious.append(j)
    return pairs, dup, spurious, matched_gt


def grade(pred, gt, ocr_text, matcher):
    """-> list of item dicts."""
    o = ocr_ctx(ocr_text)
    pred = pred if isinstance(pred, dict) else {}
    items = []

    def add(field, kind, outcome, g=None, p=None, **kw):
        it = dict(field=field, kind=kind, outcome=outcome, gt=g, pred=p)
        if outcome in ("wrong", "spurious", "spurious_row", "duplicate"):
            it["grounded"] = present(kw.pop("gfield", field), p, o)
        if outcome in ("wrong", "missing"):
            it["gt_in_ocr"] = present(kw.pop("gfield", field), g, o)
        kw.pop("gfield", None)
        items.append(it)

    for f in SCALARS:
        g, p = sv(gt.get(f)), sv(pred.get(f))
        if not g and not p:
            continue
        if not g:
            add(f, "scalar", "spurious", g, p)
        elif not p:
            add(f, "scalar", "missing", g, p)
        else:
            add(f, "scalar", "correct" if matcher(f, p, g) else "wrong", g, p)

    for lf, role in ROLES.items():
        attrs = ["name", "relation_name", "address"]
        gr, pr = rows_of(gt.get(lf), attrs), rows_of(pred.get(lf), attrs)
        pairs, dup, spur, matched_gt = _match_rows(pr, gr, lambda p, g: match_party(matcher, p, g))
        for j, gi in pairs:
            add(f"{role}_name", "row", "correct", item_text(gr[gi]), item_text(pr[j]))
            for a, nm in (("relation_name", "relation"), ("address", "address")):
                g, p = gr[gi].get(a, ""), pr[j].get(a, "")
                if g:
                    add(f"{role}_{nm}", "attr", "missing" if not p else ("correct" if matcher(a, p, g) else "wrong"), g, p, gfield=a)
        for gi, g in enumerate(gr):
            if gi not in matched_gt:
                add(f"{role}_name", "row", "missing", item_text(g), None, gfield="name")
        for j in dup:
            add(f"{role}_name", "row", "duplicate", None, item_text(pr[j]), gfield="name")
        for j in spur:
            add(f"{role}_name", "row", "spurious_row", None, item_text(pr[j]), gfield="name")

    attrs = ["village", "khata", "plot", "area"]
    gr, pr = rows_of(gt.get("property_details"), attrs), rows_of(pred.get("property_details"), attrs)
    pairs, dup, spur, matched_gt = _match_rows(pr, gr, lambda p, g: match_prop(matcher, p, g))
    for j, gi in pairs:
        add("plot", "row", "correct", item_text(gr[gi]), item_text(pr[j]))
        for a in ("village", "khata", "area"):
            g, p = gr[gi].get(a, ""), pr[j].get(a, "")
            if g:
                add(a, "attr", "missing" if not p else ("correct" if matcher(a, p, g) else "wrong"), g, p)
    for gi, g in enumerate(gr):
        if gi not in matched_gt:
            add("plot", "row", "missing", item_text(g), None, gfield="village")
    for j in dup:
        add("plot", "row", "duplicate", None, item_text(pr[j]), gfield="village")
    for j in spur:
        add("plot", "row", "spurious_row", None, item_text(pr[j]), gfield="village")
    return items


def hallucinated(it):
    if it["outcome"] not in ("wrong", "spurious", "spurious_row"):
        return False
    if it["outcome"] == "spurious" and it["kind"] == "scalar":
        return True            # filled a field whose GT is empty
    return it.get("grounded") is False


def counts(items):
    """Overall counts from primary items (scalars + rows)."""
    c = Counter()
    for it in items:
        if it["kind"] == "attr":
            continue
        o = it["outcome"]
        c["tp"] += o == "correct"
        c["fp"] += o in ("wrong", "spurious", "spurious_row", "duplicate")
        c["fn"] += o in ("wrong", "missing")
        c["wrong"] += o == "wrong"
        c["missing"] += o == "missing"
        c["spurious"] += o in ("spurious", "spurious_row")
        c["duplicate"] += o == "duplicate"
        c["halluc"] += hallucinated(it)
    return dict(c)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def per_field(all_items):
    d = defaultdict(Counter)
    for it in all_items:
        c = d[it["field"]]
        c[it["outcome"]] += 1
        c["halluc"] += hallucinated(it)
    out = {}
    for f, c in d.items():
        tp, wrong, miss = c["correct"], c["wrong"], c["missing"]
        fp = wrong + c["spurious"] + c["spurious_row"] + c["duplicate"]
        p, r, f1 = prf(tp, fp, wrong + miss)
        out[f] = dict(n_gt=tp + wrong + miss, p=round(p, 3), r=round(r, 3), f1=round(f1, 3), halluc=c["halluc"],
                      correct=tp, wrong=wrong, missing=miss, spurious=c["spurious"] + c["spurious_row"], duplicate=c["duplicate"])
    return out


def categorize(it, rec, gt_pages=None):
    """Heuristic root-cause label for a non-correct item (None if correct)."""
    o = it["outcome"]
    if o == "correct":
        return None
    if not rec.get("valid_json", True):
        return "json_failure"
    f = it["field"]
    pages_used = set(rec.get("pages_used") or [])
    if o in ("missing", "wrong") and gt_pages and pages_used:
        gp = gt_pages.get(f.split("_")[0] if f.startswith(("seller", "buyer")) else f)
        if gp and not (set(gp) & pages_used):
            return "context_too_short"
    if o in ("missing", "wrong") and rec.get("prompt_tokens", 0) > 24000:
        return "context_too_long"
    if o == "duplicate":
        return "duplicate_entity"
    if hallucinated(it):
        return "hallucination"
    if o == "missing":
        return "missed_evidence" if it.get("gt_in_ocr") else "ocr_error_or_absent"
    if o == "wrong":
        return "wrong_value_selected" if it.get("gt_in_ocr") else "ocr_error_or_absent"
    if o in ("spurious", "spurious_row"):
        return "extra_value_grounded_in_ocr"
    return "other"


def lenient_matcher():
    return L.lenient_match


def strict_matcher():
    return S.match
