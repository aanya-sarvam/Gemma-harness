"""Stage 3: score every run dir against ground truth.

Writes <scores>/summary.csv and <scores>/per_field.csv.
"""
import argparse
import csv
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

ODIA_DIGITS = str.maketrans("୦୧୨୩୪୫୬୭୮୯", "0123456789")
THRESH = 0.85


def norm(x):
    s = unicodedata.normalize("NFC", str(x)).translate(ODIA_DIGITS).lower()
    s = re.sub(r"[\s\.,;:\-/()'\"|]+", " ", s)
    return s.strip()


def _date(x):
    m = re.match(r"^\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{2,4})\s*$", str(x).translate(ODIA_DIGITS))
    if not m:
        return None
    d, mo, y = m.groups()
    return (int(d), int(mo), int(y) + (1900 if len(y) == 2 else 0))


def _tok_f1(a, b):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    common = sum((__import__("collections").Counter(ta) & __import__("collections").Counter(tb)).values())
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def match(field, pred, gt):
    da, db = _date(pred), _date(gt)
    if da or db:
        return da == db
    a, b = norm(pred), norm(gt)
    if a == b:
        return True
    long_field = any(k in field for k in ("address", "relation_name", "village", "plot"))
    if long_field:
        return _tok_f1(a, b) >= 0.6 or SequenceMatcher(None, a, b).ratio() >= 0.75
    return SequenceMatcher(None, a, b).ratio() >= THRESH


def empty(v):
    return v is None or v == "" or v == [] or v == {}


def flatten(obj, prefix=""):
    """Dicts -> dotted paths; lists and scalars are leaves."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: obj}


def item_str(x):
    if isinstance(x, dict):
        keys = ("name", "relation_name", "address") if "name" in x else sorted(x)
        return " ".join(str(x[k]) for k in keys if k in x and not empty(x[k]))
    return str(x)


def score_list(field, pred, gt):
    p = [item_str(x) for x in (pred if isinstance(pred, list) else [pred] if not empty(pred) else [])]
    g = [item_str(x) for x in gt]
    used, tp = set(), 0
    for gi in g:
        for j, pj in enumerate(p):
            if j not in used and match("address" if field.startswith(("seller", "buyer")) else "plot", pj, gi):
                used.add(j)
                tp += 1
                break
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def load_gt(path):
    # Adapter point: if your ground-truth files use a different shape, convert them to schema.json's shape here.
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--gt-dir", required=True, help="<reg_no>.json ground truth")
    ap.add_argument("--out", default="scores")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, per_field = [], []
    for run in sorted(p for p in Path(a.runs).iterdir() if p.is_dir()):
        recs = [json.loads(f.read_text(encoding="utf-8")) for f in run.glob("*.json") if f.name != "cfg.json"]
        if not recs:
            continue
        field_scores, scalar, lists = {}, [], []
        n_gt = 0
        for r in recs:
            gp = Path(a.gt_dir) / f"{r['reg_no']}.json"
            if not gp.exists():
                continue
            n_gt += 1
            gt = flatten(load_gt(gp))
            pred = flatten(r["parsed"]) if isinstance(r.get("parsed"), dict) else {}
            for f, gv in gt.items():
                if empty(gv):
                    continue
                pv = pred.get(f)
                if isinstance(gv, list):
                    s = score_list(f, pv, gv)
                    lists.append(s)
                else:
                    s = float(not empty(pv) and match(f, pv, gv))
                    scalar.append(s)
                field_scores.setdefault(f, []).append(s)
        allv = scalar + lists
        mean = lambda v: round(sum(v) / len(v), 4) if v else ""
        summary.append({
            "cfg_tag": run.name, "n_runs": len(recs), "n_with_gt": n_gt,
            "parse_ok": mean([float(r["ok"]) for r in recs]),
            "field_score": mean(allv), "scalar_acc": mean(scalar), "list_f1": mean(lists),
            "avg_prompt_tokens": mean([r["prompt_tokens"] for r in recs]),
            "avg_latency_s": mean([r["latency_s"] for r in recs]),
            "n_errors": sum(bool(r["errors"]) for r in recs),
        })
        for f, v in sorted(field_scores.items()):
            per_field.append({"cfg_tag": run.name, "field": f, "n": len(v), "score": mean(v)})

    summary.sort(key=lambda r: r["field_score"] or 0, reverse=True)
    for name, rows in (("summary.csv", summary), ("per_field.csv", per_field)):
        if rows:
            with open(out / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    for r in summary:
        print(f"{r['field_score']!s:>7}  parse={r['parse_ok']}  tok={r['avg_prompt_tokens']}  "
              f"lat={r['avg_latency_s']}s  err={r['n_errors']}  {r['cfg_tag']}")


if __name__ == "__main__":
    main()
