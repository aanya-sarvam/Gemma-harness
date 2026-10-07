"""Re-merge per-call outputs with different strategies and score them (no API calls).

Usage: python score2.py --runs runs --gt-dir gt --merge first,vote --out scores2
Adds, beyond score.py: how often a scalar is answered, precision when answered, and false-fill rate
(scalar answered where GT is empty).
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import score as S
from extract import merge as merge_first

SCALARS = ["deed_type", "district", "office", "registration_date", "presentation_date",
           "consideration_amount", "old_reg_no"]
PARTY = ("seller_details", "buyer_details")


def s(x):
    return "" if x is None else str(x).strip()


def clean_row(r):
    return {k: s(v) for k, v in r.items()} if isinstance(r, dict) else {"name": s(r)}


def same_row(field, a, b):
    if field == "property_details":
        pa, pb = S.norm(a.get("plot", "")), S.norm(b.get("plot", ""))
        if pa and pb:
            return pa == pb and (not a.get("khata") or not b.get("khata") or S.norm(a["khata"]) == S.norm(b["khata"]))
        return S.norm(S.item_str(a)) == S.norm(S.item_str(b))
    na, nb = S.norm(a.get("name", "")), S.norm(b.get("name", ""))
    if na and nb:
        return S._tok_f1(na, nb) >= 0.6 or S.SequenceMatcher(None, na, nb).ratio() >= 0.8
    # nameless partial row (e.g. only an address): same person if the details overlap
    ta, tb = S.norm(S.item_str(a)), S.norm(S.item_str(b))
    return bool(ta) and bool(tb) and S._tok_f1(ta, tb) >= 0.8


def merge_rows(field, parts):
    rows = []
    for lst in parts:
        for r in (lst if isinstance(lst, list) else []):
            r = clean_row(r)
            if not any(r.values()):
                continue
            for e in rows:
                if same_row(field, e, r):
                    for k, v in r.items():  # keep the fuller value per subfield
                        if len(v) > len(e.get(k, "")):
                            e[k] = v
                    break
            else:
                rows.append(dict(r))
    return rows


def merge_vote(parts):
    """Scalars: majority of normalised non-empty values (ties -> earliest call); lists: fuzzy row merge."""
    parts = [p for p in parts if isinstance(p, dict)]
    if not parts:
        return None
    out = {}
    for f in SCALARS:
        vals = [s(p.get(f)) for p in parts if not S.empty(p.get(f)) and s(p.get(f)).lower() not in ("null", "none")]
        if not vals:
            out[f] = None
            continue
        cnt = Counter(S.norm(v) for v in vals)
        best = max(cnt.values())
        out[f] = next(v for v in vals if cnt[S.norm(v)] == best)
    for f in PARTY + ("property_details",):
        out[f] = merge_rows(f, [p.get(f) for p in parts])
    return out


def merged(rec, how):
    calls = [c.get("parsed") for c in rec.get("calls", [])]
    if how == "first":
        return rec["parsed"] if len(calls) <= 1 else merge_first(calls)
    if how == "vote":
        return merge_vote(calls)
    raise ValueError(how)


def evaluate(recs, gt_dir, how):
    sc, li, ans, corr, gtn, ff, ffn = [], [], 0, 0, 0, 0, 0
    per = {}
    for r in recs:
        gp = Path(gt_dir) / f"{r['reg_no']}.json"
        if not gp.exists():
            continue
        gt = S.flatten(json.loads(gp.read_text(encoding="utf-8")))
        p = merged(r, how)
        pred = S.flatten(p) if isinstance(p, dict) else {}
        for f, gv in gt.items():
            pv = pred.get(f)
            if S.empty(gv):
                if f in SCALARS:
                    ffn += 1
                    ff += not S.empty(pv)
                continue
            if isinstance(gv, list):
                v = S.score_list(f, pv, gv)
                li.append(v)
            else:
                v = float(not S.empty(pv) and S.match(f, pv, gv))
                sc.append(v)
                gtn += 1
                ans += not S.empty(pv)
                corr += v
            per.setdefault(f, []).append(v)
    m = lambda v: round(sum(v) / len(v), 4) if v else ""
    return dict(field_score=m(sc + li), scalar_acc=m(sc), list_f1=m(li),
                answered=round(ans / gtn, 3) if gtn else "", precision=round(corr / ans, 3) if ans else "",
                false_fill=round(ff / ffn, 3) if ffn else ""), {f: m(v) for f, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gt-dir", default="gt")
    ap.add_argument("--merge", default="first,vote")
    ap.add_argument("--only", help="substring filter on run dir name")
    ap.add_argument("--out", default="scores2")
    ap.add_argument("--lenient", action="store_true", help="forgive format / honorific / Odia-vs-Latin differences")
    a = ap.parse_args()
    if a.lenient:
        import lenient
        lenient.enable()
    rows, fields = [], []
    for root in a.runs:
        for run in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            if a.only and a.only not in run.name:
                continue
            recs = [json.loads(f.read_text(encoding="utf-8")) for f in run.glob("*.json") if f.name != "cfg.json"]
            if not recs:
                continue
            for how in a.merge.split(","):
                res, per = evaluate(recs, a.gt_dir, how)
                rows.append(dict(cfg=run.name.replace("__img-all__ocr-matched__interleaved__px1280", "")
                                 .replace("__js-schema__t0.0", "").replace("__js-object__t0.0", "*"),
                                 merge=how, n=len(recs), **res,
                                 tokens=round(sum(r["prompt_tokens"] for r in recs) / len(recs)),
                                 calls=round(sum(len(r.get("calls", [0])) for r in recs) / len(recs), 1)))
                fields += [dict(cfg=rows[-1]["cfg"], merge=how, field=f, score=v) for f, v in per.items()]
    rows.sort(key=lambda r: r["field_score"] or 0, reverse=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary.csv", rows), ("per_field.csv", fields)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"{'score':>6} {'scal':>6} {'list':>6} {'ans':>5} {'prec':>5} {'ffill':>5} | {'calls':>5} {'tok':>6}  merge  cfg")
    for r in rows:
        print(f"{r['field_score']:>6} {r['scalar_acc']:>6} {r['list_f1']:>6} {r['answered']:>5} {r['precision']:>5} "
              f"{r['false_fill']:>5} | {r['calls']:>5} {r['tokens']:>6}  {r['merge']:<5}  {r['cfg']}")


if __name__ == "__main__":
    main()
