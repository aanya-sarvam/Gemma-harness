"""Aggregate evalkit_runs into comparison tables, per-field tables and error analysis.

python -m evalkit.report [--runs evalkit_runs] [--metric lenient|strict] [--out evalkit_report]
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from evalkit import metrics as M


def load(runs):
    out = {}
    for d in sorted(Path(runs).iterdir()):
        rf, cf = d / "records.jsonl", d / "cfg.json"
        if rf.exists() and cf.exists():
            recs = [json.loads(l) for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()]
            out[d.name] = dict(cfg=json.loads(cf.read_text(encoding="utf-8")), recs=recs)
    return out


def agg(recs, metric):
    tp = fp = fn = wrong = missing = halluc = spur = dup = 0
    for r in recs:
        c = r["metrics"][metric]["counts"]
        tp += c.get("tp", 0); fp += c.get("fp", 0); fn += c.get("fn", 0)
        wrong += c.get("wrong", 0); missing += c.get("missing", 0); halluc += c.get("halluc", 0)
        spur += c.get("spurious", 0); dup += c.get("duplicate", 0)
    p, r_, f1 = M.prf(tp, fp, fn)
    n = len(recs)
    return dict(n=n, p=p, r=r_, f1=f1, tp=tp, wrong=wrong, missing=missing, halluc=halluc, spurious=spur, dup=dup,
                valid=sum(x["valid_json_calls"] for x in recs) / max(1, sum(x["total_calls"] for x in recs)),
                lat=statistics.mean(x["latency"] for x in recs), tin=statistics.mean(x["input_tokens"] for x in recs),
                tout=statistics.mean(x["output_tokens"] for x in recs), calls=statistics.mean(x["n_calls"] for x in recs),
                f1s=[x["metrics"][metric]["f1"] for x in recs])


def items_by_field(recs, filtered=False):
    """Recompute per-field metrics from stored items (graded with the lenient matcher)."""
    its = []
    for r in recs:
        its += [i for i in r["items"]]
    return M.per_field(its)


def paired(a_recs, b_recs, metric):
    """Mean paired F1 difference b - a across common deeds, with a bootstrap-free sign count."""
    a = {r["deed_id"]: r["metrics"][metric]["f1"] for r in a_recs}
    b = {r["deed_id"]: r["metrics"][metric]["f1"] for r in b_recs}
    d = [b[k] - a[k] for k in a if k in b]
    if not d:
        return None
    se = statistics.pstdev(d) / (len(d) ** 0.5) if len(d) > 1 else 0
    return statistics.mean(d), se, sum(x > 0 for x in d), sum(x < 0 for x in d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="evalkit_runs")
    ap.add_argument("--metric", default="lenient")
    ap.add_argument("--out", default="evalkit_report")
    ap.add_argument("--control", default="A7_full_boundaries")
    a = ap.parse_args()
    data = load(a.runs)
    # virtual rows for evidence-filtered predictions
    for k in list(data):
        recs = data[k]["recs"]
        if recs and "lenient_filtered" in recs[0]["metrics"]:
            data[k + "+filter"] = dict(cfg=data[k]["cfg"], recs=[dict(r, metrics={"lenient": r["metrics"]["lenient_filtered"], "strict": r["metrics"]["strict_filtered"]}) for r in recs])
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    metric = a.metric
    L = []
    L.append(f"## Comparison ({metric} matcher; micro-averaged over deeds; n deeds per row shown)\n")
    L.append("| Experiment | Context | Images | Strategy | n | Precision | Recall | F1 | Wrong | Missing | Halluc. | Spurious | Dup | Valid JSON | Calls | In tok | Out tok | Latency s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for k, v in data.items():
        g = agg(v["recs"], metric)
        rows[k] = g
        c = v["cfg"]["cfg"]
        strat = "+".join(x for x in [c["groups"] if c["groups"] != "all" else "", ("retr:" + c["retrieve"]) if c["retrieve"] else "",
                                       ("verify:" + c["verify"]) if c["verify"] else "", c["output"] if c["output"] != "plain" else "",
                                       c["prompt"] if c["prompt"] != "minimal" else "", c["merge"] if c["units"] != "full" else ""] if x) or "single-pass"
        ctx = f"{c['units']}/{c['ocr_fmt']}" + (f"/{c['ocr_variant']}" if c["ocr_variant"] != "raw" else "")
        L.append(f"| {k} | {ctx} | {c['images']} | {strat} | {g['n']} | {g['p']:.3f} | {g['r']:.3f} | {g['f1']:.3f} | {g['wrong']} | {g['missing']} | "
                 f"{g['halluc']} | {g['spurious']} | {g['dup']} | {g['valid']:.3f} | {g['calls']:.1f} | {g['tin']:.0f} | {g['tout']:.0f} | {g['lat']:.1f} |")
    # paired differences vs control
    if a.control in data:
        L.append(f"\n## Paired F1 difference vs control {a.control} (mean, s.e., deeds better/worse)\n")
        L.append("| Experiment | dF1 | s.e. | better | worse | verdict |")
        L.append("|---|---|---|---|---|---|")
        for k, v in data.items():
            if k == a.control or k.endswith("+filter") is False and False:
                continue
            pr = paired(data[a.control]["recs"], v["recs"], metric)
            if pr:
                d, se, b, w = pr
                verdict = "inconclusive" if abs(d) < 2 * se else ("better" if d > 0 else "worse")
                L.append(f"| {k} | {d:+.3f} | {se:.3f} | {b} | {w} | {verdict} |")
    # per-field table
    L.append("\n## Per-field F1 (precision/recall in brackets); lenient matcher\n")
    cols = [("Buyer", ["buyer_name"]), ("Seller", ["seller_name"]), ("Plot", ["plot"]), ("Khata", ["khata"]), ("Village", ["village"]),
            ("Address", ["seller_address", "buyer_address"]), ("Relation", ["seller_relation", "buyer_relation"]),
            ("Dates", ["registration_date", "presentation_date"]), ("Amount", ["consideration_amount"]), ("Office", ["office"]),
            ("District", ["district"]), ("DeedType", ["deed_type"]), ("OldRegNo", ["old_reg_no"])]
    L.append("| Approach | " + " | ".join(c for c, _ in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    pf_all = {}
    for k, v in data.items():
        its = []
        for r in v["recs"]:
            its += r["items"]
        pf = M.per_field(its)
        pf_all[k] = pf
        cells = []
        for _, fs in cols:
            tp = sum(pf.get(f, {}).get("correct", 0) for f in fs)
            wr = sum(pf.get(f, {}).get("wrong", 0) for f in fs)
            ms = sum(pf.get(f, {}).get("missing", 0) for f in fs)
            sp = sum(pf.get(f, {}).get("spurious", 0) + pf.get(f, {}).get("duplicate", 0) for f in fs)
            p, r_, f1 = M.prf(tp, wr + sp, wr + ms)
            cells.append(f"{f1:.2f} ({p:.2f}/{r_:.2f})")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    # error analysis
    L.append("\n## Error categories (items, summed over deeds; heuristic labels)\n")
    cats = sorted({i.get("category") for v in data.values() for r in v["recs"] for i in r["items"] if i.get("category")})
    L.append("| Experiment | " + " | ".join(cats) + " |")
    L.append("|---|" + "---|" * len(cats))
    examples = defaultdict(list)
    for k, v in data.items():
        if k.endswith("+filter"):
            continue
        c = Counter(i["category"] for r in v["recs"] for i in r["items"] if i.get("category"))
        L.append(f"| {k} | " + " | ".join(str(c.get(x, 0)) for x in cats) + " |")
        for r in v["recs"]:
            for i in r["items"]:
                cat = i.get("category")
                if cat and len(examples[cat]) < 6 and k in (a.control, "B6_field_by_field", "A2_page1", "D_all"):
                    examples[cat].append(dict(experiment=k, deed=r["deed_id"], field=i["field"], gt=i.get("gt"), pred=i.get("pred"), outcome=i["outcome"]))
    (out / "error_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "COMPARISON.md").write_text("\n".join(L), encoding="utf-8")
    (out / "per_field.json").write_text(json.dumps(pf_all, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
