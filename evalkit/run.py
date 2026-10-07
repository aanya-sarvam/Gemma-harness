"""Run experiments:  python -m evalkit.run --experiments A1_full_plain,A2_page1 --limit 3 --out evalkit_runs

Writes <out>/<experiment_id>/cfg.json and records.jsonl (one line per deed, resumable). Records follow the required schema.
"""
import argparse
import concurrent.futures as cf
import json
import os
import traceback
from pathlib import Path

from evalkit import metrics as M
from evalkit.data import load_deeds
from evalkit.experiments import EXPERIMENTS
from evalkit.llm import LLM
from evalkit.strategies import run_pipeline


def grade_both(pred, deed):
    out, items_len = {}, None
    for name, m in (("lenient", M.lenient_matcher()), ("strict", M.strict_matcher())):
        items = M.grade(pred, deed.gt, deed.ocr_all(), m)
        c = M.counts(items)
        p, r, f1 = M.prf(c.get("tp", 0), c.get("fp", 0), c.get("fn", 0))
        out[name] = dict(counts=c, precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4))
        if name == "lenient":
            items_len = items
    return out, items_len


def run_one(exp, deed, llm, root="data"):
    cfg = exp["cfg"]
    try:
        res = run_pipeline(cfg, deed, llm)
    except Exception as e:  # keep the run going; recorded as an error
        res = dict(prediction={}, pages_used=[], calls=[], extra={"exception": traceback.format_exc()[-1500:]})
        res["calls"] = [dict(tag="crash", prompt_tokens=0, completion_tokens=0, latency_s=0, error=f"{type(e).__name__}: {e}", valid_json=False)]
    calls = res["calls"]
    metrics, items = grade_both(res["prediction"], deed)
    extra = res["extra"]
    if extra.get("pred_filtered") is not None:
        m2, _ = grade_both(extra["pred_filtered"], deed)
        metrics["lenient_filtered"], metrics["strict_filtered"] = m2["lenient"], m2["strict"]
    gp = {}
    # optional: <root>/gt_pages/<reg_no>.json = {page_no: [field, ...]} sharpens the "which page should this
    # have come from" categorisation in metrics.categorize; skipped entirely if not present.
    gpf = Path(root, "gt_pages", deed.reg + ".json")
    if gpf.exists():
        for pg, fields in json.loads(gpf.read_text(encoding="utf-8")).items():
            for f in fields:
                gp.setdefault(f, []).append(int(pg))
    rec_ctx = dict(valid_json=all(c.get("valid_json", True) for c in calls), pages_used=res["pages_used"],
                   prompt_tokens=max([c.get("prompt_tokens", 0) for c in calls] or [0]))  # largest single call
    cats = [M.categorize(it, rec_ctx, gp) for it in items]
    return dict(experiment_id=exp["id"], deed_id=deed.reg, model=os.environ.get("GEMMA_MODEL", "gemma4"),
                prompt_version=f"{cfg['prompt']}|{cfg['output']}|{cfg['order']}",
                context_strategy=f"units={cfg['units']} ocr={cfg['ocr_fmt']}/{cfg['ocr_variant']} groups={cfg['groups']} "
                                 f"retrieve={cfg['retrieve']} verify={cfg['verify']} merge={cfg['merge']} images={cfg['images']}",
                pages_used=res["pages_used"], n_pages=deed.n, n_calls=len(calls),
                input_tokens=sum(c.get("prompt_tokens", 0) for c in calls),
                output_tokens=sum(c.get("completion_tokens", 0) for c in calls),
                latency=round(sum(c.get("latency_s", 0) for c in calls), 2),
                valid_json_calls=sum(bool(c.get("valid_json")) for c in calls), total_calls=len(calls),
                prediction=res["prediction"], ground_truth=deed.gt, metrics=metrics,
                errors=[c["error"] for c in calls if c.get("error")],
                items=[dict(it, category=cat) for it, cat in zip(items, cats)],
                extra={k: v for k, v in extra.items()}, calls=calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", required=True, help="comma-separated ids, or 'all' / a group name")
    ap.add_argument("--root", default="data", help="dataset root: <root>/deeds/, <root>/gt/, optionally <root>/gt_pages/")
    ap.add_argument("--deeds", default="data/deeds.txt")
    ap.add_argument("--ocr-dir", default="data/ocr")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="evalkit_runs")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--force", action="store_true", help="ignore existing records")
    a = ap.parse_args()
    ids = []
    for t in a.experiments.split(","):
        if t == "all":
            ids += list(EXPERIMENTS)
        elif t in EXPERIMENTS:
            ids.append(t)
        else:
            ids += [k for k, v in EXPERIMENTS.items() if v["group"] == t]
    deeds = load_deeds(root=a.root, ocr_dir=a.ocr_dir, deeds_file=a.deeds, limit=a.limit)
    llm = LLM(timeout=a.timeout)
    for eid in ids:
        exp = EXPERIMENTS[eid]
        d = Path(a.out) / eid
        d.mkdir(parents=True, exist_ok=True)
        (d / "cfg.json").write_text(json.dumps({k: exp[k] for k in ("id", "group", "hypothesis", "change", "expected", "cfg")},
                                               indent=1, ensure_ascii=False), encoding="utf-8")
        rf = d / "records.jsonl"
        done = set()
        if rf.exists() and not a.force:
            done = {json.loads(l)["deed_id"] for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()}
        todo = [x for x in deeds if x.reg not in done]
        print(f"[{eid}] {len(todo)} deeds to run", flush=True)
        with cf.ThreadPoolExecutor(a.workers) as ex, open(rf, "a", encoding="utf-8") as fh:
            for rec in ex.map(lambda x: run_one(exp, x, llm, a.root), todo):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
        recs = [json.loads(l) for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()]
        bad = sum(1 for r in recs if r["errors"] or r["valid_json_calls"] < r["total_calls"])
        print(f"[{eid}] done: {len(recs)} records, {bad} with errors/invalid json", flush=True)


if __name__ == "__main__":
    main()
