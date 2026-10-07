"""Re-grade stored records offline (no API calls) after a metrics/categorisation change.

python -m evalkit.regrade [--runs evalkit_runs]
"""
import argparse
import json
from pathlib import Path

from evalkit import metrics as M
from evalkit.data import load_deeds
from evalkit.run import grade_both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="evalkit_runs")
    ap.add_argument("--root", default="data", help="dataset root: <root>/deeds/, <root>/gt/, optionally <root>/gt_pages/")
    ap.add_argument("--ocr-dir", default="data/ocr")
    ap.add_argument("--deeds", default="data/deeds.txt")
    a = ap.parse_args()
    deeds = {d.reg: d for d in load_deeds(root=a.root, ocr_dir=a.ocr_dir, deeds_file=a.deeds)}
    for rf in sorted(Path(a.runs).glob("*/records.jsonl")):
        recs = [json.loads(l) for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in recs:
            d = deeds[r["deed_id"]]
            metrics, items = grade_both(r["prediction"], d)
            pf = (r.get("extra") or {}).get("pred_filtered")
            if pf is not None:
                m2, _ = grade_both(pf, d)
                metrics["lenient_filtered"], metrics["strict_filtered"] = m2["lenient"], m2["strict"]
            gp = {}
            gpf = Path(a.root, "gt_pages", d.reg + ".json")
            if gpf.exists():
                for pg, fields in json.loads(gpf.read_text(encoding="utf-8")).items():
                    for f in fields:
                        gp.setdefault(f, []).append(int(pg))
            ctx = dict(valid_json=all(c.get("valid_json", True) for c in r["calls"]), pages_used=r["pages_used"],
                       prompt_tokens=max([c.get("prompt_tokens", 0) for c in r["calls"]] or [0]))
            r["metrics"] = metrics
            r["items"] = [dict(it, category=M.categorize(it, ctx, gp)) for it in items]
        rf.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
        print("regraded", rf.parent.name, len(recs))


if __name__ == "__main__":
    main()
