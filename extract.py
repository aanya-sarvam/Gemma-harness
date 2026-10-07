"""Stage 2: run each config in the grid over each deed against Gemma. Resumable per (cfg, deed).

Output: <out>/<cfg_tag>/cfg.json and <out>/<cfg_tag>/<reg_no>.json
"""
import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path

from build_messages import build, text_only, with_defaults
from common import load_env, cfg_tag, extract_json, list_deeds, list_pages, load_digitised, load_ocr, read_ids
from gemma_client import GemmaClient

load_env()


def merge(parts):
    """Merge chunked outputs: dicts recurse, lists union (dedup), scalars first non-empty."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if all(isinstance(p, dict) for p in parts):
        keys = []
        for p in parts:
            keys += [k for k in p if k not in keys]
        return {k: merge([p.get(k) for p in parts]) for k in keys}
    if all(isinstance(p, list) for p in parts):
        seen, out = set(), []
        for p in parts:
            for x in p:
                s = json.dumps(x, ensure_ascii=False, sort_keys=True)
                if s not in seen:
                    seen.add(s)
                    out.append(x)
        return out
    for p in parts:
        if p not in ("", [], {}):
            return p
    return parts[0]


MAX_BODY = int(9.5 * 1024 * 1024)  # some endpoints reject >10 MB bodies (413); base64 images count


def fit(pages, ocr, schema, c, digitised, limit=MAX_BODY):
    """Build calls; if any body exceeds the limit, shrink image size until all fit."""
    size = lambda calls: max(len(json.dumps(m, ensure_ascii=False).encode()) for m, _ in calls)
    calls = build(pages, ocr, schema, c, digitised)
    if size(calls) <= limit or c["images"] == "none":
        return calls, None
    for side in (1024, 896, 768, 640, 512, 384):
        if side >= (c["max_side"] or 10**9):
            continue
        calls = build(pages, ocr, schema, {**c, "max_side": side, "jpeg_quality": 75}, digitised)
        if size(calls) <= limit:
            return calls, side
    return calls, -1  # still too big; the API will 413 and it is recorded


def run_one(client, deed_dir, ocr_dir, dig_dir, schema, json_schema, c, out_path):
    reg = deed_dir.name
    pages = list_pages(deed_dir)
    ocr = load_ocr(ocr_dir, reg, pages)
    digitised = load_digitised(dig_dir, reg)
    calls, downscaled = fit(pages, ocr, schema, c, digitised)
    results = []
    for msgs, meta in calls:
        r = client.chat(msgs, c["temperature"], c["max_tokens"], c["json_mode"], json_schema)
        r["parsed"] = extract_json(r["text"])
        r.update(meta)
        results.append(r)
    parsed = results[0]["parsed"] if len(results) == 1 else merge([r["parsed"] for r in results])
    rec = {
        "reg_no": reg, "cfg_tag": cfg_tag(c), "n_pages": len(pages),
        "missing_ocr_pages": [i + 1 for i, t in enumerate(ocr) if t is None],
        "downscaled_to": downscaled,
        "missing_digitised": c["ocr"] == "digitised" and digitised is None,
        "ok": parsed is not None and all(r["error"] is None for r in results),
        "errors": [r["error"] for r in results if r["error"]],
        "prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in results),
        "completion_tokens": sum(r["usage"].get("completion_tokens", 0) for r in results),
        "latency_s": round(sum(r["latency_s"] for r in results), 2),
        "parsed": parsed, "calls": results,
    }
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deeds-dir", required=True)
    ap.add_argument("--ocr-dir", default="ocr", help="per-page OCR txts (only for ocr=all/matched)")
    ap.add_argument("--digitised-dir", help="<reg_no>.json Gemini digitised output (for ocr=digitised)")
    ap.add_argument("--schema", required=True, help="schema/field spec shown to the model in the prompt")
    ap.add_argument("--json-schema", help="JSON Schema for guided decoding (json_mode=schema)")
    ap.add_argument("--grid", required=True, help="JSON file: a list of config dicts (see with_defaults in build_messages.py for fields)")
    ap.add_argument("--only", help="comma-separated indices into the grid to run")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--base-url", default=os.environ.get("GEMMA_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("GEMMA_MODEL", "gemma4"))
    ap.add_argument("--api-key", default=os.environ.get("GEMMA_API_KEY"))
    ap.add_argument("--auth-header", default="api-subscription-key",
                    help="custom header name for the API key, or 'Authorization' for Bearer auth (e.g. vLLM)")
    ap.add_argument("--stream", action="store_true", help="use SSE streaming")
    ap.add_argument("--timeout", type=int, default=600, help="per-request timeout (s); a hung call otherwise blocks a worker")
    ap.add_argument("--deeds", help="file of reg_nos to restrict to")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retry-failed", action="store_true", help="rerun records with ok=false")
    ap.add_argument("--dry-run", action="store_true", help="print prompt for first deed per cfg and exit")
    a = ap.parse_args()

    schema = json.loads(Path(a.schema).read_text(encoding="utf-8"))
    json_schema = json.loads(Path(a.json_schema).read_text(encoding="utf-8")) if a.json_schema else None
    grid = [with_defaults(g) for g in json.loads(Path(a.grid).read_text(encoding="utf-8"))]
    if a.only:
        grid = [grid[int(i)] for i in a.only.split(",")]
    deeds = list_deeds(a.deeds_dir, read_ids(a.deeds), a.limit)
    client = GemmaClient(a.base_url, a.model, a.api_key, auth_header=a.auth_header, stream=a.stream,
                         timeout=a.timeout)

    if a.dry_run:
        for c in grid:
            pages = list_pages(deeds[0])
            calls = build(pages, load_ocr(a.ocr_dir, deeds[0].name, pages), schema, c,
                          load_digitised(a.digitised_dir, deeds[0].name))
            print(f"\n######## {cfg_tag(c)} | {len(calls)} call(s)")
            print(text_only(calls[0][0]))
        return

    for c in grid:
        tag = cfg_tag(c)
        d = Path(a.out) / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / "cfg.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
        todo = []
        for deed in deeds:
            op = d / f"{deed.name}.json"
            if op.exists():
                if not a.retry_failed:
                    continue
                if json.loads(op.read_text(encoding="utf-8")).get("ok"):
                    continue
            todo.append((deed, op))
        print(f"[{tag}] {len(todo)} deeds to run")
        ok = fail = 0
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(run_one, client, deed, a.ocr_dir, a.digitised_dir, schema, json_schema, c, op)
                    for deed, op in todo]
            for f in cf.as_completed(futs):
                try:
                    rec = f.result()
                    ok += rec["ok"]
                    fail += not rec["ok"]
                    if rec["errors"]:
                        print(f"  {rec['reg_no']}: {rec['errors'][0][:200]}")
                except Exception as e:
                    fail += 1
                    print(f"  crash: {type(e).__name__}: {e}")
        print(f"[{tag}] ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
