"""Config-driven extraction pipelines. One function, run_pipeline(cfg, deed, llm), interprets an experiment config:

  units     full | page | chunkN | slideN         how pages are grouped per call (within the pages a group may see)
  ocr_fmt   plain | boundaries | json | none     how OCR is rendered ("none" = images only)
  ocr_variant raw | clean | flat
  prompt    key of prompts.PARTS ; output plain|evidence|confidence|ambiguity|candidates ; order schema_first|ocr_first
  groups    all | field_by_field | field_group     one call per group of schema keys
  retrieve  None | llm | fields | keyword          which pages each group gets
  verify    None | verify | evidence               second pass over the draft
  merge     vote | first | llm                     how per-unit predictions are combined
  images    0-3                                    page images attached to each call (first N pages of the unit)
"""
import json
import re

import ocr_coverage as C
import score as S
import score2 as S2
from build_messages import encode_image
from extract import merge as merge_first

from evalkit.data import render
from evalkit.prompts import LISTS, SCALARS, SYSTEM, build_user, shape, OUTPUT_NOTES, FORMAT, TASK

ALL_KEYS = SCALARS + list(LISTS)
GROUPS = {
    "all": [("all fields", ALL_KEYS)],
    "reg_split": [("registration_date", ["registration_date"]), ("all other fields", [k for k in ALL_KEYS if k != "registration_date"])],
    "field_by_field": [(k, [k]) for k in ALL_KEYS],
    "field_group": [("deed_type, district, office", ["deed_type", "district", "office"]),
                    ("registration_date, presentation_date, consideration_amount, old_reg_no",
                     ["registration_date", "presentation_date", "consideration_amount", "old_reg_no"]),
                    ("seller_details", ["seller_details"]), ("buyer_details", ["buyer_details"]),
                    ("property_details", ["property_details"])],
}
FIELD2KEY = {"seller_name": "seller_details", "seller_relation": "seller_details", "seller_address": "seller_details",
             "buyer_name": "buyer_details", "buyer_relation": "buyer_details", "buyer_address": "buyer_details",
             "plot": "property_details", "khata": "property_details", "village": "property_details", "area": "property_details"}
FIELD_NAMES = SCALARS + list(FIELD2KEY)

KEYWORDS = {  # non-LLM page filter (C4); any hit selects the page
    "parties": ["ବିକ୍ରେତା", "ଖରିଦ", "କ୍ରେତା", "ଦାତା", "ଗ୍ରହୀତା", "ପିତା", "ସ୍ୱାମୀ", "ଜାତି", "ପେଷା", "ସାକିନ", "seller", "vendor", "purchaser", "vendee",
                "buyer", "executant", "claimant", "son of", "s/o", "w/o", "caste", "occupation", "resident"],
    "property": ["ଖାତା", "ପ୍ଲଟ", "ମୌଜା", "ଅନୁସୂଚୀ", "ଏକର", "ଡିସିମିଲ", "ଡ଼ି", "khata", "plot", "mouza", "schedule", "acre", "decimal", "dismil"],
    "endorsement": ["registered", "registering officer", "sub-registrar", "sub registrar", "presented", "presentation", "fee", "stamp",
                    "ନିବନ୍ଧ", "ଦାଖଲ", "ରେଜିଷ୍ଟ୍ରି", "ଟଙ୍କା", "rs.", "rs ", "document no", "copy of"],
}
GROUP_KW = {"seller_details": "parties", "buyer_details": "parties", "property_details": "property"}


# ---- helpers ------------------------------------------------------------------------------------------------------
def units_of(pages, mode):
    pages = list(pages)
    if mode == "full" or len(pages) <= 1:
        return [pages]
    if mode == "page":
        return [[p] for p in pages]
    m = re.fullmatch(r"chunk(\d+)", mode)
    if m:
        n = int(m.group(1))
        return [pages[i:i + n] for i in range(0, len(pages), n)]
    m = re.fullmatch(r"slide(\d+)", mode)
    if m:
        n = int(m.group(1))
        if len(pages) <= n:
            return [pages]
        out, i = [], 0
        while True:
            out.append(pages[i:i + n])
            if i + n >= len(pages):
                break
            i += n - 1
        return out
    raise ValueError(mode)


def to_json_schema(ex):
    if isinstance(ex, dict):
        return {"type": "object", "properties": {k: to_json_schema(v) for k, v in ex.items()}, "required": list(ex)}
    if isinstance(ex, list):
        return {"type": "array", "items": to_json_schema(ex[0]) if ex else {}}
    return {"type": ["string", "null"]}


def norm_out(parsed, out):
    """Model output (any format) -> (plain prediction, side info)."""
    side = {"evidence": {}, "confidence": {}, "candidates": {}}
    if not isinstance(parsed, dict):
        return {}, side
    pred = {}
    for k, v in parsed.items():
        if k in SCALARS:
            if isinstance(v, dict):
                pred[k] = v.get("selected") if out == "candidates" else v.get("value")
                for s_, key in (("evidence", "evidence"), ("confidence", "confidence"), ("candidates", "candidates")):
                    if v.get(key):
                        side[s_][k] = v[key]
            else:
                pred[k] = v
        elif k in LISTS:
            rows = [v] if isinstance(v, dict) else (v if isinstance(v, list) else [])
            clean = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                clean.append({a: r.get(a) for a in LISTS[k]})
                for s_ in ("evidence", "confidence"):
                    side[s_].setdefault(k, []).append(r.get(s_))
            pred[k] = clean
    return pred, side


def ev_ok(ev, o_toks):
    if not ev or not isinstance(ev, str):
        return False
    t = S.norm(ev).split()
    return bool(t) and C.tok_recall(" ".join(t), o_toks) >= 0.8


def filter_by_evidence(pred, side, ocr_text):
    toks = set(S.norm(ocr_text).split())
    out = {}
    for k, v in pred.items():
        if k in SCALARS:
            out[k] = v if (v in (None, "") or ev_ok(side["evidence"].get(k), toks)) else None
        else:
            evs = side["evidence"].get(k, [])
            out[k] = [r for i, r in enumerate(v) if i < len(evs) and ev_ok(evs[i], toks)]
    return out


def attach_images(user_text, deed, idx, n_img, cfg):
    if not n_img:
        return user_text
    content = [{"type": "text", "text": user_text}]
    for i in idx[:min(n_img, 3)]:
        content += [{"type": "text", "text": f"[Page {i + 1} image]"},
                    {"type": "image_url", "image_url": {"url": encode_image(deed.pages[i], cfg.get("max_side", 1280), 85)}}]
    return content


def call_extract(cfg, deed, idx, llm, keys, label, tag):
    out = cfg.get("output", "plain")
    block = "" if cfg.get("ocr_fmt") == "none" else render(deed, idx, cfg.get("ocr_fmt", "boundaries"), cfg.get("ocr_variant", "raw"))
    user = build_user(cfg, block, keys, len(idx), None if label == "all fields" else label, cfg.get("order", "schema_first"))
    if cfg.get("ocr_fmt") == "none":
        user = user.replace("OCR text of the deed", "Page images of the deed (no OCR text provided)")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": attach_images(user, deed, idx, cfg.get("images", 0), cfg)}]
    js = to_json_schema(shape(keys, out)) if cfg.get("api_mode") == "schema" else None
    parsed, rec = llm.call(msgs, tag=tag, temperature=cfg.get("temperature", 0.0), max_tokens=cfg.get("max_tokens", 4096),
                           mode=cfg.get("api_mode", "object"), json_schema=js)
    pred, side = norm_out(parsed, out)
    rec["pages"] = [i + 1 for i in idx]
    return pred, side, rec


def merge_preds(preds, how):
    preds = [p for p in preds if p]
    if not preds:
        return {}
    if len(preds) == 1:
        return preds[0]
    return (S2.merge_vote(preds) if how == "vote" else merge_first(preds)) or {}


def llm_json(llm, system, user, tag, cfg):
    parsed, rec = llm.call([{"role": "system", "content": system}, {"role": "user", "content": user}], tag=tag,
                           temperature=cfg.get("temperature", 0.0), max_tokens=cfg.get("max_tokens", 4096))
    return parsed, rec


# ---- retrieval ----------------------------------------------------------------------------------------------------
def retrieve_llm(cfg, deed, llm, calls):
    user = (TASK.split(".")[0] + ".\n\nBelow is the OCR text of every page of a deed. Identify which pages contain information relevant to "
            "extracting these fields: deed title, district, registering office, registration and presentation dates, consideration amount, "
            "deed number, sellers/executants, buyers/claimants (names, relations, addresses) and the property schedule (village, khata, plot, area). "
            "Answer with the relevant page numbers as {\"pages\": [..]}.\n\n" + render(deed, range(deed.n), "boundaries"))
    parsed, rec = llm_json(llm, SYSTEM, user, "retrieve_pages", cfg)
    calls.append(rec)
    try:
        pages = sorted({int(p) - 1 for p in parsed["pages"] if 0 < int(p) <= deed.n})
    except Exception:
        pages = []
    return pages or list(range(deed.n))


def retrieve_fields(cfg, deed, llm, calls):
    """C3: which fields does each page support? -> {top-level key: pages}."""
    user = ("Below is the OCR text of every page of a deed. For each page list which of these fields the page supports (contains a value for): "
            + ", ".join(FIELD_NAMES) + ". Answer as {\"pages\": [{\"page\": 1, \"fields\": [..]}, ..]}.\n\n"
            + render(deed, range(deed.n), "boundaries"))
    parsed, rec = llm_json(llm, SYSTEM, user, "retrieve_fields", cfg)
    calls.append(rec)
    m = {k: set() for k in ALL_KEYS}
    try:
        for e in parsed["pages"]:
            p = int(e["page"]) - 1
            for f in e.get("fields", []):
                k = FIELD2KEY.get(f, f)
                if k in m and 0 <= p < deed.n:
                    m[k].add(p)
    except Exception:
        pass
    return {k: (sorted(v) or list(range(deed.n))) for k, v in m.items()}


def retrieve_keyword(deed, keys):
    def hit(i, kws):
        t = deed.ocr[i].lower()
        return any(k.lower() in t for k in kws)
    sel = set()
    for k in keys:
        kws = KEYWORDS[GROUP_KW.get(k, "endorsement")]
        sel |= {i for i in range(deed.n) if hit(i, kws)}
    return sorted(sel) or list(range(deed.n))


# ---- verification -------------------------------------------------------------------------------------------------
def verify_pass(cfg, deed, pred, llm, calls, kind):
    ocr = render(deed, range(deed.n), cfg.get("ocr_fmt", "boundaries"), cfg.get("ocr_variant", "raw"))
    draft = json.dumps(pred, ensure_ascii=False, indent=1)
    if kind == "verify":
        user = ("Below are the OCR text of a deed and a draft extraction. Check every value of the draft against the OCR text. Set to null or "
                "remove any value that the OCR does not support, and correct values that were misread or misplaced (for example a wrong date, or "
                "a person under the wrong role). Add a value only if it is explicitly stated in the OCR. Return the corrected record in the same "
                "structure as the draft.\n\nDraft:\n" + draft + "\n\n" + FORMAT + "\n\nOCR text of the deed:\n" + ocr)
        parsed, rec = llm_json(llm, SYSTEM, user, "verify", cfg)
        calls.append(rec)
        out, _ = norm_out(parsed, "plain")
        return out if out else pred, None
    user = ("Below are the OCR text of a deed and a draft extraction. For every non-empty value in the draft, copy the exact supporting text from "
            "the OCR as evidence. " + OUTPUT_NOTES["evidence"] + " Keep the values unchanged; do not add new ones.\n\nSchema:\n"
            + json.dumps(shape(ALL_KEYS, "evidence"), ensure_ascii=False, indent=1) + "\n\nDraft:\n" + draft + "\n\nOCR text of the deed:\n" + ocr)
    parsed, rec = llm_json(llm, SYSTEM, user, "evidence_verify", cfg)
    calls.append(rec)
    p2, side = norm_out(parsed, "evidence")
    filt = filter_by_evidence(p2, side, deed.ocr_all()) if p2 else pred
    return filt, {"evidence": side["evidence"], "unfiltered": p2}


def llm_merge(cfg, deed, unit_preds, llm, calls):
    user = ("Below are extraction results obtained from different page groups of the same deed. Merge them into one record. Merge entries that "
            "refer to the same person or the same property; do not merge unless the entries clearly refer to the same entity; do not add values that "
            "are not in the results. Prefer the registrar's endorsement for dates. Return one record with the same structure.\n\nSchema:\n"
            + json.dumps(shape(ALL_KEYS, "plain"), ensure_ascii=False, indent=1) + "\n\nResults:\n"
            + "\n".join(f"Result {i + 1}: {json.dumps(p, ensure_ascii=False)}" for i, p in enumerate(unit_preds)))
    parsed, rec = llm_json(llm, SYSTEM, user, "llm_merge", cfg)
    calls.append(rec)
    out, _ = norm_out(parsed, "plain")
    return out or merge_preds(unit_preds, "vote")


# ---- main ---------------------------------------------------------------------------------------------------------
def run_pipeline(cfg, deed, llm):
    calls, extra, used = [], {}, set()
    groups = GROUPS[cfg.get("groups", "all")]
    allp = list(range(deed.n))
    ret = cfg.get("retrieve")
    gpages = {g: allp for g, _ in groups}
    if ret == "llm":
        sel = retrieve_llm(cfg, deed, llm, calls)
        gpages = {g: sel for g, _ in groups}
        extra["selected_pages"] = [p + 1 for p in sel]
    elif ret == "fields":
        fm = retrieve_fields(cfg, deed, llm, calls)
        gpages = {g: sorted({p for k in keys for p in fm[k]}) for g, keys in groups}
        extra["selected_pages"] = {g: [p + 1 for p in v] for g, v in gpages.items()}
    elif ret in ("reg_last", "reg_kw"):
        # registration_date asked of the endorsement page only; every other field sees all pages
        if ret == "reg_last":
            reg_pages = [deed.n - 1]
        else:
            reg_pages = [i for i in range(deed.n) if any(k in deed.ocr[i].lower() for k in ("registered", "regist", "ନିବନ୍ଧ", "ରେଜିଷ୍ଟ"))
                         and i >= deed.n - 2] or [deed.n - 1]
        gpages = {g: (reg_pages if g == "registration_date" else allp) for g, _ in groups}
        extra["selected_pages"] = {g: [p + 1 for p in v] for g, v in gpages.items()}
    elif ret == "keyword":
        gpages = {g: retrieve_keyword(deed, keys) for g, keys in groups}
        extra["selected_pages"] = {g: [p + 1 for p in v] for g, v in gpages.items()}

    final, side_all, unit_all = {}, {"evidence": {}, "confidence": {}, "candidates": {}}, []
    for g, keys in groups:
        gp = []
        for u in units_of(gpages[g], cfg.get("units", "full")):
            for _s in range(cfg.get("samples", 1)):          # self-consistency: repeated sampled calls, merged by vote
                pred, side, rec = call_extract(cfg, deed, u, llm, keys, g, f"extract:{g}")
                calls.append(rec)
                used |= set(u)
                gp.append(pred)
                unit_all.append(pred)
                for s_ in side:
                    for k, v in side[s_].items():
                        if len(gp) == 1 or k not in side_all[s_]:
                            side_all[s_][k] = v
        if cfg.get("merge") == "llm" and len(gp) > 1 and cfg.get("groups", "all") == "all":
            m = llm_merge(cfg, deed, gp, llm, calls)
        else:
            m = merge_preds(gp, cfg.get("merge", "vote"))
        for k in keys:
            if k in m:
                final[k] = m[k]
    extra["unit_predictions"] = unit_all if len(unit_all) > 1 else None
    extra["side"] = side_all if any(side_all.values()) else None
    if cfg.get("output") == "evidence" and extra["side"]:
        extra["pred_filtered"] = filter_by_evidence(final, side_all, deed.ocr_all())
    if cfg.get("verify"):
        final, vx = verify_pass(cfg, deed, final, llm, calls, cfg["verify"])
        if vx:
            extra["verify"] = vx
    return dict(prediction=final, pages_used=sorted(p + 1 for p in used), calls=calls, extra=extra)
