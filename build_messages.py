"""Builds OpenAI-style chat messages for Gemma from pages + OCR + schema, per config."""
import base64
import io
import json

from PIL import Image, ImageOps

from prompts import FMT, PROMPTS

DEFAULTS = dict(
    prompt="basic",
    images="all",        # none | all | first_n | pages
    n_images=3,          # for first_n
    pages=None,          # for images=pages: 1-based page numbers
    ocr="all",           # all | matched (only pages whose image is sent) | none | digitised (whole-deed JSON, optional)
    layout="interleaved",  # interleaved | grouped (images then OCR) | ocr_first (OCR then images)
    max_side=1280,       # resize longest side; 0 = original
    jpeg_quality=85,
    split="single",      # single | chunk
    chunk_size=2,        # pages per call when split=chunk
    json_mode="schema",  # none | object | schema (vLLM guided decoding)
    temperature=0.0,
    max_tokens=4096,
    system_role=True,    # False = fold system prompt into first user turn
)


def with_defaults(cfg):
    c = dict(DEFAULTS)
    c.update(cfg)
    return c


def encode_image(path, max_side, quality):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def encode_tiles(path, n, max_side, quality, overlap=0.06):
    """Cut the page into n horizontal strips (small overlap) and encode each; the model then sees each strip at higher zoom."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    step = h / n
    out = []
    for k in range(n):
        top, bot = max(0, int(k * step - overlap * step)), min(h, int((k + 1) * step + overlap * step))
        t = im.crop((0, top, w, bot))
        if max_side:
            t.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        t.save(buf, "JPEG", quality=quality)
        out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return out


def _img_indices(n, c):
    if c["images"] == "none":
        return set()
    if c["images"] == "all":
        return set(range(n))
    if c["images"] == "first_n":
        return set(range(min(n, c["n_images"])))
    if c["images"] == "pages":
        return {p - 1 for p in (c["pages"] or []) if 0 < p <= n}
    raise ValueError(c["images"])


def _ocr_indices(n, img_idx, c, ocr):
    if c["ocr"] in ("none", "digitised"):
        idx = set()
    elif c["ocr"] in ("all", "deed"):  # "deed": every call carries the OCR of the whole deed (see build)
        idx = set(range(n))
    elif c["ocr"] == "matched":
        idx = set(img_idx)
    else:
        raise ValueError(c["ocr"])
    return {i for i in idx if ocr[i]}


def _describe(n_img, n_ocr, dig=False):
    parts = []
    if n_img:
        parts.append(f"{n_img} page image(s)")
    if dig:
        parts.append("the digitised JSON (OCR output) of the whole deed")
    if n_ocr:
        parts.append(f"OCR text for {n_ocr} page(s)")
    return " and ".join(parts) or "no content"


def build(pages, ocr, schema, c, digitised=None):
    """Returns list of (messages, meta), one per API call."""
    n = len(pages)
    img_all = _img_indices(n, c)
    ocr_all = _ocr_indices(n, img_all, c, ocr)
    use_dig = c["ocr"] == "digitised" and bool(digitised)
    dig_part = {"type": "text", "text": "[Digitised JSON (OCR of the whole deed)]\n" + (digitised or "")}
    groups = [list(range(n))] if c["split"] == "single" else \
        [list(range(i, min(n, i + c["chunk_size"]))) for i in range(0, n, c["chunk_size"])]
    p = PROMPTS[c["prompt"]]
    schema_str = json.dumps(schema, ensure_ascii=False, indent=1)

    calls = []
    for g in groups:
        img_idx = [i for i in g if i in img_all]
        ocr_idx = [i for i in (range(n) if c["ocr"] == "deed" else g) if i in ocr_all]
        header = p["user"].format(schema=schema_str, n_pages=len(g),
                                  fmt=FMT["json"] if c["json_mode"] == "none" else FMT["structured"],
                                  sources=_describe(len(img_idx), len(ocr_idx), use_dig))
        if not c["system_role"]:
            header = p["system"] + "\n\n" + header

        def img_parts(i):
            nt = c.get("img_tiles", 1)
            if nt > 1:
                out = []
                for k, url in enumerate(encode_tiles(pages[i], nt, c["max_side"], c["jpeg_quality"]), 1):
                    out += [{"type": "text", "text": f"[Page {i + 1} image, part {k} of {nt} (top to bottom, slight overlap)]"},
                            {"type": "image_url", "image_url": {"url": url}}]
                return out
            return [{"type": "text", "text": f"[Page {i + 1} image]"},
                    {"type": "image_url",
                     "image_url": {"url": encode_image(pages[i], c["max_side"], c["jpeg_quality"])}}]

        def ocr_part(i):
            return {"type": "text", "text": f"[Page {i + 1} OCR]\n{ocr[i]}"}

        content = [{"type": "text", "text": header}]
        if use_dig and c["layout"] == "ocr_first":
            content.append(dig_part)
        if c["ocr"] == "deed":  # whole-deed OCR first, then only this call's page images
            content += [ocr_part(i) for i in ocr_idx]
            for i in img_idx:
                content += img_parts(i)
        elif c["layout"] == "interleaved":
            for i in g:
                if i in img_idx:
                    content += img_parts(i)
                if i in ocr_idx:
                    content.append(ocr_part(i))
        elif c["layout"] == "grouped":
            for i in img_idx:
                content += img_parts(i)
            content += [ocr_part(i) for i in ocr_idx]
        elif c["layout"] == "ocr_first":
            content += [ocr_part(i) for i in ocr_idx]
            for i in img_idx:
                content += img_parts(i)
        else:
            raise ValueError(c["layout"])
        if use_dig and c["layout"] != "ocr_first":
            content.append(dig_part)
        if c["json_mode"] == "none":
            content.append({"type": "text", "text": "Return only the JSON object."})

        msgs = [{"role": "user", "content": content}]
        if c["system_role"]:
            msgs.insert(0, {"role": "system", "content": p["system"]})
        meta = {"pages": [i + 1 for i in g], "img_pages": [i + 1 for i in img_idx],
                "ocr_pages": [i + 1 for i in ocr_idx],
                "ocr_chars": sum(len(ocr[i]) for i in ocr_idx),
                "digitised_chars": len(digitised) if use_dig else 0}
        calls.append((msgs, meta))
    return calls


def text_only(messages):
    """Printable view of messages with images elided (for --dry-run)."""
    out = []
    for m in messages:
        if isinstance(m["content"], str):
            out.append(f"--- {m['role']} ---\n{m['content']}")
            continue
        out.append(f"--- {m['role']} ---")
        for part in m["content"]:
            out.append(part["text"] if part["type"] == "text" else "<IMAGE>")
    return "\n".join(out)
