import json
import re
from pathlib import Path

def load_env(path=".env"):
    """Load KEY=VALUE lines from .env into os.environ (existing vars win)."""
    import os
    f = Path(__file__).parent / path
    if f.exists():
        for l in f.read_text(encoding="utf-8").splitlines():
            if "=" in l and not l.lstrip().startswith("#"):
                k, v = l.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def natkey(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_ids(path):
    if not path:
        return None
    return {l.strip() for l in open(path, encoding="utf-8") if l.strip()}


def list_deeds(root, only=None, limit=None):
    ds = sorted((p for p in Path(root).iterdir() if p.is_dir()), key=lambda p: natkey(p.name))
    if only:
        ds = [d for d in ds if d.name in only]
    return ds[:limit] if limit else ds


def list_pages(deed_dir):
    return sorted((p for p in Path(deed_dir).iterdir() if p.suffix.lower() in IMG_EXTS),
                  key=lambda p: natkey(p.name))


def load_ocr(ocr_dir, reg_no, pages):
    out = []
    for p in pages:
        f = Path(ocr_dir) / reg_no / f"{p.stem}.txt"
        out.append(f.read_text(encoding="utf-8") if f.exists() else None)
    return out


def load_digitised(dig_dir, reg_no):
    """Gemini digitised JSON for the whole deed: <dig_dir>/<reg_no>.json -> compact string, or None."""
    if not dig_dir:
        return None
    f = Path(dig_dir) / f"{reg_no}.json"
    if not f.exists():
        return None
    return json.dumps(json.loads(f.read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":"))


def extract_json(text):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


def cfg_tag(c):
    img = c["images"] + (str(c["n_images"]) if c["images"] == "first_n" else "")
    split = "single" if c["split"] == "single" else f"chunk{c['chunk_size']}"
    if c.get("img_tiles", 1) > 1:
        split += f"_tiles{c['img_tiles']}"
    return "__".join([
        c["prompt"], f"img-{img}", f"ocr-{c['ocr']}", c["layout"],
        f"px{c['max_side']}", split, f"js-{c['json_mode']}", f"t{c['temperature']}",
    ])
