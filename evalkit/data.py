"""Deed loading (images, OCR, GT) and OCR rendering variants."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from common import list_pages


@dataclass
class Deed:
    reg: str
    pages: list            # image paths
    ocr: list              # per-page OCR text ("" if missing)
    gt: dict
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.pages)

    def ocr_all(self, idx=None):
        idx = range(self.n) if idx is None else idx
        return "\n".join(self.ocr[i] for i in idx)


def load_deeds(root="data", ocr_dir="data/ocr", deeds_file="data/deeds.txt", limit=None, only=None):
    regs = [l.strip() for l in open(deeds_file, encoding="utf-8-sig") if l.strip()]
    if only:
        regs = [r for r in regs if r in only]
    if limit:
        regs = regs[:limit]
    out = []
    for r in regs:
        pages = list_pages(Path(root, "deeds", r))
        ocr = []
        for p in pages:
            f = Path(ocr_dir, r, p.stem + ".txt")
            ocr.append(f.read_text(encoding="utf-8") if f.exists() else "")
        gt = json.loads(Path(root, "gt", r + ".json").read_text(encoding="utf-8"))
        out.append(Deed(r, pages, ocr, gt))
    return out


# ---- OCR text variants (H) ----------------------------------------------------------------------------------------
def light_clean(t):
    """H2: whitespace / empty-table-row artefacts only; content untouched."""
    t = re.sub(r"[ \t]+", " ", t)
    t = "\n".join(l.rstrip() for l in t.split("\n"))
    t = "\n".join(l for l in t.split("\n") if not re.fullmatch(r"[\s|\-:]*", l) or not l.strip())  # drop pure-separator rows
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def flatten(t):
    """H4: drop layout (line breaks and table pipes)."""
    t = t.replace("|", " ")
    return re.sub(r"\s+", " ", t).strip()


def render(deed, idx, fmt="boundaries", text_variant="raw"):
    """OCR of pages `idx` as the model sees it. fmt: plain | boundaries | json | flat_boundaries."""
    tv = {"raw": lambda x: x, "clean": light_clean, "flat": flatten}[text_variant]
    texts = [(i, tv(deed.ocr[i])) for i in idx]
    if fmt == "plain":
        return "\n\n".join(t for _, t in texts)
    if fmt == "json":
        return json.dumps([{"page": i + 1, "text": t} for i, t in texts], ensure_ascii=False, indent=1)
    return "\n\n".join(f"PAGE {i + 1}:\n{t}" for i, t in texts)
