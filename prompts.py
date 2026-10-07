"""Prompt variants. Add new ones here and reference by key in a sweep config.

user template placeholders: {schema}, {n_pages}, {sources}
"""

# With response_format set, some endpoints return blank output if the prompt also has a JSON instruction,
# so the wording is switched per json_mode (see build_messages).
FMT = {
    "json": "Output ONE JSON object matching the schema exactly. No markdown, no commentary.",
    "structured": "Fill every field of the schema exactly.",
}

_RULES = """Rules:
- {fmt}
- Use null for fields not present in the document. Never invent values.
- Keep values in the script they appear in on the document.
- Dates as DD-MM-YYYY with Arabic digits.
- For list fields (parties, properties), list each distinct entry once."""

PROMPTS = {
    "basic": {
        "system": "You extract structured fields from scanned Odisha land registration deeds.",
        "user": "This deed has {n_pages} page(s). You are given {sources}.\n\n"
                "Schema:\n{schema}\n\n" + _RULES,
    },
    "ocr_primary": {
        "system": "You extract structured fields from scanned Odisha land registration deeds.",
        "user": "This deed has {n_pages} page(s). You are given {sources}.\n"
                "Treat the digitised OCR output (text/JSON) as the primary source. Use page images only to resolve "
                "[illegible] spans, OCR errors in names/numbers, and table structure.\n\n"
                "Schema:\n{schema}\n\n" + _RULES,
    },
    "image_primary": {
        "system": "You extract structured fields from scanned Odisha land registration deeds.",
        "user": "This deed has {n_pages} page(s). You are given {sources}.\n"
                "Read the page images as the primary source. The digitised OCR output is a machine "
                "transcription that may contain errors; use it as a reading aid only.\n\n"
                "Schema:\n{schema}\n\n" + _RULES,
    },
}

_SYS = "You extract structured fields from scanned Odisha land registration deeds."

_JSON_GUIDE = """The digitised JSON for each page is a first-pass extraction: a list of fields with
id, item_index, attr (name / relation_name / address for parties; village / khata / plot / area for properties),
odia_text (as read on the page) and english_value (a transliteration/translation). It can contain errors,
misassigned fields or missing entries."""

_ODIA_RULES = """Rules:
- {fmt}
- Copy text values in ODIA SCRIPT exactly as they appear on the page (use odia_text / the image). Do NOT translate or transliterate to English.
- Amounts, khata, plot and area keep the digits as written on the document (Odia digits stay Odia).
- Dates as DD-MM-YYYY with Arabic digits.
- seller_details / buyer_details: one object per person; name, relation_name (relation word plus relative's name, e.g. father/husband) and address (village, post, thana, district, caste, occupation joined by ' ; ').
- property_details: one object per property row (village, khata, plot, area).
- deed_type is the deed title as written; district and office as written.
- Use null for fields not present. Never invent values. List each distinct entry once."""

PROMPTS["odia_strict"] = {
    "system": _SYS,
    "user": "This deed has {n_pages} page(s). You are given {sources}.\n" + _JSON_GUIDE +
            "\nUse the images as the source of truth and the digitised JSON as a strong hint.\n\n"
            "Schema:\n{schema}\n\n" + _ODIA_RULES,
}
PROMPTS["json_primary"] = {
    "system": _SYS,
    "user": "This deed has {n_pages} page(s). You are given {sources}.\n" + _JSON_GUIDE +
            "\nTreat the digitised JSON as the primary source: merge the per-page fields into one deed-level record, "
            "fixing obvious errors. Use the images only to verify or fill gaps.\n\n"
            "Schema:\n{schema}\n\n" + _ODIA_RULES,
}
PROMPTS["verify_images"] = {
    "system": _SYS,
    "user": "This deed has {n_pages} page(s). You are given {sources}.\n" + _JSON_GUIDE +
            "\nCompare the JSON with the page images, correct wrong or missing values, then output the final "
            "deed-level record. Where JSON and image disagree, trust the image.\n\n"
            "Schema:\n{schema}\n\n" + _ODIA_RULES,
}

_ODIA_RULES_IMG = _ODIA_RULES.replace("(use odia_text / the image)", "(read them from the page)")

# Image-only / raw-transcription variants: no field-level Gemini JSON involved.
PROMPTS["images_only"] = {
    "system": _SYS,
    "user": "This deed has {n_pages} page(s) in this request. You are given {sources}. Some fields may be on pages "
            "not shown in this request; leave those null.\n\nSchema:\n{schema}\n\n" + _ODIA_RULES_IMG,
}
PROMPTS["ocr_text"] = {
    "system": _SYS,
    "user": "This deed has {n_pages} page(s) in this request. You are given {sources}. The OCR text is a raw "
            "machine transcription of each page (it may contain errors; tables use ' | ' between cells). "
            "Use the images to check names, numbers and table structure. Some fields may be on pages "
            "not shown in this request; leave those null.\n\nSchema:\n{schema}\n\n" + _ODIA_RULES_IMG,
}
_FIELD_GUIDE = """Where to find things in an Odisha deed:
- deed_type: the deed title near the top (e.g. sale deed / gift deed / mortgage).
- district, office: the registering office and district in the stamp, endorsement or registration seal.
- registration_date, presentation_date: in the registrar's endorsement; write each as DD-MM-YYYY.
- consideration_amount: the sale/consideration amount, as written on the document.
- old_reg_no: the deed's serial / document number as written.
- seller_details: the executant(s), introduced by words like ବିକ୍ରେତା (seller). buyer_details: the claimant(s), ଖରିଦାର / କ୍ରେତା (buyer).
  Each person is one object: name; relation_name = relation word (ପିତା father, ସ୍ୱାମୀ husband, ...) plus the relative's name; address = residence, post, thana, district, caste and occupation joined by ' ; '.
- property_details: the schedule of land (ଅନୁସୂଚୀ): one object per row with village/mouza, khata, plot, area."""

_FEWSHOT = """Example of the expected output shape (fictional values, for format only):
{"deed_type": "ବିକ୍ରୟ କବଲା", "district": "ଉଦାହରଣ", "office": "ଉଦାହରଣ", "registration_date": "05-03-1990",
 "presentation_date": "05-03-1990", "consideration_amount": "୫୦୦୦", "old_reg_no": "12",
 "seller_details": [{"name": "ରାମ ଦାସ", "relation_name": "ପିତା ହରି ଦାସ", "address": "ଗ୍ରାମ ଉଦାହରଣ ; ଥାନା ଉଦାହରଣ ; ଜାତି ଚାଷୀ"}],
 "buyer_details": [{"name": "ଶ୍ୟାମ ସାହୁ", "relation_name": "ପିତା ମଧୁ ସାହୁ", "address": "ଗ୍ରାମ ଉଦାହରଣ ; ଥାନା ଉଦାହରଣ"}],
 "property_details": [{"village": "ଉଦାହରଣ", "khata": "୧୨", "plot": "୩୪", "area": "୦.୫୦"}]}"""

_FEWSHOT = _FEWSHOT.replace("{", "{{").replace("}", "}}")  # literal braces: prompts are str.format templates

PROMPTS["ocr_locate_verify"] = {
    "system": _SYS,
    "user": "This deed has {n_pages} page(s) in this request. You are given {sources}. The OCR text is a raw "
            "machine transcription of each page (tables use ' | ' between cells). Use the OCR as a guide to spot "
            "where each field is on the page. Then read the value from the page image and check it against the OCR: "
            "if they differ, trust the image; if the image is unclear, fall back to the OCR. Some fields may be on "
            "pages not shown in this request; leave those null.\n\nSchema:\n{schema}\n\n" + _ODIA_RULES_IMG,
}
PROMPTS["ocr_field_guide"] = {
    "system": _SYS,
    "user": "This deed has {n_pages} page(s) in this request. You are given {sources}. The OCR text is a raw "
            "machine transcription of each page (tables use ' | ' between cells). Treat the OCR as the main "
            "source and use the images to fix OCR errors. Some fields may be on pages not shown in this "
            "request; leave those null.\n\n" + _FIELD_GUIDE + "\n\nSchema:\n{schema}\n\n" + _ODIA_RULES_IMG,
}
PROMPTS["ocr_fewshot"] = {
    "system": _SYS,
    "user": "This deed has {n_pages} page(s) in this request. You are given {sources}. The OCR text is a raw "
            "machine transcription of each page (tables use ' | ' between cells). Treat the OCR as the main "
            "source and use the images to fix OCR errors. Some fields may be on pages not shown in this "
            "request; leave those null.\n\n" + _FEWSHOT + "\n\nSchema:\n{schema}\n\n" + _ODIA_RULES_IMG,
}
PROMPTS["ocr_alltext"] = {
    "system": _SYS,
    "user": "This is one deed with {n_pages} page(s). You are given {sources}: the raw machine OCR of ALL pages of "
            "the deed, in page order (tables use ' | ' between cells), and possibly some page images. The OCR may "
            "contain errors and [illegible] spans; use any images to check names and numbers. Combine information "
            "from all pages into one deed-level record. Pages often repeat the same party or date (copies, "
            "endorsements): list each person and each property once. Prefer the registrar's registration and "
            "presentation dates over dates of copying, certification or stamps.\n\nSchema:\n{schema}\n\n"
            + _ODIA_RULES_IMG,
}
_SURE = ("Fill a field ONLY if its complete value is explicitly written on the page(s) shown in this request and you "
         "are confident you read it correctly. If a value is missing, cut off, illegible, or would have to be "
         "guessed or inferred from other pages, use null (or an empty list). It is expected that most fields are "
         "null on most pages: covers, endorsements, signature pages and continuation pages hold only a few fields. "
         "Do not copy dates from stamps or copy/certification lines unless they are labelled as the registration or "
         "presentation date. Include a person only if their name is legible; include a property row only if it "
         "has at least a plot or khata number.")
PROMPTS["ocr_sure"] = {
    "system": _SYS,
    "user": "This deed has several pages; this request shows {n_pages} of them. You are given {sources}. The OCR text "
            "is a raw machine transcription of each page (tables use ' | ' between cells). Treat the OCR as the main "
            "source and use the images to fix OCR errors.\n" + _SURE + "\n\nSchema:\n{schema}\n\n" + _ODIA_RULES_IMG,
}
PROMPTS["ocr_sure_ctx"] = {
    "system": _SYS,
    "user": "This is one deed. You are given {sources}. The OCR of ALL pages is provided only as context to help "
            "you understand the deed (who the parties are, which page is which). The page image(s) [Page N image] "
            "are the pages you are extracting from: extract only what is written on THOSE pages, using the OCR of "
            "the same pages as the main text source.\n" + _SURE + "\n\nSchema:\n{schema}\n\n" + _ODIA_RULES_IMG,
}
_DATE_AMOUNT_RULES = """Extra rules for dates and amounts:
- registration_date is the date in the registrar's endorsement / registration seal (usually near the end of the deed). presentation_date is the date the deed was presented for registration. Do NOT use the execution/writing date, the "compared by", "copied by" or fee-payment dates, or dates of earlier endorsements or old stamps.
- If you cannot tell which date is which, use null rather than guessing.
- consideration_amount is the sale/consideration money of the deed, not stamp duty, registration fees or totals of fees. If the amount is written both in figures and in words, they must agree; if they disagree or the digits are unclear, use null.
- Read digits one by one from the page image; do not round or infer."""
PROMPTS["ocr_fewshot_rules"] = {
    "system": _SYS,
    "user": PROMPTS["ocr_fewshot"]["user"].replace("\n\nSchema:", "\n\n" + _DATE_AMOUNT_RULES + "\n\nSchema:", 1),
}
for _base in ("ocr_alltext", "ocr_sure"):
    PROMPTS[_base + "_rules"] = {
        "system": _SYS,
        "user": PROMPTS[_base]["user"].replace("\n\nSchema:", "\n\n" + _DATE_AMOUNT_RULES + "\n\nSchema:", 1),
    }
PROMPTS["ocr_minimal"] = {
    "system": _SYS,
    "user": "Extract the deed fields from these {n_pages} page(s): {sources}. Keep Odia script. Use null when "
            "a field is not on these pages.\n\nSchema:\n{schema}\n\n- {fmt}",
}
PROMPTS["ocr_primary_odia"] = {
    "system": _SYS,
    "user": "This deed has {n_pages} page(s) in this request. You are given {sources}. The OCR text is a raw "
            "machine transcription of each page (tables use ' | ' between cells). Treat the OCR text as the "
            "primary source. Use the page images only to fix OCR errors in names and numbers, resolve [illegible] "
            "spans and recover table structure. Some fields may be on pages not shown in this request; leave "
            "those null.\n\nSchema:\n{schema}\n\n" + _ODIA_RULES_IMG,
}

