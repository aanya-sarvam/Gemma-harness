"""Prompt building blocks. A prompt = ordered list of parts; experiments change one part at a time.

The harness found that some endpoints return blank output when the prompt contains an explicit JSON instruction together
with response_format; wording therefore avoids the word JSON (evalkit.llm retries once without response_format if blank).
"""
import json

SYSTEM = "You extract structured fields from the OCR text of scanned Odisha land registration deeds."

TASK = ("Extract the fields described by the schema from the OCR text of an Odisha land registration deed. "
        "The OCR was produced automatically and may contain errors and [illegible] spans.")

FORMAT = """Rules:
- Fill every field of the schema exactly; use null when a value is not present.
- Copy text values in the script they appear in (Odia script stays Odia); do not translate or transliterate.
- Dates as DD-MM-YYYY with Arabic digits. Amounts, khata, plot and area keep the digits as written.
- seller_details / buyer_details: one object per person (name, relation_name = relation word plus relative's name, address).
- property_details: one object per property row (village, khata, plot, area). List each distinct entry once."""

DEFS = """Field definitions:
- deed_type: the deed title (sale deed, mortgage, gift, ...). Not the deed number.
- district / office: the district and the registering (sub-registrar) office, from the stamp, seal or endorsement.
- registration_date: the date in the registrar's registration endorsement/seal. NOT the execution date, NOT dates of copying, comparing or fee payment.
- presentation_date: the date the deed was presented for registration.
- consideration_amount: the sale / mortgage / gift consideration money of the deed. NOT stamp duty, registration fee or fee totals.
- old_reg_no: the deed's own serial / document number as written.
- seller_details: executants (seller / vendor / mortgagor / donor), one object per person; buyer_details: claimants (purchaser / vendee / mortgagee / donee). Witnesses, identifiers and the presentant are NOT parties.
- relation_name: relation word (father, husband, ...) plus the relative's name. address: residence, post, police station, district, caste, occupation.
- property_details: one object per land row of the schedule: village/mouza, khata, plot, area."""

ANTI = """Use only information explicitly present in the OCR text. Never infer, guess, reconstruct or fill in missing values.
If a value is not stated in the text, use null."""

CONSERVATIVE = ("Only extract a value when it is explicitly supported by the provided OCR. Do not infer, guess, reconstruct, or "
                "hallucinate missing values. If insufficient evidence exists, return null.")

STEPS = """Work as follows: (1) locate the evidence for each field in the text; (2) identify the candidate values; (3) resolve duplicates and conflicts; (4) output only the final structured result."""

_EX1 = ("PAGE 1:\nSale deed. Executant: Ram Das, son of Hari Das, village Udaharan, P.S. Udaharan, Dist. Puri, caste Chasi. "
        "Claimant: Shyam Sahu, son of Madhu Sahu, village Udaharan. Consideration Rs. 5000/-. Stamp Rs. 250. "
        "Schedule: Mouza Udaharan, Khata 12, Plot 34, Area Ac.0.50.\n\nPAGE 2:\nPresented on 05-03-1990. Registered on 06-03-1990 "
        "at Sub-Registrar office Puri. Document No. 12.")
_OUT1 = ('{"deed_type": "Sale deed", "district": "Puri", "office": "Puri", "registration_date": "06-03-1990", '
         '"presentation_date": "05-03-1990", "consideration_amount": "5000", "old_reg_no": "12", '
         '"seller_details": [{"name": "Ram Das", "relation_name": "son of Hari Das", "address": "village Udaharan, P.S. Udaharan, Dist. Puri, caste Chasi"}], '
         '"buyer_details": [{"name": "Shyam Sahu", "relation_name": "son of Madhu Sahu", "address": "village Udaharan"}], '
         '"property_details": [{"village": "Udaharan", "khata": "12", "plot": "34", "area": "Ac.0.50"}]}')
_EX2 = ("PAGE 1:\nMortgage deed written on 01-02-1995. Mortgagor Sita Devi w/o Gopal Devi of Ganjam. Mortgagee Bank. "
        "Loan Rs. 20,000/-. Witness: Hari Nayak.\nFee paid Rs. 400. Compared by Raju on 03-02-1995.")
_OUT2 = ('{"deed_type": "Mortgage deed", "district": "Ganjam", "office": null, "registration_date": null, "presentation_date": null, '
         '"consideration_amount": "20,000", "old_reg_no": null, '
         '"seller_details": [{"name": "Sita Devi", "relation_name": "w/o Gopal Devi", "address": "Ganjam"}], '
         '"buyer_details": [{"name": "Bank", "relation_name": null, "address": null}], "property_details": []}')

FEWSHOT = ("Examples (fictional, for format and behaviour only).\n\nExample 1 input:\n" + _EX1 + "\nExample 1 output:\n" + _OUT1 +
           "\n\nExample 2 input:\n" + _EX2 + "\nExample 2 output:\n" + _OUT2 +
           "\n(Example 2 shows null for values that are not stated: the registration date is null because only a writing date and a "
           "'compared by' date appear; the fee is not the consideration; the witness is not a party.)")

NEGATIVE = """Common mistakes to avoid (do NOT do these):
- Using the date the deed was written/executed, or a 'copied by' / 'compared by' / fee date, as the registration date.
- Using stamp duty, registration fee or a fee total as consideration_amount.
- Guessing the district from the office name, or the office from the district, when it is not written.
- Listing witnesses, identifiers or the presentant as sellers or buyers.
- Splitting one person into several rows (name in one row, father's name in another), or repeating the same person from different pages.
- Writing a plausible value when the OCR is unreadable: use null instead."""

DATERULE = """Dates and deed number:
- The registrar's endorsement normally shows two different dates: the date the deed was presented (earlier) and the date it was registered (the same day or later). registration_date is the later registration date, not the presentation date. If the endorsement gives only one date, use it for both fields.
- The date on which the deed was written/executed, and dates of copying, comparing or fee payment, are neither registration_date nor presentation_date.
- old_reg_no is the deed's own number in the registrar's book: look for 'Copy of document No.', 'Document No.', 'Deed No.' or a number written in the header. Do not leave it null when such a number is present."""

PARTS = {
    "daterule": [DATERULE],
    "fewshot_daterule": [FEWSHOT, DATERULE],
    "minimal": [],
    "defs": [DEFS],
    "anti": [ANTI],
    "defs_anti": [DEFS, ANTI],
    "conservative": [CONSERVATIVE],
    "steps": [STEPS],
    "fewshot": [FEWSHOT],
    "negative": [NEGATIVE],
    "fewshot_negative": [FEWSHOT, NEGATIVE],
    "defs_anti_fewshot_negative": [DEFS, ANTI, FEWSHOT, NEGATIVE],
}

# ---- output formats (E / B3 / B4 / B5) -----------------------------------------------------------------------------
SCALARS = ["deed_type", "district", "office", "registration_date", "presentation_date", "consideration_amount", "old_reg_no"]
LISTS = {"seller_details": ["name", "relation_name", "address"], "buyer_details": ["name", "relation_name", "address"],
         "property_details": ["village", "khata", "plot", "area"]}


def shape(schema_keys, output):
    """Example structure shown to the model for a given output format."""
    s = {}
    for k in schema_keys:
        if k in SCALARS:
            s[k] = {"plain": "", "evidence": {"value": "", "evidence": ""},
                    "confidence": {"value": "", "confidence": "high|medium|low"},
                    "ambiguity": {"value": "", "candidates": [], "reason": ""},
                    "candidates": {"candidates": [], "selected": "", "evidence": ""}}[output]
        else:
            row = {a: "" for a in LISTS[k]}
            if output == "evidence":
                row["evidence"] = ""
            elif output == "confidence":
                row["confidence"] = "high|medium|low"
            s[k] = [row]
    return s


OUTPUT_NOTES = {
    "plain": "Fill every field of the schema exactly.",
    "evidence": "For every scalar field give an object with the value and the exact supporting text copied from the OCR as evidence; "
                "every list entry also carries an evidence field with copied OCR text. If a value is not supported, set value to null "
                "and evidence to null.",
    "confidence": "For every scalar field give the value and a confidence (high, medium or low); every list entry carries a confidence too.",
    "ambiguity": "If several candidate values exist and the OCR does not resolve which is correct, do not choose: set value to null and "
                 "list the candidates with a short reason. Otherwise give the value and leave candidates empty.",
    "candidates": "For every scalar field list the candidate values found in the text, the selected one (or null) and the supporting OCR text.",
}


def build_user(cfg, ocr_block, schema_keys, n_pages, group_label=None, order="schema_first", schema_text=None):
    out = cfg.get("output", "plain")
    sch = shape(schema_keys, out)
    schema_str = schema_text or json.dumps(sch, ensure_ascii=False, indent=1)
    head = TASK + (f" Extract only: {group_label}." if group_label else "")
    parts = [head, "Schema:\n" + schema_str, OUTPUT_NOTES[out]]
    if cfg.get("format", True):
        parts.append(FORMAT)
    parts += PARTS[cfg.get("prompt", "minimal")]
    ctx = f"OCR text of the deed ({n_pages} page(s)):\n{ocr_block}"
    if order == "schema_first":
        return "\n\n".join(parts + [ctx])
    return "\n\n".join([head, ctx] + parts[1:])
