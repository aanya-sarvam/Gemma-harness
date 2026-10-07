"""Experiment registry. Every experiment = BASE config + overrides, with an explicit hypothesis.

CONTROL is A7 (full OCR, page boundaries, minimal prompt, plain output): later experiments change ONE thing relative to it
unless 'change' says otherwise. Add a new experiment by appending one exp(...) call; no engine code needs to change.
"""
BASE = dict(units="full", ocr_fmt="boundaries", ocr_variant="raw", prompt="minimal", output="plain", groups="all",
            retrieve=None, verify=None, merge="vote", images=0, order="schema_first", api_mode="object",
            temperature=0.0, max_tokens=4096, max_side=1280)

EXPERIMENTS = {}


def exp(id, group, hypothesis, change, expected, **over):
    cfg = dict(BASE)
    cfg.update(over)
    EXPERIMENTS[id] = dict(id=id, group=group, hypothesis=hypothesis, change=change, expected=expected, cfg=cfg)


# ---- A: context ---------------------------------------------------------------------------------------------------
exp("A1_full_plain", "context", "A whole-deed OCR with no page labels is a sound baseline.",
    "Whole OCR concatenated without page markers.", "Baseline; slightly worse than A7 if page labels help.",
    ocr_fmt="plain")
exp("A7_full_boundaries", "context", "Explicit page boundaries help the model place facts (e.g. endorsement on the last page).",
    "A1 + 'PAGE n:' labels. THIS IS THE CONTROL.", "Equal or better than A1, esp. dates/office.", ocr_fmt="boundaries")
exp("A8_full_json_pages", "context", "A structured [{page,text}] input is no better than labelled text.",
    "Pages passed as a list of objects.", "About equal to A7.", ocr_fmt="json")
exp("A2_page1", "context", "One page per call reduces context overload but loses cross-page facts.",
    "Per-page calls, merged by vote.", "Higher precision on page-local fields, lower recall on dates/amount (different pages).",
    units="page")
exp("A3_chunk2", "context", "2-page chunks recover some cross-page context.", "chunk size 2.", "Between A2 and A7.", units="chunk2")
exp("A4_chunk4", "context", "4-page chunks cover almost whole deeds, so behave like A7.", "chunk size 4.", "About equal to A7.", units="chunk4")
exp("A5_chunk8", "context", "8-page chunks equal the full deed for our deeds (max 6 pages): a sanity check.", "chunk size 8.",
    "Identical to A7 (expected: no difference).", units="chunk8")
exp("A6_slide4", "context", "Overlapping windows help when facts straddle a boundary.", "Windows of 4 pages, stride 3.",
    "Equal or slightly better than A4 on 5-6 page deeds only.", units="slide4")

# ---- B: extraction strategy ---------------------------------------------------------------------------------------
exp("B2_conservative", "strategy", "Telling the model to extract only explicitly supported values lowers hallucinations and recall.",
    "A7 + conservative instruction.", "Fewer hallucinations/spurious values, lower recall.", prompt="conservative")
exp("B3_evidence", "strategy", "Requiring a quote per value grounds answers; unsupported values can then be dropped.",
    "A7 + value+evidence output (also scored after dropping values whose evidence is not in the OCR: +filter).",
    "Higher precision after filtering, lower recall.", output="evidence")
exp("B4_confidence", "strategy", "Self-reported confidence is informative but not reliable.", "A7 + confidence labels.",
    "Same accuracy as A7; confidence only weakly predicts correctness.", output="confidence")
exp("B5_ambiguity", "strategy", "Letting the model return candidates instead of choosing reduces wrong picks on dates.",
    "A7 + ambiguity-aware output.", "Fewer wrong dates, more nulls.", output="ambiguity")
exp("B6_field_by_field", "strategy", "Focused one-field calls beat a single complex call.", "10 calls, one per top-level field, full OCR each.",
    "Higher accuracy on hard fields (amount, buyers); 10x cost.", groups="field_by_field")
exp("B7_field_group", "strategy", "Grouped fields keep most of the benefit of B6 at lower cost.", "5 calls, one per field group.",
    "Between A7 and B6.", groups="field_group")

# ---- C: retrieval -------------------------------------------------------------------------------------------------
exp("C2_retrieve_llm", "retrieval", "Dropping irrelevant pages reduces distraction.", "Gemma selects relevant pages, then extracts from them.",
    "Equal or better precision; risk of dropping a needed page.", retrieve="llm")
exp("C3_retrieve_fields", "retrieval", "Per-group page selection from a page->fields map is more precise than one page set.",
    "Gemma maps pages to fields; each field group sees only its pages (field_group calls).", "Better precision than C2 at more calls.",
    retrieve="fields", groups="field_group")
exp("C4_retrieve_keyword", "retrieval", "A keyword filter is a cheap baseline for page selection.", "Keyword page filter per field group, then extract.",
    "Worse than model selection on handwritten/irregular pages.", retrieve="keyword", groups="field_group")

# ---- D: prompting -------------------------------------------------------------------------------------------------
exp("D2_defs", "prompt", "Field definitions fix misunderstanding of fields (e.g. fee vs consideration; witness vs party).",
    "A7 + detailed field definitions.", "Better dates/amount/parties.", prompt="defs")
exp("D3_anti", "prompt", "Explicit no-guessing instructions reduce hallucinations.", "A7 + anti-hallucination block.",
    "Fewer spurious values.", prompt="anti")
exp("D4_fewshot", "prompt", "Worked examples show the output convention and null behaviour.", "A7 + 2 fictional examples.",
    "Better format conformity; small accuracy change.", prompt="fewshot")
exp("D5_negative", "prompt", "Naming typical mistakes reduces them.", "A7 + list of mistakes to avoid.", "Fewer wrong dates/amounts/parties.",
    prompt="negative")
exp("D6_fewshot_negative", "prompt", "Positive plus negative guidance is additive.", "A7 + examples + mistakes.", "Best of D4/D5.",
    prompt="fewshot_negative")
exp("D7_steps", "prompt", "Step-by-step instructions help candidate resolution.", "A7 + step instructions (no reasoning shown).",
    "Small gain on dates.", prompt="steps")
exp("D9_ocr_first", "prompt", "Order of schema and OCR matters.", "OCR before schema/instructions.", "Small difference either way.",
    order="ocr_first")
exp("D_all", "prompt", "Definitions + anti-hallucination + examples + mistakes together.", "A7 + all prompt parts.", "Best prompt.",
    prompt="defs_anti_fewshot_negative")

# ---- F: multi-pass ------------------------------------------------------------------------------------------------
exp("F2_verify", "multipass", "A second pass that checks each value against the OCR corrects wrong values.", "A7 then verify pass.",
    "Higher precision; added latency.", verify="verify")
exp("F3_evidence_verify", "multipass", "Asking for supporting text for every value removes unsupported ones.",
    "A7 then evidence pass; values without grounded evidence dropped.", "Higher precision, lower recall.", verify="evidence")
exp("F5_retrieve_extract_verify", "multipass", "Retrieval + verification stack gains.", "C2 + F2.", "Best precision; most calls.",
    retrieve="llm", verify="verify")

# ---- G: images ----------------------------------------------------------------------------------------------------
exp("G1_images_only", "images", "Gemma cannot read handwriting from page images alone.", "No OCR; first 3 page images, 1280px.",
    "Far worse than OCR.", ocr_fmt="none", images=3)
exp("G3_ocr_1img", "images", "A single page image adds little to OCR.", "A7 + first page image.", "About equal to A7.", images=1)
exp("G3_ocr_3img", "images", "Up to 3 images add little to OCR.", "A7 + first 3 page images.", "About equal to A7.", images=3)

# ---- H: OCR representation ----------------------------------------------------------------------------------------
exp("H2_clean_ocr", "ocr", "Light whitespace cleaning does not change extraction.", "A7 with light-cleaned OCR.", "About equal.",
    ocr_variant="clean")
exp("H4_flat_ocr", "ocr", "Layout (line breaks, table pipes) carries information about table structure.", "A7 with layout removed.",
    "Worse on property tables and parties.", ocr_variant="flat")

# ---- J / K / L ----------------------------------------------------------------------------------------------------
exp("J3_page_calls_llm_merge", "dedup", "An LLM entity-merge across per-page outputs beats vote merge.", "A2 + LLM merge.",
    "Fewer duplicate entities; risk of dropped ones.", units="page", merge="llm")
exp("K1_temp03", "params", "Small temperature does not help extraction.", "A7 at temperature 0.3.", "Equal or slightly worse.",
    temperature=0.3)
for _i in (1, 2, 3):
    exp(f"L4_repeat{_i}", "stability", "Identical runs at temperature 0 are stable.", "A7 repeated (same config).",
        "Near-identical results.", ocr_fmt="boundaries")


# ---- round 2: targeted at the failure modes found in round 1 (registration/presentation confusion, null deed number) ------
exp("R2a_daterule", "round2", "Round 1: 13/19 wrong registration dates were the presentation date and old_reg_no was null in 11/13 misses. An explicit rule for the two dates and the deed number fixes this.",
    "A7 + date/deed-number rule (no examples).", "Registration date and old_reg_no improve; other fields unchanged.", prompt="daterule")
exp("R2b_fewshot_daterule", "round2", "Few-shot (which already fixed old_reg_no) plus the date rule is additive.", "D4 + date/deed-number rule.",
    "Best of D4 and R2a.", prompt="fewshot_daterule")
exp("R2c_candidates_daterule", "round2", "Listing all candidate dates before choosing makes the choice explicit and corrects it.",
    "R2b with candidate-listing output (candidates, selected, evidence).", "Better dates; maybe more output tokens.",
    prompt="fewshot_daterule", output="candidates")
exp("R2d_vote3", "round2", "Majority vote over 3 sampled runs removes random selection errors.", "R2b, 3 samples at temperature 0.5, vote merge.",
    "Small gain only if errors are random rather than systematic.", prompt="fewshot_daterule", samples=3, temperature=0.5)
# ---- round 3: registration date asked of the endorsement page only (finding: it sits on the last page; earlier pages repeat the presentation date) ----
exp("R3a_reg_last_page", "round3", "Round 2: registration_date stays ~0.5 because earlier pages repeat the presentation date as a distractor. Asking for it from the last (endorsement) page only removes the distractor.",
    "R2b, but registration_date is a separate call that sees only the last page; all other fields see the full OCR.",
    "registration_date improves markedly; other fields unchanged.", prompt="fewshot_daterule", groups="reg_split", retrieve="reg_last")
exp("R3b_reg_kw_pages", "round3", "Using the last two pages that mention 'Registered' is more robust than the last page when the endorsement is not last.",
    "As R3a but pages chosen by keyword among the last two pages (fallback: last page).", "Equal to R3a or slightly better.",
    prompt="fewshot_daterule", groups="reg_split", retrieve="reg_kw")
exp("R3c_candidates_reg_last", "round3", "R2c (candidate listing) and R3a (registration date from the last page) were each tried on top of R2b but never together; if their gains are separate, the combination should beat both.",
    "R3a with candidate-listing output (candidates, selected, evidence) in both calls.", "Equal to or slightly better than R3a (0.505).",
    prompt="fewshot_daterule", output="candidates", groups="reg_split", retrieve="reg_last")