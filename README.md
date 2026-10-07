# Deed-extraction evaluation harness

Evaluates how well a model extracts structured fields from scanned Indian land-registration deeds
(Odia + English), given OCR text and a target JSON schema. Runs your model over a set of deeds, grades
every field against a human-validated answer key, and reports accuracy plus a breakdown of failure causes.

See **[reports/evaluation-report.pdf](reports/evaluation-report.pdf)** for example results and how to read
the metrics.

## What it expects as input

A deed set laid out like this (you provide this; it is not included in this repo):

```
<root>/deeds/<reg_no>/<page_file>.{jpg,png,...}   scanned page images, one folder per deed
<root>/gt/<reg_no>.json                           ground-truth record per deed, matching schema.json
<root>/gt_pages/<reg_no>.json                     optional: {"<page_no>": ["field_name", ...]}, sharpens
                                                   error categorisation; skipped entirely if absent
<ocr-dir>/<reg_no>/<page_stem>.txt                OCR text per page (one .txt per image, same stem)
<deeds-file>                                      one reg_no per line, listing which deeds to evaluate
```

`--root` (default `data`), `--ocr-dir` (default `data/ocr`) and `--deeds` (default `data/deeds.txt`) are all
separately overridable flags on `evalkit.run` / `evalkit.regrade`.

`schema.json` shows the target JSON shape (7 scalar fields + 3 list fields: sellers, buyers, property
rows) that both the ground truth and every model prediction must match.

## Setup

```
pip install -r requirements.txt
```

Point the harness at your model's OpenAI-compatible chat-completions endpoint:

```
export GEMMA_BASE_URL=http://<host>:<port>/v1       # required
export GEMMA_MODEL=<served-model-name>              # default: gemma4
export GEMMA_API_KEY=<key-if-your-endpoint-needs-one>
```

(or put them in a `.env` file in this directory; it's loaded automatically and gitignored)

## Running the harness

```
python -m evalkit.run --experiments all --root data --deeds data/deeds.txt --ocr-dir data/ocr --out evalkit_runs
python -m evalkit.report --runs evalkit_runs --out evalkit_report
```

- `--experiments` takes a single id, a comma-separated list, `all`, or a group name (`context`, `prompt`,
  `images`, `verification`, `voting` — see `evalkit/experiments.py` for the full registry of configurations
  and what each one changes).
- `evalkit.run` writes one record per (configuration, deed) to `evalkit_runs/<id>/records.jsonl` — the raw
  model output, parsed JSON, and per-field grading.
- `evalkit.report` aggregates those records into summary tables (precision/recall/F1 per configuration,
  strict and lenient).
- To re-score existing run records without re-calling the model (e.g. after a scoring fix):
  `python -m evalkit.regrade --runs evalkit_runs`.

## How it works

```
OCR text (per page, page-labelled) ---> prompt built per configuration ---> your model ---> JSON record
                                                                                    |
                                                                                    v
                                                                   graded field-by-field against ground truth
```

- `evalkit/experiments.py` — registry of every configuration (what changes relative to the baseline: how
  much context the model sees, prompt wording, extra verification calls, images on/off, voting, etc.)
- `evalkit/data.py` — loads a deed's images, OCR text and ground truth
- `evalkit/strategies.py` — builds and runs the model call(s) for one configuration on one deed
- `evalkit/llm.py` + `gemma_client.py` — the HTTP client to your model's endpoint
- `evalkit/metrics.py` + `score.py` / `score2.py` / `lenient.py` / `ocr_coverage.py` — field-level scoring
  (strict matching, a lenient variant that forgives script/formatting differences, and a proxy for whether
  the correct value is even present in the OCR text)
- `build_messages.py` / `prompts.py` / `extract.py` — message/prompt construction shared with the scoring
  and strategy code above
