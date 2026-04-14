# autorig cut optimizer

Autonomous optimization of the lasso-cut character-part pipeline.
The agent modifies the pipeline code, runs it, evaluates output quality,
and keeps or discards each change — looping forever.

---

## Setup

Before starting, read these files completely:

- `problem.md` — the problem statement and goals
- `lasso_batch.py` — the main pipeline (primary file to modify)
- `processing/refine_part.py` — polygon masking and raw crop extraction
- `processing/simple_background.py` — flood-fill background removal
- `processing/genai_background.py` — alternative GenAI-based background removal
- `evaluate.py` — fixed evaluation script (do NOT modify)

Then:

1. **Agree on a run tag** with the user (e.g. `apr14`). The branch
   `autorig-opt/<tag>` must not already exist.
2. **Create the branch:** `git checkout -b autorig-opt/<tag>`
3. **Initialize results.tsv** with just the header row (see Logging below).
4. **Establish the baseline** by running the full pipeline and evaluating
   before touching anything.

---

## The optimization target

After every pipeline run, evaluate with:

```bash
python evaluate.py > eval.log 2>&1
grep "^overall_score:" eval.log
```

**Higher is better.** Current baseline: `overall_score: 0.740653`

The six metrics and what they measure:

| Metric | What it catches |
|---|---|
| `color_fidelity` | Did Gemini draw the right content? Low = hallucinated or wrong body part |
| `edge_sharpness` | Are alpha edges hard and clean? Low = ragged fringe |
| `connected_components` | Is the part one piece? Low = fragmented |
| `interior_holes` | Are occluded areas filled in? Low = hollow part |
| `area_ratio` | Is output the right scale vs the lasso polygon? Low = wrong size |
| `white_residue` | Is the white background fully removed? Low = bg contamination |

**Focus for this run: cut quality.**
`color_fidelity` is the most variable metric (range 0.045–0.899 in the
baseline) and is the primary signal. `area_ratio` is broken (0.000 on most
parts — a known scale problem to be fixed in a separate run). Do not treat
`area_ratio` as a signal to optimize right now.

---

## What you CAN modify

**`lasso_batch.py`** — the main pipeline. Everything in here is fair game:
- The Gemini prompt text inside `_refine_with_gemini()`
- Which images / how much context you send to Gemini
- The Gemini model name and generation config (temperature, etc.)
- Image preprocessing before sending to Gemini (resize, crop, annotate)
- Post-processing after receiving from Gemini
- The `_cleanup_part()` logic and parameters
- The order of operations (e.g. remove background before or after Gemini)

**`processing/refine_part.py`** — raw crop extraction:
- `MASK_PAD` and `SEARCH_PAD` constants
- How the search region and crop box are computed

**`processing/simple_background.py`** — background removal:
- Flood-fill tolerance
- Blur radius and edge cleanup

**`processing/genai_background.py`** — GenAI background removal:
- This is an alternative background removal path you may want to use
  instead of or in addition to `simple_background`

## What you CANNOT modify

- `evaluate.py` — fixed ground truth, never touch this
- `resources/` — source images and lasso coordinate JSON files
- `problem.md`

---

## Running the pipeline

```bash
# Full run — all 5 characters (recommended before keeping a change)
python lasso_batch.py > run.log 2>&1

# Fast iteration — single character (use to test a direction quickly)
python lasso_batch.py tank > run.log 2>&1
```

If a run crashes, check:
```bash
tail -n 40 run.log
```

**Speed note:** each full run makes ~60 Gemini API calls (12 parts × 5
characters). Use single-character runs to explore a direction, then do a
full run before deciding to keep.

---

## Evaluating

```bash
# All characters
python evaluate.py > eval.log 2>&1
grep "^overall_score:" eval.log

# Single character
python evaluate.py tank
```

---

## Logging results

`results.tsv` — tab-separated, NOT comma-separated. Do not commit this file.

```
commit	overall_score	status	description
```

| Column | Format |
|---|---|
| commit | 7-char short hash |
| overall_score | 6 decimal places (e.g. `0.740653`) |
| status | `keep`, `discard`, or `crash` |
| description | short description of what changed and why |

Example:

```
commit	overall_score	status	description
a1b2c3d	0.740653	keep	baseline
b2c3d4e	0.762100	keep	prompt: ask Gemini to preserve exact pixel region
c3d4e5f	0.731000	discard	removed full-character context image hurt fidelity
d4e5f6g	0.000000	0.0	crash	tried genai bg removal - missing temp files
```

---

## The experiment loop

LOOP FOREVER once the baseline is established:

1. Check git state: current branch and latest commit
2. Form a hypothesis — what specific change might improve `color_fidelity`?
3. Edit the file(s)
4. `git commit -m "experiment: <description>"`
5. Run pipeline on one character first: `python lasso_batch.py tank > run.log 2>&1`
6. If promising, run full pipeline: `python lasso_batch.py > run.log 2>&1`
7. Evaluate: `python evaluate.py > eval.log 2>&1 && grep "^overall_score:" eval.log`
8. Log the result to `results.tsv`
9. If `overall_score` improved → **keep**, advance the branch
10. If equal or worse → `git reset --hard HEAD~1` then restore outputs with a fresh run

**Experiment cap: stop after 25 experiments** (not counting the baseline).
Log the final summary and wait for the human. Do not ask during the run
whether to continue — only stop at 25 or if manually interrupted.

If you run out of ideas before 25, re-read the files, look at which
specific parts have the lowest `color_fidelity`, and reason about why
Gemini might be producing wrong content for those parts.

**Crashes:** If a run crashes (API error, missing import, etc.) and it is
something trivial to fix, fix it and re-run. If the idea is fundamentally
broken, log `crash`, reset, and move on.

---

## Key observations from the baseline run

```
overall_score: 0.740653

giraffe  0.7240   hippo  0.7138   papa  0.7727   tank  0.7300   toby  0.7628
```

- `color_fidelity` drives most of the variation. Worst parts by fidelity:
  - `papa/left_forearm`  0.045  — severe hallucination
  - `hippo/right_hand`   0.068  — severe hallucination
  - `hippo/left_hand`    0.072  — severe hallucination
  - `hippo/right_thigh`  0.074  — severe hallucination

- `edge_sharpness` is uniformly 0.97–1.00 — background removal already good,
  this metric will mainly catch regressions

- `area_ratio` is 0.000 for almost all parts — a known scale bug being fixed
  separately. Ignore it as an optimization target for this run.

- `connected_components`, `interior_holes`, `white_residue` are all near 1.0
  — healthy, keep them that way

---

## Suggested starting hypotheses

These are starting points — not a fixed plan. Reason about the problem and
try your own ideas too.

1. **Prompt framing** — the current prompt asks Gemini to return the part
   "with a white background". This may cause it to redraw the part from
   scratch (hallucination) rather than extracting what's there. Try asking
   it to extract and clean, not redraw.

2. **Context provided** — currently sends the full character image + raw
   crop. Gemini may be confused by so much context. Try sending only the
   raw crop, or a tightly cropped region around the part.

3. **Role of the full image** — alternatively, the full image helps Gemini
   understand what the part should look like. Try different orderings or
   descriptions of the two images.

4. **Temperature** — currently 0 (deterministic). A small non-zero value
   might help on parts where Gemini is confidently wrong.

5. **Explicit constraints** — tell Gemini explicitly: do not add body parts
   that aren't visible, do not change the art style, do not change the scale.

6. **Background removal approach** — `processing/genai_background.py`
   provides an alternative (GenAI-based) background removal. It may produce
   cleaner alpha on parts where `simple_background` fails.
