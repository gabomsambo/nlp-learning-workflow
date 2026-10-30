# Measuring the fact-checker

The fact-checker flags claims in a podcast script that the paper does not support.
A prompt tweak, a cheaper model or a changed severity rule can quietly make it worse:
miss the wrong number it used to catch, or cry wolf until nobody reads the flags.
This is how that gets measured instead of guessed.

| File | What it is |
|---|---|
| `evals/factcheck/calibration.json` | Hand-labelled claims (the answer key) |
| `evals/factcheck/baseline.json` | The numbers the committed pipeline scored, and the gate's floor |
| `scripts/factcheck_eval.py` | Runs a verifier on the labelled scripts, scores it, gates it |
| `nlp_pillars/factcheck/evaluation.py` | The scoring itself. Pure, offline, unit-tested |

## Before you change a fact-check prompt, model or severity rule

1. Run the free checks. They need no key and cost nothing:

       python scripts/factcheck_eval.py --dry-run          # dataset is valid (this also runs in CI)
       python scripts/factcheck_eval.py --check-sources    # needs the database: scripts and quotes still match

2. Make the change.
3. Run the paid gate. It makes 2 full passes over the five scripts (`--runs`) and pools them, about $1–1.5 in all (the baseline records the real figure). It needs `ANTHROPIC_API_KEY` and the database (`SUPABASE_URL`, `SUPABASE_KEY`):

       python scripts/factcheck_eval.py --live

   To try a different model without touching code, override the route for the run:

       python scripts/factcheck_eval.py --live --models rejudge=claude-opus-5-5

   `--budget` (default $3) stops the run before it spends more, and nothing is saved on
   a failed call.
4. Read the table and the two lists under it (missed flags, false alarms). They name
   the script, line and claim, so you can open the script and see what changed.
5. Exit code `0` means serious-error precision and recall are within tolerance of the
   baseline. `1` means one dropped: do not merge that change as is. `2` means the run
   could not be trusted (invalid dataset, a script or paper that changed since it was
   labelled, a baseline measured on different labels, a failed model call). Two is never
   a pass.
6. If the change is meant to alter the numbers (a better verifier), re-measure and commit
   the new baseline in the same PR, so a reviewer sees the before and after:

       python scripts/factcheck_eval.py --live --write-baseline

CI runs only `--dry-run` and the unit tests (`tests/test_factcheck_eval.py`). It never
calls a model. That test also fails if `calibration.json` was edited without re-measuring
`baseline.json`.

Free ways to look without spending anything:

    python scripts/factcheck_eval.py --from-stored --scripts <script id prefix>   # score the checks already in the database
    python scripts/factcheck_eval.py --predictions out.json                       # score any verifier's saved output

## What is scored

A **label** is one clause of one script line with a verdict decided by reading the whole
paper: `supported`, `partially_supported`, `contradicted`, `unsupported` or
`not_checkable`, plus scope, category and error type. Its **severity** (`high`, `medium`,
`low`, `none`) is not typed in. It is computed by `nlp_pillars/factcheck/severity.py`, the
same table the product uses, and the dataset check rejects a label that disagrees. Changing
that table therefore breaks the dataset check instead of quietly shifting the metrics.

A **prediction** is a claim the verifier produced, with its line, span and severity. It
matches a label on the same line when their character spans overlap by at least half of
the shorter one. Matching is many-to-many, so an extractor that merges two clauses into
one claim is not punished for it.

Each script lists `coverage` windows. Inside them the labelling is exhaustive: every
checkable claim is labelled, supported ones included. So a flag inside a window that
matches no label is a **false alarm**. Predictions outside the windows cannot be judged
and are counted separately (`unscored`).

Per severity, exactly:

* **recall**: of the labels with that severity, how many got a prediction with that severity;
* **precision**: of the predictions with that severity, how many sit on a label with that severity.

So a `high` prediction on a claim labelled `medium` is a false positive for high and a
miss for medium. `any` pools the three flagged levels ("was it flagged at all"). The
**gate** is `high.precision` and `high.recall`, the serious errors: a wrong number, a
result attributed to the wrong system, a specific detail the paper never gives.

A claim the verifier never extracted counts as a miss (`got: not extracted`). That is
deliberate: extraction is half of the pipeline, and a flag that is never raised protects
nobody. `--extras ignore` drops the false-alarm penalty for unlabelled claims if you want
to look at severity agreement alone.

### How much to trust a number

The set is small on purpose, and the report says so. It has **186 labels, of which only 8
are `high`**. One missed serious error moves high recall by 12.5 points, and the report
prints a Wilson 95% interval next to each recall. Two unchanged runs differed by 0.15 in high precision, so live runs pool 2 passes and the default tolerance is 0.10. A check that fails twice is scored as no flags and printed as FAILED CHECK. The models
are not deterministic either (Claude rejects `temperature`), so two runs of the same
pipeline can differ by a claim or two; run twice before blaming a change for a one-claim
swing. The honest way to tighten this is more labels, especially `high` ones (below).

The labels were made by Claude models, so they share blind spots with the verifier they
grade. What stands against that: the first labeller only ever saw the script and the paper
(never a fact-check result), every non-supported label carries a verbatim paper quote or
a note saying what was searched for, a second reader re-checked each one, and the
research report's hand audits (the DeepSeek-VL ablation numbers, "six of eight" in the
fastText episode, the AML "no validation" line) were re-derived from the paper text rather
than copied.

## Scoring another verifier, or a generation change

`--predictions FILE` scores anything that can write this JSON, with no code change here:

    {"verifier": "my-minicheck-run", "cost_usd": 0.0,
     "predictions": {"<script_id>": [
         {"line_index": 138, "span_start": 40, "span_end": 98, "severity": "high"}, ...]}}

`span_text` can stand in for the offsets. `--verifier package.module:fn` runs a callable
`fn(script_text, paper_text, chunks, context) -> {"claims": [...], "cost_usd": 0.0}` over
every labelled script and scores it. The same table and the same gate apply, so a
cross-family judge, a retrieval-based checker or an ensemble is compared on the same labels.

The labels belong to these exact script texts, so a *generation* variant (a changed
podcast prompt) cannot be scored line by line. Use the calibrated verifier as the
yardstick instead: run it on scripts made by each variant and compare the flagged share
(`counts.high / counts.claims` in `podcast_fact_checks`). The gate above is what says that
yardstick is still trustworthy.

## Growing the set

* **Dismissals are the intake.** In Listen, a reader can dismiss a flag with a note
  ("the paper says this on p. 4"). It is stored on the claim and every dismiss and restore
  is appended to `podcast_claim_reviews` (migration 031). Export them:

      python scripts/factcheck_eval.py --export-dismissals candidates.json

  A dismissal says the flag was wrong, not what the right verdict is, so each one is a
  candidate to be read against the paper, never a label by itself.
* **Add a label** to a script in `calibration.json`: line, a verbatim `span_text`, the
  verdict, scope, category and error type, one to three verbatim paper quotes (or an
  `absence_note` for `unsupported`), a `note`, and `provenance` with a labeller and a
  second reader. Keep a window exhaustive: if you add a label inside a window, every other
  checkable claim in that window must already have one. A new window needs the whole
  window labelled.
* `python scripts/factcheck_eval.py --dry-run` validates the shape and the severity
  arithmetic; `--check-sources` confirms the script hash, the line text and every quote
  against the database.
* Changing any label changes `dataset_sha256`, so re-measure the baseline in the same PR.
* Never label from a fact-check's own output. That makes the set agree with the system it
  is supposed to catch out.

## Provenance of the first set

186 labels from five scripts (DeepSeek-VL, the AMLworld paper, fastText, Minimalist
Grammars, Graph Feature Preprocessor), labelled on 2026-10-30 against the stored
`paper_texts`. Per label: `provenance.labelled_by`, `basis`, `date`, `audited_by` and
`audit` (`agreed`, `changed` or `added`). One label was changed on audit: the AML line
"there's no validation of … network topology against real AML data" was first labelled
`partially_supported` (the paper compares laundering-motif rates with real Italian bank
data, which the labeller counted as partial validation) and was changed to `contradicted`,
matching the research report's hand audit.
