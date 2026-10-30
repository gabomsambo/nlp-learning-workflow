# The evidence-ledger podcast writer

`PODCAST_WRITER_MODE=evidence|free` chooses how a podcast script is written. `free` is the
original writer (four Ground Pack extractions, then the script from the Ground Pack and the
whole paper) and is the default. `evidence` writes from a ledger of facts the paper states,
each tied to a quote that code checked, and every spoken line cites the ledger entries it
relies on. Evidence-only: the script may not add background from outside the paper. (A
clearly labelled background channel could be added later; the tag channel already exists.)

Why: `data/nlp-factcheck-prevention/report.md` (2026-10-30). Checking after the fact cannot be
the only defence, and on three papers writing from a ledger cut flagged problems per 1,000
words by 72% and the fact-check bill by 63-79%. That was one sample per arm. This writer
stays off by default until the gate below passes on the real thing.

## What it does

```
paper text (+ repaired tables)                                     PODCAST_WRITER_MODE=evidence
   │
   ├─► ledger call (Haiku 5.5, one structured call, whole paper)
   │      quote · where · statement · kind, quote first so it copies before it paraphrases
   │      CODE checks each quote against the paper; a statement whose numbers its quote
   │      does not contain is dropped too            ──► F2, F3, ...   (F1 = title/authors/year
   │                                                                     from the database)
   ├─► paper_tables rows (status spliced, from table repair)  ──► T1, T2, ...  column=value,
   │      no model turns them into facts; a checkmark grid keeps its blanks as (empty)
   │
   └─► script call (Sonnet 5): the ledger only, no Ground Pack, no raw paper
          every [HOST] line ends [[F12,T3]], or [[-]] when it states no paper fact
                │
        parse + strip the tags ──► stored `script` (clean) + `line_citations` {line index: ids}
```

* Two calls instead of five. The Ground Pack stages show `skipped` with the reason.
* **The stored script is the stripped one.** Fact-check windows, TTS, read-along alignment
  and the phone reader all read `script`, so none of them can see a tag.
  `line_citations` is keyed by that script's 0-based line index. `cue_parser.spoken_line_text`,
  `parse_podcast_script` and `prepare_text_for_indextts` also strip id-list tags, in case a
  row predates stripping or was edited by hand. Only text matching the id grammar is stripped.
* Tags are ASCII `[[...]]`. The first experiment used Unicode brackets and the model put a stray
  character in front of the closing one.
* Prompts are versioned: `ledger-v1` and `evidence-v1` (`nlp_pillars/evidence/prompts.py`),
  stored with every ledger. The writer's system prompt is the free one with its GROUNDING RULES
  block replaced, derived at import so output format, TTS rules and the settings paragraph
  cannot drift; if the free prompt loses the markers, import fails.
* `python scripts/render_podcast_prompts.py writer_mode=evidence` (or `both`) prints both modes'
  prompts without calling a model.
* The ledger model is the `podcast_ledger` route (`LLM_PODCAST_LEDGER_MODEL`, default
  `claude-haiku-5-5`).

### Stored

`docs/migrations/035_podcast_evidence_writer.sql` (hand-applied, **not applied** by this PR)
adds `podcast_scripts.writer_mode`, `ledger` and `line_citations`. The ledger holds every
entry with its quote and how it matched (`exact`, `markup`, `fuzzy`), what failed
verification and why, the models and costs, and the writer-time measurements below. Until it
is applied an evidence-mode script still saves: `add_podcast_script()` drops the three keys,
logs the migration path, and only the ledger record is lost. A free script never sends them,
so the free writer saves exactly as before on a database without 035.

### Failure behaviour

* The ledger call failing, or fewer than 5 verified facts (besides the identity entry), stops
  the run **before** the script call and says why. Evidence mode never falls back to the free
  writer on its own: the person asked for a script every line of which cites the paper.
* A line without a tag, a tag naming an id the ledger does not have, a drop rate over 10%, and
  table rows that did not fit the ledger (240 are offered) become `source_material.warnings`.
* A paper nobody has run table repair on has no `T#` entries; that is normal, and a failed
  read of `paper_tables` is logged and treated the same.

### Measured on every script

`ledger.stats`: spoken, tagged, `[[-]]` and untagged lines; unknown ids; narrative and table
entries used; and `number_violations`, the lines stating a number (outside small whole numbers)
that their cited entries do not state. That last one is informational: it over-flags spoken
years and `7B`-style forms. Table-row use is reported separately from narrative-entry use
because no episode cites 60% of a 240-row table.

## The gate

`scripts/podcast_writer_gate.py` runs both arms on the same papers, the app's own fact-check
engine checks every script against the same (table-repaired) paper text, and the serious
flags are hand-labelled blind to the arm. The arithmetic is `nlp_pillars/evidence/gate.py`
(unit-tested); the runner is paid and is never run by CI.

| Criterion | Threshold |
|---|---|
| Papers x samples per arm | at least 8 x 2 |
| Hand-confirmed serious (high+medium) flags per 1,000 spoken words | evidence no higher than free |
| Spoken words | at least 85% of free |
| Narrative ledger entries used | at least 60% |
| Untagged spoken lines | 0 |
| Quotes dropped by verification | at most 10% of those proposed |

`flags` writes the serious flags shuffled with the arm hidden (`flags_blind.md`). A label is
`real` (the line states something the paper does not support, contradicts or materially
overstates), `minor` (right that it goes past the paper, not serious) or `wrong` (the paper
supports it). Precision is real over all labelled serious flags, strict and with minor counted.

**Recall is not measured on the new scripts.** Exhaustive labelling of every claim is what
`evals/factcheck/calibration.json` did for five scripts, and the labels there belong to those
exact script texts. The same calibrated checker judges both arms, so its measured recall
(`evals/factcheck/baseline.json`: high-severity precision 0.67 and recall 0.25 on 16 labels,
medium recall 0.25) is the yardstick for both, and misses are invisible in both. With recall
that low, every "per 1,000 words" figure here is a floor on the true error rate, and a
difference between the arms is only as trustworthy as the assumption that the checker misses
both arms' errors equally. `factcheck_eval.py --dry-run` is the free check that the yardstick
itself is intact.

```
python scripts/podcast_writer_gate.py plan     --out DIR
python scripts/podcast_writer_gate.py prepare  --out DIR      # table pages, cents; no database writes
python scripts/podcast_writer_gate.py generate --out DIR      # PAID
python scripts/podcast_writer_gate.py check    --out DIR      # PAID
python scripts/podcast_writer_gate.py flags    --out DIR      # free; then label them
python scripts/podcast_writer_gate.py report   --out DIR --labels labels.json
```

The Anthropic key is read from `--env-file` at run time (that variable only). The free arm's
Ground Pack runs on Claude Haiku 5.5 here so the two arms differ only in the writer; with
DeepSeek funded the production Ground Pack would be cheaper, not different in kind.

## First measurement (2026-10-30): the gate did NOT pass

8 library papers x 2 samples x 2 arms = 32 scripts, written by the real `PodcastAgent` over the
same table-repaired text (the repair was done in memory by `prepare`; the library's stored text
was not touched). Free arm Ground Pack on Claude Haiku 5.5. 73 serious flags labelled blind to
the arm by one reader (Claude), so the labels share the checker's family and are one person's
reading, not an independent audit. Spend $6.32 of the $8 cap, every call logged.

| | free | evidence |
|---|---|---|
| spoken words | 47,934 | 44,033 (91.9%; worst paper 74%, the credit-card paper whose tables were flagged, not spliced) |
| claims checked | 1,685 | 1,740 |
| high / medium / low flags | 9 / 25 / 117 | 15 / 24 / 62 |
| all flags per 1,000 words | 3.15 | 2.29 |
| **serious (high+medium) per 1,000 words, engine** | **0.71** | **0.89** |
| serious flags read as real / minor / wrong | 9 / 23 / 2 | 10 / 23 / 6 |
| **confirmed serious per 1,000 words** | **0.19** | **0.23** (paired difference +0.05, 95% CI -0.16 to +0.29 over 8 papers) |
| generation cost | $2.22 | $1.47 |
| fact-check cost | $1.12 | $1.32 |

| Criterion | Result |
|---|---|
| 8 papers x 2 samples | pass |
| confirmed serious flags no higher than free | **fail** (0.23 vs 0.19; the interval spans zero) |
| spoken words >= 85% of free | pass pooled (91.9%); 1 of 8 papers at 74% |
| narrative ledger entries used >= 60% | pass (97.8% mean, 93.5% min) |
| 0 untagged lines | **fail**: 14 of 1,781 spoken lines in 3 of 16 scripts (7 were `... [TRANSITION]` lines that took a cue instead of a tag) |
| quote drop rate <= 10% | **fail**: 99 of 782 proposed (12.7%); 98 of 830 (11.8%) on a re-run of the ledger step alone after the number-reading fix below |

What it says, plainly:

* The engine's low-severity flags fell by about 42% per 1,000 words, which is the report's
  mechanism working: invented framing and outside knowledge mostly disappear. **The serious
  flags did not fall.** The report's 3-paper, 1-sample result (-72% flags, 8 -> 4 high+medium)
  did not reproduce at this size, and neither did the 63-79% cheaper fact-check: the evidence
  scripts carry more claims per word (39.5 vs 35.2 per 1,000) so more reach the Sonnet
  re-judge.
* What the evidence arm's serious errors look like: a ledger statement that misreads its
  quote and the writer repeating it (credit-card paper: a 6.0% / 85.4% *relative* gain
  spoken as PR-AUC values, three real flags), a quoted definition the writer contradicts
  (placement), and small attribution slips. 4 of its 10 real errors are in the one paper where
  table repair flagged every table and no `T#` rows existed.
* Free-arm real errors were the report's classes: overgeneralised "most prior work" and
  "fourfold in both cases" (derived numbers), a method misdescribed.
* Not measured: recall on the new scripts (see above), a listener test, and more than one
  labeller.

What to change before measuring again (not done here, each needs a paid re-run):
`evidence-v2` rule that every `[HOST]` line takes a tag even when it ends in a cue (the 7
transition lines); the ledger builder dropping fewer entries (49 quotes still not found, most
of the rest are statements that add a number the quote lacks); a ledger-statement check on the
*relative vs absolute* wording of percentages; and running table repair first on papers whose
tables were flagged. Do **not** flip the default.

## Flipping the default

Not done in this PR, and the gate above did not pass. When the gate passes, set `podcast_writer_mode` to `evidence` in
`nlp_pillars/config.py` (and apply migration 035 first so the ledger is recorded), in a
follow-up that carries the new measurements.
