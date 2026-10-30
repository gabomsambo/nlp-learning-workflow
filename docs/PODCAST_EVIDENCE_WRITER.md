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
(`evals/factcheck/baseline.json`) is the yardstick for both, and misses are invisible in both.
`factcheck_eval.py --dry-run` is the free check that the yardstick itself is intact.

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

## Flipping the default

Not done in this PR. When the gate passes, set `podcast_writer_mode` to `evidence` in
`nlp_pillars/config.py` (and apply migration 035 first so the ledger is recorded), in a
follow-up that carries the new measurements.
