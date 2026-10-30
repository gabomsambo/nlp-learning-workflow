# Bridge-sentence check (M8)

The hand-labelled check that has to pass before a bridge sentence is shown
unreviewed (paper-map scout report, section 7, row M8).

**What it is.** 43 pairs of arXiv papers, run through the production pass
(`map_enrich.enrich`, via `scripts/eval_bridges.py`) with the real model
(`claude-haiku-5-5`), and every output read by hand against the abstracts and
the citing sentence it was given.

- 29 pairs where the first paper cites the second, with the citing sentence
  from the paper's HTML (ar5iv) as the stored-evidence stand-in for Semantic
  Scholar's `contexts`. Citing papers: BERT Rediscovers the Classical NLP Pipeline (Tenney),
  Attention is not Explanation (Jain and Wallace), What Does BERT Look At
  (Clark), Designing and Interpreting Probes with Control Tasks (Hewitt and
  Liang), Information-Theoretic Probing (Pimentel), and its MDL counterpart
  (Voita and Titov).
- 12 pairs with no citation between them (two references of one paper): the
  abstracts-only path.
- 2 pairs picked to have little in common (the model should say so or abstain).

**What it is not.** It is not the captain's library: the sandbox that built this
could not reach that database, and keyless Semantic Scholar answered 429 for
the whole session. The pairs are from the same neighbourhood the report
measured (interpretability, attention), not library↔library pairs. There is
one reader (the engineer who wrote the pass), not the captain, and no lesson
relation was exercised (the unit tests cover that path). The second run's prompt was written
after the first run's failures and scored on the same 43 pairs, so it is not a
held-out result.

## Results

| | Run 1 | Run 2 (prompt v1) | Run 3 (prompt v2, shipped) |
|---|---|---|---|
| bridges kept by the checks | 8 of 43 | 40 of 43 | 36 of 43 |
| faithful | | 32 | 35 |
| overreach (supported core, one claim goes beyond the sources) | | 6 | 1 |
| wrong (a claim no source supports) | | 2 | 0 |
| rejected by the checks | 35 | 3 | 7 |
| cost | $0.0105 | $0.0105 | $0.0112 |

- **Run 1** found a bug in the number check: it rejected "Tenney et al. 2019"
  because 2019 is in no source. Years in the pair's own labels are now allowed
  (a regression test pins it). The 8 it kept are not scored.
- **Run 2** showed what the checks do and do not prove. Every kept bridge had
  real quotes and no invented number, and still 2 of 40 stated something no
  source says (a made-up "the citation is for the 2018 paper, not the 2017
  one"; "internal representations of syntax are central" to a paper that
  evaluates predictions) and 6 went a step past the sources (*introduces*,
  *builds on*, *the kind of opaque model*; equating ELMo with Peters et al.'s
  representations from outside knowledge). Prompt v2 forbids those moves.
- **Run 3**: 35 of 36 kept are faithful, 1 overreaches (ELMo again), none is
  wrong. Of the 7 rejected, 3 were right to reject (two leak the prompt's
  source marker "[S3]" into the sentence, one has no connection to state) and
  4 were rejected only for being over 320 characters while the content was
  faithful. Two kept sentences are true but noisy (they remark on a preprint
  year); one hedges that the sources show no connection beyond a shared topic.
- A kept bridge is therefore *not* proof against a confident slip, which is why
  every bridge ships with `generated: true`, `reviewed: false` and its quotes,
  and why the page should say "written by a model" next to it.

**Recommendation.** Show bridges only with the quotes beside them and the
"written by a model, not reviewed" mark, until the captain has read about 30 of
their own library pairs. `MAP_ENRICHMENT_ENABLED=false` removes them entirely.

## Files

- `pairs.json`: the 43 pairs (arXiv ids, labels, the citing sentence).
- `labels.json`: the run-3 output for each pair and the verdict.
- `fetch_items.py`: rebuilds the items file (abstracts from the arXiv API).
- `../../scripts/eval_bridges.py`: runs the pass; one paid call per item,
  `--max-cost` guarded, `--dry-run` prices it.
