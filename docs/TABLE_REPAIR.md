# Table repair at ingestion

`pymupdf4llm` (layout off) turns most tables into loose text rows and drops
some. Measured on 11 library papers, 321 of 3,432 printed table numbers were
missing from the extracted text; the AML paper's results tables (2, 3, 13, 14)
kept a caption and a header row. The podcast writer and the fact-checker read
the same text, so neither could see what the extractor lost.

`nlp_pillars/table_repair.py` repairs this in a non-fatal `tables` stage that
runs right after `ingest` (daily run, selected papers and uploads):

1. **Pages.** Only pages whose PDF text layer has a line that starts like a
   caption (`Table 9:`, `Table 9 |`, `Table 9.`) are read, at most 20 per paper.
2. **Transcribe.** One `claude-haiku-5-5` call per page (route `table_repair`,
   `LLM_TABLE_REPAIR_MODEL` overrides), page rendered at 130 dpi, structured
   output. Checkmark -> `yes`, blank -> empty.
3. **Agree.** Every numeral in a transcribed cell must occur among the numerals
   of that page's PDF text layer. This is the second path. It is deliberately
   not the extracted markdown: that is the lossy side (AML scores 64% against
   it), so measuring against it would block exactly the papers that need the
   fix. `missing_before` records how many cells the markdown lacked.
4. **Splice.** Only when the paper-level agreement is at least 95% **and** the
   table's own is. Below that the tables are *flagged*: stored in
   `paper_tables`, text untouched. The pipe table replaces the garbled rows
   beside the caption (above or below); with nothing there it follows the
   caption. Only lines made entirely of the table's own words and numbers are
   replaced. Mostly-empty `|Col1|` plot tables are stripped. A splice that
   would pass `MAX_FULL_TEXT_CHARS` is refused (`too_large`). Tables with fewer
   than 3 numbers cannot be checked and are not spliced (`unverified`).

A spliced paper's `paper_texts.extractor` gets a `+tables` suffix. Its chunks
are replaced, re-embedded and re-projected exactly as a re-ingest does.

## Settings

`TABLE_REPAIR_ENABLED` (default true), `LLM_TABLE_REPAIR_MODEL`,
`ANTHROPIC_API_KEY`. Without a key, a PDF or any captions the stage is
`skipped` with the reason. A failure is a `failed` stage and the paper
carries on with its original text.

## Migration 031 (hand-applied)

`docs/migrations/031_paper_tables.sql` creates `paper_tables` (one row per
table read: page, caption, columns, rows, model, agreement, status) and
`replace_paper_tables()`. Until it is applied the repaired text is still
stored; the provenance rows are not (logged at WARNING), and
`text repair-tables` names the file. (025 was already taken.)

## Commands

    python -m nlp_pillars.cli text repair-tables --paper <id> --dry-run
    python -m nlp_pillars.cli text repair-tables --paper <id> [--model claude-sonnet-5-5]
    python -m nlp_pillars.cli text repair-tables --all --dry-run      # counts only
    python -m nlp_pillars.cli text repair-tables --all [--limit N] [--max-cost 2.0] [--force]

`--paper --dry-run` reads the pages (a fraction of a cent to a few cents) and
prints per-table agreement and what would be spliced; it writes nothing.
`--all` skips papers that already have `paper_tables` rows unless `--force`,
and stops at `--max-cost`. `text backfill` re-extracts the raw text and undoes
a splice; run `repair-tables --all --force` after it.

Measured cost: DeepSeek-VL 8 pages $0.0055, AMLworld 7 pages $0.0039.
