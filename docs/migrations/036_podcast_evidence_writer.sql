-- Migration 036: the evidence-ledger podcast writer's record.
--
-- PODCAST_WRITER_MODE=evidence (nlp_pillars/evidence/) writes a script from a
-- ledger of quote-verified facts and ends every [HOST] line with the ledger ids
-- it relies on. The stored `script` is the STRIPPED script (what is fact-checked,
-- narrated and aligned); these columns keep what it was written from.
--
--   writer_mode      'free' (Ground Pack writer) or 'evidence'
--   ledger           the verified ledger: entries (F# narrative facts with their
--                    quotes, T# table rows), what failed quote verification,
--                    prompt versions, model and cost, and the writer-time
--                    measurements (untagged lines, entries used, ...)
--   line_citations   {"<0-based line index of script>": ["F12", "T3"]}; an empty
--                    list is a line tagged [[-]] (it states no paper fact)
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same columns for a brand-new volume.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/036_podcast_evidence_writer.sql
--
-- Until it is applied the free writer is unaffected, and an evidence-mode script
-- still saves: add_podcast_script() retries without these keys, logs the
-- migration path, and only the ledger record is lost.
-- Idempotent.

ALTER TABLE podcast_scripts
    ADD COLUMN IF NOT EXISTS writer_mode TEXT NOT NULL DEFAULT 'free',
    ADD COLUMN IF NOT EXISTS ledger JSONB,
    ADD COLUMN IF NOT EXISTS line_citations JSONB;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'podcast_scripts_writer_mode_check'
    ) THEN
        ALTER TABLE podcast_scripts
            ADD CONSTRAINT podcast_scripts_writer_mode_check
            CHECK (writer_mode IN ('free', 'evidence'));
    END IF;
END
$$;

COMMENT ON COLUMN podcast_scripts.writer_mode IS
    'free = Ground Pack writer, evidence = ledger writer (nlp_pillars/evidence).';
COMMENT ON COLUMN podcast_scripts.ledger IS
    'Evidence mode: the verified ledger the script was written from. NULL for free scripts.';
COMMENT ON COLUMN podcast_scripts.line_citations IS
    'Evidence mode: 0-based line index of the stored script -> ledger ids cited; [] = tagged [[-]].';

NOTIFY pgrst, 'reload schema';
