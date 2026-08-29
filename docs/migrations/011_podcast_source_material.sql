-- ==========================================================================
-- 011: podcast_scripts.source_material
--
-- Records what a podcast script was actually written from.
--
-- Before this, a script built from the full paper and a script built from a
-- title alone were indistinguishable in the database, in the API response and
-- on the page. Generation against a paper whose PDF could not be read produced
-- five model calls (~$0.27) and a fluent, confident script whose entire factual
-- basis was the title, one author name and the placeholder string
-- "[Full text not available - using abstract and notes only]".
--
-- podcast_agent now refuses outright when there is no body, no abstract and no
-- notes. This column is for the case that still proceeds: a script written from
-- an abstract and/or an extracted notes row, which is legitimate but thin.
--
-- Shape (nlp_pillars/schemas.py::SourceMaterial):
--   {"level": "full" | "partial",
--    "full_text_chars": int,
--    "has_abstract": bool,
--    "has_notes": bool,
--    "warnings": [str, ...]}
--
-- Rows written before this migration have NULL, which _dict_to_podcast_script
-- reads back as level="full" — what every historical row was assumed to be.
--
-- Apply by hand; there is no migration-tracking table:
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--       < docs/migrations/011_podcast_source_material.sql
--
-- Until it is applied, add_podcast_script() retries the insert without the key
-- and logs this path — the script is still saved, only its provenance is lost.
-- ==========================================================================

ALTER TABLE podcast_scripts
    ADD COLUMN IF NOT EXISTS source_material JSONB;

COMMENT ON COLUMN podcast_scripts.source_material IS
    'What the script was written from; NULL on pre-011 rows, read back as level="full"';

-- PostgREST caches the schema and answers PGRST204 for a column it has not
-- seen. Without this it keeps rejecting the new key until it is restarted.
NOTIFY pgrst, 'reload schema';
