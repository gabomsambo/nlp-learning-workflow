-- Migration 032: tables read from the page images at ingestion.
--
-- (The report that specified this work called it 025, then 031 collided on main;
-- 032 is the next free number.)
--
-- pymupdf4llm turns most tables into loose text rows and drops some. The
-- `tables` ingestion stage (nlp_pillars/table_repair.py) reads each table page
-- as an image, checks every number against the page's PDF text layer, and
-- splices the agreeing tables into paper_texts.full_text. This table is the
-- record of that work: one row per table read, spliced or not.
--
--   status = spliced     in the stored text
--            flagged     agreement under 95%: stored here, NOT in the text
--            unverified  too few numbers to check
--            unplaced    caption not found in the text, or a repeated label
--            too_large   splicing would pass MAX_FULL_TEXT_CHARS
--
-- paper_texts.extractor gets a "+tables" suffix when any table was spliced.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same objects for a brand-new volume.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/032_paper_tables.sql
--
-- Until it is applied the stage still repairs the text; only the record is
-- lost (logged at WARNING), and `text repair-tables` names this file.
-- Safe to re-run.

CREATE TABLE IF NOT EXISTS paper_tables (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id       TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    table_index    INTEGER NOT NULL CHECK (table_index >= 0),
    label          TEXT NOT NULL,
    page           INTEGER NOT NULL CHECK (page >= 1),
    caption        TEXT NOT NULL DEFAULT '',
    columns        JSONB NOT NULL DEFAULT '[]'::jsonb,
    rows           JSONB NOT NULL DEFAULT '[]'::jsonb,
    model          TEXT NOT NULL,
    agreement      NUMERIC(5, 4) CHECK (agreement IS NULL OR agreement BETWEEN 0 AND 1),
    numerals       INTEGER NOT NULL DEFAULT 0 CHECK (numerals >= 0),
    matched        INTEGER NOT NULL DEFAULT 0 CHECK (matched >= 0),
    missing_before INTEGER NOT NULL DEFAULT 0 CHECK (missing_before >= 0),
    status         TEXT NOT NULL
        CHECK (status IN ('spliced', 'flagged', 'unverified', 'unplaced', 'too_large')),
    reason         TEXT NOT NULL DEFAULT '',
    cost_usd       NUMERIC(12, 6) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT paper_tables_columns_array CHECK (jsonb_typeof(columns) = 'array'),
    CONSTRAINT paper_tables_rows_array CHECK (jsonb_typeof(rows) = 'array')
);

CREATE UNIQUE INDEX IF NOT EXISTS paper_tables_paper_index
    ON paper_tables (paper_id, table_index);

COMMENT ON TABLE paper_tables IS
    'Tables transcribed from page images at ingestion, with page, caption, rows, model and numeric agreement against the PDF text layer.';

-- One transaction: delete the paper's rows, insert the new set. An empty
-- array clears the paper's rows.
CREATE OR REPLACE FUNCTION replace_paper_tables(
    p_paper_id TEXT,
    p_tables JSONB
) RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    inserted INTEGER;
BEGIN
    IF p_paper_id IS NULL OR btrim(p_paper_id) = '' THEN
        RAISE EXCEPTION 'paper tables write missing paper_id';
    END IF;
    IF p_tables IS NULL OR jsonb_typeof(p_tables) <> 'array' THEN
        RAISE EXCEPTION 'paper tables for % must be a JSON array', p_paper_id;
    END IF;

    DELETE FROM paper_tables WHERE paper_id = p_paper_id;

    INSERT INTO paper_tables (
        paper_id, table_index, label, page, caption, columns, rows, model,
        agreement, numerals, matched, missing_before, status, reason, cost_usd
    )
    SELECT
        p_paper_id,
        (elem->>'table_index')::INTEGER,
        elem->>'label',
        (elem->>'page')::INTEGER,
        COALESCE(elem->>'caption', ''),
        COALESCE(elem->'columns', '[]'::jsonb),
        COALESCE(elem->'rows', '[]'::jsonb),
        elem->>'model',
        (elem->>'agreement')::NUMERIC,
        COALESCE((elem->>'numerals')::INTEGER, 0),
        COALESCE((elem->>'matched')::INTEGER, 0),
        COALESCE((elem->>'missing_before')::INTEGER, 0),
        elem->>'status',
        COALESCE(elem->>'reason', ''),
        COALESCE((elem->>'cost_usd')::NUMERIC, 0)
    FROM jsonb_array_elements(p_tables) AS elem;

    GET DIAGNOSTICS inserted = ROW_COUNT;
    RETURN inserted;
END;
$$;

-- Explicit grants: default privileges do not cover a hand-applied migration.
GRANT SELECT, INSERT, UPDATE, DELETE ON paper_tables TO web_anon;
GRANT EXECUTE ON FUNCTION replace_paper_tables(TEXT, JSONB) TO web_anon;

NOTIFY pgrst, 'reload schema';
