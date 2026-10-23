-- Migration 017: persist paper text and chunks in Postgres.
--
-- Postgres is the source of truth for paper text. Qdrant payloads carry only
-- ids and a length, so a suspended or deleted cluster cannot be rebuilt from
-- the database. These two tables hold what ingest already extracted:
--
--   paper_texts  — one row per paper: the full text and which extractor produced it
--   paper_chunks — the semchunk 250/25 cl100k cut upsert_text() already uses
--
-- No embedding column. Embeddings are a later change and depend on a model
-- decision; paper_chunks.id is a surrogate key so that table can reference
-- these rows without a rewrite.
--
-- Re-ingest replaces a paper's chunks in one transaction (delete, then insert)
-- inside replace_paper_text_and_chunks(). A shorter re-chunk must not leave a
-- stale tail. The unique index is (paper_id, chunk_index, chunker_version) so
-- a future chunker config can be stored beside this one; today's writer still
-- deletes every chunk for the paper before inserting the new set.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same objects for a brand-new volume; init scripts only fire when
-- the data directory is empty.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/017_paper_text.sql
--
-- After it is applied, backfill papers that were ingested before this existed.
-- Do not run the backfill until this file has been applied:
--
--   python -m nlp_pillars.cli text backfill --dry-run
--   python -m nlp_pillars.cli text backfill
--
-- Until it is applied, text writes fail the persist stage and name this file.
-- Safe to re-run.

CREATE TABLE IF NOT EXISTS paper_texts (
    paper_id     TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    full_text    TEXT NOT NULL,
    extractor    TEXT NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- id is unused by this migration's writer and exists so a later embeddings
-- table can reference a chunk. Do not add an embedding column here.
CREATE TABLE IF NOT EXISTS paper_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id        TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL CHECK (chunk_index >= 0),
    text            TEXT NOT NULL,
    token_count     INTEGER NOT NULL CHECK (token_count >= 0),
    chunker         TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    chunk_size      INTEGER NOT NULL CHECK (chunk_size > 0),
    chunk_overlap   INTEGER NOT NULL CHECK (chunk_overlap >= 0),
    encoding        TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS paper_chunks_paper_idx_version
    ON paper_chunks (paper_id, chunk_index, chunker_version);

COMMENT ON TABLE paper_texts IS
    'Full text of a paper. Source of truth; vector indexes are rebuilt from here.';
COMMENT ON TABLE paper_chunks IS
    'Chunk text for one chunker config. No embedding column; id is for a later FK.';
COMMENT ON COLUMN paper_chunks.chunker_version IS
    'Chunker identity including its config, e.g. semchunk==4.1.1|250/25|cl100k_base';

-- One transaction: a failed insert rolls back the delete, so a paper cannot
-- be left with its old tail gone and its new chunks missing.
CREATE OR REPLACE FUNCTION replace_paper_text_and_chunks(
    p_paper_id TEXT,
    p_full_text TEXT,
    p_extractor TEXT,
    p_chunks JSONB
) RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    inserted INTEGER;
BEGIN
    IF p_paper_id IS NULL OR btrim(p_paper_id) = '' THEN
        RAISE EXCEPTION 'paper text write missing paper_id';
    END IF;
    IF p_full_text IS NULL OR btrim(p_full_text) = '' THEN
        RAISE EXCEPTION 'refusing to store empty paper text for %', p_paper_id;
    END IF;
    IF p_extractor IS NULL OR btrim(p_extractor) = '' THEN
        RAISE EXCEPTION 'paper text write for % missing extractor', p_paper_id;
    END IF;
    IF p_chunks IS NULL
       OR jsonb_typeof(p_chunks) <> 'array'
       OR jsonb_array_length(p_chunks) = 0 THEN
        RAISE EXCEPTION 'refusing to replace chunks for % with an empty set', p_paper_id;
    END IF;

    INSERT INTO paper_texts (paper_id, full_text, extractor, extracted_at)
    VALUES (p_paper_id, p_full_text, p_extractor, now())
    ON CONFLICT (paper_id) DO UPDATE
        SET full_text = EXCLUDED.full_text,
            extractor = EXCLUDED.extractor,
            extracted_at = now();

    DELETE FROM paper_chunks WHERE paper_id = p_paper_id;

    INSERT INTO paper_chunks (
        paper_id, chunk_index, text, token_count,
        chunker, chunker_version, chunk_size, chunk_overlap, encoding,
        content_hash
    )
    SELECT
        p_paper_id,
        (elem->>'chunk_index')::INTEGER,
        elem->>'text',
        (elem->>'token_count')::INTEGER,
        elem->>'chunker',
        elem->>'chunker_version',
        (elem->>'chunk_size')::INTEGER,
        (elem->>'chunk_overlap')::INTEGER,
        elem->>'encoding',
        elem->>'content_hash'
    FROM jsonb_array_elements(p_chunks) AS elem;

    GET DIAGNOSTICS inserted = ROW_COUNT;
    IF inserted <> jsonb_array_length(p_chunks) THEN
        RAISE EXCEPTION 'stored % chunks for %, expected %',
            inserted, p_paper_id, jsonb_array_length(p_chunks);
    END IF;
    RETURN inserted;
END;
$$;

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand. Without these, PostgREST answers 401 / 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON paper_texts TO web_anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON paper_chunks TO web_anon;
GRANT EXECUTE ON FUNCTION replace_paper_text_and_chunks(TEXT, TEXT, TEXT, JSONB) TO web_anon;

-- PostgREST caches table metadata. A new table can answer GET with 200 and
-- POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
