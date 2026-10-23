-- Migration 018: store chunk embeddings in Postgres.
--
-- paper_chunks (migration 017) holds the text. This table holds one vector
-- per chunk per embedding model. Qdrant is still written by the existing
-- upsert and is not a projection of these rows yet. Deleting a paper's
-- chunks (re-ingest replaces them) deletes the vectors with them
-- (ON DELETE CASCADE), and the new chunk ids are embedded again.
--
-- Storage is real[] (PostgreSQL float4, the width ONNX and Qdrant use),
-- not bytea and not pgvector:
--   * real[] round-trips through PostgREST as a JSON array, so the next
--     projector can read the vector without a binary codec.
--   * bytea of the same float32 values is only smaller by the array header
--     and is opaque in the REST API.
--   * pgvector would mean a different Postgres image. These rows are the
--     source copy, not the search index. The index is rebuilt from them.
-- dims is stored beside the array and checked against its length so a
-- 768-d vector cannot be recorded as a 1536-d one.
--
-- The unique key is (chunk_id, model_id). A second backfill inserts nothing
-- for pairs that already exist. model_id includes the pinned ONNX revision
-- (see nlp_pillars/embeddings.py) so a weight change does not mix vectors.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same objects for a brand-new volume; init scripts only fire
-- when the data directory is empty.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/018_chunk_embeddings.sql
--
-- After it is applied, embed chunks that were stored before this existed.
-- Do not run the backfill until this file has been applied:
--
--   python -m nlp_pillars.cli embeddings backfill --dry-run
--   python -m nlp_pillars.cli embeddings backfill
--
-- Until it is applied, embedding writes fail the persist stage and name
-- this file. Safe to re-run.

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id   UUID NOT NULL REFERENCES paper_chunks(id) ON DELETE CASCADE,
    model_id   TEXT NOT NULL CHECK (btrim(model_id) <> ''),
    dims       INTEGER NOT NULL CHECK (dims > 0),
    embedding  REAL[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunk_embeddings_dims_match CHECK (cardinality(embedding) = dims)
);

CREATE UNIQUE INDEX IF NOT EXISTS chunk_embeddings_chunk_model
    ON chunk_embeddings (chunk_id, model_id);

COMMENT ON TABLE chunk_embeddings IS
    'One float32 vector per chunk per model. real[] so PostgREST returns JSON. Not the search index.';
COMMENT ON COLUMN chunk_embeddings.embedding IS
    'float4[] (PostgreSQL real[]). Chosen over bytea so the array is readable without a codec, and over pgvector so the Postgres image stays unchanged.';
COMMENT ON COLUMN chunk_embeddings.model_id IS
    'Provider model id including the pinned revision, e.g. BAAI/bge-base-en-v1.5@<onnx sha>.';

-- Chunks that have no row yet for this model. p_paper_id NULL is every paper.
-- Limit and offset page the result; the client walks offset until a short page.
CREATE OR REPLACE FUNCTION chunks_missing_embeddings(
    p_model_id TEXT,
    p_paper_id TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 200,
    p_offset INTEGER DEFAULT 0
) RETURNS TABLE (
    chunk_id UUID,
    paper_id TEXT,
    chunk_index INTEGER,
    chunk_text TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_model_id IS NULL OR btrim(p_model_id) = '' THEN
        RAISE EXCEPTION 'chunks_missing_embeddings missing model_id';
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        RAISE EXCEPTION 'chunks_missing_embeddings limit must be positive';
    END IF;
    IF p_offset IS NULL OR p_offset < 0 THEN
        RAISE EXCEPTION 'chunks_missing_embeddings offset must be >= 0';
    END IF;

    RETURN QUERY
    SELECT c.id, c.paper_id, c.chunk_index, c.text
    FROM paper_chunks c
    WHERE (p_paper_id IS NULL OR c.paper_id = p_paper_id)
      AND NOT EXISTS (
          SELECT 1
          FROM chunk_embeddings e
          WHERE e.chunk_id = c.id
            AND e.model_id = p_model_id
      )
    ORDER BY c.paper_id, c.chunk_index, c.id
    LIMIT p_limit OFFSET p_offset;
END;
$$;

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand. Without these, PostgREST answers 401 / 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON chunk_embeddings TO web_anon;
GRANT EXECUTE ON FUNCTION chunks_missing_embeddings(TEXT, TEXT, INTEGER, INTEGER) TO web_anon;

-- PostgREST caches table metadata. A new table can answer GET with 200 and
-- POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
