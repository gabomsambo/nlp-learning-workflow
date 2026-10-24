-- Migration 019: which chunks are already projected into which Qdrant.
--
-- Postgres (paper_chunks + chunk_embeddings) is the source of truth.
-- Local Qdrant and Qdrant Cloud are rebuildable copies. This table is the
-- cursor: one row per (target, chunk, model) that has been upserted.
-- A target that is down simply has no row yet, so the next sync retries it.
--
-- There is deliberately NO foreign key to paper_chunks. Re-ingest deletes
-- the chunk rows (and the embeddings, ON DELETE CASCADE). The state row has
-- to survive that delete so the projector can remove the Qdrant point on
-- the next sync. A foreign key would either block the chunk delete or, with
-- ON DELETE CASCADE, forget which point to drop — the stale-point bug.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same objects for a brand-new volume; init scripts only fire
-- when the data directory is empty.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/019_vector_index_state.sql
--
-- Until it is applied, `python -m nlp_pillars.cli vectors sync` and the
-- vector stage name this file. Safe to re-run.

CREATE TABLE IF NOT EXISTS vector_index_state (
    target     TEXT NOT NULL CHECK (target IN ('local', 'cloud')),
    chunk_id   UUID NOT NULL,
    model_id   TEXT NOT NULL CHECK (btrim(model_id) <> ''),
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (target, chunk_id, model_id)
);

CREATE INDEX IF NOT EXISTS vector_index_state_target_model
    ON vector_index_state (target, model_id);

COMMENT ON TABLE vector_index_state IS
    'Chunks projected into one Qdrant target. Not a foreign key: the row outlives the chunk so the point can be deleted.';
COMMENT ON COLUMN vector_index_state.target IS
    'local (always-on Qdrant in compose) or cloud (QDRANT_CLOUD_URL, optional).';
COMMENT ON COLUMN vector_index_state.chunk_id IS
    'paper_chunks.id at the time of projection. uuid5 of this value is the Qdrant point id.';

-- Chunks that have an embedding for this model, joined to the paper's pillar.
-- p_paper_id NULL is the whole library. The projector pages with limit/offset.
CREATE OR REPLACE FUNCTION chunks_with_embeddings(
    p_model_id TEXT,
    p_paper_id TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 200,
    p_offset INTEGER DEFAULT 0
) RETURNS TABLE (
    chunk_id UUID,
    paper_id TEXT,
    pillar_id TEXT,
    chunk_index INTEGER,
    chunk_text TEXT,
    dims INTEGER,
    embedding REAL[]
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_model_id IS NULL OR btrim(p_model_id) = '' THEN
        RAISE EXCEPTION 'chunks_with_embeddings missing model_id';
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        RAISE EXCEPTION 'chunks_with_embeddings limit must be positive';
    END IF;
    IF p_offset IS NULL OR p_offset < 0 THEN
        RAISE EXCEPTION 'chunks_with_embeddings offset must be >= 0';
    END IF;

    RETURN QUERY
    SELECT c.id, c.paper_id, p.pillar_id, c.chunk_index, c.text, e.dims, e.embedding
    FROM paper_chunks c
    JOIN chunk_embeddings e
      ON e.chunk_id = c.id
     AND e.model_id = p_model_id
    JOIN papers p
      ON p.id = c.paper_id
    WHERE p_paper_id IS NULL OR c.paper_id = p_paper_id
    ORDER BY c.paper_id, c.chunk_index, c.id
    LIMIT p_limit OFFSET p_offset;
END;
$$;

-- State rows whose chunk has been deleted. The projector deletes those points
-- and then these rows. Paged the same way as chunks_with_embeddings.
CREATE OR REPLACE FUNCTION projection_stale_chunks(
    p_target TEXT,
    p_model_id TEXT,
    p_limit INTEGER DEFAULT 200,
    p_offset INTEGER DEFAULT 0
) RETURNS TABLE (
    chunk_id UUID
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_target IS NULL OR p_target NOT IN ('local', 'cloud') THEN
        RAISE EXCEPTION 'projection_stale_chunks target must be local or cloud';
    END IF;
    IF p_model_id IS NULL OR btrim(p_model_id) = '' THEN
        RAISE EXCEPTION 'projection_stale_chunks missing model_id';
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        RAISE EXCEPTION 'projection_stale_chunks limit must be positive';
    END IF;
    IF p_offset IS NULL OR p_offset < 0 THEN
        RAISE EXCEPTION 'projection_stale_chunks offset must be >= 0';
    END IF;

    RETURN QUERY
    SELECT s.chunk_id
    FROM vector_index_state s
    WHERE s.target = p_target
      AND s.model_id = p_model_id
      AND NOT EXISTS (
          SELECT 1 FROM paper_chunks c WHERE c.id = s.chunk_id
      )
    ORDER BY s.chunk_id
    LIMIT p_limit OFFSET p_offset;
END;
$$;

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand. Without these, PostgREST answers 401 / 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON vector_index_state TO web_anon;
GRANT EXECUTE ON FUNCTION chunks_with_embeddings(TEXT, TEXT, INTEGER, INTEGER) TO web_anon;
GRANT EXECUTE ON FUNCTION projection_stale_chunks(TEXT, TEXT, INTEGER, INTEGER) TO web_anon;

-- PostgREST caches table metadata. A new table can answer GET with 200 and
-- POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
