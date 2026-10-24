-- Migration 021: one title+abstract vector per paper.
--
-- Chunk embeddings (migration 018) stay the search copy. Graph similarity
-- uses this table instead. Mean-pooled chunk vectors on the measured library
-- sat in 0.79–0.97 (std 0.042), so a top-k cut still linked unrelated papers.
-- Title+abstract vectors span 0.43–0.84 (std 0.093). See RELATED_MIN_COSINE
-- in nlp_pillars/graph.py.
--
-- ref_kind is paper (this library) or external_paper (a later citation
-- neighbourhood). No foreign key: the other end is not required to be a row
-- in papers, and deleting a library paper leaves a vector the graph ignores.
--
-- Storage is real[] (PostgreSQL float4), same reason as chunk_embeddings:
-- PostgREST returns a JSON array, and the row is a rebuildable copy rather
-- than a search index. dims is checked against the array length.
--
-- The unique key is (ref_kind, ref_id, model_id, source). A second
-- `embeddings papers` run inserts nothing for a triple that already exists.
-- source defaults to title_abstract, the only source this command writes.
-- model_id includes the pinned ONNX revision so a weight change does not
-- mix with the previous vectors.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same objects for a brand-new volume; init scripts only fire
-- when the data directory is empty.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/021_paper_vectors.sql
--
-- After it is applied, embed library papers that were stored before this:
--
--   python -m nlp_pillars.cli embeddings papers --dry-run
--   python -m nlp_pillars.cli embeddings papers
--
-- The command always uses the pinned local BAAI/bge-base-en-v1.5 model.
-- It does not follow EMBEDDING_PROVIDER and does not call OpenAI.
-- Until this file is applied, that command fails and names it. Graph build
-- still runs; related_to edges are skipped and the report says so.
-- Safe to re-run.

CREATE TABLE IF NOT EXISTS paper_vectors (
    ref_kind   TEXT NOT NULL CHECK (ref_kind IN ('paper', 'external_paper')),
    ref_id     TEXT NOT NULL CHECK (btrim(ref_id) <> ''),
    model_id   TEXT NOT NULL CHECK (btrim(model_id) <> ''),
    source     TEXT NOT NULL DEFAULT 'title_abstract' CHECK (btrim(source) <> ''),
    dims       INTEGER NOT NULL CHECK (dims > 0),
    vec        REAL[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT paper_vectors_dims_match CHECK (cardinality(vec) = dims)
);

CREATE UNIQUE INDEX IF NOT EXISTS paper_vectors_ref_model_source
    ON paper_vectors (ref_kind, ref_id, model_id, source);

COMMENT ON TABLE paper_vectors IS
    'One float32 title+abstract vector per paper per model. Rebuildable. Not the chunk index.';
COMMENT ON COLUMN paper_vectors.vec IS
    'float4[] (PostgreSQL real[]). Chosen over bytea so the array is readable without a codec, and over pgvector so the Postgres image stays unchanged.';
COMMENT ON COLUMN paper_vectors.model_id IS
    'Provider model id including the pinned revision, e.g. BAAI/bge-base-en-v1.5@<onnx sha>.';
COMMENT ON COLUMN paper_vectors.source IS
    'What was embedded. title_abstract is the graph related_to input.';
COMMENT ON COLUMN paper_vectors.ref_kind IS
    'paper for a library row, external_paper for a paper that is not in the library.';

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand. Without this, PostgREST answers 401 / 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON paper_vectors TO web_anon;

-- PostgREST caches table metadata. A new table can answer GET with 200 and
-- POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
