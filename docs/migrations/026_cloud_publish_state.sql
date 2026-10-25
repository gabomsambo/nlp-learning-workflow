-- Migration 026: what the cloud reader already holds.
--
-- `cloud export` writes a bundle of content-hashed documents. The publisher
-- (a later change) uploads only the documents whose hash differs from the row
-- here, then records the new hash. One row per published key: a JSON document
-- under papers/ or episodes/, library.json, schedule.json, or an audio/<sha>.mp3.
--
-- The cloud copy is a projection and Postgres stays the source of truth, the
-- same relationship vector_index_state (019) has with Qdrant: losing this
-- table costs one full re-upload, nothing else. There is no foreign key,
-- because a deleted paper's row must outlive the paper until the publisher
-- has told the cloud to drop the document.
--
-- 025 is reserved for the review-log change (review_logs.device_id / origin),
-- which may land before or after this file; the two do not touch each other.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same objects for a brand-new volume; init scripts only fire
-- when the data directory is empty.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/026_cloud_publish_state.sql
--
-- `cloud export` does not read or write this table, so it works without the
-- migration. Safe to re-run.

CREATE TABLE IF NOT EXISTS cloud_publish_state (
    doc_key      TEXT PRIMARY KEY CHECK (btrim(doc_key) <> ''),
    content_hash TEXT NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    byte_size    BIGINT NOT NULL DEFAULT 0 CHECK (byte_size >= 0),
    published_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

COMMENT ON TABLE cloud_publish_state IS
    'One row per document the cloud reader holds: its content hash and when it was uploaded. The cloud is a rebuildable projection.';
COMMENT ON COLUMN cloud_publish_state.doc_key IS
    'Bundle path, e.g. papers/arxiv_3a1706.03762.json or audio/<sha256>.mp3.';
COMMENT ON COLUMN cloud_publish_state.content_hash IS
    'sha256 (hex) of the bytes uploaded, as listed in the bundle manifest.';
COMMENT ON COLUMN cloud_publish_state.byte_size IS
    'Size of the uploaded object, summed to keep the bucket under its cap.';

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand. Without this, PostgREST answers 401 / 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON cloud_publish_state TO web_anon;

-- PostgREST caches table metadata. A new table can answer GET with 200 and
-- POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
