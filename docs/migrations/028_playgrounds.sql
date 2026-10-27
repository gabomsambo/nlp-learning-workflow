-- Migration 028: Paper Playground. A paper's notebook, its study guide, and the
-- explain pass that writes the guide as a background run.
--
-- The stored shape is documented in docs/PLAYGROUND.md.
--
-- MUST BE RUN BY HAND against any database that already exists:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/028_playgrounds.sql
--
-- Until it is applied the Playground tab says so and names this file: an import
-- fails before anything is downloaded, and POST /api/playgrounds/{id}/explain
-- answers 503 because create_pipeline_run's insert of kind='playground_explain'
-- fails the pre-028 CHECK with 23514.
--
-- schema.sql carries the same objects for a brand-new volume.
--
-- Safe to re-run.

CREATE TABLE IF NOT EXISTS playgrounds (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- One current playground per paper. A re-import replaces the row's
    -- notebook in place (same id) and clears the guide.
    paper_id      TEXT NOT NULL UNIQUE REFERENCES papers(id) ON DELETE CASCADE,
    title         TEXT NOT NULL,
    -- 'notebook' = imported .ipynb. 'generated' is reserved for a later
    -- playground written for a paper that has no code.
    source_kind   TEXT NOT NULL DEFAULT 'notebook'
        CHECK (source_kind IN ('notebook', 'generated')),
    -- The raw .ipynb JSON as uploaded, saved outputs included.
    notebook      JSONB NOT NULL,
    notebook_name TEXT,
    -- The explain pass result: [{id, title, goal, cells: [...]}].
    steps         JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- One record per dataset URL found at import:
    -- {url, path, bytes, sha256} or {url, error}.
    datasets      JSONB NOT NULL DEFAULT '[]'::jsonb,
    status        TEXT NOT NULL DEFAULT 'imported'
        CHECK (status IN ('imported', 'ready', 'partial', 'failed')),
    -- Model(s) that wrote the guide, after any fallback.
    model         TEXT,
    -- Why the last pass was partial or failed. NULL when it was clean.
    error         TEXT,
    created_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    updated_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now())
);

COMMENT ON TABLE playgrounds IS
    'Paper Playground: one notebook per paper, its study guide (steps) and cached dataset records. See docs/PLAYGROUND.md.';

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand.
GRANT SELECT, INSERT, UPDATE, DELETE ON playgrounds TO web_anon;

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_kind_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_kind_check
    CHECK (kind IN ('run_daily', 'process_selected', 'discover',
                    'podcast_audio', 'upload', 'podcast_script',
                    'podcast_factcheck', 'playground_explain'));

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_trigger_source_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_trigger_source_check
    CHECK (trigger_source IN ('ui_pipeline', 'ui_select', 'ui_discover',
                              'scheduler', 'ui_podcast_audio', 'ui_upload',
                              'ui_podcast_script', 'ui_podcast_factcheck',
                              'auto_podcast_factcheck', 'ui_playground'));

-- Exempt playground_explain from the one-active-run-per-pillar guard, same
-- reasoning as upload (015), podcast_script (016) and podcast_factcheck (023):
-- it is scoped to one paper's notebook, not to the pillar's queue.
DROP INDEX IF EXISTS pipeline_runs_one_active_per_pillar;
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_per_pillar
    ON pipeline_runs (pillar_id)
    WHERE status IN ('pending', 'running')
      AND kind NOT IN ('upload', 'podcast_script', 'podcast_factcheck',
                       'playground_explain');

-- One active explain pass per playground. The insert must store
-- result.playground_id or this index cannot see it. 23505 here means "attach
-- to the pass already going", not the pillar guard.
DROP INDEX IF EXISTS pipeline_runs_one_active_explain_per_playground;
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_explain_per_playground
    ON pipeline_runs ((result->>'playground_id'))
    WHERE kind = 'playground_explain'
      AND status IN ('pending', 'running');

-- PostgREST caches table metadata; without this a new table can answer GET
-- with 200 and POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
