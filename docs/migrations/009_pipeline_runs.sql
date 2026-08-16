-- Migration 009: track pipeline runs and their stages.
--
-- Why: POST /pipeline/run and POST /api/pillars/{id}/select used to run the whole
-- synchronous Orchestrator inside the request handler, so the browser sat on
-- "Running..." for minutes with no way to tell work from a hang, and the single
-- uvicorn process was frozen for every other request while it happened. Those
-- endpoints now return immediately with a run id and the work happens on a worker
-- thread; these two tables are what the run id points at.
--
-- Postgres rather than an in-memory registry, deliberately: the scheduler runs the
-- same pipeline in a SEPARATE container (docker-compose.yml, `command: python -m
-- nlp_pillars.scheduler`), so anything held in the webui process could never show a
-- nightly run. This also survives a restart and gives run history for free.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql carries
-- the same DDL, but init scripts in docker-entrypoint-initdb.d only fire when the
-- data directory is empty, so editing schema.sql does nothing to a live database:
--
--   docker compose exec -T db psql -U nlp -d nlp -f - < docs/migrations/009_pipeline_runs.sql
--
-- Safe to re-run: every statement is IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pillar_id        TEXT NOT NULL,
    trigger_source   TEXT NOT NULL
        CHECK (trigger_source IN ('ui_pipeline', 'ui_select', 'scheduler')),
    kind             TEXT NOT NULL
        CHECK (kind IN ('run_daily', 'process_selected')),
    -- NOT NULL with a default, and every lookup filters on this rather than on a
    -- timestamp being NULL. TableQuery.eq() in nlp_pillars/db.py renders Python None
    -- as the literal string "None", so `eq('finished_at', None)` becomes
    -- finished_at=eq.None and matches zero rows without erroring. Nothing in this
    -- schema may be filtered while nullable.
    status           TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed',
                          'cancelled', 'interrupted')),
    current_stage    TEXT,
    papers_processed INTEGER NOT NULL DEFAULT 0,
    papers_failed    INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ,
    -- Refreshed on every stage transition. Lets a human tell "slow" from "wedged"
    -- without waiting for a terminal status.
    heartbeat_at     TIMESTAMPTZ
);

-- One row per stage, pre-seeded at 'pending' when the run is created so the UI has a
-- complete skeleton to render before any work starts.
--
-- A child table rather than a JSONB column on pipeline_runs, and that is not a style
-- preference: PostgREST cannot partially update a JSONB column. There is no path
-- syntax on PATCH and no `set.` filter, so a stages array would mean read the row,
-- mutate in Python, write the whole column back — eleven times per run, from a worker
-- thread, while the browser polls the same row. That loses updates. One row per stage
-- makes each transition an independent single-row PATCH.
CREATE TABLE IF NOT EXISTS pipeline_run_stages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    seq         SMALLINT NOT NULL,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    detail      TEXT,
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    UNIQUE (run_id, seq)
);

-- The concurrency guard, enforced by the database because a read-then-write check in
-- Python races. A second insert for a pillar that already has a pending/running row
-- comes back through PostgREST as HTTP 409 with code 23505, which the route turns into
-- "a run is already in progress". Once the first run reaches a terminal status the
-- partial index stops covering it and the next run inserts cleanly.
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_per_pillar
    ON pipeline_runs (pillar_id) WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_pillar
    ON pipeline_runs (pillar_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_active
    ON pipeline_runs (status) WHERE status IN ('pending', 'running');
CREATE INDEX IF NOT EXISTS idx_pipeline_run_stages_run
    ON pipeline_run_stages (run_id, seq);

COMMENT ON TABLE pipeline_runs IS
    'One row per pipeline execution, from any trigger. status=interrupted means the '
    'process died mid-run and the startup sweep found the row, NOT that the pipeline '
    'failed - nothing can write its own epitaph after being killed.';
COMMENT ON TABLE pipeline_run_stages IS
    'Per-stage progress for a run. Seeded pending at run creation; seq matches the '
    'Step N ordering in nlp_pillars/orchestrator.py.';

-- Explicit, not relying on the ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and only cover objects created afterwards by
-- that same role, which is not true of a migration applied later by hand. Measured:
-- without this, PostgREST answers 401 with 42501 "permission denied for table".
GRANT SELECT, INSERT, UPDATE, DELETE ON pipeline_runs TO web_anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON pipeline_run_stages TO web_anon;

-- PostgREST caches table metadata, and the failure is asymmetric: measured against
-- this stack, a brand-new table answers GET with 200 but POST with 404 until the
-- cache reloads. "My SELECT works but my INSERT 404s" means this line was skipped.
NOTIFY pgrst, 'reload schema';
