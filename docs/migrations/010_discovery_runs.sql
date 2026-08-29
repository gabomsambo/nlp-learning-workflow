-- Migration 010: let paper discovery be a tracked background run.
--
-- Why: POST /api/pillars/{id}/discover used to answer synchronously, so the page sat
-- on one static "Discovering papers…" line for ~30 seconds with no way to tell work
-- from a hang, and every failure underneath it — the query LLM falling back, a
-- rate-limited source, an unreachable vector store — arrived as a plausible-looking
-- zero. Discovery now goes through the same pipeline_runs machinery as the daily run
-- and the process-selected run, which is what these three changes are for.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql carries the
-- same DDL, but docker-entrypoint-initdb.d only fires on an empty data directory:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/010_discovery_runs.sql
--
-- Safe to re-run: the constraints are dropped before being re-added and the column is
-- IF NOT EXISTS.

-- 1 + 2. Two CHECK constraints stood in the way. Named explicitly rather than left to
-- Postgres's generated names, so a future migration can find them.
ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_kind_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_kind_check
    CHECK (kind IN ('run_daily', 'process_selected', 'discover'));

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_trigger_source_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_trigger_source_check
    CHECK (trigger_source IN ('ui_pipeline', 'ui_select', 'ui_discover', 'scheduler'));

-- 3. Where a run's payload lands. Only discovery writes it today: the candidate list
-- the user has to choose from, which is the whole product of a discovery run and has
-- nowhere else to live now that the request no longer carries it back.
--
-- On the run row rather than in a webui-local dict for the same reason migration 009
-- put runs in Postgres at all: an in-memory result dies with the process, and a run
-- whose progress survives a reload but whose *result* does not is worse than either.
-- It rides along on the poll response the browser already makes, so finishing a run
-- costs no extra round trip and reopening ?run=<id> re-renders the candidates.
--
-- JSONB, not a child table: unlike pipeline_run_stages this is written exactly once,
-- at the end, in a single PATCH — the read-modify-write problem that forced stages
-- into their own table does not arise.
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS result JSONB;

COMMENT ON COLUMN pipeline_runs.result IS
    'Terminal payload for runs that produce one. discover runs store '
    '{"candidates": [...], "sources_used": [...]}; run_daily and process_selected '
    'leave it NULL.';

-- Table-level grants from 009 already cover a new column, so there is nothing to
-- re-GRANT. PostgREST's schema cache, however, does NOT notice a new column: it keeps
-- answering PGRST204 "column does not exist" on writes until it reloads.
NOTIFY pgrst, 'reload schema';
