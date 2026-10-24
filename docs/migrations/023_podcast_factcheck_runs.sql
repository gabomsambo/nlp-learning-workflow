-- Migration 023: fact-checking a podcast script is its own background run.
--
-- MUST BE RUN BY HAND against any database that already exists:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/023_podcast_factcheck_runs.sql
--
-- Until it is applied, POST /api/podcast/{id}/factcheck and the automatic
-- check after a saved script cannot start: create_pipeline_run inserts
-- kind='podcast_factcheck', which the pre-023 CHECK rejects with 23514.
-- The route answers 503 naming this file.
--
-- The result tables are migration 022, not this file. A check that cannot
-- save names docs/migrations/022_podcast_fact_checks.sql.
--
-- Safe to re-run.

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_kind_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_kind_check
    CHECK (kind IN ('run_daily', 'process_selected', 'discover',
                    'podcast_audio', 'upload', 'podcast_script',
                    'podcast_factcheck'));

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_trigger_source_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_trigger_source_check
    CHECK (trigger_source IN ('ui_pipeline', 'ui_select', 'ui_discover',
                              'scheduler', 'ui_podcast_audio', 'ui_upload',
                              'ui_podcast_script', 'ui_podcast_factcheck',
                              'auto_podcast_factcheck'));

-- Exempt podcast_factcheck from the one-active-run-per-pillar guard, same
-- reasoning as upload (015) and podcast_script (016). A fact-check is one
-- script, not a second writer on the pillar queue. Refusing it because a
-- discovery run is in flight would refuse work for a conflict that does
-- not exist. Two fact-checks on one pillar may overlap; one script may not
-- (the index below).
DROP INDEX IF EXISTS pipeline_runs_one_active_per_pillar;
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_per_pillar
    ON pipeline_runs (pillar_id)
    WHERE status IN ('pending', 'running')
      AND kind NOT IN ('upload', 'podcast_script', 'podcast_factcheck');

-- One active fact-check per script. The insert must store
-- result.script_id or this index cannot see the script (NULL does not
-- conflict with NULL). A finished check drops out of the predicate, so a
-- re-check is allowed. 23505 here is "attach to the run already going",
-- not the pillar guard.
DROP INDEX IF EXISTS pipeline_runs_one_active_factcheck_per_script;
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_factcheck_per_script
    ON pipeline_runs ((result->>'script_id'))
    WHERE kind = 'podcast_factcheck'
      AND status IN ('pending', 'running');

NOTIFY pgrst, 'reload schema';
