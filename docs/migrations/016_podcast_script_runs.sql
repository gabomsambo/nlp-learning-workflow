-- Migration 016: podcast script generation is a background run with real stages.
--
-- MUST BE RUN BY HAND against any database that already exists:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/016_podcast_script_runs.sql
--
-- Until it is applied EVERY script generation fails: the insert in
-- create_pipeline_run carries kind='podcast_script' / trigger_source='ui_podcast_script',
-- which the pre-016 CHECK constraints reject with 23514. The route reports that as a
-- 503 naming this file rather than as a generic 500, but the generation does not start.
--
-- Safe to re-run.

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_kind_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_kind_check
    CHECK (kind IN ('run_daily', 'process_selected', 'discover',
                    'podcast_audio', 'upload', 'podcast_script'));

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_trigger_source_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_trigger_source_check
    CHECK (trigger_source IN ('ui_pipeline', 'ui_select', 'ui_discover',
                              'scheduler', 'ui_podcast_audio', 'ui_upload',
                              'ui_podcast_script'));

-- Exempt podcast_script from the one-active-run-per-pillar guard, same reasoning
-- as upload (migration 015): script generation is scoped to one paper the user
-- handed over, not to a pillar's discovery/queue writers. Refusing it because a
-- discovery or TTS run is in flight would refuse work for a conflict that does
-- not exist. Two script runs on one pillar may also overlap; concurrency is
-- bounded by webui/app.py::_MAX_CONCURRENT_RUNS.
DROP INDEX IF EXISTS pipeline_runs_one_active_per_pillar;
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_per_pillar
    ON pipeline_runs (pillar_id)
    WHERE status IN ('pending', 'running')
      AND kind NOT IN ('upload', 'podcast_script');

NOTIFY pgrst, 'reload schema';
