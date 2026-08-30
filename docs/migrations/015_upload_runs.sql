-- Migration 015: manual paper uploads are background runs.
--
-- MUST BE RUN BY HAND against any database that already exists:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/015_upload_runs.sql
--
-- Until it is applied EVERY upload fails: the insert in create_pipeline_run carries
-- kind='upload' / trigger_source='ui_upload', which the pre-015 CHECK constraints
-- reject with 23514. The route reports that as a 503 naming this file rather than as
-- a generic 500, but the upload does not happen.
--
-- Safe to re-run.

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_kind_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_kind_check
    CHECK (kind IN ('run_daily', 'process_selected', 'discover',
                    'podcast_audio', 'upload'));

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_trigger_source_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_trigger_source_check
    CHECK (trigger_source IN ('ui_pipeline', 'ui_select', 'ui_discover',
                              'scheduler', 'ui_podcast_audio', 'ui_upload'));

-- The one-active-run-per-pillar guard now EXEMPTS uploads, and that exemption is the
-- point of this migration rather than a detail of it.
--
-- The guard exists because discovery, a daily run and a selection all drive the same
-- pillar through the same pipeline, and starting a second while the first is mid-flight
-- means two writers competing over one pillar's queue and papers. An upload is not
-- that: the user hands over one specific paper, and the work is scoped to that paper's
-- own row. Inheriting the guard would have meant the upload button answering 409
-- "a pipeline run is already in progress" for the ~30 seconds a discovery run takes —
-- refusing work the user explicitly asked for, for a conflict that does not exist.
--
-- Written as a predicate on the index rather than as a check in Python deliberately,
-- for the same reason the original guard was: a read-then-write test races, and the
-- database is the only place the rule can actually be enforced. Uploads are therefore
-- unconstrained here — two uploads to one pillar may run at once, which is what a user
-- pasting two URLs in a row expects. Concurrency is bounded by the scheduler's thread
-- pool (webui/app.py::_MAX_CONCURRENT_RUNS), not by this index; a run beyond that
-- waits at 'pending' with its row already visible, which is the honest rendering of
-- "queued".
DROP INDEX IF EXISTS pipeline_runs_one_active_per_pillar;
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_per_pillar
    ON pipeline_runs (pillar_id)
    WHERE status IN ('pending', 'running') AND kind <> 'upload';

-- PostgREST caches table metadata; without this it keeps answering with the old
-- constraint knowledge and a fresh kind value is rejected until the cache reloads.
NOTIFY pgrst, 'reload schema';
