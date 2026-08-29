-- Migration 014: podcast audio generation runs and stored audio metadata.
--
-- MUST BE RUN BY HAND against any database that already exists:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/014_podcast_audio.sql
--
-- Safe to re-run.

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_kind_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_kind_check
    CHECK (kind IN ('run_daily', 'process_selected', 'discover', 'podcast_audio'));

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_trigger_source_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_trigger_source_check
    CHECK (trigger_source IN ('ui_pipeline', 'ui_select', 'ui_discover',
                              'scheduler', 'ui_podcast_audio'));

ALTER TABLE podcast_scripts ADD COLUMN IF NOT EXISTS audio_metadata JSONB;

COMMENT ON COLUMN podcast_scripts.audio_metadata IS
    'Generated episode audio: engine, voice, file path/name, duration. '
    'See schemas.py::AudioMetadata.';

NOTIFY pgrst, 'reload schema';
