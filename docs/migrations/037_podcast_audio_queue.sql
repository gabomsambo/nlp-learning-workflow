-- Migration 037: durable, per-script podcast audio queue and shared Studio preference.
--
-- Audio synthesis is scoped to one saved script, so it must not occupy the
-- one-active-run-per-pillar slot. A second partial unique index protects one active
-- request per script while allowing multiple episodes in the same pillar to wait.
-- The preference table stores the last confirmed usable voice selected in Studio.
--
-- This migration depends only on the base pipeline_runs / podcast_scripts schema. It
-- is independent of 036_podcast_evidence_writer.sql.
--
-- MUST BE RUN BY HAND against existing databases after merge. Do not apply it as part
-- of application startup. It is idempotent.

CREATE TABLE IF NOT EXISTS podcast_settings (
    id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id = TRUE),
    last_voice_path TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

INSERT INTO podcast_settings (id)
VALUES (TRUE)
ON CONFLICT (id) DO NOTHING;

GRANT SELECT, INSERT, UPDATE, DELETE ON podcast_settings TO web_anon;

DROP INDEX IF EXISTS pipeline_runs_one_active_per_pillar;
CREATE UNIQUE INDEX pipeline_runs_one_active_per_pillar
    ON pipeline_runs (pillar_id)
    WHERE status IN ('pending', 'running')
      AND kind NOT IN ('upload', 'podcast_script', 'podcast_audio',
                       'podcast_factcheck', 'playground_explain',
                       'playground_align', 'map_build');

CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_podcast_audio_per_script
    ON pipeline_runs ((result->>'script_id'))
    WHERE kind = 'podcast_audio'
      AND status IN ('pending', 'running');

NOTIFY pgrst, 'reload schema';
