-- Migration 038: allow podcast scripts with no paper, for imported practice drills.
--
-- POST /api/podcast/scripts/import and `nlp_pillars.cli podcast import` store
-- custom practice text as a normal podcast_scripts row with
-- source_material.kind = 'drill'. Everything downstream (Listen, /tts/generate,
-- read-along alignment, cloud publish) already reads these rows; the only thing
-- missing is that paper_id was NOT NULL.
--
-- MUST BE RUN BY HAND against any database that already exists.
-- schema.sql carries the nullable column for a brand-new volume.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/038_podcast_drill_scripts.sql
--
-- Idempotent.

ALTER TABLE podcast_scripts
    ALTER COLUMN paper_id DROP NOT NULL;

COMMENT ON COLUMN podcast_scripts.paper_id IS
    'Source paper id; NULL for an imported practice drill (source_material.kind = drill).';

NOTIFY pgrst, 'reload schema';
