-- ==========================================================================
-- 012: podcast_scripts.options
--
-- Records what a podcast script was aimed at: field/domain, audience, episode
-- length and tone.
--
-- Before this, those four things were hardcoded in the prompts — an NLP paper
-- read by a Computer Science & Linguistics graduate student, ~30 minutes, in a
-- fixed "TWIML/Neutral/Lex vibe". Now they are chosen per generation, which
-- means two scripts for the same paper can legitimately differ, and without
-- this column there is no way to tell whether the settings or the model made
-- the difference.
--
-- Shape (nlp_pillars/schemas.py::PodcastOptions):
--   {"choices": {
--       "field":    {"key": "field", "preset": "nlp", "custom": null,
--                    "label": "Natural Language Processing"},
--       "audience": {...}, "length": {...}, "tone": {...}}}
--
-- Keyed by option key rather than four named columns on purpose: the registry
-- in nlp_pillars/podcast_options.py is designed so a fifth option is a data
-- change, and a fifth option must not need a sixth migration.
--
-- A preset choice stores "preset"; a free-text choice stores "custom" (already
-- sanitized to one short line). "label" is stored in both cases so an old row
-- still says what it was aimed at even after the preset list moves on.
--
-- Rows written before this migration have NULL, which _dict_to_podcast_script
-- reads back as an empty PodcastOptions — correctly, since those scripts were
-- generated when there was nothing to choose.
--
-- Apply by hand; there is no migration-tracking table:
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--       < docs/migrations/012_podcast_options.sql
--
-- Until it is applied, add_podcast_script() retries the insert without the key
-- and logs this path — the script is still saved, only the record of what it
-- was aimed at is lost. Same degradation as 011's source_material.
-- ==========================================================================

ALTER TABLE podcast_scripts
    ADD COLUMN IF NOT EXISTS options JSONB;

COMMENT ON COLUMN podcast_scripts.options IS
    'What the script was aimed at (field/audience/length/tone); NULL on pre-012 rows';

-- PostgREST caches the schema and answers PGRST204 for a column it has not
-- seen. Without this it keeps rejecting the new key until it is restarted.
NOTIFY pgrst, 'reload schema';
