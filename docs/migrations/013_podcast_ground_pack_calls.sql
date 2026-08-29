-- ==========================================================================
-- 013: podcast_scripts.ground_pack_calls
--
-- Records which provider and model produced each Ground Pack section.
--
-- Calls 1–4 now route to DeepSeek (deepseek-v4-flash) with Claude fallback;
-- call 5 stays on Claude. Without this column, a quality shift is not
-- attributable to provider versus prompt — the same gap 011 and 012 closed
-- for source material and episode options.
--
-- Shape (nlp_pillars/schemas.py::GroundPackCallRecord), keyed by section:
--   {"facts_outline": {
--       "section": "facts_outline",
--       "provider": "deepseek",
--       "model": "deepseek-v4-flash",
--       "fallback": false,
--       "fallback_reason": null,
--       "input_tokens": 507,
--       "output_tokens": 1097,
--       "finish_reason": "stop"
--    }, ...}
--
-- Rows written before this migration have NULL, which _dict_to_podcast_script
-- reads back as an empty dict.
--
-- Apply by hand; there is no migration-tracking table:
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--       < docs/migrations/013_podcast_ground_pack_calls.sql
--
-- Until it is applied, add_podcast_script() retries the insert without the key
-- and logs this path — the script is still saved, only the provenance is lost.
-- ==========================================================================

ALTER TABLE podcast_scripts
    ADD COLUMN IF NOT EXISTS ground_pack_calls JSONB;

COMMENT ON COLUMN podcast_scripts.ground_pack_calls IS
    'Per-section model provenance for Ground Pack extraction; NULL on pre-013 rows';

NOTIFY pgrst, 'reload schema';
