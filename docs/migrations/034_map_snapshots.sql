-- Migration 034: the paper map's snapshots, and the map_build run kind.
--
-- The connections layer (nlp_pillars/map_build.py) is a projection, like
-- graph_edges and the Qdrant index: one JSONB snapshot per pillar computed
-- from works, work_citations (031), paper_vectors (021) and paper_queue. It
-- can be deleted and rebuilt at any time and never holds anything the learner
-- typed.
--
--   map_snapshots   one row per scope (a pillar id). payload is the whole
--                   map: works with positions and scores, citation arcs with
--                   SPC weights, communities, lineage, suggestions, queue
--                   verdicts, insights and evidence. positions repeats the
--                   layout so the next build can warm-start without reading
--                   the payload, and layout_key says which graph it was laid
--                   out for: an identical graph keeps the same positions. payload_hash is sha256 of the canonical
--                   payload JSON; params_hash of the build's constants.
--   put_map_snapshot  the one writer. Upserts the row and reports whether
--                   the payload changed.
--
-- It also adds the 'map_build' pipeline run kind (seven stages, see
-- MAP_BUILD_STAGES in nlp_pillars/schemas.py) and the 'ui_map' trigger
-- source for runs the web app starts. A map build reads the pillar and writes
-- only its snapshot, so it is exempt from the one-active-run-per-pillar guard
-- (same reasoning as 015 / 016 / 023 / 028 / 029); a separate index keeps one
-- active map build per pillar.
--
-- MUST BE RUN BY HAND against any database that already exists, after 031:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/034_map_snapshots.sql
--
-- Until it is applied, `python -m nlp_pillars.cli map build` fails at the
-- write and names this file; `map build --dry-run` works without it.
--
-- schema.sql carries the same objects for a brand-new volume.
--
-- Safe to re-run.

CREATE TABLE IF NOT EXISTS map_snapshots (
    scope        TEXT PRIMARY KEY CHECK (btrim(scope) <> ''),
    built_at     TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    params_hash  TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    model_id     TEXT,
    run_id       UUID,
    stats        JSONB NOT NULL DEFAULT '{}'::jsonb,
    positions    JSONB NOT NULL DEFAULT '{}'::jsonb,
    layout_key   TEXT,
    payload      JSONB NOT NULL,
    CONSTRAINT map_snapshots_payload_object CHECK (jsonb_typeof(payload) = 'object'),
    CONSTRAINT map_snapshots_positions_object CHECK (jsonb_typeof(positions) = 'object')
);

COMMENT ON TABLE map_snapshots IS
    'The paper map per scope (pillar id): a rebuildable projection written by `map build`. See nlp_pillars/map_build.py.';
COMMENT ON COLUMN map_snapshots.positions IS
    'node id -> [x, y] from payload.nodes, so the next build can warm-start its layout cheaply.';
COMMENT ON COLUMN map_snapshots.layout_key IS
    'Hash of the graph the positions were laid out for. An identical graph reuses them unchanged, so an unchanged map keeps its payload_hash.';

-- Returns {scope, built_at, changed}. changed is false when the stored payload
-- already had this hash: the map did not move, only built_at did.
CREATE OR REPLACE FUNCTION put_map_snapshot(
    p_scope        TEXT,
    p_params_hash  TEXT,
    p_payload_hash TEXT,
    p_model_id     TEXT,
    p_stats        JSONB,
    p_positions    JSONB,
    p_layout_key   TEXT,
    p_payload      JSONB,
    p_run_id       UUID
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    previous TEXT;
    stamped  TIMESTAMPTZ := timezone('utc', now());
BEGIN
    SELECT payload_hash INTO previous FROM map_snapshots WHERE scope = p_scope;
    INSERT INTO map_snapshots (
        scope, built_at, params_hash, payload_hash, model_id, run_id,
        stats, positions, layout_key, payload
    )
    VALUES (
        p_scope, stamped, p_params_hash, p_payload_hash, p_model_id, p_run_id,
        COALESCE(p_stats, '{}'::jsonb), COALESCE(p_positions, '{}'::jsonb),
        p_layout_key, p_payload
    )
    ON CONFLICT (scope) DO UPDATE SET
        built_at = EXCLUDED.built_at,
        params_hash = EXCLUDED.params_hash,
        payload_hash = EXCLUDED.payload_hash,
        model_id = EXCLUDED.model_id,
        run_id = EXCLUDED.run_id,
        stats = EXCLUDED.stats,
        positions = EXCLUDED.positions,
        layout_key = EXCLUDED.layout_key,
        payload = EXCLUDED.payload;
    RETURN jsonb_build_object(
        'scope', p_scope,
        'built_at', stamped,
        'changed', previous IS DISTINCT FROM p_payload_hash
    );
END;
$$;

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_kind_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_kind_check
    CHECK (kind IN ('run_daily', 'process_selected', 'discover',
                    'podcast_audio', 'upload', 'podcast_script',
                    'podcast_factcheck', 'playground_explain',
                    'playground_align', 'map_build'));

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_trigger_source_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_trigger_source_check
    CHECK (trigger_source IN ('ui_pipeline', 'ui_select', 'ui_discover',
                              'scheduler', 'ui_podcast_audio', 'ui_upload',
                              'ui_podcast_script', 'ui_podcast_factcheck',
                              'auto_podcast_factcheck', 'ui_playground',
                              'ui_map'));

-- Exempt map_build from the one-active-run-per-pillar guard: an ingest that
-- finishes during a daily run must be able to queue the map's rebuild.
DROP INDEX IF EXISTS pipeline_runs_one_active_per_pillar;
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_per_pillar
    ON pipeline_runs (pillar_id)
    WHERE status IN ('pending', 'running')
      AND kind NOT IN ('upload', 'podcast_script', 'podcast_factcheck',
                       'playground_explain', 'playground_align', 'map_build');

-- One active map build per pillar. A second request while one is pending or
-- running is refused (HTTP 409 / 23505).
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_map_build
    ON pipeline_runs (pillar_id)
    WHERE kind = 'map_build'
      AND status IN ('pending', 'running');

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand. Without this, PostgREST answers 401 / 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON map_snapshots TO web_anon;
GRANT EXECUTE ON FUNCTION put_map_snapshot(TEXT, TEXT, TEXT, TEXT, JSONB, JSONB, TEXT, JSONB, UUID) TO web_anon;

-- PostgREST caches table metadata. A new table can answer GET with 200 and
-- POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
