-- Migration 029: Playground follow-along. Links a podcast episode to a
-- playground, and the alignment pass that maps the episode's script lines to
-- the playground's code cells, run as a background pipeline run.
--
-- The stored shape is documented in docs/PLAYGROUND.md ("Follow-along").
--
-- MUST BE RUN BY HAND against any database that already exists, after 028:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/029_playground_links.sql
--
-- Until it is applied the Listen page's code panel says so and names this
-- file, `python -m nlp_pillars.cli playground link` refuses before any model
-- call, and POST /api/follow-along answers 503.
--
-- schema.sql carries the same objects for a brand-new volume.
--
-- Safe to re-run.

CREATE TABLE IF NOT EXISTS playground_links (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    script_id     UUID NOT NULL REFERENCES podcast_scripts(id) ON DELETE CASCADE,
    -- Any playground, not only the one on the episode's own paper.
    playground_id UUID NOT NULL REFERENCES playgrounds(id) ON DELETE CASCADE,
    -- The alignment: [{line_start, line_end, cell_index, step_id}], ordered,
    -- non-overlapping, over script.splitlines() indexes.
    mapping       JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Model(s) that wrote the mapping, after any fallback.
    model         TEXT,
    -- 'pending' until a pass has stored a mapping, then 'ready'. 'failed'
    -- when no pass has produced one.
    status        TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'ready', 'failed')),
    -- Why the last pass failed, or what it dropped. NULL when it was clean.
    error         TEXT,
    -- sha256 of the script text the mapping indexes. A script edited since
    -- has different line numbers, and the panel says so instead of guessing.
    script_sha256 TEXT,
    created_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    updated_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    UNIQUE (script_id, playground_id)
);

CREATE INDEX IF NOT EXISTS idx_playground_links_playground
    ON playground_links (playground_id);

COMMENT ON TABLE playground_links IS
    'Playground follow-along: which script lines of an episode are about which code cell of a playground. See docs/PLAYGROUND.md.';

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand.
GRANT SELECT, INSERT, UPDATE, DELETE ON playground_links TO web_anon;

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_kind_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_kind_check
    CHECK (kind IN ('run_daily', 'process_selected', 'discover',
                    'podcast_audio', 'upload', 'podcast_script',
                    'podcast_factcheck', 'playground_explain',
                    'playground_align'));

-- The trigger source is 028's 'ui_playground'; no change needed there.

-- Exempt playground_align from the one-active-run-per-pillar guard, same
-- reasoning as 015 / 016 / 023 / 028: it is scoped to one episode and one
-- notebook, not to the pillar's queue.
DROP INDEX IF EXISTS pipeline_runs_one_active_per_pillar;
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_per_pillar
    ON pipeline_runs (pillar_id)
    WHERE status IN ('pending', 'running')
      AND kind NOT IN ('upload', 'podcast_script', 'podcast_factcheck',
                       'playground_explain', 'playground_align');

-- One active alignment pass per (episode, playground). The insert must store
-- result.script_id and result.playground_id or this index cannot see it.
DROP INDEX IF EXISTS pipeline_runs_one_active_align_per_link;
CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_one_active_align_per_link
    ON pipeline_runs ((result->>'script_id'), (result->>'playground_id'))
    WHERE kind = 'playground_align'
      AND status IN ('pending', 'running');

-- PostgREST caches table metadata; without this a new table can answer GET
-- with 200 and POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
