-- Migration 020: learner knowledge graph.
--
-- Postgres stays the source of truth. These three tables are a projection:
-- `python -m nlp_pillars.cli graph build` deletes the rows it owns and writes
-- them again. A second build does not add a second copy of an edge.
--
--   concepts        one row per canonical concept name
--   paper_concepts  which library paper covers which concept
--   graph_edges     one row per directed edge, unique on its endpoints and kind
--
-- Structural edges (a paper's pillar, authors, lesson, cards, episode) are
-- stored here too, with provenance derived:structure, and removed on rebuild.
-- They are still derived: the columns they came from remain the truth.
-- Citation edges copy paper_citations. Embedding edges are mean-pooled from
-- chunk_embeddings in Python. Provenance derived:% and semantic_scholar are
-- replaced wholesale. provenance user and llm:% are left in place.
--
-- Concept endpoints in graph_edges are the canonical name, not concepts.id.
-- The uuid is only the key paper_concepts uses. A rebuild can allocate new
-- uuids without changing the graph's node ids.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same objects for a brand-new volume; init scripts only fire
-- when the data directory is empty.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/020_knowledge_graph.sql
--
-- Until it is applied, graph build and GET /api/graph fail and name this file.
-- Safe to re-run.

CREATE TABLE IF NOT EXISTS concepts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL CHECK (btrim(name) <> ''),
    aliases     JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT concepts_aliases_array CHECK (jsonb_typeof(aliases) = 'array')
);

CREATE UNIQUE INDEX IF NOT EXISTS concepts_name_key ON concepts (name);

CREATE TABLE IF NOT EXISTS paper_concepts (
    paper_id   TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    source     TEXT NOT NULL CHECK (btrim(source) <> ''),
    weight     REAL,
    PRIMARY KEY (paper_id, concept_id)
);

CREATE INDEX IF NOT EXISTS paper_concepts_concept_id
    ON paper_concepts (concept_id);

-- No foreign keys: an endpoint is a paper, an author name, a concept name,
-- a lesson uuid, or a paper that is not in the library.
CREATE TABLE IF NOT EXISTS graph_edges (
    src_kind   TEXT NOT NULL CHECK (btrim(src_kind) <> ''),
    src_id     TEXT NOT NULL CHECK (btrim(src_id) <> ''),
    dst_kind   TEXT NOT NULL CHECK (btrim(dst_kind) <> ''),
    dst_id     TEXT NOT NULL CHECK (btrim(dst_id) <> ''),
    kind       TEXT NOT NULL CHECK (btrim(kind) <> ''),
    weight     REAL,
    provenance TEXT NOT NULL CHECK (btrim(provenance) <> ''),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

-- The edge identity. A rebuild inserts this key again and changes nothing
-- about how many rows exist. ON CONFLICT in replace_derived_graph uses it.
CREATE UNIQUE INDEX IF NOT EXISTS graph_edges_edge_key
    ON graph_edges (src_kind, src_id, dst_kind, dst_id, kind);

CREATE INDEX IF NOT EXISTS graph_edges_src ON graph_edges (src_kind, src_id);
CREATE INDEX IF NOT EXISTS graph_edges_dst ON graph_edges (dst_kind, dst_id);
CREATE INDEX IF NOT EXISTS graph_edges_provenance ON graph_edges (provenance);

COMMENT ON TABLE concepts IS
    'Canonical concept names built from notes.key_terms. Rebuildable.';
COMMENT ON TABLE paper_concepts IS
    'Paper to concept. source key_terms is replaced by graph build.';
COMMENT ON TABLE graph_edges IS
    'Directed learner-graph edges. Unique per endpoints and kind. Derived rows are rebuilt.';
COMMENT ON COLUMN graph_edges.provenance IS
    'derived:structure, derived:key_terms, derived:embedding:<model>, semantic_scholar, user, or llm:<model>.';

-- One transaction. A failure rolls the delete back, so a crashed build does
-- not leave the graph empty.
CREATE OR REPLACE FUNCTION replace_derived_graph(
    p_concepts JSONB,
    p_links JSONB,
    p_edges JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    concept_count INTEGER;
    link_count INTEGER;
    edge_count INTEGER;
    got INTEGER;
BEGIN
    IF p_concepts IS NULL OR jsonb_typeof(p_concepts) <> 'array' THEN
        RAISE EXCEPTION 'replace_derived_graph concepts must be a JSON array';
    END IF;
    IF p_links IS NULL OR jsonb_typeof(p_links) <> 'array' THEN
        RAISE EXCEPTION 'replace_derived_graph links must be a JSON array';
    END IF;
    IF p_edges IS NULL OR jsonb_typeof(p_edges) <> 'array' THEN
        RAISE EXCEPTION 'replace_derived_graph edges must be a JSON array';
    END IF;

    INSERT INTO concepts (name, aliases)
    SELECT btrim(elem->>'name'),
           CASE
               WHEN jsonb_typeof(elem->'aliases') = 'array' THEN elem->'aliases'
               ELSE '[]'::jsonb
           END
    FROM jsonb_array_elements(p_concepts) AS elem
    WHERE btrim(elem->>'name') <> ''
    ON CONFLICT (name) DO UPDATE
        SET aliases = EXCLUDED.aliases,
            updated_at = timezone('utc', now());

    DELETE FROM paper_concepts WHERE source = 'key_terms';

    INSERT INTO paper_concepts (paper_id, concept_id, source, weight)
    SELECT btrim(elem->>'paper_id'),
           c.id,
           COALESCE(NULLIF(btrim(elem->>'source'), ''), 'key_terms'),
           (elem->>'weight')::real
    FROM jsonb_array_elements(p_links) AS elem
    JOIN concepts c ON c.name = btrim(elem->>'name')
    WHERE btrim(elem->>'paper_id') <> ''
      AND btrim(elem->>'name') <> ''
    ON CONFLICT (paper_id, concept_id) DO UPDATE
        SET source = EXCLUDED.source,
            weight = EXCLUDED.weight;

    GET DIAGNOSTICS link_count = ROW_COUNT;
    IF link_count <> jsonb_array_length(p_links) THEN
        RAISE EXCEPTION
            'stored % paper_concepts, expected % (a name or paper_id did not match)',
            link_count, jsonb_array_length(p_links);
    END IF;

    DELETE FROM concepts c
    WHERE NOT EXISTS (
        SELECT 1 FROM paper_concepts pc WHERE pc.concept_id = c.id
    );

    DELETE FROM graph_edges
    WHERE provenance LIKE 'derived:%'
       OR provenance = 'semantic_scholar';

    INSERT INTO graph_edges (
        src_kind, src_id, dst_kind, dst_id, kind, weight, provenance
    )
    SELECT btrim(elem->>'src_kind'),
           btrim(elem->>'src_id'),
           btrim(elem->>'dst_kind'),
           btrim(elem->>'dst_id'),
           btrim(elem->>'kind'),
           (elem->>'weight')::real,
           btrim(elem->>'provenance')
    FROM jsonb_array_elements(p_edges) AS elem
    ON CONFLICT (src_kind, src_id, dst_kind, dst_id, kind) DO UPDATE
        SET weight = EXCLUDED.weight,
            provenance = EXCLUDED.provenance;

    GET DIAGNOSTICS edge_count = ROW_COUNT;
    IF edge_count <> jsonb_array_length(p_edges) THEN
        RAISE EXCEPTION
            'stored % graph edges, expected %',
            edge_count, jsonb_array_length(p_edges);
    END IF;

    SELECT count(*) INTO concept_count FROM concepts;
    SELECT count(*) INTO got FROM graph_edges;

    RETURN jsonb_build_object(
        'concepts', concept_count,
        'paper_concepts', link_count,
        'edges_written', edge_count,
        'edges', got
    );
END;
$$;

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand. Without these, PostgREST answers 401 / 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON concepts TO web_anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON paper_concepts TO web_anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON graph_edges TO web_anon;
GRANT EXECUTE ON FUNCTION replace_derived_graph(JSONB, JSONB, JSONB) TO web_anon;

-- PostgREST caches table metadata. A new table can answer GET with 200 and
-- POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
