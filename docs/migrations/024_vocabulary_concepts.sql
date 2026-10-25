-- Migration 024: vocabulary-aware concepts and FSRS mastery.
--
-- MUST BE RUN BY HAND against an existing database before a vocabulary-aware
-- summarizer or the concepts backfill can spend money:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/024_vocabulary_concepts.sql
--
-- Safe to re-run. Postgres functions are the transaction boundary: a failed
-- extraction never calls replace_paper_concepts, and a failed replacement
-- rolls back without removing the paper's previous LLM-owned concepts.

CREATE OR REPLACE FUNCTION replace_paper_concepts(
    p_paper_id TEXT,
    p_concepts JSONB,
    p_source TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    link_count INTEGER;
    edge_count INTEGER;
BEGIN
    IF btrim(COALESCE(p_paper_id, '')) = '' THEN
        RAISE EXCEPTION 'replace_paper_concepts paper_id must not be empty';
    END IF;
    IF p_concepts IS NULL OR jsonb_typeof(p_concepts) <> 'array' THEN
        RAISE EXCEPTION 'replace_paper_concepts concepts must be a JSON array';
    END IF;
    IF jsonb_array_length(p_concepts) > 10 THEN
        RAISE EXCEPTION 'replace_paper_concepts accepts at most 10 concepts';
    END IF;
    IF btrim(COALESCE(p_source, '')) NOT LIKE 'llm:%' THEN
        RAISE EXCEPTION 'replace_paper_concepts source must start with llm:';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM papers WHERE id = btrim(p_paper_id)) THEN
        RAISE EXCEPTION 'replace_paper_concepts paper % does not exist', p_paper_id;
    END IF;
    IF EXISTS (
        SELECT 1
        FROM jsonb_array_elements(p_concepts) AS elem
        WHERE btrim(COALESCE(elem->>'name', '')) = ''
    ) THEN
        RAISE EXCEPTION 'replace_paper_concepts concept names must not be empty';
    END IF;

    INSERT INTO concepts (name, aliases)
    SELECT DISTINCT btrim(elem->>'name'), '[]'::jsonb
    FROM jsonb_array_elements(p_concepts) AS elem
    ON CONFLICT (name) DO UPDATE
        SET updated_at = timezone('utc', now());

    DELETE FROM graph_edges
    WHERE src_kind = 'paper'
      AND src_id = btrim(p_paper_id)
      AND dst_kind = 'concept'
      AND kind = 'covers'
      AND provenance LIKE 'llm:%';

    DELETE FROM paper_concepts
    WHERE paper_id = btrim(p_paper_id)
      AND source LIKE 'llm:%';

    INSERT INTO paper_concepts (paper_id, concept_id, source, weight)
    SELECT btrim(p_paper_id), c.id, btrim(p_source),
           COALESCE((elem->>'weight')::real, 1.0)
    FROM jsonb_array_elements(p_concepts) AS elem
    JOIN concepts c ON c.name = btrim(elem->>'name')
    ON CONFLICT (paper_id, concept_id) DO UPDATE
        SET source = EXCLUDED.source,
            weight = EXCLUDED.weight;

    INSERT INTO graph_edges (
        src_kind, src_id, dst_kind, dst_id, kind, weight, provenance
    )
    SELECT 'paper', btrim(p_paper_id), 'concept', c.name, 'covers',
           COALESCE((elem->>'weight')::real, 1.0), btrim(p_source)
    FROM jsonb_array_elements(p_concepts) AS elem
    JOIN concepts c ON c.name = btrim(elem->>'name')
    ON CONFLICT (src_kind, src_id, dst_kind, dst_id, kind) DO UPDATE
        SET weight = EXCLUDED.weight,
            provenance = EXCLUDED.provenance;

    DELETE FROM concepts c
    WHERE NOT EXISTS (
        SELECT 1 FROM paper_concepts pc WHERE pc.concept_id = c.id
    );

    SELECT count(*) INTO link_count
    FROM paper_concepts
    WHERE paper_id = btrim(p_paper_id)
      AND source = btrim(p_source);
    SELECT count(*) INTO edge_count
    FROM graph_edges
    WHERE src_kind = 'paper'
      AND src_id = btrim(p_paper_id)
      AND dst_kind = 'concept'
      AND kind = 'covers'
      AND provenance = btrim(p_source);

    RETURN jsonb_build_object(
        'paper_id', btrim(p_paper_id),
        'concepts', link_count,
        'edges', edge_count,
        'source', btrim(p_source)
    );
END;
$$;

-- Keep paid vocabulary choices when the free key_terms graph is rebuilt.
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
            weight = EXCLUDED.weight
        WHERE paper_concepts.source NOT LIKE 'llm:%';

    GET DIAGNOSTICS link_count = ROW_COUNT;

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
            provenance = EXCLUDED.provenance
        WHERE graph_edges.provenance NOT LIKE 'llm:%';

    GET DIAGNOSTICS edge_count = ROW_COUNT;

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

CREATE OR REPLACE FUNCTION concept_mastery(
    p_user_id TEXT DEFAULT 'default_user',
    p_pillar_id TEXT DEFAULT NULL
) RETURNS TABLE (
    concept_id UUID,
    name TEXT,
    retrievability DOUBLE PRECISION,
    card_count BIGINT,
    reviewed_card_count BIGINT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        c.id AS concept_id,
        c.name,
        CASE
            WHEN count(q.id) = 0 THEN NULL
            ELSE avg(
                CASE
                    WHEN q.last_review_date IS NULL
                      OR q.stability IS NULL
                      OR q.stability <= 0 THEN 0.0
                    ELSE power(
                        1.0 + (19.0 / 81.0) *
                        greatest(
                            0.0,
                            floor(
                                extract(epoch FROM (
                                    timezone('utc', now()) - q.last_review_date
                                )) / 86400.0
                            )
                        ) / q.stability,
                        -0.5
                    )
                END
            )::double precision
        END AS retrievability,
        count(q.id) AS card_count,
        count(q.id) FILTER (
            WHERE q.last_review_date IS NOT NULL
              AND q.stability IS NOT NULL
              AND q.stability > 0
        ) AS reviewed_card_count
    FROM concepts c
    JOIN paper_concepts pc ON pc.concept_id = c.id
    JOIN papers p ON p.id = pc.paper_id
    LEFT JOIN quiz_cards q
      ON q.paper_id = pc.paper_id
     AND q.user_id = COALESCE(NULLIF(btrim(p_user_id), ''), 'default_user')
    WHERE p_pillar_id IS NULL OR p.pillar_id = p_pillar_id
    GROUP BY c.id, c.name
    ORDER BY c.name;
$$;

GRANT EXECUTE ON FUNCTION replace_paper_concepts(TEXT, JSONB, TEXT) TO web_anon;
GRANT EXECUTE ON FUNCTION replace_derived_graph(JSONB, JSONB, JSONB) TO web_anon;
GRANT EXECUTE ON FUNCTION concept_mastery(TEXT, TEXT) TO web_anon;

NOTIFY pgrst, 'reload schema';
