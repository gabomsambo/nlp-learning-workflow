-- Migration 035: the paper map's knowledge layer (M8).
--
-- Three additions, each small, none a second store:
--
--   lessons.connection_links   the lesson's connections as typed items
--                              ({work, title, relation, why}). The prose list
--                              in lessons.connections is still written, from
--                              these, so every existing reader keeps working.
--                              The map resolves the typed items against the
--                              paper's reference list on every build (a
--                              projection: nothing resolved is stored).
--
--   learner_connections        what the learner decided about a suggested
--                              connection between two works: confirmed (with
--                              an optional edited relation and phrase) or
--                              rejected. provenance is always 'user'. A map
--                              rebuild reads this table and never writes it,
--                              so a decision survives every rebuild. One row
--                              per unordered pair. put_learner_connection /
--                              delete_learner_connection are the writers
--                              (PostgREST cannot upsert on an expression index).
--
--   concepts.anchor ...        a concept's place in a controlled vocabulary
--                              (nlp_pillars/vocabulary.py): anchor is
--                              'nlp-kg:Attention', anchor_relation says whether
--                              the concept IS that field ('same') or is a kind
--                              of it ('narrower'), parent_id points at the
--                              concept anchored to the nearest ancestor field.
--                              apply_concept_anchoring writes anchors, merges
--                              concepts anchored 'same' to one field, and sets
--                              parents, in one transaction.
--
-- MUST BE RUN BY HAND against any database that already exists, after 020
-- and 024:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/035_knowledge_layer.sql
--
-- Until it is applied: `concepts anchor` and the learner-decision routes name
-- this file; new lessons still save (add_lesson drops connection_links and
-- logs, as add_podcast_script does for 011); `map build` runs without the
-- learner's decisions and says so in its warnings.
--
-- schema.sql carries the same objects for a brand-new volume.
--
-- Safe to re-run.

-- ----------------------------------------------------------------- lessons

ALTER TABLE lessons ADD COLUMN IF NOT EXISTS connection_links JSONB;

-- ---------------------------------------------------------------- concepts

ALTER TABLE concepts ADD COLUMN IF NOT EXISTS anchor TEXT;
ALTER TABLE concepts ADD COLUMN IF NOT EXISTS anchor_relation TEXT;
ALTER TABLE concepts ADD COLUMN IF NOT EXISTS anchored_at TIMESTAMPTZ;
ALTER TABLE concepts ADD COLUMN IF NOT EXISTS parent_id UUID
    REFERENCES concepts(id) ON DELETE SET NULL;

ALTER TABLE concepts DROP CONSTRAINT IF EXISTS concepts_anchor_relation_check;
ALTER TABLE concepts ADD CONSTRAINT concepts_anchor_relation_check
    CHECK (
        (anchor IS NULL AND anchor_relation IS NULL)
        OR (anchor IS NOT NULL AND anchor_relation IN ('same', 'narrower'))
    );

CREATE INDEX IF NOT EXISTS concepts_anchor ON concepts (anchor)
    WHERE anchor IS NOT NULL;
CREATE INDEX IF NOT EXISTS concepts_parent_id ON concepts (parent_id)
    WHERE parent_id IS NOT NULL;

COMMENT ON COLUMN concepts.anchor IS
    'vocab:id of the controlled-vocabulary field this concept is, or is a kind of (nlp_pillars/vocabulary.py).';
COMMENT ON COLUMN concepts.anchor_relation IS
    'same: the concept is the field (concepts with the same anchor are merged). narrower: it is a kind of the field.';
COMMENT ON COLUMN concepts.parent_id IS
    'The concept anchored to the nearest ancestor of this concept''s field, when the library has one.';

-- One transaction. p_anchors: [{name, anchor, relation}]. p_merges:
-- [{keep, drop: [name, ...]}]. p_parents: [{name, parent}] (concept names),
-- the complete parent mapping for anchored concepts.
--
-- A merge moves the dropped concept's paper links and graph edges to the kept
-- concept (graph_edges name concepts by name, not id), adds the dropped name
-- and its aliases to the kept concept's aliases, re-parents its children, and
-- deletes it. A name missing from concepts raises, so a plan made against a
-- concept list that has since changed rolls back instead of half applying.
CREATE OR REPLACE FUNCTION apply_concept_anchoring(
    p_anchors JSONB,
    p_merges  JSONB,
    p_parents JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    anchored INTEGER := 0;
    merged   INTEGER := 0;
    parented INTEGER := 0;
    got      INTEGER;
    merge_elem JSONB;
    drop_name  TEXT;
    keep_id    UUID;
    drop_id    UUID;
    keep_name  TEXT;
BEGIN
    IF p_anchors IS NULL OR jsonb_typeof(p_anchors) <> 'array'
       OR p_merges IS NULL OR jsonb_typeof(p_merges) <> 'array'
       OR p_parents IS NULL OR jsonb_typeof(p_parents) <> 'array' THEN
        RAISE EXCEPTION 'apply_concept_anchoring arguments must be JSON arrays';
    END IF;

    UPDATE concepts c
    SET anchor = btrim(elem->>'anchor'),
        anchor_relation = btrim(elem->>'relation'),
        anchored_at = timezone('utc', now()),
        updated_at = timezone('utc', now())
    FROM jsonb_array_elements(p_anchors) AS elem
    WHERE c.name = btrim(elem->>'name');
    GET DIAGNOSTICS anchored = ROW_COUNT;
    IF anchored <> jsonb_array_length(p_anchors) THEN
        RAISE EXCEPTION 'anchored % concepts, expected % (a concept name did not match)',
            anchored, jsonb_array_length(p_anchors);
    END IF;

    FOR merge_elem IN SELECT * FROM jsonb_array_elements(p_merges) LOOP
        keep_name := btrim(merge_elem->>'keep');
        SELECT id INTO keep_id FROM concepts WHERE name = keep_name;
        IF keep_id IS NULL THEN
            RAISE EXCEPTION 'merge target concept % does not exist', keep_name;
        END IF;
        FOR drop_name IN
            SELECT btrim(value) FROM jsonb_array_elements_text(merge_elem->'drop')
        LOOP
            SELECT id INTO drop_id FROM concepts WHERE name = drop_name;
            IF drop_id IS NULL THEN
                RAISE EXCEPTION 'merged concept % does not exist', drop_name;
            END IF;
            IF drop_id = keep_id THEN
                CONTINUE;
            END IF;

            INSERT INTO paper_concepts (paper_id, concept_id, source, weight)
            SELECT paper_id, keep_id, source, weight
            FROM paper_concepts WHERE concept_id = drop_id
            ON CONFLICT (paper_id, concept_id) DO UPDATE
                SET weight = GREATEST(paper_concepts.weight, EXCLUDED.weight),
                    source = CASE WHEN EXCLUDED.source LIKE 'llm:%'
                                  THEN EXCLUDED.source
                                  ELSE paper_concepts.source END;
            DELETE FROM paper_concepts WHERE concept_id = drop_id;

            INSERT INTO graph_edges (
                src_kind, src_id, dst_kind, dst_id, kind, weight, provenance
            )
            SELECT src_kind,
                   CASE WHEN src_kind = 'concept' AND src_id = drop_name
                        THEN keep_name ELSE src_id END,
                   dst_kind,
                   CASE WHEN dst_kind = 'concept' AND dst_id = drop_name
                        THEN keep_name ELSE dst_id END,
                   kind, weight, provenance
            FROM graph_edges
            WHERE (src_kind = 'concept' AND src_id = drop_name)
               OR (dst_kind = 'concept' AND dst_id = drop_name)
            ON CONFLICT (src_kind, src_id, dst_kind, dst_id, kind) DO NOTHING;
            DELETE FROM graph_edges
            WHERE (src_kind = 'concept' AND src_id = drop_name)
               OR (dst_kind = 'concept' AND dst_id = drop_name);
            DELETE FROM graph_edges
            WHERE src_kind = 'concept' AND dst_kind = 'concept'
              AND src_id = dst_id;

            UPDATE concepts SET parent_id = keep_id WHERE parent_id = drop_id;

            UPDATE concepts k
            SET aliases = (
                    SELECT COALESCE(jsonb_agg(DISTINCT alias), '[]'::jsonb)
                    FROM (
                        SELECT jsonb_array_elements_text(k.aliases) AS alias
                        UNION
                        SELECT jsonb_array_elements_text(d.aliases)
                        UNION
                        SELECT d.name
                    ) AS names
                    WHERE alias <> k.name
                ),
                updated_at = timezone('utc', now())
            FROM concepts d
            WHERE k.id = keep_id AND d.id = drop_id;

            DELETE FROM concepts WHERE id = drop_id;
            merged := merged + 1;
        END LOOP;
    END LOOP;

    UPDATE concepts SET parent_id = NULL WHERE anchor IS NOT NULL;
    UPDATE concepts c
    SET parent_id = p.id
    FROM jsonb_array_elements(p_parents) AS elem
    JOIN concepts p ON p.name = btrim(elem->>'parent')
    WHERE c.name = btrim(elem->>'name') AND p.id <> c.id;
    GET DIAGNOSTICS parented = ROW_COUNT;

    SELECT count(*) INTO got FROM concepts WHERE anchor IS NOT NULL;
    RETURN jsonb_build_object(
        'anchored', anchored,
        'merged', merged,
        'parents', parented,
        'anchored_total', got
    );
END;
$$;

-- --------------------------------------------------------- learner decisions

CREATE TABLE IF NOT EXISTS learner_connections (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    src        TEXT NOT NULL CHECK (btrim(src) <> ''),
    dst        TEXT NOT NULL CHECK (btrim(dst) <> ''),
    relation   TEXT NOT NULL DEFAULT 'related'
        CHECK (relation IN ('builds_on', 'uses', 'critiques', 'compares',
                            'alternative', 'background', 'related')),
    status     TEXT NOT NULL CHECK (status IN ('confirmed', 'rejected')),
    provenance TEXT NOT NULL DEFAULT 'user' CHECK (provenance = 'user'),
    phrase     TEXT CHECK (phrase IS NULL OR char_length(phrase) <= 400),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT learner_connections_distinct CHECK (src <> dst)
);

-- src and dst are map node ids (a works.id uuid, or 'paper:<id>' for a
-- library paper Semantic Scholar does not know). No foreign key, as with
-- graph_edges and map_snapshots: a decision about a work must outlive a
-- re-sync that renumbers nothing but could briefly miss it.
CREATE UNIQUE INDEX IF NOT EXISTS learner_connections_pair
    ON learner_connections (LEAST(src, dst), GREATEST(src, dst));

COMMENT ON TABLE learner_connections IS
    'The learner''s confirm / reject / edit of a suggested connection (provenance user). Read by map build, never written by it.';

-- Insert or update the decision for the unordered pair. Returns the row.
-- Re-deciding keeps the row id and created_at.
CREATE OR REPLACE FUNCTION put_learner_connection(
    p_src      TEXT,
    p_dst      TEXT,
    p_relation TEXT,
    p_status   TEXT,
    p_phrase   TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    found_id UUID;
    result   JSONB;
BEGIN
    SELECT id INTO found_id FROM learner_connections
    WHERE LEAST(src, dst) = LEAST(btrim(p_src), btrim(p_dst))
      AND GREATEST(src, dst) = GREATEST(btrim(p_src), btrim(p_dst));
    IF found_id IS NULL THEN
        INSERT INTO learner_connections (src, dst, relation, status, phrase)
        VALUES (btrim(p_src), btrim(p_dst), COALESCE(NULLIF(btrim(p_relation), ''), 'related'),
                p_status, NULLIF(btrim(COALESCE(p_phrase, '')), ''))
        RETURNING id INTO found_id;
    ELSE
        UPDATE learner_connections
        SET src = btrim(p_src),
            dst = btrim(p_dst),
            relation = COALESCE(NULLIF(btrim(p_relation), ''), 'related'),
            status = p_status,
            phrase = NULLIF(btrim(COALESCE(p_phrase, '')), ''),
            updated_at = timezone('utc', now())
        WHERE id = found_id;
    END IF;
    SELECT to_jsonb(l) INTO result FROM learner_connections l WHERE id = found_id;
    RETURN result;
END;
$$;

-- Forget the decision, returning the pair to "suggested". Returns the number
-- of rows removed.
CREATE OR REPLACE FUNCTION delete_learner_connection(
    p_a TEXT,
    p_b TEXT
) RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    removed INTEGER;
BEGIN
    DELETE FROM learner_connections
    WHERE LEAST(src, dst) = LEAST(btrim(p_a), btrim(p_b))
      AND GREATEST(src, dst) = GREATEST(btrim(p_a), btrim(p_b));
    GET DIAGNOSTICS removed = ROW_COUNT;
    RETURN removed;
END;
$$;

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand. Without these, PostgREST answers 401 / 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON learner_connections TO web_anon;
GRANT EXECUTE ON FUNCTION apply_concept_anchoring(JSONB, JSONB, JSONB) TO web_anon;
GRANT EXECUTE ON FUNCTION put_learner_connection(TEXT, TEXT, TEXT, TEXT, TEXT) TO web_anon;
GRANT EXECUTE ON FUNCTION delete_learner_connection(TEXT, TEXT) TO web_anon;

-- PostgREST caches table metadata. A new table can answer GET with 200 and
-- POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
