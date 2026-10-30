-- Migration 031: work identity and typed citations.
--
-- One row in `works` per scholarly work the app knows about, whether or not it
-- is in the library, and one row in `work_ids` per id that work goes by (S2
-- paperId, arXiv, DOI, ACL, CorpusId). The library stores `arxiv:1706.03762`
-- while Semantic Scholar names the same paper `1706.03762`; comparing those
-- strings is why `graph_edges` held no citation between two library papers
-- while Semantic Scholar knew four. Every id is normalised in Python
-- (nlp_pillars/work_ids.py) before it is written, so the key is
-- (kind, value) with arXiv versionless and DOIs lowercase.
--
--   works          the work: metadata, counts, when S2 last described it.
--                  title is NULL for a stub known only by an id, and
--                  fetched_at is NULL until S2 describes it.
--   work_ids       (kind, value) -> work. A key belongs to exactly one work.
--   papers.work_id the library paper's work. NULL for the captain's own
--                  documents (file: uploads) that S2 cannot know.
--   work_citations one row per citation, both ends works: is_influential,
--                  intents, up to three citing sentences, source.
--
-- `paper_citations` becomes a READ-ONLY VIEW over work_citations so callers
-- that read it keep working: the old table is renamed `paper_citations_legacy`
-- (data kept, nothing reads it), not dropped. The view has one 'outgoing' row
-- per library paper's reference and one 'incoming' row per library paper's
-- citer, in the old column shape; a citation between two library papers shows
-- in both directions with the library id (papers.id) on both ends, which is
-- how graph build now sees library->library edges. `other_citation_count` is
-- extra: graph build ranks the external endpoints it keeps by it.
-- Writers must go through nlp_pillars.works_store: `db.add_paper_citations`
-- already does, and falls back to the legacy table until this file is applied.
--
-- `apply_work_batch` is the one writer for the three tables: Python resolves
-- identity and hands over finished rows, the function writes them in a single
-- transaction. A write never blanks a stored value (COALESCE) and never
-- overwrites an edge's evidence with a thinner copy of it.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same objects for a brand-new volume; init scripts only fire
-- when the data directory is empty. No data moves: the legacy rows are
-- refetched, complete and typed, by
--
--   python -m nlp_pillars.cli works sync --dry-run
--   python -m nlp_pillars.cli works sync
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/031_works_identity.sql
--
-- Until it is applied, `works sync` fails and names this file, and citation
-- writes keep going to the old table. Safe to re-run.

CREATE TABLE IF NOT EXISTS works (
    id                         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    s2_paper_id                TEXT,
    corpus_id                  BIGINT,
    title                      TEXT,
    year                       INTEGER,
    publication_date           DATE,
    venue                      TEXT,
    authors                    JSONB NOT NULL DEFAULT '[]'::jsonb,
    author_count               INTEGER,
    abstract                   TEXT,
    tldr                       TEXT,
    citation_count             INTEGER,
    influential_citation_count INTEGER,
    reference_count            INTEGER,
    open_pdf_url               TEXT,
    fetched_at                 TIMESTAMPTZ,
    refreshed_at               TIMESTAMPTZ,
    references_synced_at       TIMESTAMPTZ,
    citers_synced_at           TIMESTAMPTZ,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT works_authors_array CHECK (jsonb_typeof(authors) = 'array')
);

CREATE UNIQUE INDEX IF NOT EXISTS works_s2_paper_id_key
    ON works (s2_paper_id) WHERE s2_paper_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS work_ids (
    kind    TEXT NOT NULL
        CHECK (kind IN ('s2', 'arxiv', 'doi', 'acl', 'corpus', 'title_hash')),
    value   TEXT NOT NULL CHECK (btrim(value) <> ''),
    work_id UUID NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    PRIMARY KEY (kind, value)
);

CREATE INDEX IF NOT EXISTS work_ids_work_id ON work_ids (work_id);

ALTER TABLE papers
    ADD COLUMN IF NOT EXISTS work_id UUID REFERENCES works(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS papers_work_id
    ON papers (work_id) WHERE work_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS work_citations (
    citing         UUID NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    cited          UUID NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    is_influential BOOLEAN NOT NULL DEFAULT FALSE,
    intents        TEXT[] NOT NULL DEFAULT '{}',
    contexts       TEXT[] NOT NULL DEFAULT '{}',
    source         TEXT NOT NULL DEFAULT 'semantic_scholar',
    fetched_at     TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    PRIMARY KEY (citing, cited),
    CONSTRAINT work_citations_not_self CHECK (citing <> cited),
    CONSTRAINT work_citations_contexts_cap CHECK (cardinality(contexts) <= 3)
);

CREATE INDEX IF NOT EXISTS work_citations_cited ON work_citations (cited);

-- The bare id the rest of the app has always shown for an external paper:
-- arXiv, else DOI, else S2 paperId (SemanticScholarTool._convert_to_paper_ref).
CREATE OR REPLACE FUNCTION work_display_id(p_work UUID) RETURNS TEXT
LANGUAGE sql STABLE
AS $$
    SELECT value
    FROM work_ids
    WHERE work_id = p_work AND kind IN ('arxiv', 'doi', 's2')
    ORDER BY CASE kind WHEN 'arxiv' THEN 1 WHEN 'doi' THEN 2 ELSE 3 END
    LIMIT 1
$$;

-- Retire the old table, once. A re-run finds the view and does nothing here.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = current_schema() AND c.relname = 'paper_citations'
          AND c.relkind IN ('r', 'p')
    ) THEN
        ALTER TABLE paper_citations RENAME TO paper_citations_legacy;
    END IF;
END
$$;

CREATE OR REPLACE VIEW paper_citations AS
    SELECT md5(p.id || '>' || wc.cited::text)            AS id,
           p.id                                           AS paper_id,
           COALESCE(op.id, work_display_id(wc.cited))     AS cited_paper_id,
           'outgoing'::text                               AS citation_direction,
           wc.is_influential                              AS is_influential,
           wc.contexts[1]                                 AS citation_context,
           wc.source                                      AS source,
           wc.fetched_at                                  AS fetched_at,
           ow.citation_count                              AS other_citation_count
    FROM work_citations wc
    JOIN papers p ON p.work_id = wc.citing
    JOIN works ow ON ow.id = wc.cited
    LEFT JOIN papers op ON op.work_id = wc.cited
    UNION ALL
    SELECT md5(p.id || '<' || wc.citing::text),
           p.id,
           COALESCE(op.id, work_display_id(wc.citing)),
           'incoming'::text,
           wc.is_influential,
           wc.contexts[1],
           wc.source,
           wc.fetched_at,
           ow.citation_count
    FROM work_citations wc
    JOIN papers p ON p.work_id = wc.cited
    JOIN works ow ON ow.id = wc.citing
    LEFT JOIN papers op ON op.work_id = wc.citing;

-- One transaction. Python has already resolved identity, so every uuid here is
-- final; the function only has to keep a write from making a stored row worse.
--   p_works   [{id, s2_paper_id, corpus_id, title, year, publication_date,
--               venue, authors[], author_count, abstract, tldr, citation_count,
--               influential_citation_count, reference_count, open_pdf_url,
--               described, references_synced, citers_synced}]
--             `described` = S2 described this work (sets fetched_at once and
--             refreshed_at every time). A stub row leaves every field NULL.
--   p_ids     [{kind, value, work_id}]
--   p_edges   [{citing, cited, is_influential, intents[], contexts[], source,
--               detail}]
--             detail=true is a typed edge from S2's references/citations
--             endpoint and replaces what is stored; detail=false is an id-only
--             edge and never overwrites one.
--   p_papers  [{id, work_id, citation_count}]  library rows to link
CREATE OR REPLACE FUNCTION apply_work_batch(
    p_works  JSONB,
    p_ids    JSONB,
    p_edges  JSONB,
    p_papers JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    works_n    INTEGER;
    ids_n      INTEGER;
    edges_n    INTEGER := 0;
    thin_n     INTEGER := 0;
    papers_n   INTEGER;
BEGIN
    IF p_works IS NULL OR jsonb_typeof(p_works) <> 'array'
       OR p_ids IS NULL OR jsonb_typeof(p_ids) <> 'array'
       OR p_edges IS NULL OR jsonb_typeof(p_edges) <> 'array'
       OR p_papers IS NULL OR jsonb_typeof(p_papers) <> 'array' THEN
        RAISE EXCEPTION 'apply_work_batch arguments must be JSON arrays';
    END IF;

    INSERT INTO works (
        id, s2_paper_id, corpus_id, title, year, publication_date, venue,
        authors, author_count, abstract, tldr, citation_count,
        influential_citation_count, reference_count, open_pdf_url,
        fetched_at, refreshed_at, references_synced_at, citers_synced_at
    )
    SELECT r.id, r.s2_paper_id, r.corpus_id, r.title, r.year,
           r.publication_date, r.venue,
           COALESCE(r.authors, '[]'::jsonb), r.author_count, r.abstract,
           r.tldr, r.citation_count, r.influential_citation_count,
           r.reference_count, r.open_pdf_url,
           CASE WHEN r.described THEN timezone('utc', now()) END,
           CASE WHEN r.described THEN timezone('utc', now()) END,
           CASE WHEN r.references_synced THEN timezone('utc', now()) END,
           CASE WHEN r.citers_synced THEN timezone('utc', now()) END
    FROM jsonb_to_recordset(p_works) AS r(
        id UUID, s2_paper_id TEXT, corpus_id BIGINT, title TEXT, year INTEGER,
        publication_date DATE, venue TEXT, authors JSONB, author_count INTEGER,
        abstract TEXT, tldr TEXT, citation_count INTEGER,
        influential_citation_count INTEGER, reference_count INTEGER,
        open_pdf_url TEXT, described BOOLEAN, references_synced BOOLEAN,
        citers_synced BOOLEAN
    )
    ON CONFLICT (id) DO UPDATE SET
        s2_paper_id = COALESCE(works.s2_paper_id, EXCLUDED.s2_paper_id),
        corpus_id = COALESCE(EXCLUDED.corpus_id, works.corpus_id),
        title = COALESCE(EXCLUDED.title, works.title),
        year = COALESCE(EXCLUDED.year, works.year),
        publication_date = COALESCE(EXCLUDED.publication_date, works.publication_date),
        venue = COALESCE(EXCLUDED.venue, works.venue),
        authors = CASE WHEN jsonb_array_length(EXCLUDED.authors) > 0
                       THEN EXCLUDED.authors ELSE works.authors END,
        author_count = COALESCE(EXCLUDED.author_count, works.author_count),
        abstract = COALESCE(EXCLUDED.abstract, works.abstract),
        tldr = COALESCE(EXCLUDED.tldr, works.tldr),
        citation_count = COALESCE(EXCLUDED.citation_count, works.citation_count),
        influential_citation_count = COALESCE(
            EXCLUDED.influential_citation_count, works.influential_citation_count),
        reference_count = COALESCE(EXCLUDED.reference_count, works.reference_count),
        open_pdf_url = COALESCE(EXCLUDED.open_pdf_url, works.open_pdf_url),
        fetched_at = COALESCE(works.fetched_at, EXCLUDED.fetched_at),
        refreshed_at = COALESCE(EXCLUDED.refreshed_at, works.refreshed_at),
        references_synced_at = COALESCE(
            EXCLUDED.references_synced_at, works.references_synced_at),
        citers_synced_at = COALESCE(EXCLUDED.citers_synced_at, works.citers_synced_at);
    GET DIAGNOSTICS works_n = ROW_COUNT;

    INSERT INTO work_ids (kind, value, work_id)
    SELECT r.kind, r.value, r.work_id
    FROM jsonb_to_recordset(p_ids) AS r(kind TEXT, value TEXT, work_id UUID)
    ON CONFLICT (kind, value) DO NOTHING;
    GET DIAGNOSTICS ids_n = ROW_COUNT;

    INSERT INTO work_citations (
        citing, cited, is_influential, intents, contexts, source
    )
    SELECT e.citing, e.cited, COALESCE(e.is_influential, FALSE),
           ARRAY(SELECT jsonb_array_elements_text(COALESCE(e.intents, '[]'::jsonb))),
           ARRAY(SELECT jsonb_array_elements_text(COALESCE(e.contexts, '[]'::jsonb))),
           COALESCE(e.source, 'semantic_scholar')
    FROM jsonb_to_recordset(p_edges) AS e(
        citing UUID, cited UUID, is_influential BOOLEAN, intents JSONB,
        contexts JSONB, source TEXT, detail BOOLEAN
    )
    WHERE COALESCE(e.detail, FALSE)
    ON CONFLICT (citing, cited) DO UPDATE SET
        is_influential = EXCLUDED.is_influential,
        intents = EXCLUDED.intents,
        contexts = EXCLUDED.contexts,
        source = EXCLUDED.source,
        fetched_at = timezone('utc', now());
    GET DIAGNOSTICS edges_n = ROW_COUNT;

    INSERT INTO work_citations (
        citing, cited, is_influential, source
    )
    SELECT e.citing, e.cited, COALESCE(e.is_influential, FALSE),
           COALESCE(e.source, 'semantic_scholar')
    FROM jsonb_to_recordset(p_edges) AS e(
        citing UUID, cited UUID, is_influential BOOLEAN, intents JSONB,
        contexts JSONB, source TEXT, detail BOOLEAN
    )
    WHERE NOT COALESCE(e.detail, FALSE)
    ON CONFLICT (citing, cited) DO NOTHING;
    GET DIAGNOSTICS thin_n = ROW_COUNT;

    UPDATE papers p
    SET work_id = COALESCE(r.work_id, p.work_id),
        citation_count = COALESCE(r.citation_count, p.citation_count)
    FROM jsonb_to_recordset(p_papers) AS r(
        id TEXT, work_id UUID, citation_count INTEGER
    )
    WHERE p.id = r.id;
    GET DIAGNOSTICS papers_n = ROW_COUNT;

    RETURN jsonb_build_object(
        'works', works_n,
        'ids_added', ids_n,
        'edges', edges_n + thin_n,
        'papers', papers_n
    );
END;
$$;

COMMENT ON TABLE works IS
    'One scholarly work, in the library or not. title NULL = a stub known only by an id; fetched_at NULL = S2 has not described it.';
COMMENT ON TABLE work_ids IS
    'Every normalised id a work goes by. (kind, value) belongs to exactly one work. Normal forms: nlp_pillars/work_ids.py.';
COMMENT ON TABLE work_citations IS
    'citing -> cited, with S2 intents, up to three citing sentences and the influential flag. The citation truth; paper_citations is a view of it.';
COMMENT ON COLUMN papers.work_id IS
    'The library paper''s work. NULL for documents Semantic Scholar cannot know (file: uploads).';
COMMENT ON VIEW paper_citations IS
    'Read-only compatibility view over work_citations in the pre-031 column shape. Write through nlp_pillars.works_store.';

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand. Without this, PostgREST answers 401 / 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON works, work_ids, work_citations TO web_anon;
GRANT SELECT ON paper_citations TO web_anon;
GRANT EXECUTE ON FUNCTION work_display_id(UUID) TO web_anon;
GRANT EXECUTE ON FUNCTION apply_work_batch(JSONB, JSONB, JSONB, JSONB) TO web_anon;

-- PostgREST caches table metadata. A new table can answer GET with 200 and
-- POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
