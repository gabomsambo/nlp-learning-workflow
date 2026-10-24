-- Migration 022: podcast fact-check results.
--
-- One row in podcast_fact_checks is one finished check of one script against
-- the stored paper text. podcast_claims holds the claims from that check.
-- The script row is not updated. A failed check writes nothing: the insert
-- function below is the only writer, and it runs after every stage succeeds.
--
-- 019 is vector_index_state and 021 is paper_vectors. This migration is 022.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same objects for a brand-new volume.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/022_podcast_fact_checks.sql
--
-- Until it is applied, `python -m nlp_pillars.cli factcheck run` cannot save
-- and names this file. Safe to re-run.

CREATE TABLE IF NOT EXISTS podcast_fact_checks (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    script_id         UUID NOT NULL REFERENCES podcast_scripts(id) ON DELETE CASCADE,
    script_sha256     TEXT NOT NULL CHECK (btrim(script_sha256) <> ''),
    paper_text_sha256 TEXT NOT NULL CHECK (btrim(paper_text_sha256) <> ''),
    status            TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    models            JSONB NOT NULL DEFAULT '{}'::jsonb,
    prompt_version    TEXT NOT NULL CHECK (btrim(prompt_version) <> ''),
    counts            JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_usd          NUMERIC(12, 6) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT podcast_fact_checks_models_object
        CHECK (jsonb_typeof(models) = 'object'),
    CONSTRAINT podcast_fact_checks_counts_object
        CHECK (jsonb_typeof(counts) = 'object')
);

CREATE INDEX IF NOT EXISTS podcast_fact_checks_script_id
    ON podcast_fact_checks (script_id, created_at DESC);

CREATE TABLE IF NOT EXISTS podcast_claims (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fact_check_id    UUID NOT NULL REFERENCES podcast_fact_checks(id) ON DELETE CASCADE,
    line_index       INTEGER NOT NULL CHECK (line_index >= 0),
    span_start       INTEGER NOT NULL CHECK (span_start >= 0),
    span_end         INTEGER NOT NULL CHECK (span_end >= span_start),
    span_text        TEXT NOT NULL,
    claim            TEXT NOT NULL CHECK (btrim(claim) <> ''),
    scope            TEXT NOT NULL CHECK (scope IN ('paper', 'background')),
    category         TEXT NOT NULL CHECK (category IN (
                         'number', 'method', 'setup', 'result', 'comparison',
                         'attribution', 'limitation', 'implication', 'background'
                     )),
    verify_marker    BOOLEAN NOT NULL DEFAULT false,
    verdict          TEXT NOT NULL CHECK (verdict IN (
                         'supported', 'partially_supported', 'contradicted',
                         'unsupported', 'not_checkable'
                     )),
    error_type       TEXT NOT NULL CHECK (error_type IN (
                         'number_mismatch', 'invented_detail', 'overstatement',
                         'wrong_attribution', 'method_misdescribed',
                         'outside_knowledge', 'other', 'none'
                     )),
    severity         TEXT NOT NULL CHECK (severity IN ('high', 'medium', 'low', 'none')),
    rationale        TEXT NOT NULL DEFAULT '',
    source_says      TEXT NOT NULL DEFAULT '',
    evidence         JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_status  TEXT NOT NULL CHECK (evidence_status IN (
                         'located', 'not_located', 'not_required'
                     )),
    judge_trail      JSONB NOT NULL DEFAULT '[]'::jsonb,
    dismissed_at     TIMESTAMPTZ,
    dismissed_note   TEXT,
    CONSTRAINT podcast_claims_evidence_array
        CHECK (jsonb_typeof(evidence) = 'array'),
    CONSTRAINT podcast_claims_judge_trail_array
        CHECK (jsonb_typeof(judge_trail) = 'array')
);

CREATE INDEX IF NOT EXISTS podcast_claims_fact_check_id
    ON podcast_claims (fact_check_id);

COMMENT ON TABLE podcast_fact_checks IS
    'One finished fact-check of a podcast script against paper_texts. Does not change the script.';
COMMENT ON TABLE podcast_claims IS
    'Claims from one fact-check. Evidence quotes are only those located in the paper text.';
COMMENT ON COLUMN podcast_fact_checks.script_sha256 IS
    'Hash of the script text that was checked. A later edit makes this stale.';
COMMENT ON COLUMN podcast_claims.evidence_status IS
    'located, not_located (quote missing; verdict unchanged), or not_required.';

-- One transaction. A claims failure rolls the header back, so a check is
-- never stored half-written. This function does not update podcast_scripts.
CREATE OR REPLACE FUNCTION insert_podcast_fact_check(
    p_check JSONB,
    p_claims JSONB
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    new_id UUID;
    inserted INTEGER;
BEGIN
    IF p_check IS NULL OR jsonb_typeof(p_check) <> 'object' THEN
        RAISE EXCEPTION 'fact check payload must be an object';
    END IF;
    IF p_claims IS NULL OR jsonb_typeof(p_claims) <> 'array' THEN
        RAISE EXCEPTION 'fact check claims must be an array';
    END IF;
    IF p_check->>'status' IS DISTINCT FROM 'succeeded' THEN
        RAISE EXCEPTION 'fact check insert only accepts status succeeded';
    END IF;

    INSERT INTO podcast_fact_checks (
        script_id, script_sha256, paper_text_sha256, status,
        models, prompt_version, counts, cost_usd
    ) VALUES (
        (p_check->>'script_id')::uuid,
        p_check->>'script_sha256',
        p_check->>'paper_text_sha256',
        p_check->>'status',
        COALESCE(p_check->'models', '{}'::jsonb),
        p_check->>'prompt_version',
        COALESCE(p_check->'counts', '{}'::jsonb),
        COALESCE((p_check->>'cost_usd')::numeric, 0)
    )
    RETURNING id INTO new_id;

    INSERT INTO podcast_claims (
        fact_check_id, line_index, span_start, span_end, span_text,
        claim, scope, category, verify_marker,
        verdict, error_type, severity, rationale, source_says,
        evidence, evidence_status, judge_trail
    )
    SELECT
        new_id,
        (elem->>'line_index')::integer,
        (elem->>'span_start')::integer,
        (elem->>'span_end')::integer,
        elem->>'span_text',
        elem->>'claim',
        elem->>'scope',
        elem->>'category',
        COALESCE((elem->>'verify_marker')::boolean, false),
        elem->>'verdict',
        elem->>'error_type',
        elem->>'severity',
        COALESCE(elem->>'rationale', ''),
        COALESCE(elem->>'source_says', ''),
        COALESCE(elem->'evidence', '[]'::jsonb),
        elem->>'evidence_status',
        COALESCE(elem->'judge_trail', '[]'::jsonb)
    FROM jsonb_array_elements(p_claims) AS elem;

    GET DIAGNOSTICS inserted = ROW_COUNT;
    IF inserted <> jsonb_array_length(p_claims) THEN
        RAISE EXCEPTION 'stored % claims, expected %',
            inserted, jsonb_array_length(p_claims);
    END IF;
    RETURN new_id;
END;
$$;

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES: those are per-grantor
-- and do not cover a migration applied later by hand.
GRANT SELECT, INSERT, UPDATE, DELETE ON podcast_fact_checks TO web_anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON podcast_claims TO web_anon;
GRANT EXECUTE ON FUNCTION insert_podcast_fact_check(JSONB, JSONB) TO web_anon;

NOTIFY pgrst, 'reload schema';
