-- Migration 031: dismiss a fact-check flag with a note, and keep the history.
--
-- podcast_claims already has dismissed_at / dismissed_note (022). This adds
-- the function that sets them and an append-only log, so "who dismissed what,
-- when, and why" survives a later restore. The note is the point: a dismissal
-- is a human saying the verifier was wrong about one claim, which is exactly
-- a calibration label (see docs/FACTCHECK_EVAL.md).
--
-- set_podcast_claim_dismissal() changes the claim and writes the log row in
-- one transaction. Dismissing needs a non-empty note (at most 1000 chars);
-- restoring clears the columns but keeps every log row. It never touches
-- verdict, severity or the script.
--
-- MUST BE RUN BY HAND against any database that already exists, after 022:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/031_claim_dismissals.sql
--
-- Until it is applied, dismissing answers 503 naming this file. schema.sql
-- carries the same objects for a brand-new volume. Safe to re-run.

CREATE TABLE IF NOT EXISTS podcast_claim_reviews (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id   UUID NOT NULL REFERENCES podcast_claims(id) ON DELETE CASCADE,
    action     TEXT NOT NULL CHECK (action IN ('dismissed', 'restored')),
    note       TEXT NOT NULL DEFAULT '' CHECK (char_length(note) <= 1000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT podcast_claim_reviews_dismiss_has_note
        CHECK (action <> 'dismissed' OR btrim(note) <> '')
);

CREATE INDEX IF NOT EXISTS podcast_claim_reviews_claim_id
    ON podcast_claim_reviews (claim_id, created_at);

COMMENT ON TABLE podcast_claim_reviews IS
    'Append-only log of dismissals and restores of fact-check claims. Rows are never updated.';

CREATE OR REPLACE FUNCTION set_podcast_claim_dismissal(
    p_claim_id UUID,
    p_dismissed BOOLEAN,
    p_note TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    clean TEXT := btrim(COALESCE(p_note, ''));
    updated podcast_claims%ROWTYPE;
BEGIN
    IF p_dismissed AND clean = '' THEN
        RAISE EXCEPTION 'a dismissal needs a note';
    END IF;
    IF char_length(clean) > 1000 THEN
        RAISE EXCEPTION 'a dismissal note is at most 1000 characters';
    END IF;

    UPDATE podcast_claims
       SET dismissed_at   = CASE WHEN p_dismissed THEN timezone('utc', now()) END,
           dismissed_note = CASE WHEN p_dismissed THEN clean END
     WHERE id = p_claim_id
    RETURNING * INTO updated;

    IF NOT FOUND THEN
        RETURN NULL;
    END IF;

    INSERT INTO podcast_claim_reviews (claim_id, action, note)
    VALUES (
        p_claim_id,
        CASE WHEN p_dismissed THEN 'dismissed' ELSE 'restored' END,
        clean
    );
    RETURN to_jsonb(updated);
END;
$$;

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES (per-grantor; see 009).
-- The log is read and appended to by the app, never edited.
GRANT SELECT, INSERT ON podcast_claim_reviews TO web_anon;
GRANT EXECUTE ON FUNCTION set_podcast_claim_dismissal(UUID, BOOLEAN, TEXT) TO web_anon;

NOTIFY pgrst, 'reload schema';
