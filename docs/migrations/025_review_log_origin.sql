-- Migration 025: where a review was made, and an index for folding a card's log.
--
-- review_logs is the source of truth for quiz scheduling; quiz_cards is a cache
-- derived from it (nlp_pillars/fsrs_fold.py). Two columns say where each event
-- came from, and one index serves the per-card read that every review does.
--
--   device_id  which device made the review. NULL for the desktop app.
--   origin     'desktop' for a review made in this app, 'cloud' for one pulled
--              from the cloud reader. Existing rows (there are none on the
--              captain's volume as of 2026-10-25) and every desktop review take
--              the default, so the desktop writer never has to send it.
--
-- review_logs.id doubles as the review's idempotency key: the API takes an
-- optional event_id and inserts it as the row id, so a repeated event is a
-- primary-key conflict and records one review. No change is needed for that.
--
-- No data migration: review_logs is empty today.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql
-- carries the same columns for a brand-new volume; init scripts only fire
-- when the data directory is empty.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/025_review_log_origin.sql
--
-- Until it is applied, reviews still work: the desktop writer omits both
-- columns. Only an event that names a device or the cloud origin needs it.
-- Safe to re-run.

ALTER TABLE review_logs ADD COLUMN IF NOT EXISTS device_id TEXT;
ALTER TABLE review_logs ADD COLUMN IF NOT EXISTS origin TEXT NOT NULL DEFAULT 'desktop';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'review_logs_origin_check'
          AND conrelid = 'review_logs'::regclass
    ) THEN
        ALTER TABLE review_logs
            ADD CONSTRAINT review_logs_origin_check CHECK (origin IN ('desktop', 'cloud'));
    END IF;
END
$$;

-- The fold reads one card's events in (review_timestamp, id) order.
CREATE INDEX IF NOT EXISTS idx_review_logs_card_fold
    ON review_logs (card_id, review_timestamp, id);

COMMENT ON COLUMN review_logs.device_id IS
    'Device that made the review; NULL for the desktop app.';
COMMENT ON COLUMN review_logs.origin IS
    'desktop = made in this app; cloud = pulled from the cloud reader.';

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand.
GRANT SELECT, INSERT, UPDATE, DELETE ON review_logs TO web_anon;

-- PostgREST caches table metadata; without this it answers PGRST204 for the
-- new columns until it reloads.
NOTIFY pgrst, 'reload schema';
