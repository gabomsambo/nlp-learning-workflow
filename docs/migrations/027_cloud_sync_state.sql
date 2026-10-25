-- Migration 027: how far the computer has folded the cloud reader's review events.
--
-- The phone posts each review to the cloud reader as an event with a `seq`
-- (arrival order). `python -m nlp_pillars.cli cloud pull-reviews` copies new events
-- into review_logs (origin = 'cloud') and re-derives the cards, and this table
-- remembers where it got to, so the next pull asks for `seq > reviews_seq` only.
--
--   reviews_seq      every event up to and including this seq is in review_logs and
--                    its card is re-derived. `cloud export` publishes this number as
--                    `through_seq`, so the phone knows which events its snapshot has.
--   reviews_pending  set to the highest seq a pull is about to apply, and set equal
--                    to reviews_seq when it is done. If they differ, a pull stopped
--                    half-way (a crash, a lost connection) and the cards may hold some
--                    of those events. `cloud export` refuses to publish until the next
--                    pull has finished the job, because a snapshot taken in between
--                    would say "through seq N" over cards that hold a different set.
--
-- Nothing here is the source of truth: review_logs is, and a pull is safe to repeat
-- (an event is a primary-key insert, so a second delivery records nothing). Losing
-- the table costs one pull from seq 0.
--
-- MUST BE RUN BY HAND against any database that already exists. schema.sql carries
-- the same table for a brand-new volume; init scripts only fire when the data
-- directory is empty.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/027_cloud_sync_state.sql
--
-- Until it is applied, `cloud export` and `cloud publish` still work (through_seq
-- is 0) and `cloud pull-reviews` refuses, naming this file. Safe to re-run.

CREATE TABLE IF NOT EXISTS cloud_sync_state (
    key        TEXT PRIMARY KEY CHECK (btrim(key) <> ''),
    value      BIGINT NOT NULL DEFAULT 0 CHECK (value >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

COMMENT ON TABLE cloud_sync_state IS
    'Cursors for syncing with the cloud reader: reviews_seq (folded through) and reviews_pending (a pull in progress).';

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand.
GRANT SELECT, INSERT, UPDATE, DELETE ON cloud_sync_state TO web_anon;

-- PostgREST caches table metadata; without this a new table can answer GET
-- with 200 and POST with 404 until the cache reloads.
NOTIFY pgrst, 'reload schema';
