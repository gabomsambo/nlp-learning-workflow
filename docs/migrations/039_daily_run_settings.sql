-- Migration 039: the daily run's settings, edited from the "Daily run" page.
--
-- Before this the daily run was controlled only by SCHEDULE_ENABLED /
-- SCHEDULE_TIME / SCHEDULE_TIMEZONE / PAPERS_PER_DAY on the `scheduler`
-- container, and changing any of them meant editing .env and recreating the
-- container. Now the page writes this one row and the scheduler re-reads it
-- every minute (nlp_pillars/daily_run.py; precedence is documented there).
--
-- One row, id = 1. No row is inserted here on purpose: until the scheduler
-- starts (or the page saves) the env vars are the first-boot defaults, and the
-- first of those writes copies them into this row. After that the row wins.
-- Papers per pillar is NOT stored here; it is the existing
-- pillars.papers_per_day column.
--
-- MUST BE RUN BY HAND against any database that already exists.
-- schema.sql carries the same table for a brand-new volume.
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/039_daily_run_settings.sql
--
-- Idempotent.

CREATE TABLE IF NOT EXISTS daily_run_settings (
    id                SMALLINT PRIMARY KEY DEFAULT 1
        CONSTRAINT daily_run_settings_singleton CHECK (id = 1),
    -- Off unless someone turns it on. Turning it on starts a daily API spend.
    enabled           BOOLEAN NOT NULL DEFAULT false,
    -- 24-hour HH:MM in `timezone`.
    run_time          TEXT NOT NULL DEFAULT '08:00'
        CONSTRAINT daily_run_settings_run_time_check
        CHECK (run_time ~ '^([01][0-9]|2[0-3]):[0-5][0-9]$'),
    -- IANA name. Validated by the app (Postgres has no IANA check that
    -- PostgREST can surface usefully); the scheduler refuses to schedule an
    -- unknown zone rather than guessing.
    timezone          TEXT NOT NULL DEFAULT 'UTC',
    -- The pillars that take part. NULL means every pillar in the database.
    pillar_ids        TEXT[],
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Written by the scheduler each time it re-reads this row, so the page can
    -- tell "on and being honoured" from "on, but no scheduler is running".
    scheduler_seen_at TIMESTAMPTZ,
    -- The most recent daily run, scheduled or "Run now": which trigger, when,
    -- and the pipeline_runs ids per pillar. Outcomes are read live from
    -- pipeline_runs, so this never disagrees with the run rows.
    last_run          JSONB
);

COMMENT ON TABLE daily_run_settings IS
    'Single-row daily-run configuration (id = 1), edited on the Daily run page '
    'and re-read by the scheduler every minute. See nlp_pillars/daily_run.py.';

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES in db/init/03-grants.sql:
-- default privileges are per-grantor and do not cover a migration applied
-- later by hand. Without this, PostgREST answers 401 / 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON daily_run_settings TO web_anon;

NOTIFY pgrst, 'reload schema';
