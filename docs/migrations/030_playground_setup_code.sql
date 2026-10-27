-- Migration 030: Playground setup code. A short block the runner runs at the
-- start of every session and after every restart, before any cell: usually
-- the imports a notebook uses but never wrote down (its import cell was
-- deleted before it was saved).
--
-- The stored shape is documented in docs/PLAYGROUND.md ("Setup code").
--
-- MUST BE RUN BY HAND against any database that already exists, after 028:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/migrations/030_playground_setup_code.sql
--
-- Until it is applied an import still works (without storing setup code),
-- the Playground tab still runs the suggested setup code, and saving an
-- edited one answers 503 naming this file.
--
-- schema.sql carries the same column for a brand-new volume.
--
-- Safe to re-run.

-- NULL: never saved, so the tab uses the suggestion computed from the
-- notebook. '' is a saved empty setup (run nothing).
ALTER TABLE playgrounds ADD COLUMN IF NOT EXISTS setup_code TEXT;

-- PostgREST caches table metadata; without this the new column answers
-- PGRST204 until the cache reloads.
NOTIFY pgrst, 'reload schema';
