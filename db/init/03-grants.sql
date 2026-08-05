-- Grant web_anon access to the schema created by 02-schema.sql.
--
-- Runs after the schema because Postgres executes docker-entrypoint-initdb.d
-- in alphabetical order: 01-roles.sh, 02-schema.sql, then this file. Grants
-- cannot be written before the tables they refer to exist.
--
-- Full CRUD, not read-only: the application inserts papers and queue entries,
-- updates quiz cards after every review, updates and deletes pillars, and
-- appends review logs. A read-only anon role would break all of that. See the
-- auth-model note at the top of 01-roles.sh for why this is acceptable here.

GRANT USAGE ON SCHEMA public TO web_anon;

GRANT SELECT, INSERT, UPDATE, DELETE
    ON ALL TABLES IN SCHEMA public
    TO web_anon;

-- Sequences are not strictly needed today — every generated key is a UUID
-- default rather than a serial — but granting them keeps a future serial
-- column from failing with a confusing permission error.
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO web_anon;

-- Apply the same grants to anything created later (e.g. a migration adding a
-- table), so the REST surface does not silently lose a table.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO web_anon;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO web_anon;
