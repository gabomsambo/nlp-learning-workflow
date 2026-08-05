#!/bin/bash
# Create the roles PostgREST authenticates as.
#
# Runs once, on first start, from Postgres's docker-entrypoint-initdb.d.
# This is a .sh rather than a .sql purely so the password can come from the
# environment instead of being hardcoded in a committed file.
#
# AUTH MODEL — deliberately minimal.
# This is a single-user tool bound to localhost. There is no user management,
# no per-row security and no JWT: PostgREST connects as `authenticator`, which
# has no privileges of its own beyond the right to switch into `web_anon`, and
# every anonymous request runs as `web_anon`. `web_anon` gets full CRUD on the
# application tables (see 03-grants.sql) because the app is the only client and
# it both reads and writes all eleven of them.
#
# Consequence worth stating plainly: anything that can reach the PostgREST port
# has full read/write access to the data. That is acceptable only because the
# port is published on the host loopback and nothing else uses it. Do not expose
# this stack to a network without adding real authentication first.

set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Anonymous role: what every unauthenticated request runs as.
    -- NOLOGIN — it is only ever reached via SET ROLE from the authenticator.
    CREATE ROLE web_anon NOLOGIN;

    -- The role PostgREST actually connects as. NOINHERIT is important: it must
    -- not passively hold web_anon's privileges, only be able to switch to them.
    CREATE ROLE authenticator NOINHERIT LOGIN PASSWORD '${AUTHENTICATOR_PASSWORD}';
    GRANT web_anon TO authenticator;
EOSQL

echo "roles: created web_anon and authenticator"
