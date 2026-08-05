# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Running the stack

`docker compose up -d --build` brings up five containers: `webui` (FastAPI on :8000),
`db` (Postgres on :5432), `postgrest` (:3000), `searxng` (:8080) and a local `qdrant`
(:6333). The build is slow and the image is large — `requirements.txt` pulls torch and
layoutparser.

`.env` is gitignored and is not in the repo; compose reads it via `env_file:`. It is also
not copied into the image (see the `COPY` lines in `Dockerfile`), so the container's
environment comes entirely from compose.

## The database is self-hosted

Postgres + PostgREST run in this compose file. The hosted Supabase project was reaped and
its host no longer resolves; `SUPABASE_URL` / `SUPABASE_KEY` are left in `.env` only so the
old project stays reachable if it is ever recovered.

Three env vars are overridden on `webui` in `docker-compose.yml`, and all three are load-
bearing — the comments there explain why. The trap: there are **two** database clients and
they resolve the URL differently. `webui/services/postgrest_client.py` reads
`POSTGREST_URL` then `SUPABASE_URL`, but `nlp_pillars/db.py::get_client()` reads
`SUPABASE_URL` **only**. Setting `POSTGREST_URL` alone silently fixes half the app.

`SUPABASE_KEY` is sent as an `Authorization: Bearer` token on every request and cannot be
disabled (`get_client()` raises if it is empty). PostgREST therefore **must** have
`PGRST_JWT_SECRET` set: with no secret, a request carrying a token fails with
`500 PGRST300 "Server lacks JWT secret"` rather than being served anonymously. The secret
and the `web_anon` token in compose are a matched pair — change them together.

Schema is applied automatically on **first** start only, from `docker-entrypoint-initdb.d`
(`db/init/01-roles.sh`, `schema.sql` mounted as `02-schema.sql`, `db/init/03-grants.sql`).
Editing `schema.sql` does nothing to an existing volume; migrate by hand or recreate
`nlp_pg_data`. Never `docker compose down -v` casually — that volume is the real data, and
`qdrant_data` is shared with any other worktree running this compose file.

## Which Qdrant the app talks to

`webui` uses `QDRANT_URL` from `.env`, which points at a managed Qdrant Cloud cluster.
The local `qdrant` service is kept only as an offline option and is **not** what the app
uses — do not re-add a `QDRANT_URL` override to the `webui` service in `docker-compose.yml`.
`depends_on: qdrant` on `webui` is therefore cosmetic and is left in place deliberately.

`nlp_pillars/vectors.py::ensure_collections()` creates the single `nlp_pillars` collection
(1536-dim, cosine). It runs from `VectorSearchTool.__init__`, so constructing an
`Orchestrator` is enough to create it. The cloud cluster is a small free tier — do not
bulk-load it.

## Sharp edges

`schema.sql` now covers all 11 tables, but it was **reconstructed from application code**,
not recovered — the old database is gone. Inferred types, defaults and constraints are
marked `INFERRED:` inline. `scripts` is a pure stub: its only mention in the codebase is a
commented-out example in a docstring (`webui/routers/api/script_download.py`), so that grep
for table names reports 11 while only 10 have live call sites.

Two pre-existing application bugs, both independent of the database backend and both left
alone deliberately:

- `TableQuery.eq()` in `nlp_pillars/db.py` renders Python `None` as the literal string
  `"None"`, so a `pillar_id IS NULL` filter becomes `pillar_id=eq.None` and matches nothing.
- `upsert_user_fsrs_parameters()` UPDATEs first and treats PostgREST's `200 []` (zero rows
  matched) as success, so it never falls through to INSERT. The table is fine — a direct
  insert of the converter's payload round-trips all 18 weights — but the upsert can only
  ever update a row that already exists.

`nlp_pillars` and `webui` are two **sibling top-level packages** under `/app` (see the
`COPY` lines in `Dockerfile`), so no relative import can ever reach from one into the
other — `webui` code must import `nlp_pillars` **absolutely**. Getting this wrong is not
loud: both quiz routers wrap their handler bodies in a bare `except Exception`, so the
resulting ImportError surfaced as a JSON 500 on `/api/quiz/*` and, in
`webui/routers/quiz.py`, as a silent fall-through to the non-FSRS `PostgrestClient`
fallback that made the page look healthy while ignoring FSRS scheduling entirely. Fixed
in `fm/nlp-quiz-api-broken`; keep new `webui` → `nlp_pillars` imports absolute.

`/app` is on `sys.path` only because it is the `WORKDIR` uvicorn starts in. A one-off
script run from elsewhere in the container needs `PYTHONPATH=/app`
(`docker compose exec -e PYTHONPATH=/app webui python /tmp/x.py`); `create_pillars.py` is
not in the image at all and must be `docker compose cp`'d in.

`requirements.txt` is entirely unpinned (`>=` everywhere), so a rebuild can silently move
the whole stack. Check resolved versions with `docker compose exec webui pip list` before
debugging anything version-sensitive. This has already bitten once: `atomic-agents` now
requires Python >= 3.12, so a host virtualenv must be `uv venv --python 3.12` even though
`pyproject.toml` still says 3.11.

## Configuration is loaded in an order that surprises people

`config.py::get_settings()` runs `load_dotenv(find_dotenv(), override=True)`. Two
consequences that cost real debugging time:

- `find_dotenv()` walks up from `config.py` itself, not from the cwd, so it finds the
  repo's `.env` no matter where you run a script from.
- `override=True` means **exported environment variables lose to `.env`**. And
  `db.py::get_client()` reads `os.environ` directly rather than `Settings`, so pointing a
  script at a different database means setting `os.environ` *after* the first
  `get_settings()` call. Setting it before, or passing it on the command line, is silently
  ignored.

## Agent LLM conventions

`summarizer_agent` / `synthesis_agent` / `quiz_agent` are real `instructor` + OpenAI
structured-output agents. Two deliberate conventions, both easy to "helpfully" undo:

- **No hand-rolled retry.** `instructor` retries internally with validation feedback
  (`max_retries` defaults to 3) and raises `InstructorRetryException`, which is *not* a
  `pydantic.ValidationError`. An `except ValidationError` retry branch is unreachable dead
  code. Each agent makes exactly one `create()` call and wraps failures `from e`.
- **Lazy singletons.** `SummarizerAgent` / `SynthesisAgent` / `QuizAgent` are `_Lazy*`
  proxies, not agent instances. `orchestrator.py` and `upload_service.py` import them by
  value at module load, so an eagerly-built `None` singleton surfaced a missing API key as
  `AttributeError: 'NoneType' object has no attribute 'run'`. The proxy builds the client on
  first `.run()` and names the missing key instead.

`quiz_agent` trusts the model's `question_type` but still overwrites `difficulty` from
`QuizGeneratorInput.difficulty_mix` — that mix is a declared caller input and seeds FSRS
initial scheduling, so it is enforced, not advisory.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
