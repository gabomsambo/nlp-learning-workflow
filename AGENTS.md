# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Running the stack

`docker compose up -d --build` brings up six containers: `webui` (FastAPI on :8000),
`db` (Postgres on :5432), `postgrest` (:3000), `searxng` (:8080), a local `qdrant`
(:6333) and `scheduler` (no ports). The build is slow and the image is large —
`requirements.txt` pulls torch and layoutparser.

`.env` is gitignored and is not in the repo; compose reads it via `env_file:`. It is also
not copied into the image (see the `COPY` lines in `Dockerfile`), so the container's
environment comes entirely from compose. That absence is load-bearing: it is why
`get_settings()`'s `load_dotenv(override=True)` cannot clobber the overrides compose sets.

`scheduler` runs the same image as `webui` with `command: python -m nlp_pillars.scheduler`.
Any *new* service reusing this image must set `healthcheck: disable: true` as it does — the
Dockerfile's `HEALTHCHECK` curls `localhost:8000/health`, which exists only under uvicorn,
so a non-webui service inherits a probe that can never pass and sits permanently
"unhealthy" while working fine.

### Running a second stack alongside an existing one

Container names, host ports and volume names are all fixed, so a plain `up -d` in a second
worktree evicts the first. Use `-p <project>` plus an uncommitted override that renames
`container_name`, every volume `name:`, and the ports. Ports need the `!override` YAML tag —
Compose *appends* list entries by default, so a plain `ports:` in an override republishes
the original host port too and collides anyway.

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

## Uploaded PDFs are retained, and are the only copy

A file upload stores `papers.url_pdf = file:///app/data/uploads/<hash>.pdf` and the podcast
agent dereferences that path to get the paper body, so the file is real data, not a cache.
It lives in the `nlp_uploads` named volume (mounted at `/app/data`, created and chowned in
the `Dockerfile` so the non-root `appuser` can write into a fresh volume) and
`upload_service.py` deletes it only when the upload never reached the database. Do not
"tidy" that directory and do not move it under `.cache/`. There is deliberately **no**
retention or cleanup policy yet — retained PDFs accumulate at ~1-5 MB/paper.

Papers added by URL keep the http URL in `url_pdf` and are re-downloaded on demand, so
only the file-upload path depends on this.

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

Three pre-existing application bugs, all left alone deliberately:

- `TableQuery.eq()` in `nlp_pillars/db.py` renders Python `None` as the literal string
  `"None"`, so a `pillar_id IS NULL` filter becomes `pillar_id=eq.None` and matches nothing.
- `upsert_user_fsrs_parameters()` UPDATEs first and treats PostgREST's `200 []` (zero rows
  matched) as success, so it never falls through to INSERT. The table is fine — a direct
  insert of the converter's payload round-trips all 18 weights — but the upsert can only
  ever update a row that already exists.
- `vectors.search_similar()` calls `client.search()`, removed in qdrant-client 1.19 (the
  pinned version), so **every** query fails with `'QdrantClient' object has no attribute
  'search'` — swallowed by its `except`, which returns `[]`. Writes are unaffected:
  `upsert_text()` works and `client.query_points(...)` over the same collection returns
  hits. Verified live 2026-08-05 against the cloud cluster.

`nlp_pillars` and `webui` are two **sibling top-level packages** under `/app` (see the
`COPY` lines in `Dockerfile`), so no relative import can ever reach from one into the
other — `webui` code must import `nlp_pillars` **absolutely**. Getting this wrong is not
loud: both quiz routers wrap their handler bodies in a bare `except Exception`, so the
resulting ImportError surfaced as a JSON 500 on `/api/quiz/*` and, in
`webui/routers/quiz.py`, as a silent fall-through to the non-FSRS `PostgrestClient`
fallback that made the page look healthy while ignoring FSRS scheduling entirely. Fixed
in `fm/nlp-quiz-api-broken`; keep new `webui` → `nlp_pillars` imports absolute.

`docker compose logs webui` shows only WARNING and above from `nlp_pillars` — uvicorn never
configures those loggers, so every `logger.info` in the pipeline (timings, extracted char
counts, token usage) is invisible there. To see them, run the code under
`docker compose exec` with `logging.basicConfig(level=logging.INFO)`.

`/app` is on `sys.path` only because it is the `WORKDIR` uvicorn starts in. A one-off
script run from elsewhere in the container needs `PYTHONPATH=/app`
(`docker compose exec -e PYTHONPATH=/app webui python /tmp/x.py`); `create_pillars.py` is
not in the image at all and must be `docker compose cp`'d in.

## Dependencies are pinned, and there are two files

`requirements.txt` (direct deps, exact `==`) and `requirements.lock.txt` (full transitive
`pip freeze`) are both installed by one `pip install` in the `Dockerfile`, so pip fails the
build if they drift apart. The lock is the one that matters: the breakages that motivated
pinning came from *transitive* packages — Starlette 1.x arrives via `fastapi` and is not
listed in `requirements.txt` at all. The upgrade/regeneration recipe lives in the header of
`requirements.txt`; follow it rather than hand-editing the lock.

Python 3.12 is the floor everywhere (`pyproject.toml`, `Dockerfile`, README): `atomic-agents`
requires >= 3.12, so a host virtualenv must be `uv venv --python 3.12`. The `Dockerfile` pins
the interpreter patch release too — a floating `3.12-slim` would undo half the point.

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

## Chunking is measured in tokens, and its fallback is not a safety net

`pdf_loader.chunk_text()` and `vectors.upsert_text()` take `chunk_size` / `chunk_overlap`
in **tokens**, counted with tiktoken `cl100k_base` — the encoding `text-embedding-3-small`
uses, so the budget is the size the embedding model actually sees. They were documented as
characters until the semchunk call was repaired; ~4 characters per token if you are
converting an old value. `_chunk_text_naive()` is still character-denominated (it cannot
count tokens) and `chunk_text` converts before calling it.

`semchunk.chunk()` takes a **required positional `token_counter`**, and has since 0.1.0 —
there is no older release whose signature accepts a call without one, so the `chunk(text,
chunk_size=...)` this code used never worked against any published version. It raised
`TypeError`, the `except` swallowed it, and every chunk written to Qdrant for months was a
fixed-offset slice: measured on a 5-page paper, 4 of 6 naive boundaries fell inside a word
versus 0 of 8 semantic ones.

So `chunk_text` now **raises** `RuntimeError` on `TypeError` from semchunk (a call-signature
bug, not bad data) and only falls back — at ERROR with a traceback — on other runtime
failures. Do not re-widen that to a blanket `except`: both callers already contain the
raise (`ingest_agent` turns it into `IngestError`, `upload_service` logs it and finishes the
upload without vectors), so nothing reaches the user as a crash.

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

`podcast_agent` is the exception to the lazy-proxy rule and deliberately so: it has no
module-level singleton (`webui/routers/api/podcast.py` constructs it per request), so there
is nothing built at import time to fail. Its `_get_full_text` is synchronous and must stay
behind `asyncio.to_thread` — `generate()` is awaited straight from a FastAPI route, and
running the PDF ingest inline froze the event loop for every other request. Measured live:
0 co-running requests served inline vs 13/13 through `to_thread`. One podcast is five
Claude calls that each carry the full paper text; measured end to end on a 5-page paper at
37.8K input + 9.6K output tokens ≈ **$0.26**.

`quiz_agent` trusts the model's `question_type` but still overwrites `difficulty` from
`QuizGeneratorInput.difficulty_mix` — that mix is a declared caller input and seeds FSRS
initial scheduling, so it is enforced, not advisory.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
