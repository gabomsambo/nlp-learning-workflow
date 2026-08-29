# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## How to implement changes (plan first)

Plan non-trivial changes before writing code. This applies to any agent working here.

1. **Research first** — read the code you are about to change and the sections of this
   file that cover it.
2. **Write the plan** to `PRPs/<feature>.md`: goal, context (files, patterns, gotchas),
   dependency-ordered tasks, and the validation gates each task must pass.
   `/prp:generate-prp` produces this shape. `/PRPs` is gitignored — plans stay local.
3. **Implement the tasks in order**, passing the gates as you go.

**One plan should be one reviewable change.** If the task list spans unrelated subsystems,
split it and ship the pieces separately. This is a cost argument, not a style preference:
review cost scales with diff size and is the most expensive validation step there is.
Measured on a sibling project running the same no-mistakes pipeline, review averaged
18.6 minutes against ~4,100-line diffs, and one oversized task took three pipeline runs
and roughly 3.7 hours to land.

**Delivery is not part of implementation.** Do not decide on your own to run `no-mistakes`
or open a PR. Whoever dispatched the work owns delivery: a firstmate crewmate follows its
brief's *Definition of done* — `direct-PR` means you push and open the PR yourself, while
`no-mistakes` means the pipeline owns review, tests, documentation, push, PR and CI, and
you must not stack your own review or PR steps on top of it.

## Running the stack

`docker compose up -d --build` starts **five** services: `webui` (FastAPI on :8000),
`db` (Postgres, host port **5434**), `postgrest` (:3000), `searxng` (:8080) and
`scheduler` (no ports). The build is slow and the image is large — `requirements.txt`
pulls torch and layoutparser.

A sixth service, the local `qdrant`, is defined but sits behind the `local-vectors`
compose profile and is **not** started: the app writes vectors to Qdrant Cloud. See
"Which Qdrant the app talks to".

Two of those five look wrong in `docker compose ps` and are not:

- **`scheduler` shows `Exited (0)`** whenever `SCHEDULE_ENABLED=false`, which is the
  committed default. `scheduler.py::main()` returns 0 immediately in that case, and the
  service is `restart: on-failure` (not `unless-stopped`) precisely so the disabled state
  stays visible instead of being restarted forever. "The scheduler container is not
  running" is therefore the expected state, not drift. To check the scheduler itself
  still works without enabling anything, run it as a throwaway:
  `docker compose run --rm --no-deps -e SCHEDULE_ENABLED=true scheduler` — it registers
  four APScheduler jobs and blocks; Ctrl-C it. Turning `SCHEDULE_ENABLED` on for real is
  a separate decision.
- **`postgrest` and `searxng` are often not recreated** by `up -d` after a rebuild. Only
  services whose config hash or image changed are replaced, and those two pin an image and
  take no code from this repo. PostgREST reconnects on its own when `db` is recreated.

The database is published on host 5434, not the usual 5432, because a developer machine
tends to have Postgres already. Only the *host* side moved: inside the compose network
postgrest still dials `db:5432`, and the container-side port must stay 5432. Reach it with
`psql -h 127.0.0.1 -p 5434`. When a port conflict is suspected, do not trust `lsof -iTCP`
— run as your own user it silently omits listeners owned by another user and will report a
busy port free. `netstat -an | grep <port>` or a throwaway `socket.bind()` tells the truth.

5434 has been the committed value since PR #14 (2026-08-16). A container still publishing
**5432** is a container older than that PR, not a compose/doc disagreement — recreating it
is the whole fix. Measured 2026-08-29: the two rival servers that originally motivated the
move (another Docker project on 5432, a native host Postgres on 5433) were both down, so a
bind test finds 5433 and 5434 free and 5432 held by `nlp_postgres` itself. 5434 stays
regardless; the point of the choice is that it survives those servers coming back.

You do not need the image to run the app locally. `uvicorn webui.app:app` from a host
virtualenv works against the compose services, starts instantly, and skips the multi-GB
build; see the dependency section for why that is the *only* sane option on Apple Silicon.

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

### `docs/migrations/` splits in two, and only the second half applies here

`nlp_pg_data`'s baseline is **`schema.sql`**, not a replay of the numbered migrations.
That makes 001-007 and 008+ two different things, and treating them as one chain wastes an
afternoon:

- **001-007 are Supabase-era history and must not be run against this database.** They
  describe the reaped hosted project's evolution, not this volume's. 001 creates `progress`
  and `daily_sessions`, which exist in neither `schema.sql` nor any live call site; 007 does
  `ALTER COLUMN host_cs DROP NOT NULL` on `podcast_scripts`, and no `host_cs`/`host_ling`
  column has ever existed here (the only surviving mention is `docker/init-db/01_init_schema.sql`,
  a legacy file nothing mounts). They are kept as history. Applying them would error or
  create dead tables.
- **008 onward are written against `schema.sql` and are the ones to apply by hand.** Each
  is idempotent (`IF NOT EXISTS`) and ends with `NOTIFY pgrst, 'reload schema'`, without
  which PostgREST keeps answering `PGRST204`/404 for the new column or table. 009 also
  carries its own explicit `GRANT`s — the `ALTER DEFAULT PRIVILEGES` in `db/init/03-grants.sql`
  is per-grantor and does not cover objects a later hand-applied migration creates.

Apply one with:

    docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - < docs/migrations/00N_*.sql

There is no migration-tracking table. The only way to know what a database has is to look:
compare `information_schema.columns` against `schema.sql`. Both 008 and 009 were applied to
the captain's volume on 2026-08-29, which brought it level with `schema.sql`'s 13 tables.

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

**Qdrant Cloud**, as of 2026-08-29 — cluster `nlp-learning-workflow-retry`, aws
eu-west-1, free/cost-optimised tier.

This reverses the 2026-08-16 entry that used to be here, and the reversal matters more
than the destination: that entry said the cloud cluster "was suspended and no longer
exists", and the second half was simply wrong. **A free-tier cluster suspended for
inactivity is not deleted.** Unsuspending this one took 20 seconds, and the `nlp_pillars`
collection came back whole — 170 points, status green, `pillar_id` payload index intact.
Nothing had to be rebuilt. Before concluding a cluster is gone, try to resume it.

The reason it read as gone is worth keeping, because it will happen again: **a suspended
cluster does not fail like a dead host.** DNS still resolves, the regional ingress still
terminates TLS, and **every** path — `/`, `/healthz`, `/collections` alike — answers a
plain-text `404 page not found`. A live cluster answers `/` with a JSON banner naming the
version. Read that 404 as "no cluster routed behind this UUID"; it is neither a wrong URL
nor proof of deletion.

`QDRANT_URL` and `QDRANT_API_KEY` are set on **both** `webui` and `scheduler`, and both
are interpolated from the single `.env` entry rather than hardcoded per service. That is
deliberate: the two containers share one collection and must never disagree about which
server holds it, and one value in one file makes them identical by construction where two
copies only look identical until someone edits one. Both use compose's required-variable
form (`${QDRANT_URL:?...}`) because the failure mode of an unset value is silent —
`vectors.py:45-47` logs a WARNING and disables vector operations, so a nightly run would
finish, report success, and write no embeddings at all. Failing at `docker compose up` is
the loud version of that. The API key is a live credential: it lives in `.env`, never in
`docker-compose.yml`.

The local `qdrant` service is still defined, still backed by `nlp_qdrant_data`, and is
now behind the **`local-vectors` compose profile**, so a plain `up -d` skips it. Keeping
it costs nothing and it is the offline option; leaving it *running* would have meant an
idle container holding 6333/6334 and looking like the thing the app uses. The
`depends_on: qdrant` entries on `webui` and `scheduler` were removed in the same change —
a dependency on a service the app no longer reads is worse than no dependency at all.
Start it deliberately with `docker compose --profile local-vectors up -d qdrant`.

Both stores hold **orphaned** points: the papers they describe were rows in the reaped
Supabase database, and `GET /api/pillars/{id}/search` drops any vector hit whose
`paper_id` is missing from the `papers` table (`if not paper: continue` in
`webui/routers/api/pillars.py`). So the points are real, count toward the collection
total, and are invisible in the UI until those papers are re-ingested. A low visible
result count is not evidence a write failed — scroll the collection and group by
`paper_id` before concluding anything. Do not delete them to tidy up.

**Free-tier clusters get suspended for inactivity, and this project has already lost one
cluster outright and had a second suspended.** Nothing currently notices: the app finds
out at the next ingest, as a stage failure. Making it loud is cheap and unbuilt — the
distinction above (plain-text 404 on every path vs. a JSON banner on `/`) is a two-line
check, and `/health` is the obvious place for it.

`nlp_pillars/vectors.py::ensure_collections()` creates the single `nlp_pillars` collection
(1536-dim, cosine) **and** the `pillar_id` keyword payload index. It runs from
`VectorSearchTool.__init__`, so constructing an `Orchestrator` is enough to create both.
The cloud cluster is a small free tier — do not bulk-load it.

That index is not optional. The cloud cluster runs Qdrant **strict mode**
(`unindexed_filtering_retrieve: false`), which answers a filtered query on an unindexed
payload key with `400 Bad Request: Index required but not found`. Every read in
`vectors.py` filters by `pillar_id` — that is the namespace isolation — so without the
index no search can succeed however it is spelled. Only `pillar_id` is indexed; filtering
or scrolling by `paper_id` still 400s, so filter that one client-side.

Read the collection with `client.query_points()`. `search()`, `search_batch()`,
`search_groups()`, `recommend()` and `discover()` were all removed in qdrant-client 1.19
(the pinned version); `query_points()` returns a `QueryResponse` whose hits are on
`.points`, where `search()` returned the hit list directly. `search_similar()` therefore
**raises** `RuntimeError` on `AttributeError`/`TypeError` or on a 4xx from the server,
rather than returning `[]` — those mean this code disagrees with its library or its server,
and reporting them as "nothing matched" is what hid the dead read path for months. Genuine
runtime failures (embedding call, transport, 5xx) still degrade to `[]`. Same precedent as
`pdf_loader.chunk_text()`; do not re-widen it to a blanket `except`. The one caller that
can surface the raise to a user is `GET /api/pillars/{id}/search`, which turns it into a
500 with the message — deliberately, since the alternative is a silent empty result page.

Mock the client with `Mock(spec=QdrantClient)` in tests. A bare `Mock()` answers any
attribute, so `tests/test_vectors.py` passed green for months against a read path calling
a method the installed library no longer had.

Stored payloads carry only `pillar_id`, `paper_id`, `chunk_index` and `len` — **no chunk
text**. `search_similar()` is a paper-level discovery API, not a snippet API; recovering
the text of a hit means re-chunking the source with the same parameters.

## There are eight pillars, and three places must agree

`create_pillars.py` is authoritative (captain's decision, 2026-08-05). The same list is
mirrored in `nlp_pillars/config.py::PILLAR_CONFIGS` — the fallback `get_pillar_config()`
uses when the database lookup fails — and in `README.md`. Change all three together, or a
database outage starts serving pillars that do not exist.

The `P1`-`P5` legacy IDs (`config.LEGACY_TO_SLUG`, `pillar_utils.LEGACY_PILLAR_MAPPING`)
predate the slug migration and now point at retired pillars. They are left in place
deliberately; `get_pillar_config("P1")` raises rather than returning something wrong.

## SearXNG serves JSON only because `settings.yml` says so

`searxng_config/settings.yml` sets `search.formats: [html, json]`. SearXNG defaults to
HTML-only and answers `?format=json` with **403**, which silently costs the app its
SearXNG discovery source: `Orchestrator._search_candidates` swallows the failure and
carries on with arXiv alone. `SearXNGTool.search()` prefers JSON (arXiv engine, real
paper IDs) and falls back to scraping the HTML UI, which returns general-web results —
tutorials and blog posts, not papers. If discovery quality drops, check that key first.

The second thing to check is whether SearXNG has benched the engine. A burst of queries
trips arXiv's rate limit, and SearXNG then suspends that engine for **an hour**
(`suspended_time=3600`). It does not error: `/search?format=json` still returns HTTP 200
with `"results": []` and the reason tucked into `"unresponsive_engines":
[["arxiv", "Suspended: too many requests"]]`. Every query looks like it simply matched
nothing. Read `unresponsive_engines` before concluding a query is bad, and space out
manual probing — a dozen curls in a row is enough to trigger it.

Query *shape* matters more than it looks, because both back ends do keyword matching.
Measured against the live arXiv API on 2026-08-16, same intent expressed two ways:
`state space models Mamba RWKV` returned 5/5 on-topic papers, while
`Exploration of state space models for natural language processing, focusing on
architectures like Mamba and RWKV` returned 0/5 — top hit "A New Strategy for the
Exploration of Venus", matched on the word *Exploration*. This is why
`discovery_agent`'s prompt spends three output instructions forcing 2-8 word keyword
queries; an LLM left to itself writes the second form every time.

## A paper identifier must resolve to a PDF, or it is not an identifier

`nlp_pillars/paper_ids.py` is the single place that answers "is this a real paper id, and
what PDF does it point at?", and every producer of ids uses it. The rule it enforces:
**discovery returns `None` rather than inventing an id**, and `add_to_paper_queue` refuses
any candidate for which `resolvable_pdf_url()` is `None`.

This is load-bearing, not stylistic. `SearXNGTool` used to mint `searxng_{hash(url) %
1000000:06d}` for URLs it could not parse; `add_to_paper_queue` recorded `source: 'arxiv'`
for every row; and `_fetch_full_paper_metadata` therefore rebuilt the download URL as
`https://arxiv.org/pdf/searxng_078015.pdf`, which 404s. The failure surfaced as "the pillar
failed at ingest today", three stages after the mistake. (`hash()` is also salted per
process, so the same URL got a different id every run and the "already queued" check never
matched.) Measured live 2026-08-06 with SearXNG's JSON API disabled: 0/2 pillars succeeded
before, 2/2 after.

`paper_queue` carries `url_pdf` so a non-arXiv paper survives the round trip — added by
`docs/migrations/008_paper_queue_url_pdf.sql`, which **must be run against any database
created before it**. `add_to_paper_queue` degrades to arXiv-only queueing and logs the
migration path if PostgREST reports the column missing.

## Long runs are background jobs, and their state is in Postgres

`POST /pipeline/run` and `POST /api/pillars/{id}/select` return **202 with a run id**
and do the work on a thread. Before 2026-08-16 both ran the synchronous `Orchestrator`
inside the request handler — `/pipeline/run` by shelling out to
`python -m nlp_pillars.cli run` and awaiting `proc.communicate()`, `/select` by calling
`process_selected_papers` directly — so the browser sat on "Running..." for minutes and
the single uvicorn process was frozen for every other request, `/health` included.

`POST /api/pillars/{id}/discover` still answers synchronously, because the user needs
those candidates in front of them to choose from, but its blocking call is now behind
`asyncio.to_thread`. Measured after the change: the call takes ~30s and `/health`
answers in 5ms during it.

Run state lives in `pipeline_runs` + `pipeline_run_stages`, **not** in memory. The
`scheduler` container is a separate process against the same database, so a webui-local
registry could never show a nightly run. `docs/migrations/009_pipeline_runs.sql` must be
run **by hand** against any existing database.

**Stages are a child table because PostgREST cannot partially update a JSONB column.**
There is no path syntax on PATCH and no `set.` filter, so an array column would mean
read-modify-write eleven times per run from a worker thread while the browser polls the
same row. One row per stage makes each transition an independent single-row PATCH.

`Orchestrator(on_stage=..., cancel=...)` — both optional, both defaulting to inert, so
the CLI, the scheduler and every existing test are unaffected. `on_stage(name, status,
detail)` fires at the eleven `Step N` boundaries that already existed. Two things worth
knowing before changing it:

- `_process_paper` has **no internal try/except**, so a paper that dies leaves its stage
  marked running. The orchestrator tracks the in-flight per-paper stage itself and the
  caller's handler closes it out via `_mark_current_paper_stage_failed`. A callback that
  only fires "before each step" cannot report *which* step failed.
- A callback that raises is caught and logged, never propagated. Losing the progress
  display is bad; losing the run with it is worse.

`RunCancelledError` is raised when a stage *starts* and the cancel event is set — never
mid-stage, so a stage always finishes the work it began and the database is not left
half-written. This is cooperative because it has to be: **nothing can interrupt
synchronous Python**. There is no `Thread.kill()`, and neither APScheduler's
`shutdown(wait=True)` nor a `ThreadPoolExecutor` will do it.

Because a killed process cannot record its own death, startup sweeps any run still
`pending`/`running` to **`interrupted`** — a status deliberately distinct from `failed`,
since all it means is "nobody is working on this any more". The sweep is scoped by
`trigger_source` so a webui restart cannot declare a live scheduler run dead.

One active run per pillar, enforced by a **partial unique index**
(`pipeline_runs_one_active_per_pillar`) rather than a check-then-insert, which races.
A second insert returns HTTP 409 / Postgres `23505`; `create_pipeline_run` turns that
into `None` and the route into a 409.

Two traps in the job machinery, both of which fail silently:

- **APScheduler's `misfire_grace_time` defaults to one second.** A job that reaches the
  executor later than that is discarded with only a WARNING — the user gets their 202
  and the run sits at `pending` forever. Always pass `misfire_grace_time=None`.
- **Do not use FastAPI `BackgroundTasks` here.** Starlette awaits them *inside* the ASGI
  cycle (`responses.py`: `await self.background()`), so a minutes-long task blocks
  graceful shutdown and its exceptions land after the response with nothing listening.

Writes go through the **synchronous** `nlp_pillars/db.py` client from the worker thread;
the poll endpoint reads through the **async** `webui/services/postgrest_client.py`.
Never cross them — calling the sync client from an `async def` handler is the original
bug. `httpx.Client` is thread-safe and designed to be shared, so the module-global
singleton is fine to use from both.

The browser side is `webui/static/run-progress.js`, shared by both pages. It polls with
a **recursive `setTimeout`, never `setInterval`** — `setInterval` queues the next tick
regardless of whether the last response arrived, so a slow server renders stages out of
order. It backs off on errors only, and reattaches after a reload from `?run=`, then
`localStorage`, then `GET /api/pipeline-runs/active`. That last lookup is deliberately
**not** filtered by pillar: filtering by whatever the dropdown shows is how a run
appears to vanish after a refresh.

More background, with sources and measurements:
`PRPs/ai_docs/background-jobs-and-postgrest-gotchas.md`.

### Reporting a run's outcome honestly

Four rules the run-status path now enforces, each of which was previously a lie the UI
told the user:

- **`PipelineResult.success` is not the run status.** It is `len(papers_processed) > 0`,
  so a run that legitimately finds nothing arrives as `success=False` with an empty
  `errors` list. `run_service._terminal_status()` maps that to **succeeded**; only
  `errors` makes a run failed. `success` itself is left alone — the CLI, the scheduler
  and fourteen tests read it and its meaning is right for them.
- **`RunCancelledError` subclasses `Exception`**, so any broad `except Exception` on the
  cancel path needs a narrower `except RunCancelledError: raise` above it.
  `run_daily` lacked one and recorded cancellations as failures while
  `process_selected_papers`, which has no wrapper, was always correct.
- **`create_pipeline_run()` returns `None` only for the one-active-per-pillar
  conflict** and raises `PipelineRunCreateError` for everything else. Every `None` used
  to become HTTP 409, so a missing table or a bad grant told the user "a run is already
  in progress".
- **A 404 from `GET /api/pipeline-runs/{id}` means the run does not exist, and nothing
  else.** `postgrest_client.get_pipeline_run()` no longer swallows exceptions: a missing
  row is already `200` with an empty array, so the old blanket `except` could only hide
  real failures. The route answers **503** for those, because the browser treats 404 as
  "forget this run" and one blip used to detach the UI from a live job.

`upsert_text()` returning 0 is ambiguous — empty text and a dead Qdrant look identical —
so `_process_paper` marks VECTORS **failed** when it gets 0 chunks from non-empty text.
Non-fatal: the lesson and quiz are already written.

### The browser side, and why it is not just a status line

`webui/static/run-progress.js` renders rows with `createElement` + `textContent`, never
`innerHTML` with a template literal. Its `escapeHtml()` escapes `& < >` and **not
quotes** — it serializes a text node, and the HTML spec only escapes quotes in attribute
mode — so `title="${escapeHtml(x)}"` is still injectable and was measured injecting an
`onmouseover`. Use `setAttribute` for attributes; `escapeAttr()` in `pillar_detail.html`
covers the string-building case, and neither is safe for `on*=` handlers, which need a
data attribute plus a delegated listener.

`onGone` (the 404 path) **must** call `onFinished`. It is the only thing that re-enables
each page's button, and without it a stale run id left the UI disabled on "Running…"
with a blank panel and no recovery but a reload.

Stage names are `StageName` slugs; `STAGE_LABEL` maps them to prose so nobody is shown
`pop_queue`. Stage rows are per-**run**, not per-paper, so a failure in one paper is
overwritten when the next paper re-enters that stage — the run-level `error` carries
every failure (`_summarise_errors`), not just the first. A terminal run never renders a
stage as still running: `displayStatus()` resolves a left-over `running` to `unknown`.

The one-sentence summary lives in a `role="status" aria-live="polite"` element rendered
**empty** by the template — a live region announces changes, not content already present
at load. The stage list is `aria-live="off"`; re-announcing eleven rows a second is
unusable.

### JS tests exist now, and they are deliberately narrow

`tests/js/run-progress.test.js`, run by `node --test tests/js/*.test.js` in its own CI
job. Stdlib only — no npm, no bundler, no `package.json` — reached through a CommonJS
guard at the bottom of `run-progress.js` that the browser ignores. Use the glob, not
`node --test tests/js/`: the directory form resolves as a module and dies with
`MODULE_NOT_FOUND` when the repo path contains a space.

It covers the pure helpers only. The polling loop, `AbortController` teardown and the
404 path need a DOM and are **not** covered; jsdom means npm and Playwright means a
browser download, both rejected. Do not read that job's green tick as "the frontend is
tested".

## Sharp edges

`schema.sql` now covers all 11 tables, but it was **reconstructed from application code**,
not recovered — the old database is gone. Inferred types, defaults and constraints are
marked `INFERRED:` inline. `scripts` is a pure stub: its only mention in the codebase is a
commented-out example in a docstring (`webui/routers/api/script_download.py`), so that grep
for table names reports 11 while only 10 have live call sites.

Two pre-existing application bugs, both left alone deliberately:

- `TableQuery.eq()` in `nlp_pillars/db.py` renders Python `None` as the literal string
  `"None"`, so a `pillar_id IS NULL` filter becomes `pillar_id=eq.None` and matches nothing.
- `upsert_user_fsrs_parameters()` UPDATEs first and treats PostgREST's `200 []` (zero rows
  matched) as success, so it never falls through to INSERT. The table is fine — a direct
  insert of the converter's payload round-trips all 18 weights — but the upsert can only
  ever update a row that already exists.

Two others of the same shape were fixed in PR #17, and the resulting contracts matter:

- **`get_pillars()` raises `PillarLookupError`; it no longer returns `[]` on failure.**
  An empty list now means the table is empty and nothing else. Collapsing the two is
  what made two fallbacks dead code — `cli.get_valid_pillars()` and
  `scheduler.run_all_pillars()` both wrap the call in an `except` that could never
  fire, so an unreachable database made the CLI reject every `--pillar` with
  "Valid pillars: " and nothing after it, and made the scheduler advise seeding a
  database that was merely down. Render paths that would rather show an empty dropdown
  than a 500 call **`get_pillars_or_empty()`**, which makes swallowing the failure a
  visible choice at the call site instead of everyone's silent default. All eight page
  routers and both `pillar_utils` helpers use the degrading variant; the CLI and the
  scheduler use the raising one.
- **`_paper_ref_to_dict()` rejects a missing `pillar_id` or `paper.id`.** Its None-filter
  exists so optional metadata (venue, year, abstract) is omitted rather than written as
  NULL, but it applied to every key — so a `None` pillar_id was quietly stripped and the
  row inserted with no pillar at all, against the one invariant this schema has.
  `add_paper()` catches the `ValueError` and returns `False`, so a daily run records the
  paper as failed and surfaces it rather than aborting mid-run.

`upsert_text()` wraps its whole body in `except Exception: return 0`, and that body calls
`chunk_text()` — so the `RuntimeError` PR #8 added there is caught and reported as "0 chunks
upserted". Both `upsert_text()` callers (`orchestrator.py`, `upload_service.py`) invoke it
un-guarded, so making it propagate would abort a daily run mid-paper. Left alone; know that
a chunker contract break shows up here as a zero, not an exception.

`webui/routers/api/script_download.py` is dead code and must **not** be registered in
`webui/app.py`: importing it aborts application startup (FastAPI rejects its
`Optional[BackgroundTasks]` annotation at decoration time), and its `get_script_from_db()`
is an unimplemented stub returning `None`. Working script download already exists at
`GET /api/podcast/{script_id}/download`. Its module docstring carries the detail.

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

**The lock is linux/amd64 and cannot be installed on a Mac.** Its own header says so, and it
carries sixteen `nvidia-*` packages plus `triton`. On an Apple-Silicon host install
`requirements.txt` instead — `uv pip install -r requirements.txt` into a 3.12 venv — and
accept that the resolution is the direct pins, not the locked transitive set. CI runs on
`ubuntu-latest`, which *is* amd64, so `.github/workflows/tests.yml` installs the lock and
keeps the guarantee where it can be kept.

Building the *image* on Apple Silicon is worse than it looks. Those CUDA packages do publish
`aarch64` wheels, so the build does not fail — it quietly downloads gigabytes
(`nvidia_cublas` alone is 542 MB) of libraries that cannot be used without an NVIDIA GPU.
Run the app from a host venv instead unless you specifically need to test the image.

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

## PDF extraction: the first two extractors in the chain do not do what the code implies

`pdf_loader.extract_text()` tries four extractors in order. The one that actually runs is
the **third**, `pymupdf4llm`:

- `layout-parser` is listed first and **always fails** — `layoutparser` 0.3.4 exposes
  `Detectron2LayoutModel` only when `detectron2` is installed, and it is not (it is not on
  PyPI). Every extraction logs `module layoutparser has no attribute Detectron2LayoutModel`
  at ERROR before falling through. That is expected, not a new break. It still costs a full
  PDF→image render of every page first, so extraction is slower than it looks.
- `pymupdf4llm` was imported but never declared as a dependency until `fm/nlp-pymupdf4llm`,
  so for the project's whole history extraction silently landed on `pypdf`.

`pymupdf4llm` 1.28 enables a layout/OCR mode by default, and `pdf_loader` turns it **off**
at import. Leaving it on drops prose from real papers (all of section 3.1 of arXiv:1706.03762;
~13% of word instances in arXiv:2106.09685) — the comment at the import has the measurements.
Do not "restore" it for its nicer tables; there is no knob to keep the layout pass without
the OCR pass.

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

`discovery_agent` became a real LLM agent on 2026-08-16; before that it was a stub that
pasted stopword-stripped pillar goals into three fixed templates and never called a model.
It is the cheapest agent in the project by a wide margin. Measured over three
representative calls on gpt-4o-mini (cold start, daily run with five recent paper ids, and
user-steered with two priority topics), usage barely moves: **451-526 input tokens and
129-133 output**, averaging 492 in / 132 out. The prompt is a fixed system message plus a
pillar's focus areas — it does not scale with paper text, which is why it is so stable.

At $0.15/$0.60 per million input/output tokens that is **~$0.00015 per call**, about 6,500
calls to the dollar. Eight pillars discovering once a day is **$0.0012/day, ~$0.45/year** —
less than two podcasts. Re-derive from the token counts rather than trusting the dollar
figure; those rates were current when this was written and the token counts are the part
that will not drift.

Both callers guard the call and fall back to `Orchestrator._fallback_queries`, so no
discovery failure — missing key, rate limit, network — can stop a daily run or 500 the
discovery API. That fallback returns the pillar's own focus areas, reordered by
`DiscoveryAgent._blend_topics` so user topics come first. It deliberately does *not*
interpolate the pillar slug the way the old fallback did: "recent advances
neural-architectures-language" is not a phrase in any paper.

`quiz_agent` trusts the model's `question_type` but still overwrites `difficulty` from
`QuizGeneratorInput.difficulty_mix` — that mix is a declared caller input and seeds FSRS
initial scheduling, so it is enforced, not advisory.

## The test suite runs in CI, and it is green — keep it that way

`.github/workflows/tests.yml` (added 2026-08-16) is the first CI this project has ever had
that runs `pytest`. `daily.yml` is unrelated and its schedule is disabled.

It was knowingly red for its first three PRs — 252 passed / 28 failed / 10 errors at PR
#14, rising to 275 / 28 / 10 by PR #15. PR #16 cleared all of it: **353 passed, 0 failed,
0 errors**. A failing test is now a real signal. Do not add to the red, and do **not**
reach for `continue-on-error`, `|| true`, `--ignore` or `-k` if you break something — a
suppressed suite is the exact condition this workflow was added to end.

What the 38 turned out to be, since the diagnosis is worth keeping:

- **10 errors, one cause.** Three `Lesson` fixtures predating the schema gaining required
  `title` / `content`. Each raised at construction, so those tests had not actually run in
  a long time. `scripts/smoke_local.py` had the same bug and had been silently reporting
  `Success: False`.
- **15 in `tests/test_db.py`.** The file predates the rewrite from a supabase-py-style
  client to the hand-rolled `PostgRESTClient`. Mocks returned a bare `Mock` where the code
  subscripts `response['error']`, and several tests asserted a `ValueError` validation
  path and an `.upsert()` method that have never existed here.
- **5 in `tests/test_cli.py`.** Not mocked at all; they really dialled PostgREST. The
  trap underneath: `cli.get_valid_pillars()` only falls back to `PILLAR_CONFIGS` on an
  *exception*, but `db.get_pillars()` swallows its own connection errors and returns `[]`
  — so an unreachable database yields an empty valid-pillar list instead of the fallback,
  and every pillar looks invalid.
- **The rest** were qdrant-client 1.19 normalising point ids to UUID form, three
  `*_no_client` tests that depended on execution order and on `QDRANT_URL` being unset,
  and a stale assertion on a legacy `P2` pillar id.

**The suite is hermetic now, and that is load-bearing.** It previously gave different
answers locally and on CI, in both directions, and the difference was always the
environment lying:

- the `test_cli.py` tests passed locally *only* when the compose stack happened to be up
- `test_vectors.py::test_get_vector_size_failure_fallback` failed locally because
  `get_settings()`'s `load_dotenv(override=True)` means the real `OPENAI_API_KEY` in
  `.env` beats any placeholder you export, so the embedding call it needed to *fail*
  succeeded against the live API — **billing the project on every local run**

Both are fixed at the source: nothing in the suite reaches a database or the network, and
no test decides which branch it is exercising by looking at your shell. Local and CI now
agree exactly (353/353 both ways, measured). If you add a test that only passes with the
stack up or with a real key present, you have reintroduced the problem.

To check a change the way CI sees it, run without `.env` and against dead ports — that is
all the runner is:

    OPENAI_API_KEY=test-key-not-real ANTHROPIC_API_KEY=test-key-not-real \
    DEFAULT_MODEL=gpt-4o-mini \
    SUPABASE_URL=http://localhost:3000 SUPABASE_KEY=test-token-not-real \
    POSTGREST_URL=http://localhost:3000 \
    QDRANT_URL=http://localhost:6333 QDRANT_API_KEY="" \
    SEARXNG_URL=http://localhost:8080 pytest tests/ -q

Copy that list from `.github/workflows/tests.yml` rather than trimming it. An earlier,
shorter version of this recipe omitted `SEARXNG_URL`, and the omission fails exactly one
test — `test_discovery_and_search.py::TestSearXNGTool::test_init`, where
`SearXNGTool.base_url` falls back to `settings.searxng_url` and is `None`. Green in CI,
red locally, for a reason that has nothing to do with the code under test.

…from a copy of the tree with no `.env` in it, since `find_dotenv()` walks up from
`config.py` and will find the repo's own regardless of your cwd.

The suite also only imports because `pyproject.toml` sets `pythonpath = ["."]`. There is no
`conftest.py`, the project is not pip-installed into the test environment, and `tests/` has
no `__init__.py`. Before that line, plain `pytest` failed every module with
`ModuleNotFoundError: No module named 'nlp_pillars'` and only `python -m pytest` worked,
because `-m` adds the working directory to `sys.path` as a side effect.

The `guard` job is separate, stdlib-only, and **must stay green**. It runs
`scripts/check_bare_slugs.py`, which fails the build on a pillar slug written without
quotes:

    pillar_id=models-architectures      # subtraction of two undefined names
    pillar_id="models-architectures"    # what was meant

This is not hypothetical tidiness. The slug migration dropped the quotes in **73** places
across `tests/test_discovery_and_search.py`, `tests/integration/test_orchestrator.py` and
`tests/e2e/test_smoke.py`, and every one of them survived thirteen merged PRs. The bad form
parses, compiles, imports, and passes ruff; it dies only at runtime, with `NameError: name
'models' is not defined` raised from whichever line *uses* the fixture rather than the line
that defines it. Nothing ran those three files, so nothing said so. Detection uses
`tokenize`, not a regex, so slugs inside strings and comments are never flagged.

The quotes were added and the slugs left pointing at the retired pre-migration ids
(`models-architectures` and friends, from `config.LEGACY_TO_SLUG`). That is deliberate and
matches the seven test files that were already correct — all of them use quoted legacy
slugs throughout. Re-pointing them at five of the current eight pillars would be an
invention rather than a migration, which is the same reasoning `config.py` gives for
leaving `LEGACY_TO_SLUG` alone. Retiring those ids from the tests is a separate job.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
