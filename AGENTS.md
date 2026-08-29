# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## How to implement changes (plan first)

Plan non-trivial changes before writing code. This applies to any agent working here.

1. **Research first** — read the code you are about to change and the sections of this
   file that cover it.
2. **Write the plan** to `PRPs/<feature>.md`: goal, context (files, patterns, gotchas),
   dependency-ordered tasks, and the validation gates each task must pass.
   The `generate-prp` skill produces this shape — `/prp:generate-prp` on Claude,
   `$generate-prp` on Codex. `/PRPs` is gitignored — plans stay local.
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
**011 (`podcast_scripts.source_material`) has NOT been applied** — verified against the live
PostgREST on 2026-08-29, which answers `PGRST204 Could not find the 'source_material' column`.
`add_podcast_script()` detects exactly that and retries without the key, so podcasts still
save; only the record of what they were written from is lost until it is run.

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

### An upload reports two facts, and the checkboxes default on

`UploadResponse.success` means the paper reached `papers`; `pipeline_ok` /
`pipeline_errors` report the follow-on processing separately. They used to be one fact:
`_run_full_pipeline` wrapped its whole body in `except Exception`, appended
`pipeline_error: {e}` to `actions_triggered` as a pseudo-action, and the route still
answered `success=True` — so the page printed "uploaded successfully! Triggered:
pipeline_error: ...". Each stage now fails independently into `PipelineOutcome.errors`
(a failed lesson no longer takes the quiz with it), and `upsert_text` returning 0 for a
non-empty paper is recorded as a failure, matching `Orchestrator._process_paper`. A
failed pipeline is **not** an upload error: the paper is in the library and re-uploading
is the wrong remedy, which is what the page's `#upload-result` banner says.

`run_summarizer` and `generate_quiz` default **True** now, matching discovery's hardcoded
`enable_quiz=True` at `run_service.py`. The quiz genuinely needs the summarizer — it is
built from its `PaperNote` — so the two checkboxes move together in the UI and the
combination is reported rather than silently ignored server-side. An unchecked box is
absent from `FormData`, so `pillar_detail.html` sends both flags explicitly; omitting them
would now mean the *opposite* of what the user chose.

`_create_paper_ref_from_url` parses the PDF only when nothing else can name the paper.
An arXiv id in the URL skips it, because `_enrich_from_arxiv` overwrites the guessed title
unconditionally and the real ingest parses the file again anyway — 8.4 seconds thrown away
per arXiv upload, measured on the captain's 4.71 MB PDF. The fallback still runs for
non-arXiv URLs (and before the S2 lookup, which searches by title) and when an arXiv
lookup was expected but failed.

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

`upsert_text()` embeds in **batches of `EMBED_BATCH_SIZE` (100)**, not one request per
chunk — the captain's 448-chunk paper cost 448 sequential OpenAI round trips and was the
dominant term in a 3m20s upload. Two contracts hold it together and neither is optional:
`_embed_batch()` places each vector by the response item's `index` rather than trusting
arrival order (a reshuffle is invisible in the stored data — payloads carry no chunk
text), and any batch failure abandons the **whole** upsert. The per-chunk loop it
replaced logged a warning and `continue`d, so a paper could be written with an arbitrary
subset of itself embedded and still return a plausible count. `upsert_text` is therefore
all-or-nothing: a non-zero return means the paper is fully represented, 0 means nothing
was written.

Stored payloads carry only `pillar_id`, `paper_id`, `chunk_index` and `len` — **no chunk
text**. `search_similar()` is a paper-level discovery API, not a snippet API; recovering
the text of a hit means re-chunking the source with the same parameters.

## The discovery candidate payload is the paper's permanent record

`webui/services/discovery_results.py` looks like a view-model and is not one. The dicts
it builds are stored in `pipeline_runs.result`, rendered by `discovery.html`, and then
**posted back to `/select` unchanged**, which persists them via `run_service._to_paper_refs`
-> `orchestrator` -> `db.upsert_paper`. Anything shortened there is shortened in `papers`
forever. It used to cap abstracts at 300 characters and author lists at 3, commented as
display caps — for two fields the candidates table does not render at all. Measured on the
captain's library: `2403.05525` carries 3 authors and a 303-character abstract cut
mid-sentence; a URL-uploaded paper in the same pillar carries all 319 authors and 1538
characters, because that path never went through this serialiser.

**Truncate at render time in the template, never in the serialiser.** Adding a field means
touching three places or it is silently dropped: the serialiser, `PaperData` in
`webui/routers/api/discovery.py` (FastAPI discards undeclared keys — that is why `venue`
was NULL on every discovered paper), and the `selectedPapers` map in `discovery.html`.
`tests/webui/test_discovery_metadata_roundtrip.py` walks all three and greps the template
for the JS link no Python test can otherwise reach.

Fixed forward only. Rows written before this still carry the truncation; re-resolving them
is a separate captain decision.

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

`POST /api/pillars/{id}/discover` is a background job too, as of 2026-08-29. It used to
answer synchronously — deliberately, because the user needs the candidates in front of
them to choose from — with its blocking call behind `asyncio.to_thread`. That reasoning
was sound and was still reversed: ~30 seconds behind one static "Discovering papers…"
line cost more than the early return bought, and moving it onto `pipeline_runs` also
bought reload-survival and a cancel button. The candidate list now comes back in
`pipeline_runs.result` on the poll the browser already makes, so finishing a run costs
no extra round trip and `?run=<id>` re-renders the candidates after a reload.

`docs/migrations/010_discovery_runs.sql` must be run **by hand**: `kind` and
`trigger_source` are CHECK-constrained, so `discover`/`ui_discover` are rejected until it
is applied, and `result` does not exist. It is `result`'s only home — JSONB on the run row
rather than a child table, because unlike stages it is written exactly once, at the end.

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

Discovery has the same ambiguity in three places, and resolves it the same way: **a
discovery step that says "0 found" means genuinely zero results, never "this failed".**
Each source therefore returns a `SourceResult(candidates, failures)` rather than a bare
list — do not "simplify" it back, the second field is the entire point. The three that
used to be invisible, all of which arrived as a plausible zero:

- **Query generation fell back.** `run_discovery_with_selection` catches anything from
  `DiscoveryAgent.run` and uses `_fallback_queries`. The stage is marked failed and says
  so, rather than presenting the pillar's focus areas as if the model wrote them.
- **A source errored.** `_friendly_source_error` names rate limiting specifically — it is
  the common one and the only one with an obvious remedy (see the SearXNG section: a
  suspended engine still answers HTTP 200 with an empty `results`).
- **The vector store was unreachable.** `VectorSearchTool.search_similar_papers` lets
  `search_similar()`'s `RuntimeError` propagate instead of degrading to `[]`.

Exception text is put on screen through `_first_line`, which trims the parts that are not
a reason: tenacity's `RetryError[<Future …>]` wrapper (via `_unwrap_retry`), httpx's
` for url …` tail, and instructor's dangling `<failed_attempts>` open tag.

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

## The paper-detail modal is shared, and it is the only paper detail surface

`webui/static/paper-modal.js` + `webui/templates/_paper_modal.html`, included by
`papers.html` and `pillar_detail.html`. Both call `PaperModal.init({triggerSelector})`
and nothing else. One endpoint behind it — `GET /papers/details/{paper_id}`, which returns
`{paper, notes, lessons, quiz_cards}` — so a lesson row and a quiz-card row open the *paper*
they came from, keyed by their `paper_id`. Do not add a second detail surface; the pillar
page had inert `<div>`s for exactly as long as it had no shared one to point at.

`triggerSelector` must be narrow. The pillar page's search results carry `data-paper-id` on
their **enqueue** buttons, so a generic `[data-paper-id]` trigger swallows those clicks; the
page passes `.activity-item-link`.

Nothing in that module escapes anything, on purpose: every node is `createElement` +
`textContent` and every attribute is `setAttribute`. The version it replaced built the whole
modal from template literals into `innerHTML`. Measured on the pre-change page with a hostile
payload through the details endpoint: **19 injected handlers fired and a `javascript:` href
went live**; the same payload against the new renderer fires none and produces no `<a>` at
all. `url_pdf` is scheme-checked before it reaches an href — `textContent` does not protect a
URL, and `file://` is allowed because uploaded PDFs are stored that way.

Quiz answers are **visible by default** and hidden by one control for the whole section, not
per card (captain's call, 2026-08-29): these cards are read as a paper summary far more often
than they are self-tested, so per-card reveal was friction on the common path. The choice
persists at `localStorage['nlp:quizAnswersVisible']` across cards, papers, reloads and both
pages. Hiding uses the `hidden` attribute, never a `display:none` class — a class leaves the
answer in the accessibility tree, so a screen reader still reads out what the user hid.

## JS tests exist now, and they are deliberately narrow

`tests/js/run-progress.test.js` and `tests/js/paper-modal.test.js`, run by
`node --test tests/js/*.test.js` in its own CI job. Stdlib only — no npm, no bundler, no
`package.json` — reached through a CommonJS guard at the bottom of each module that the
browser ignores. Use the glob, not `node --test tests/js/`: the directory form resolves as
a module and dies with `MODULE_NOT_FOUND` when the repo path contains a space.

They cover the pure helpers only. The polling loop, `AbortController` teardown, the 404
path, and everything in `paper-modal.js` that touches the DOM — the renderer, the fetch,
the answers toggle — need a DOM and are **not** covered; jsdom means npm and Playwright
means a browser download, both rejected. Do not read that job's green tick as "the
frontend is tested".

### Counts on the pillar page are totals, and a failed count is not zero

`PostgrestClient.counts_for_pillar()` reads real per-pillar totals from
`Content-Range` under `Prefer: count=exact` (the client sets that header on every
request), one `select=id&limit=1` per table. Use it rather than lengthening a list:
`/pillars/{id}` used to derive "Papers Processed" from `len(get_papers(limit=5))`,
so the number could never exceed 5.

It **raises `CountUnavailableError`** where its neighbours `get_quick_stats()` and
`counts_by_pillar()` degrade to `0`, and `_require_count_from_content_range()` raises
where `_parse_count_from_content_range()` answers `0` for a missing header. That
split is the point: an empty pillar and an unreachable database are different facts,
and the pillar page renders the second as an explicit unknown (`—`, "Count
unavailable"), never as `0`. Do not "simplify" the raising pair into the lenient one.

"Quiz Cards" there counts **all** cards for the pillar, not the due subset —
`get_quiz_cards_for_review()` is a due query and belongs on the review page. The
progress-bar denominators are display goals (`PROGRESS_GOALS` in
`webui/routers/pillars.py`), nothing enforces them, and the bar is clamped to 100%.

### Pico v2 namespaces its CSS variables, so most of the templates' colours do nothing

`base.html` loads `@picocss/pico@2`, whose custom properties are all `--pico-`
prefixed. Every bare `var(--primary)`, `var(--muted-color)`, `var(--border-color)`,
`var(--card-background-color)` and `var(--secondary-background)` in
`webui/templates/` is therefore undefined: the declaration is dropped at computed-value
time and the element falls back to inherited or transparent. Measured on
`pillar_detail.html` — the Progress Overview bars rendered as nothing at all, and the
"cards" have no background or border. `webui/static/styles.css` defines no variables of
its own. The fix per declaration is `var(--primary, var(--pico-primary))`; only the
progress-bar rules carry it so far. Repointing the rest is a separate change.

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
0 co-running requests served inline vs 13/13 through `to_thread`. One podcast is five model
calls (DeepSeek for the four Ground Pack extractions, Claude for synthesis); measured end
to end on a 5-page paper at 37.8K input + 9.6K output tokens ≈ **$0.26** when all five
were Claude — synthesis still dominates the bill.

**Ground Pack extraction routes to DeepSeek; synthesis stays on Claude.** Model ids live
in `nlp_pillars/podcast_models.py` and override through `PODCAST_EXTRACTION_MODEL` /
`PODCAST_SYNTHESIS_MODEL` in `.env` (defaults: `deepseek-v4-flash` and
`claude-sonnet-4-5-20250929`). Do not use `deepseek-v4-pro` for extraction — it is a
reasoning model that burns `max_tokens` on thinking and truncates before writing the
table. `DEEPSEEK_API_KEY` is required in compose (`:?` on `webui` and `scheduler`, same
pattern as `QDRANT_API_KEY`); it lives only in `.env`, never in the repo.

Each extraction call tries DeepSeek first and **falls back to Claude loudly** on any
failure, recording `fallback=true` and `fallback_reason` on the row. Truncated
(`finish_reason` `length`/`max_tokens`) or empty extraction output is never passed to
synthesis — it triggers fallback, and raises `GroundPackExtractionError` if Claude fails
too (HTTP 500). Per-section provenance is stored in `podcast_scripts.ground_pack_calls`
(JSONB, `docs/migrations/013_…`, hand-applied); same degradation loop as 011/012 when
the column is missing.

### Podcast generation refuses rather than inventing, and never destroys a script

Three contracts, each of which replaced a measured lie. Do not soften any of them.

- **No source material is a hard failure, raised before the first model call.**
  `generate()` calls `_assess_source_material()` right after the paper/notes/full-text
  reads. No body **and** no abstract **and** no notes raises
  `InsufficientSourceMaterialError`; the route answers **422** with the reason. Before
  this, `_get_full_text`'s `except: return ""` became the placeholder
  `[Full text not available…]` and all five calls ran — ~$0.27 of fluent, confident
  script whose entire factual basis was the title, reported as a green success. Paper
  `file:2dd76e910fbc` in the captain's database is exactly that case and is the fixture:
  reproduce with the model calls stubbed, never live.
- **Partial material proceeds and is recorded.** A body-less paper with an abstract
  and/or a notes row still generates — a notes row carries the problem, method and
  limitations prompts 3 and 4 ask for — but `SourceMaterial` (level, warnings) rides on
  `PodcastScript`, into the response, onto the page and into
  `podcast_scripts.source_material` (JSONB, `docs/migrations/011_…`, hand-applied).
  "Thin" and "complete" looking identical is the bug; equating them one level up is the
  same bug.
- **A failed insert must not destroy the script.** `add_podcast_script()` raises
  `PodcastScriptSaveError`; the route answers **200 with `saved: false`** and the full
  script in the body, and the page renders it under a "NOT saved" banner with a
  client-side download. A 5xx here sends a paid-for artifact into generic error handling
  and loses it. `_get_full_text` returns `FullTextResult(text, error)` so the reason a
  body is missing survives the `except` — do not collapse it back to `str`.

`get_podcast_script_by_id()` / `get_podcast_scripts()` / `get_all_papers()` **raise** on
read failure (`PodcastScriptLookupError`, `PaperLookupError`); `None` and `[]` mean
genuinely absent. Same precedent as `PillarLookupError`. Routes map the error to **503**,
never 404 — a malformed id used to produce PostgREST `400 invalid input syntax for type
uuid` and reach the user as "Script not found". The `/podcast` page renders an explicit
banner instead of an empty dropdown plus "No podcast scripts generated yet".

Still deliberately unfixed on this path: the fake progress bar (tracked as
`nlp-podcast-progress`) and the `innerHTML` sink that renders the script body. The
"30-60 seconds" label is fixed — it says ~4 minutes, which is what 238s rounds to.

### What a podcast is aimed at is configurable; what it may say is not

`nlp_pillars/podcast_options.py` owns four knobs — **field/domain, audience, episode
length, tone** — as a registry of `OptionSpec`s. **Adding a fifth is a data change:**
append a spec, reference its variables from a prompt template in `podcast_agent.py`.
No signature change, no schema change (`PodcastOptions.choices` is keyed by option
key), no migration.

Every default reproduces the aiming the prompts hardcoded before this existed — an NLP
paper for a graduate student, ~30 minutes, the "TWIML/Neutral/Lex vibe" tone — so
`PodcastAgent()` with no options behaves as it always did. `tests/test_podcast_options.py`
pins the pre-change fragments verbatim; `scripts/render_podcast_prompts.py` prints all
five prompts for any option set without calling a model, which is how a prompt change is
reviewed as a diff rather than by spending $0.27.

Three rules hold this together, and each of them is asserted:

- **Trusted text is interpolated; user text is not.** A preset's `vars` are written in
  that file and go straight into instruction sentences. Free text is sanitized to one
  short line and appears ONLY inside the delimited `=== EPISODE SETTINGS ===` block in
  the *user* message, followed by the precedence note, then the rules. A custom *field*
  therefore contributes a pointer ("the field named in the EPISODE SETTINGS block"), not
  its own words — otherwise it reaches instruction slots through the audience templates,
  which name the field. That leak was found by the tests, not by review.
- **The grounding rules and output format are not options.** "Use ONLY information found
  in the provided paper", the `[VERIFY]:` marker, "no external facts",
  numbers-only-if-present, the `[HOST]:` line format and the `[MUSIC]/[SFX]/[PAUSE]/
  [TRANSITION]` cue vocabulary are fixed, restated after the settings block, and tested
  against hostile free text. Do not make any of them configurable.
- **Call 5's system prompt is constant and interpolates nothing.** Role, grounding,
  format and TTS live there; the options, Ground Pack and paper are in the user message.
  That is deliberate — a constant system prefix is the prerequisite for the prompt
  caching that would take ~22% off every podcast.

Temperatures are set per call and were previously unset (so all five ran at the API
default of 1.0): extraction 0.1 (calls 1 and 3), analysis 0.4 (calls 2 and 4), script 0.8.

`docs/migrations/012_podcast_options.sql` adds `podcast_scripts.options` and **must be
run by hand**; until it is, `add_podcast_script()` drops the key, logs it and still saves
the script — the same degradation 011 has, now looped so a database behind by both
migrations still saves.

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

## Podcast episode audio (IndexTTS v1)

IndexTTS runs on the **host GPU**, never in Docker. The webui calls it via
`gradio_client` at `INDEXTTS_URL` (default `http://host.docker.internal:7861`).
Liveness is **not** a port check — probe `/gradio_api/info` for `/gen_single` with
the exact 24-parameter contract in `nlp_pillars/tts/indextts_client.py`.

Voice references come from the captain's folder, mounted read-only at `/voices`
(`VOICES_DIR`). Scanning and preflight live in `nlp_pillars/tts/voice_library.py`.
All podcast cues (`[HOST]:`, `[PAUSE]`, `[MUSIC]`, etc.) are stripped in
`nlp_pillars/tts/cue_parser.py` before any text reaches the model. IndexTTS then
splits each chunk again internally (120 tokens, on `-` and sentence punctuation).
Em-dashes in script prose normalize to `-` and can isolate a trailing quote into a
one-token segment that crashes the codec — `prepare_text_for_indextts()` in
`nlp_pillars/tts/text_prep.py` replaces em/en dashes with `, ` before every
`/gen_single` call. A failed chunk aborts the episode (no partial MP3): the
`tts_synthesize` stage is marked `failed` with `chunk N/M: {reason}` and the run
error carries the same via `ChunkSynthesisError`.

Audio generation is a fourth `pipeline_runs` kind (`podcast_audio` /
`ui_podcast_audio`). MP3s land in `/app/data/podcast_audio/` on the `nlp_uploads`
volume; metadata is `podcast_scripts.audio_metadata` (migration 014, hand-applied).
Rebuild the webui image after compose changes — it adds `ffmpeg`, `gradio_client`,
`extra_hosts`, and the voices mount.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
