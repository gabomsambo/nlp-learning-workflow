# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Running the stack

`docker compose up -d --build` brings up three containers: `webui` (FastAPI on :8000),
`searxng` (:8080) and a local `qdrant` (:6333). The build is slow and the image is large —
`requirements.txt` pulls torch and layoutparser.

`.env` is gitignored and is not in the repo; compose reads it via `env_file:`. It is also
not copied into the image (see the `COPY` lines in `Dockerfile`), so the container's
environment comes entirely from compose.

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

`schema.sql` is stale: it defines 5 tables while the code queries 11. Do not treat it as
the authoritative schema — `grep -rhoE "table\(['\"][a-z_]+['\"]\)" nlp_pillars/ webui/`
gives the real list.

`requirements.txt` is entirely unpinned (`>=` everywhere), so a rebuild can silently move
the whole stack. Check resolved versions with `docker compose exec webui pip list` before
debugging anything version-sensitive.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
