# Paper Playground

Study the code that goes with a paper, cell by cell. This file is the feature's
contract: what is stored, where datasets are cached, and how the sandbox finds
them. Later PRs extend it; keep it short and exact.

| Part | Where |
|---|---|
| Table | `playgrounds`, created by `docs/migrations/028_playgrounds.sql` (hand-applied) |
| Import, datasets, explain pass | `nlp_pillars/playground/` |
| API | `webui/routers/api/playgrounds.py` |
| Tab | `webui/templates/paper.html` (`#playground`), `webui/static/playground.js` |
| Prompt | `nlp_pillars/playground/explain.py::SYSTEM_PROMPT`, copied in `docs/PROMPTS.md` |
| Sandbox (running code) | [`docs/PLAYGROUND_RUNNER.md`](PLAYGROUND_RUNNER.md) |

Until migration 028 is applied, the tab and the CLI say so and name the file;
nothing is downloaded and no model is called.

## Stored shape

One row per paper (`UNIQUE (paper_id)`). A re-import replaces the notebook in
place, keeps the row id, and clears `steps`.

| Column | Type | Meaning |
|---|---|---|
| `id` | uuid | Playground id. Also the dataset directory name. |
| `paper_id` | text | The paper. Deleting the paper deletes the playground. |
| `title` | text | First markdown heading, else the file name. |
| `source_kind` | text | `notebook`. `generated` is reserved for papers without code. |
| `notebook` | jsonb | The `.ipynb` as uploaded (nbformat 4), saved outputs included. |
| `notebook_name` | text | The uploaded file name. |
| `steps` | jsonb | The explain pass result, below. `[]` until one has run. |
| `datasets` | jsonb | `[{url, path, bytes, sha256}]`, or `{url, error}` for a failed download. |
| `status` | text | `imported`, `ready`, `partial` or `failed`. |
| `model` | text | Model(s) that wrote `steps`, after any fallback. |
| `error` | text | Why the last pass was partial or failed. |
| `created_at`, `updated_at` | timestamptz | |

`steps`:

```json
[
  {
    "id": "step-1",
    "title": "Load the points",
    "goal": "Read the sample points and look at the first rows.",
    "cells": [
      {
        "cell_index": 2,
        "explanation": "Reads the points file into a dataframe and shows the first five rows to check the columns.",
        "line_notes": [{"line": 3, "note": "head() shows five rows unless you pass a number."}],
        "tries": [
          {
            "label": "Show ten rows",
            "code": "<the full modified cell>",
            "expect": "The table grows to ten rows; nothing else changes."
          }
        ],
        "quiz": "What would head() show if the file had only three rows?"
      }
    ]
  }
]
```

Rules the explain pass enforces before anything is stored
(`nlp_pillars/playground/explain.py::validate_draft`):

- `cell_index` is the 0-based index into the notebook's `cells`, and names a
  code cell with source. Each appears once, in notebook order.
- `explanation` is one sentence, at most 40 words (the prompt asks for 25).
- `line_notes[].line` is 1-based within the cell and exists; one note per line.
- `tries` is 0 to 3 items; `code` is the whole modified cell and differs from
  the original.
- `quiz` is one question.
- Step ids are assigned by the app (`step-1`, `step-2`, …), not the model.

A pass that covers some cells is stored as `partial` with `error` listing what
is missing. A pass with nothing usable keeps any earlier `steps` and records
`error`; with no earlier guide the status is `failed`. The notebook stays
viewable in every case.

## Explain pass

`POST /api/playgrounds/{id}/explain` starts a `pipeline_runs` job of kind
`playground_explain` (trigger `ui_playground`) behind `require_spend_room`.
Stages: `playground_prepare`, `playground_explain`, `playground_persist`.
`result.playground_id` is set at insert; migration 028 allows one active pass
per playground and exempts the kind from the one-run-per-pillar guard.

The notebook is split into parts of at most 12 code cells (and about 60,000
prompt characters); each part is one call to `LLM_PLAYGROUND_MODEL` (default
`claude-sonnet-5`, Sonnet 5.5 as fallback). The model sees markdown, code with
line numbers, and saved outputs (text first, HTML tables as text, figures as a
placeholder, 3,000 characters per output). The explain stage detail carries the
running cost, which is what the daily spend cap adds up.

## Import

- Web: the Playground tab's file control, `POST /api/playgrounds/import?paper_id=…`
  (multipart `file`).
- CLI: `python -m nlp_pillars.cli playground import <paper_id> <path.ipynb>`.

Only nbformat 4 `.ipynb` files up to `PLAYGROUND_NOTEBOOK_MAX_BYTES`
(default 25 MB) are accepted. Saved outputs shown on the tab: stream text,
`text/plain`, `text/html` and `image/png`; error tracebacks are shown as text;
any other output type is named, not shown.

`text/html` is untrusted. The server turns it into a tag tree
(`notebook.py::sanitize_html`): tables and basic formatting only, no
attributes but numeric `colspan`/`rowspan`, and `script`, `style`, `iframe`,
`svg` and similar are dropped with their content. The browser builds that tree
with `createElement`. Markdown cells are shown as plain text; their images are
not loaded.

## Dataset cache

At import every quoted `http(s)` URL in a code cell is downloaded, including
URLs on commented-out lines. Hosts that resolve to private, loopback or
link-local addresses are refused. Up to `PLAYGROUND_DATA_MAX_BYTES` (default
1 GB) per playground; a file that would exceed it is not kept. A failed download
is recorded in `datasets` with its reason and does not fail the import.
Re-importing reuses a file whose URL, size and hash are already recorded, and
removes files the new notebook no longer names.

Layout, inside the `playground_data` Docker volume (`nlp_playground_data`),
mounted read-write in `webui` at `PLAYGROUND_DATA_DIR=/app/playground-data`:

```
/<playground_id>/manifest.json
/<playground_id>/files/<first 12 hex of sha256(url)>-<last URL path segment>
```

`manifest.json` is a flat object from URL to path relative to
`/<playground_id>/`, listing only complete downloads:

```json
{"https://example.org/data/points.csv": "files/3fa2c0d1e9b2-points.csv"}
```

The file name and this shape are the contract with the sandbox. It mounts the
same volume read-only at `/data` and serves a URL from
`/data/<playground_id>/<path>` instead of the network.

## Running code

The `playground-runner` service runs notebook code in a locked-down kernel and
reads this cache through `manifest.json`. Its API, limits and offline-data
rules are [`docs/PLAYGROUND_RUNNER.md`](PLAYGROUND_RUNNER.md). The Playground
tab does not call it yet; running and editing cells from the tab is the next
change.
