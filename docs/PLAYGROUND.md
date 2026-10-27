# Paper Playground

Study the code that goes with a paper, cell by cell. This file is the feature's
contract: what is stored, where datasets are cached, and how the sandbox finds
them. Later PRs extend it; keep it short and exact.

| Part | Where |
|---|---|
| Table | `playgrounds`, created by `docs/migrations/028_playgrounds.sql` (hand-applied) |
| Import, datasets, explain pass | `nlp_pillars/playground/` |
| API | `webui/routers/api/playgrounds.py` |
| Tab | `webui/templates/paper.html` (`#playground`), `webui/static/playground.js` (render), `playground-core.js` (pure logic), `playground-session.js` (runner session) |
| Explain the difference | `nlp_pillars/playground/difference.py`, `POST /api/playgrounds/{id}/explain-difference` |
| Prompt | `nlp_pillars/playground/explain.py::SYSTEM_PROMPT`, copied in `docs/PROMPTS.md` |
| Sandbox (running code) | [`docs/PLAYGROUND_RUNNER.md`](PLAYGROUND_RUNNER.md) |
| Follow-along (Listen page) | `playground_links`, `docs/migrations/029_playground_links.sql` (hand-applied); `nlp_pillars/playground/{follow,links}.py`; `webui/routers/api/follow_along.py`; `webui/static/follow-along.js` |

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
rules are [`docs/PLAYGROUND_RUNNER.md`](PLAYGROUND_RUNNER.md). The tab reaches
it only through webui's `/api/playground/runner` proxy.

**Session.** Opening the tab on a playground starts one kernel for it. The
session bar shows Starting, Ready, Busy, Stopped or Runner not running (with
the runner's own start hint), plus Restart and Stop. Leaving the page closes
the session (`DELETE` with `keepalive`). A page reloaded while its kernel was
still starting cannot close that kernel; when the runner then answers 429, the
tab closes the least recently used idle session of the same playground and
retries once. Sessions of other playgrounds are never touched.

**Runs.** The runner runs one execution per session, so the tab queues runs:
a cell clicked while another runs waits its turn. Interrupt, Restart and Stop
drop what is waiting. "Run up to here" runs, in order, every earlier code cell
that has not run cleanly in this kernel, then the cell; it stops at the first
cell that fails. A restart, a stop, an expired session (404) or a kernel the
runner had to restart empties that record. A run that hits the time limit
offers to run again with double the limit, up to `max_exec_timeout_s`.
Shift+Enter runs the focused cell (or the try it is in).

Before sending, the tab removes indentation that every non-blank,
non-comment line of a cell shares. Older Jupyter did this itself, so saved
notebooks can hold cells such as `"  x = 1\n  print(x)"` that ran there and
are an `IndentationError` for the runner's IPython.

**Outputs.** The execute proxy turns runner outputs into the same view shape
as saved ones (`notebook.py::runner_output_view`), so `text/html` goes through
the same sanitizer and raw HTML never reaches the browser. A live output
replaces the saved one in place; "Compare with saved output" shows both.

**Editing.** Edit turns the code into a textarea with line numbers (no editor
library). Tab indents, Shift+Tab outdents, Escape then Tab leaves the editor.
Line notes belong to the original code and are hidden while editing. "Reset
to original" drops the edit; a compact diff shows what changed.

**Tries.** "Run this try" runs the try's code in the same kernel and shows
Before (the cell's latest live output, else its saved output) beside After.
Changed lines of text outputs, and changed cells of tables with the same
shape, are highlighted. A try changes kernel state, so the card says so and
offers "Undo this try": restart, then run up to this cell.

**Explain the difference** sends both pieces of code, both outputs (as text,
cut to 3,000 characters each) and the cell's explanation to
`LLM_PLAYGROUND_DIFF_MODEL` (default `claude-haiku-5-5`, no fallback) behind
`require_spend_room`. The answer is shown under the Before / After and is not
stored. Its cost is logged at WARNING; it is not added to the daily spend
total, which only counts stored run stages and fact-checks.

## Follow-along

While an episode plays on the Listen page, the code card the narrator is
talking about shows beside the transcript (a drawer under the player on a
phone). An episode can follow **any** playground, not only the one on its own
paper: two study episodes made from study-guide papers can both follow one
notebook attached to another paper.

Until migration 029 is applied, the code panel, the API and the CLI say so and
name `docs/migrations/029_playground_links.sql`; no model is called.

### Stored shape

`playground_links`, one row per (episode, playground) (`UNIQUE (script_id,
playground_id)`). Deleting either side deletes the link.

| Column | Type | Meaning |
|---|---|---|
| `id` | uuid | Link id. |
| `script_id` | uuid | The episode (`podcast_scripts.id`). |
| `playground_id` | uuid | The playground it follows. |
| `mapping` | jsonb | The spans, below. `[]` until a pass has stored one. |
| `model` | text | Model that wrote `mapping`, after any fallback. |
| `status` | text | `pending` until a pass stores a mapping, then `ready`. `failed` when no pass has. |
| `error` | text | Why the last pass failed, or which spans it dropped. A failed retry on a `ready` link keeps the mapping and says `The last attempt failed: …`. |
| `script_sha256` | text | sha256 of the script text the mapping indexes. |
| `created_at`, `updated_at` | timestamptz | |

`mapping`:

```json
[
  {"line_start": 12, "line_end": 15, "cell_index": 2, "step_id": "step-1"},
  {"line_start": 31, "line_end": 31, "cell_index": 7, "step_id": "step-3"}
]
```

Line numbers are `script.splitlines()` indexes, the numbering the read-along
timeline (`audio_metadata.timeline`) uses. Lines about no particular code have
no span.

### Alignment pass

`POST /api/follow-along` with `{script_id, playground_id}` starts a
`pipeline_runs` job of kind `playground_align` (trigger `ui_playground`)
behind `require_spend_room`. Stages: `playground_align_prepare`,
`playground_align`, `playground_align_persist`. `result.script_id` and
`result.playground_id` are set at insert; migration 029 allows one active pass
per pair and exempts the kind from the one-run-per-pillar guard. The CLI runs
the same pass in the foreground:
`python -m nlp_pillars.cli playground link <script_id> <playground_id>`.

Before any call it refuses an unknown episode or playground (404), a
playground with no study guide or a script with no spoken lines (422).

One call to `LLM_PLAYGROUND_ALIGN_MODEL` (default `claude-haiku-5-5`, Sonnet
5.5 as fallback, thinking off). The model sees the spoken lines as
`L<n>: text` and the study steps: step id, title and goal, then each cell's
number, explanation and code (30 lines or 1,200 characters per cell). The
prompt is `nlp_pillars/playground/follow.py::SYSTEM_PROMPT`, copied in
`docs/PROMPTS.md`.

Rules enforced before anything is stored (`follow.py::validate_spans`):

- An answer that is not the `AlignDraft` schema is malformed and fails the pass.
- Both ends of a span are spoken lines the model was shown, and start <= end.
- `cell_index` is a cell of the study guide and `step_id` is the step that
  lists it.
- Spans do not overlap; of two that do, the earlier one is kept. Spans are
  stored in script order.

A span that breaks a rule is dropped and named in `error`. A pass whose every
span was dropped fails. A pass with no spans at all is a `ready` link with an
empty mapping. The align stage detail carries the call's cost, which is what
the daily spend cap adds up.

### Playback

`GET /api/follow-along?script_id=…` returns the episode's links (each with its
playground's paper and title), the playground on the episode's own paper (the
picker's default) and the latest pass. `GET /api/follow-along/playgrounds`
lists papers with a playground for the picker. The cards themselves come from
`GET /api/playgrounds?paper_id=…`, rendered with `playground.js`'s
`renderCode`/`renderOutput`, so the sanitizer is the Playground tab's. The
Listen page loads `playground-core.js` and `playground.js` but not
`playground-session.js`: the panel shows saved outputs and never starts a
runner session.

`follow-along.js` maps playback time to a line (`ReadAlong.positionAt`), the
line to a span, and the span to a card: step title, explanation, code with line
numbers and saved output. Between spans the last card stays, dimmed; before
the first span there is none. Seeking lands on the same card as playing there.
`Pin` stops auto-follow; unpinning snaps back to the current position. The
card links to `/papers/<id>?cell=<n>#playground`, which scrolls the Playground
tab to that cell.

No follow-along is offered, and the panel says why, when the episode has no
audio, no read-along timeline (`python -m nlp_pillars.cli timeline backfill`),
or a timeline for an older version of the script. A link whose
`script_sha256` no longer matches the script is offered again rather than
followed. Spans whose cell is no longer in the study guide are skipped.
