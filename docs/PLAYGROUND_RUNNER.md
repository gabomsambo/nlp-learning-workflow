# Playground runner

The playground runner executes notebook code for paper playgrounds. It is a
separate compose service (`playground-runner`, container
`nlp_playground_runner`) built from `docker/playground-runner/`. Each session
is one Jupyter kernel, so variables persist from one cell to the next the way
they do in a notebook.

The web app talks to it through `webui/services/playground_runner_client.py`
and exposes the same calls as JSON under `/api/playground/runner/...`. The
runner has no host port; nothing but the web app can reach it.

It starts with the stack (`./workstation up`) and shows up in
`./workstation status` as **Playground**, and on Activity as
**Playground runner** (non-fatal: a stopped runner does not turn the shell
chip red). Idle, it is one small uvicorn process. A kernel starts only when a
session is created.

## Lockdown

Every item is enforced in `docker-compose.yml` or the image, and checked by
`tests/playground_runner/test_compose_lockdown.py` (static) and
`tests/playground_runner/test_docker_integration.py` (inside a real container).

| Control | How |
|---|---|
| No network egress | The runner is attached only to `playground`, a compose network with `internal: true`: Docker gives it no route off the host and no external DNS. webui joins that network to reach it. |
| No host port | The service has no `ports:`. |
| No sockets from notebook code | Each kernel starts through `playground_runner.kernel_launcher`, which installs a seccomp filter refusing `socket()` for `AF_INET`, `AF_INET6` and `AF_PACKET` (`EACCES`) and the `io_uring_*` calls. The network blocks the outside world; this blocks the one thing that network does connect to (webui) and the runner's own API. Kernels talk to the server over Unix-domain sockets (ZeroMQ `ipc` transport). |
| Own image | Built from `docker/playground-runner/` only: no app code, no `.env`, no keys. |
| Non-root | User `runner`, uid 10001. |
| Read-only root | `read_only: true`. Writable: `/tmp` (tmpfs, `PLAYGROUND_TMP_SIZE`, default 512m) and `/work`, the kernels' working directory (tmpfs, `PLAYGROUND_WORK_SIZE`, default 2g). Both are RAM and count toward the memory limit. |
| No capabilities | `cap_drop: [ALL]`, `security_opt: no-new-privileges:true`. |
| Memory | `PLAYGROUND_RUNNER_MEMORY`, default `6g`, with swap set equal (no swap). Each kernel raises its own OOM score, so the kernel that ran out of memory is the process killed, and the server reports it. |
| CPU | `PLAYGROUND_RUNNER_CPUS`, default `4`. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` follow it. |
| Processes | `PLAYGROUND_RUNNER_PIDS`, default `512`. |
| Shared memory | `PLAYGROUND_SHM_SIZE`, default `1g`. joblib (scikit-learn `n_jobs`) memory-maps large arrays into `/dev/shm`; Docker's 64 MB default is smaller than one 1M-row float matrix. |
| Execution time | `PLAYGROUND_EXEC_TIMEOUT_S`, default `180`. A request may ask for up to `PLAYGROUND_MAX_EXEC_TIMEOUT_S` (default `900`). On the deadline the kernel is interrupted (SIGINT) and the answer is `status: "timeout"`; variables from earlier cells are kept. If the code ignores the interrupt for `PLAYGROUND_INTERRUPT_GRACE_S` (default `10`), the kernel is restarted and `kernel_restarted` is `true`. |
| Output size | `PLAYGROUND_MAX_OUTPUT_BYTES`, default `2000000` per execution. See [Output cap](#output-cap). |
| Code size | `PLAYGROUND_MAX_CODE_BYTES`, default `1000000`. |
| Sessions | At most `PLAYGROUND_MAX_SESSIONS` (default `3`). A session idle for `PLAYGROUND_IDLE_TIMEOUT_S` (default `1800`) is closed; a session running code is never reaped. |

The limits are sized for the target workload: pandas aggregation over about
one million rows and several scikit-learn `IsolationForest` fits. Set any of
the variables above in `.env` and recreate the service
(`docker compose up -d playground-runner`).

Known gap: output printed from inside a forked child process (for example a
`multiprocessing` worker) is not shown. ipykernel forwards it over a TCP
loopback socket, which the seccomp filter refuses. Output from the cell itself,
including `!shell` commands, is unaffected.

## Offline datasets

The runner mounts the `playground_data` volume (Docker name
`nlp_playground_data`) read-only at `/data`. The web app's import step writes
each playground's files there:

```
/data/<playground_id>/manifest.json     {"<url>": "<path relative to this directory>", ...}
/data/<playground_id>/<cached files>
```

Every kernel runs a startup hook (`playground_runner/offline_data.py`) for its
playground. It patches `pandas.read_csv`, `read_json`, `read_excel`,
`read_parquet`, `read_table` and `urllib.request.urlopen` (which also covers
`urllib.request.urlretrieve`). A URL in the manifest is read from the cached
file, so

```python
df = pd.read_csv("https://raw.githubusercontent.com/<owner>/<repo>/main/transactions.csv")
```

runs unchanged. A URL that is not in the manifest raises
`OfflineDatasetError` (a subclass of `urllib.error.URLError`):

```
This playground runs offline, and https://… is not in the dataset cache for playground '<id>'.
Only URLs listed in the playground's manifest.json can be read. Re-import the playground so
the dataset is cached, or read a file that is already cached.
```

Matching ignores a `#fragment` and surrounding whitespace, nothing else. A
manifest entry pointing outside its playground directory is ignored. Readers
imported by a path other than the `pandas` module attribute (for example
`from pandas.io.parsers import read_csv`) are not patched; they fail with a
network error instead.

`playground_id` must match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$` and must not
contain `..`. A playground with no directory under `/data` still gets a
session; every URL read then fails with the message above.

## API

The runner listens on port 8765 inside the `playground` network. webui proxies
each call under `/api/playground/runner` with the same body and status codes
(see [Through webui](#through-webui)). Bodies are JSON
(`Content-Type: application/json`). Errors are `{"detail": "<message>"}`.

### `GET /health`

```json
{
  "status": "ok",
  "sessions": 1,
  "max_sessions": 3,
  "limits": {
    "exec_timeout_s": 180.0,
    "max_exec_timeout_s": 900.0,
    "max_output_bytes": 2000000,
    "max_code_bytes": 1000000,
    "idle_timeout_s": 1800.0,
    "memory_limit_bytes": 6442450944
  },
  "kernel_seccomp": true,
  "python": "3.12.13",
  "packages": {"numpy": "…", "pandas": "…", "scipy": "…", "scikit-learn": "…",
               "matplotlib": "…", "ipykernel": "…", "jupyter_client": "…"}
}
```

`memory_limit_bytes` is the container's cgroup limit, or `null` when there is
none. `packages` are the versions every kernel imports; results such as an
`IsolationForest` score can differ between scikit-learn versions, so this is
the place to look when a number does not match the original notebook.

### `GET /sessions`

```json
{"sessions": [<session>, …], "max_sessions": 3, "idle_timeout_s": 1800.0}
```

A `<session>` is:

| Field | Type | Meaning |
|---|---|---|
| `session_id` | string | 24 hex characters. |
| `playground_id` | string | As given at creation. |
| `created_at` | string | ISO 8601, UTC. |
| `last_active_at` | string | ISO 8601, UTC. Idle reaping counts from here. |
| `busy` | bool | Code is running now. |
| `execution_count` | int | Last execution number; 0 after a restart. |

### `POST /sessions`

Request: `{"playground_id": "<id>"}`. Starts a kernel (a few seconds) and
answers **201** with a `<session>`. Errors: **422** bad `playground_id`,
**429** `max_sessions` already open, **503** the kernel could not start.

### `POST /sessions/{session_id}/execute`

Request:

| Field | Type | Meaning |
|---|---|---|
| `code` | string, required | Python source; IPython syntax (`%magic`, `!cmd`) works. |
| `timeout_s` | number, optional | Seconds, > 0 and ≤ `max_exec_timeout_s`. Default `exec_timeout_s`. |

Response **200** (also for code that raised; check `status`):

| Field | Type | Meaning |
|---|---|---|
| `status` | `"ok"` \| `"error"` \| `"timeout"` | `error`: the code raised, was interrupted, or the kernel died. `timeout`: stopped at the deadline. |
| `execution_ms` | int | Wall time, milliseconds. |
| `execution_count` | int \| null | The cell number (`In [n]`). `null` when the kernel died first. |
| `outputs` | list | In order; shapes below. |
| `truncated` | bool | Some output was cut or dropped by the output cap. |
| `dropped_outputs` | int | Outputs dropped whole by the cap. |
| `interrupted` | bool | `POST …/interrupt` stopped this execution. |
| `kernel_restarted` | bool | The kernel was restarted during this execution (ignored interrupt or kernel death); every variable is gone. |

Output shapes:

| `type` | Fields | From |
|---|---|---|
| `stream` | `name` (`stdout`/`stderr`), `text` | `print`, warnings, logging. Consecutive chunks of one stream are merged. ANSI codes removed. |
| `text` | `text` | `text/plain` result, e.g. a cell ending in an expression. |
| `html` | `html` (raw), `text` (plain fallback or `null`), `untrusted: true` | `text/html`, e.g. a DataFrame. **Untrusted**: render it sandboxed or sanitized, never with `innerHTML` into the app page. |
| `image` | `mime` (`image/png`), `data` (base64), `text` (fallback or `null`) | `plt.show()` and figures left open at the end of a cell (matplotlib uses the non-interactive inline backend). |
| `error` | `ename`, `evalue`, `traceback` (plain text, newline-joined, ANSI codes removed) | An exception. Also produced by the runner itself: `TimeoutError` (deadline), `KernelDied` (process killed, usually out of memory), `Cancelled` (session restarted or closed mid-run). |

Any output cut to fit carries `"truncated": true`.

Errors: **404** unknown session, **409** the session is already running code,
**413** `code` too large, **422** bad `timeout_s` or missing `code`.

### `POST /sessions/{session_id}/interrupt`

Sends SIGINT to the kernel. Answers `{"session_id": "…", "interrupted": bool}`;
`interrupted` is `true` when code was running. The running execute then
returns `status: "error"`, `interrupted: true` and a `KeyboardInterrupt`
error output. **404** unknown session.

### `POST /sessions/{session_id}/restart`

Stops any running code, starts a fresh kernel (all variables gone) and re-runs
the offline hook. Answers `{"session_id": "…", "restarted": true}`. **404**,
**503** when the new kernel does not start.

### `DELETE /sessions/{session_id}`

Stops any running code, shuts the kernel down and removes its working
directory. Answers `{"session_id": "…", "deleted": true}`. **404** unknown
session.

### Output cap

Each execution may return at most `max_output_bytes`, counted over text,
HTML, base64 image data and error text. When the cap is reached, the output in
progress is cut (`truncated: true`, with a closing
`[output truncated: limit of N bytes reached]` line), later outputs are
dropped and counted in `dropped_outputs`, and an image or HTML output that
does not fit falls back to its plain-text form. Error outputs are always kept
so the reason a cell failed is never lost; their traceback is cut to 64 KB.

## Through webui

| webui route | Runner call |
|---|---|
| `GET /api/playground/runner/health` | `GET /health` |
| `GET /api/playground/runner/sessions` | `GET /sessions` |
| `POST /api/playground/runner/sessions` | `POST /sessions` |
| `POST /api/playground/runner/sessions/{id}/execute` | `POST /sessions/{id}/execute` |
| `POST /api/playground/runner/sessions/{id}/interrupt` | `POST /sessions/{id}/interrupt` |
| `POST /api/playground/runner/sessions/{id}/restart` | `POST /sessions/{id}/restart` |
| `DELETE /api/playground/runner/sessions/{id}` | `DELETE /sessions/{id}` |

Bodies pass through unchanged, with one exception: execute's `outputs` come
back in the Playground tab's view shape (`kind`: `stream`, `text`, `error`,
`html` as a sanitized tag tree, `png`), the same shape as a saved output
(`nlp_pillars/playground/notebook.py::runner_output_view`). The runner's raw
`html` never reaches the browser. Status codes: the runner's 4xx pass through
with its `detail`; **503** when the runner cannot be reached (the `detail`
names the URL and says to run `./workstation up` or
`docker compose up -d playground-runner`) or could not start a kernel;
**502** for any other runner 5xx; **504** when the runner did not answer in
time (the wait is the execution timeout plus 120 s). webui checks `{id}`
against `^[A-Za-z0-9_-]{1,64}$` and bodies before calling the runner.

`PLAYGROUND_RUNNER_URL` (default `http://playground-runner:8765`) sets where
webui looks. A webui started from a host virtualenv cannot reach the runner,
which has no host port; run the stack in Docker, or for local development run
the image with a port published on loopback only.

Remote requests (through the Cloudflare tunnel) reach these routes only with a
verified Access token, like every other route (`webui/access.py`). Cross-site
pages cannot call them: the routes take JSON bodies only, and the app answers
no CORS preflight.

## Changing the image

`docker/playground-runner/requirements.in` lists the direct dependencies;
`requirements.txt` is the full pinned set, regenerated with the `uv pip
compile` command in the `.in` file's header. Rebuild with
`docker compose build playground-runner` (or `./workstation up --build`).

## Tests

| File | Needs | Runs in CI |
|---|---|---|
| `tests/playground_runner/test_outputs.py`, `test_offline_data.py`, `test_api.py` | the app's test environment | `pytest`, `playground runner` jobs |
| `tests/playground_runner/test_compose_lockdown.py` | PyYAML | `pytest` job |
| `tests/playground_runner/test_kernel_live.py` | the runner's requirements (real kernels) | `playground runner` job; skips elsewhere |
| `tests/playground_runner/test_docker_integration.py` | Docker | wherever Docker is available; deselect with `-m "not docker"` |
| `tests/webui/test_playground_runner_api.py` | the app's test environment | `pytest` job |

To run the live-kernel tests locally:

```bash
uv venv --python 3.12 /tmp/runner-venv
uv pip install --python /tmp/runner-venv/bin/python -r docker/playground-runner/requirements.txt \
  pytest==9.1.1 pytest-asyncio==1.4.0 pytest-cov==7.1.0 httpx==0.28.1
/tmp/runner-venv/bin/python -m pytest tests/playground_runner/test_kernel_live.py
```
