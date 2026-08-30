# Manual upload as a background run — verification screenshots

Captured live for the "move manual paper upload off the web request" change. The
"before" shots are the **deployed** `nlp_webui` container (an image built before PR #30,
so it still double-parses the PDF and embeds one chunk at a time); the "after" shots are
a host `uvicorn webui.app:app` on port 8010 running this branch against the same compose
services and the same database.

Kept in the repo only so they render in the pull request; nothing references them from
the app, and they can be deleted once the PR is merged and read.

| File | What it shows |
|---|---|
| `01-before-form.png` | The upload tab as it was. |
| `02-before-frozen.png` | The old behaviour, mid-upload: a disabled `Uploading...` button and nothing else. |
| `03-after-form.png` | The upload tab on this branch, at rest — the progress panel is hidden until there is a run. |
| `04-after-midrun.png` | Mid-run: `Step 2 of 8 — Looking up paper details`, per-stage rows with live durations, a cancel button. Started from the form; the POST answered 202 in 7.7 ms. |
| `05-after-complete.png` | The finished run: every stage with the real number it produced (`61,177 characters extracted`, `5 card(s)`, `102 chunk(s) indexed`) and the outcome box naming the paper that is now in the library. |
| `06-after-skipped-steps.png` | The same upload with both post-upload actions off: `Summarizing`, `Writing the lesson` and `Building quiz cards` render as `skipped` **with their reasons**, not as rows left pending. |
| `07-after-failed-upload.png` | A 404 URL. The failing stage is a red ✗ with `Client error '404 Not Found'`, every stage it never reached is closed out with `the PDF could not be fetched`, and the outcome box says `Nothing was added to the library.` The second history row below it is the same failure **before** the exception-chain fix, still showing tenacity's `RetryError[<Future at 0x… >]`. |

`before-health-probe.txt` is a `curl` of `GET /health` against the **old** container every
2 seconds across the upload in `02-before-frozen.png`. It answers in ~5 ms until the
upload starts and then times out at 5 seconds continuously, for as long as the probe ran.
The container's own healthcheck failed with it and Docker restarted `nlp_webui`
(`RestartCount` went 0 → 1). On this branch, `GET /health` on the same process during an
upload answered 200 in 0.3–0.9 ms.
