# Discovery progress — verification screenshots

Captured live for the "show what paper discovery is doing while it does it" change,
against a host `uvicorn webui.app:app` on the running compose services. Kept in the repo
only so they render in the pull request; nothing references them from the app, and they
can be deleted once the PR is merged and read.

| File | What it shows |
|---|---|
| `01-before.png` | The old behaviour: `Discovering papers…`, alone, for ~30 seconds. |
| `02-after-midrun.png` | Mid-run — steps, per-source counts, the generated queries, a cancel button. |
| `03-source-failed-honestly.png` | A real Semantic Scholar 403 as a red ✗ with its reason, beside a genuine `0 found`. |
| `04-query-fallback.png` | Query generation falling back, forced with an invalid `OPENAI_API_KEY`. |
| `05-xss-neutralised.png` | A hostile stored candidate rendering as literal text: no handler fired. |
