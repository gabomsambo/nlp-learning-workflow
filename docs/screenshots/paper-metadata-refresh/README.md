# Paper metadata refresh screenshots

## Intended captures

1. `01-before-thin-metadata.png` — paper detail modal for a discovery-ingested row with truncated authors/abstract (e.g. `arxiv:1706.03762`, three authors).
2. `02-after-refresh.png` — same modal after **Refresh metadata**, showing the success banner and repaired author list.

## Live exercise (2026-08-30)

Screenshots could not be captured in this worktree: `chrome-devtools-axi` is installed but no Chrome/Chromium binary is present (`Could not find Google Chrome executable for channel 'stable'`).

The feature was exercised live against the captain's library via the Python service and API:

- **`2403.05525` (DeepSeek-VL)** — refreshed from 3 authors / 303-char abstract / `venue=NULL` to **15 authors**, **1851-char abstract**, **`arXiv:cs.AI`**.
- **`arxiv:1706.03762` (Attention Is All You Need)** — left thin for a follow-up UI capture; API path verified with unit tests and host uvicorn on `:8001`.

To reproduce UI screenshots once Chrome is available:

```bash
# from repo root, against compose PostgREST on :3000
SUPABASE_URL=http://localhost:3000 POSTGREST_URL=http://localhost:3000 \
  SUPABASE_KEY='<web_anon token>' \
  .venv/bin/uvicorn webui.app:app --host 127.0.0.1 --port 8001

CHROME_DEVTOOLS_AXI_HEADED=1 chrome-devtools-axi run <<'EOF'
open "http://127.0.0.1:8001/papers?pillar=neural-architectures-language"
wait 3000
click @<paper-link-uid>
wait 2000
screenshot docs/screenshots/paper-metadata-refresh/01-before-thin-metadata.png
click @<refresh-btn-uid>
wait 15000
screenshot docs/screenshots/paper-metadata-refresh/02-after-refresh.png
EOF
```
