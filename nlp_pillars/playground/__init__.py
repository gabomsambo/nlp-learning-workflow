"""Paper Playground: study a paper's notebook cell by cell.

The contract (stored shape, dataset cache layout, sandbox hand-off) is
docs/PLAYGROUND.md. This package is the import, the dataset cache, the
explain pass and the store; the web routes live in
webui/routers/api/playgrounds.py. ``follow`` and ``links`` are the Listen
page's follow-along (webui/routers/api/follow_along.py).
"""

MIGRATION = "docs/migrations/028_playgrounds.sql"

#: Follow-along: episode-to-playground links and the alignment pass.
FOLLOW_MIGRATION = "docs/migrations/029_playground_links.sql"

#: Setup code run before any cell (docs/PLAYGROUND.md, "Setup code").
SETUP_MIGRATION = "docs/migrations/030_playground_setup_code.sql"
