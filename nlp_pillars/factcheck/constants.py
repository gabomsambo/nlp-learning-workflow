"""Shared names for the fact-check engine and its migrations."""

PROMPT_VERSION = "v2"
MIGRATION = "docs/migrations/022_podcast_fact_checks.sql"
RUNS_MIGRATION = "docs/migrations/023_podcast_factcheck_runs.sql"
DISMISSAL_MIGRATION = "docs/migrations/031_claim_dismissals.sql"

KIND = "podcast_factcheck"
TRIGGER_UI = "ui_podcast_factcheck"
TRIGGER_AUTO = "auto_podcast_factcheck"

# StageName values. run-progress.js labels the same slugs.
STAGE_EXTRACT = "factcheck_extract"
STAGE_SCREEN = "factcheck_screen"
STAGE_REJUDGE = "factcheck_rejudge"
STAGE_ANCHOR = "factcheck_anchor"
STAGE_PERSIST = "factcheck_persist"
STAGES = (
    STAGE_EXTRACT,
    STAGE_SCREEN,
    STAGE_REJUDGE,
    STAGE_ANCHOR,
    STAGE_PERSIST,
)

# Measured range for the seven scripts already in the library. Haiku 5.5
# extracts and screens; Sonnet 5.5 re-judges only what the screen flagged.
BACKFILL_COST = (
    "Expected cost for the 7 existing scripts is about $0.6–1.3 total "
    "(Haiku 5.5 extract and screen, Sonnet 5.5 re-judge of flagged claims only)."
)
