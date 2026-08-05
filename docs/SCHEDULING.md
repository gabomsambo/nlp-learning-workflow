# Scheduling Guide

This document explains how to set up automated daily runs of the NLP Learning Workflow.

**The daily run now fires from the app's own scheduler, not from GitHub Actions.** See
[The built-in scheduler](#the-built-in-scheduler-current-method) below. The GitHub Actions
section that follows it is kept because the workflow file is kept, but its schedule trigger
is commented out and it cannot work against the current self-hosted database.

## The built-in scheduler (current method)

`nlp_pillars/scheduler.py` runs as the `scheduler` service in `docker-compose.yml` — the
same image as `webui`, started with `python -m nlp_pillars.scheduler`. It is its own
service on purpose: a `webui` crash or restart must not silently stop the daily run.

It runs two things on one APScheduler instance:

1. **The daily learning run**, at `SCHEDULE_TIME` in `SCHEDULE_TIMEZONE`, for **every
   pillar in the database** (`create_pillars.py` seeds eight). Pillars run sequentially and
   independently — one failing does not stop the rest, matching the Action's
   `fail-fast: false`.
2. **The FSRS optimization and maintenance jobs** in `nlp_pillars/services/background_jobs.py`.
   These were written long ago but `start_background_jobs()` had no callers, so they had
   never actually run anywhere. They do now.

### Settings

All read from `.env` (see `Settings` in `nlp_pillars/config.py`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCHEDULE_ENABLED` | `false` | Master off switch. When false the service logs why and exits 0. |
| `SCHEDULE_TIME` | `08:00` | 24-hour `HH:MM`. |
| `SCHEDULE_TIMEZONE` | `UTC` | IANA name, e.g. `America/New_York`. DST is handled by the zone. |
| `PAPERS_PER_DAY` | `1` | Papers per pillar per run — the whole run's API-spend knob. The per-pillar `papers_per_day` column in the database is not consulted. |

An unparseable `SCHEDULE_TIME` or unknown `SCHEDULE_TIMEZONE` makes the service exit 1
rather than fall back to a default — a silently wrong run time is worse than a container
that refuses to start.

The service uses `restart: on-failure`, not `unless-stopped`, so the deliberate exit-0 when
disabled is not restarted into a loop. Disabled looks like `Exited (0)` in
`docker compose ps`; that is the expected off state, not a failure.

```bash
docker compose up -d scheduler         # start it
docker compose logs -f scheduler       # watch it; it logs each job's next run time at startup
docker compose restart scheduler       # re-read .env after changing SCHEDULE_*
```

### Missed runs are skipped, not caught up

The jobstore is APScheduler's default in-memory one, so **a run missed while the stack was
down does not happen later**. On start, the cron trigger computes its next fire time from
now. Stack down at 08:00 and back up at 11:00 means there is no run that day; the next one
is 08:00 the following day.

This is the accepted cost of scheduling locally instead of on a hosted runner: the
scheduler only runs while the machine is on and the stack is up. There is deliberately no
wake-on-schedule, system service, or catch-up mechanism. To run a missed day by hand:

```bash
docker compose exec scheduler python -c \
  "import logging; logging.basicConfig(level=logging.INFO); \
   from nlp_pillars.scheduler import run_all_pillars; run_all_pillars()"
```

## GitHub Actions Workflow

### Overview

> **The schedule trigger on this workflow is commented out and the workflow no longer runs
> daily.** The database is Postgres + PostgREST bound to `127.0.0.1` on the owner's machine
> (`docker-compose.yml`), which GitHub's runners cannot reach, so every scheduled run would
> fail. The file, its steps and its `secrets.*` references are kept intact and it can still
> be triggered by hand from the Actions tab. Uncommenting the `schedule:` block re-enables
> it — that is the only change needed — but it is only useful once the database is reachable
> from GitHub's runners again.
>
> Note also that the matrix below still lists `P1`–`P5`. Those pillar IDs no longer exist:
> `create_pillars.py` seeds eight pillars with slug IDs and `cli.py` validates against the
> database, so the matrix would need updating too before this workflow could do useful work.

The workflow was designed to run the NLP learning pipeline daily at **08:00 America/New_York** for all five learning pillars (P1-P5) in parallel.

**Workflow Location:** `.github/workflows/daily.yml`

### DST Handling

Since GitHub Actions only supports UTC cron expressions, we use two separate cron entries to handle Daylight Saving Time transitions:

- **`0 12 * 3-11 *`** → 08:00 EDT (March–November)
- **`0 13 * 12,1,2 *`** → 08:00 EST (December–February)

This ensures the workflow always runs at 08:00 local time in the America/New_York timezone, regardless of whether DST is in effect.

### Required GitHub Secrets

Configure the following secrets in your repository settings (`Settings > Secrets and variables > Actions`):

#### Required Secrets
- **`OPENAI_API_KEY`** - OpenAI API key for LLM operations
- **`SUPABASE_URL`** - Supabase project URL for database operations
- **`SUPABASE_KEY`** - Supabase service role key for database access

#### Optional Secrets
- **`QDRANT_URL`** - Qdrant vector database URL (defaults to localhost if not set)
- **`QDRANT_API_KEY`** - Qdrant API key for cloud instances
- **`SEARXNG_URL`** - SearXNG instance URL for web searches (defaults to localhost if not set)

### Manual Trigger

You can manually trigger the workflow from the GitHub Actions tab:

1. Navigate to **Actions** tab in your repository
2. Select **Daily NLP Pillar Run** workflow
3. Click **Run workflow** button
4. Choose the branch and click **Run workflow**

This will execute the workflow immediately for all pillars, useful for testing or catch-up runs.

### Matrix Strategy

The workflow uses a matrix strategy to run all five pillars (P1-P5) in parallel jobs with `fail-fast: false`, meaning:
- If one pillar fails, others continue running
- Each pillar runs independently with its own logs
- Total execution time is roughly the time of the slowest pillar

## Local Testing with Act

You can test the GitHub Actions workflow locally using [act](https://github.com/nektos/act):

```bash
# Install act (macOS)
brew install act

# Run the workflow locally (requires Docker)
act schedule

# Run with secrets file
act schedule --secret-file .secrets
```

Create a `.secrets` file with your environment variables:
```
OPENAI_API_KEY=your_key_here
SUPABASE_URL=your_url_here
SUPABASE_KEY=your_key_here
```

## System Crontab for Self-Hosted/Local Runs

### Basic Crontab Example

For Linux/macOS systems, you can set up a system cron job to run the workflow locally:

```bash
# Edit crontab
crontab -e

# Add entry for 08:00 daily (single pillar example)
0 8 * * * cd /path/to/NLPWorkflow && /path/to/venv/bin/python -m nlp_pillars.cli run --pillar P1 --papers 1 >> /var/log/nlpworkflow.log 2>&1
```

### Pillar Rotation Examples

#### Daily Rotation (5-day cycle)
```bash
# Monday → P1
0 8 * * 1 cd /path/to/NLPWorkflow && /path/to/venv/bin/python -m nlp_pillars.cli run --pillar P1 --papers 1 >> /var/log/nlpworkflow.log 2>&1

# Tuesday → P2  
0 8 * * 2 cd /path/to/NLPWorkflow && /path/to/venv/bin/python -m nlp_pillars.cli run --pillar P2 --papers 1 >> /var/log/nlpworkflow.log 2>&1

# Wednesday → P3
0 8 * * 3 cd /path/to/NLPWorkflow && /path/to/venv/bin/python -m nlp_pillars.cli run --pillar P3 --papers 1 >> /var/log/nlpworkflow.log 2>&1

# Thursday → P4
0 8 * * 4 cd /path/to/NLPWorkflow && /path/to/venv/bin/python -m nlp_pillars.cli run --pillar P4 --papers 1 >> /var/log/nlpworkflow.log 2>&1

# Friday → P5
0 8 * * 5 cd /path/to/NLPWorkflow && /path/to/venv/bin/python -m nlp_pillars.cli run --pillar P5 --papers 1 >> /var/log/nlpworkflow.log 2>&1
```

#### Weekly Rotation (all pillars, one day per week)
```bash
# Sunday → Run all pillars
0 8 * * 0 cd /path/to/NLPWorkflow && for pillar in P1 P2 P3 P4 P5; do /path/to/venv/bin/python -m nlp_pillars.cli run --pillar $pillar --papers 1; done >> /var/log/nlpworkflow.log 2>&1
```

### Environment Setup for Cron

Since cron runs with a minimal environment, you may need to source your environment variables:

```bash
# Create a script: ~/scripts/run_nlp_workflow.sh
#!/bin/bash
source /path/to/.env
cd /path/to/NLPWorkflow
/path/to/venv/bin/python -m nlp_pillars.cli run --pillar "$1" --papers 1

# Make executable
chmod +x ~/scripts/run_nlp_workflow.sh

# Use in crontab
0 8 * * * ~/scripts/run_nlp_workflow.sh P1 >> /var/log/nlpworkflow.log 2>&1
```

### Log Management

For production deployments, consider log rotation:

```bash
# Add to /etc/logrotate.d/nlpworkflow
/var/log/nlpworkflow.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 username username
}
```

## Troubleshooting

### Common Issues

1. **Missing Secrets**: Ensure all required secrets are configured in GitHub repository settings
2. **Permission Errors**: Verify the service account keys have proper permissions
3. **Rate Limits**: OpenAI and other APIs have rate limits; consider spacing out pillars if needed
4. **Timezone Confusion**: Remember that cron times are in UTC for GitHub Actions, local time for system cron

### Monitoring

- **GitHub Actions**: Check the Actions tab for workflow run history and logs
- **Local Cron**: Monitor `/var/log/nlpworkflow.log` for execution logs
- **Database**: Use the CLI `status` command to verify papers are being processed

### Manual Execution

Test individual pillars manually:

```bash
# Test a single pillar
python -m nlp_pillars.cli run --pillar P1 --papers 1

# Check status
python -m nlp_pillars.cli status --pillar P1

# Review due cards
python -m nlp_pillars.cli review --pillar P1
```
