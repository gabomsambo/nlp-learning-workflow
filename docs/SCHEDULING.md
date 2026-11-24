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

**Turn it on and off on the Daily run page** (`/daily-run/`, under Workspace in the
sidebar). That page sets everything about the run: on/off, time, timezone, which pillars
take part and how many papers each gets (the pillar's own `papers_per_day`). It also shows
the next run, the last run with links to each pillar's pipeline run, an estimated daily
cost, the daily spend cap, and a **Run now** button. Nothing runs on its own until it is
turned on there.

The settings live in the `daily_run_settings` row (`docs/migrations/039_daily_run_settings.sql`,
applied by hand). The scheduler re-reads it every minute and again when the job fires, so a
change takes effect within a minute with no rebuild or restart.

While the run is **on**, one APScheduler instance runs:

1. **The daily learning run**, at the page's time in its timezone, for the ticked pillars.
   Pillars run one after another and independently — one failing does not stop the rest.
   Each pillar is a `pipeline_runs` row (trigger `scheduler`), visible on Activity. Today's
   spend is checked against `DAILY_SPEND_CAP_USD` before every pillar; once the cap is
   reached the remaining pillars are skipped and say so.
2. **The FSRS optimization and maintenance jobs** in `nlp_pillars/services/background_jobs.py`.
3. **The weekly map refresh**, when `WEEKLY_MAP_REFRESH_ENABLED` is true.

While it is **off**, the process stays up and idles: no daily job, the FSRS jobs paused,
no weekly refresh. `docker compose logs scheduler` says `Daily run is OFF`.

### Settings and precedence

The precedence is documented once, in `nlp_pillars/daily_run.py`:

1. The `daily_run_settings` row, once it exists, wins.
2. With no row yet, `SCHEDULE_ENABLED` (default `false`), `SCHEDULE_TIME` (`08:00`) and
   `SCHEDULE_TIMEZONE` (`UTC`) from `.env` apply, with every pillar taking part. The
   scheduler copies them into the row when it starts (and so does the first save from the
   page); after that, editing them in `.env` does nothing and the scheduler logs a warning
   if they disagree with the row.
3. If migration 039 is not applied, the `.env` values apply as before and the page says so.

`PAPERS_PER_DAY` is no longer read by the daily run. An unparseable time or unknown
timezone is never guessed at: the run is not scheduled and the page shows why.

The service uses `restart: unless-stopped`. It no longer exits when the run is off, so a
stopped scheduler container means a crash or a manual stop.

```bash
docker compose up -d scheduler         # start it
docker compose logs -f scheduler       # watch it; it logs each change it applies
```

### Missed runs are skipped, not caught up

The jobstore is APScheduler's default in-memory one, so **a run missed while the stack was
down does not happen later**. On start, the cron trigger computes its next fire time from
now. Stack down at 08:00 and back up at 11:00 means there is no run that day; the next one
is 08:00 the following day.

This is the accepted cost of scheduling locally instead of on a hosted runner: the
scheduler only runs while the machine is on and the stack is up. There is deliberately no
wake-on-schedule, system service, or catch-up mechanism. To run a missed day by hand, use
**Run now** on the Daily run page.

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
