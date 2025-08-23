# Scheduling Guide

This document explains how to set up automated daily runs of the NLP Learning Workflow using GitHub Actions or system cron jobs.

## GitHub Actions Workflow

### Overview

The automated scheduling system runs the NLP learning pipeline daily at **08:00 America/New_York** for all five learning pillars (P1-P5) in parallel.

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
