"""Daily spend: reading the cost the app already records, and the cap on it."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from nlp_pillars import spend


class FakeQuery:
    def __init__(self, table, rows, error, log):
        self.table, self.rows, self.error, self.log = table, rows, error, log
        self.params = {}

    def select(self, columns="*"):
        self.params["select"] = columns
        return self

    def limit(self, count):
        self.params["limit"] = str(count)
        return self

    def execute(self):
        self.log.append((self.table, dict(self.params)))
        if self.error:
            return {"data": None, "error": self.error}
        return {"data": self.rows, "error": None}


class FakeClient:
    def __init__(self, stages=(), checks=(), stage_error=None, check_error=None):
        self.stages, self.checks = list(stages), list(checks)
        self.stage_error, self.check_error = stage_error, check_error
        self.log = []

    def table(self, name):
        if name == "pipeline_run_stages":
            return FakeQuery(name, self.stages, self.stage_error, self.log)
        assert name == "podcast_fact_checks"
        return FakeQuery(name, self.checks, self.check_error, self.log)


NOON = datetime(2026, 10, 25, 12, 30, tzinfo=timezone.utc)


def test_stage_details_are_summed():
    details = [
        "claude-haiku-5-5 · 61,204 in · 1,880 out · $0.009500",
        "claude-haiku-5-5 · 2,100 in · 900 out · $0.000660",
        "chunking 12 chunks",
        None,
        "",
    ]
    assert spend.parse_stage_costs(details) == pytest.approx(0.01016)


def test_a_dollar_sign_without_digits_is_not_a_cost():
    assert spend.parse_stage_costs(["costs $ nothing", "$.5", "price in $US"]) == 0


def test_spend_adds_stages_and_factchecks_since_the_day_started():
    client = FakeClient(
        stages=[{"detail": "m · $0.500000"}, {"detail": "m · $0.250000"}],
        checks=[{"cost_usd": "0.75"}, {"cost_usd": None}],
    )
    with patch.object(spend.db, "get_client", return_value=client):
        budget = spend.current_budget(NOON)

    assert budget.spent_usd == pytest.approx(1.5)
    assert budget.cap_usd == 2.0
    assert budget.remaining_usd == pytest.approx(0.5)
    assert not budget.exhausted
    stage_query = next(p for t, p in client.log if t == "pipeline_run_stages")
    assert stage_query["finished_at"].startswith("gte.2026-10-25T00:00:00")
    assert stage_query["detail"] == "like.*$*"
    check_query = next(p for t, p in client.log if t == "podcast_fact_checks")
    assert check_query["created_at"].startswith("gte.2026-10-25T00:00:00")


def test_cap_is_reached_at_exactly_the_cap():
    client = FakeClient(stages=[{"detail": "m · $2.000000"}])
    with patch.object(spend.db, "get_client", return_value=client):
        budget = spend.current_budget(NOON)
    assert budget.exhausted
    assert budget.remaining_usd == 0


def test_cap_defaults_to_two_dollars_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv("DAILY_SPEND_CAP_USD", raising=False)
    assert spend.daily_cap_usd() == 2.0
    monkeypatch.setenv("DAILY_SPEND_CAP_USD", "5.5")
    assert spend.daily_cap_usd() == 5.5
    monkeypatch.setenv("DAILY_SPEND_CAP_USD", "0")
    assert spend.daily_cap_usd() == 0


@pytest.mark.parametrize("raw", ["", "lots", "-1", "nan"])
def test_unusable_cap_falls_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv("DAILY_SPEND_CAP_USD", raw)
    assert spend.daily_cap_usd() == 2.0


def test_a_zero_cap_refuses_everything(monkeypatch):
    monkeypatch.setenv("DAILY_SPEND_CAP_USD", "0")
    with patch.object(spend.db, "get_client", return_value=FakeClient()):
        assert spend.current_budget(NOON).exhausted


def test_unreadable_stages_raise_instead_of_reading_as_zero():
    client = FakeClient(stage_error={"message": "connection refused"})
    with patch.object(spend.db, "get_client", return_value=client):
        with pytest.raises(spend.SpendUnavailableError):
            spend.current_budget(NOON)


def test_missing_factcheck_table_counts_as_zero():
    client = FakeClient(
        stages=[{"detail": "m · $0.100000"}],
        check_error={"code": "PGRST205", "message": "Could not find the table"},
    )
    with patch.object(spend.db, "get_client", return_value=client):
        assert spend.current_budget(NOON).spent_usd == pytest.approx(0.1)


def test_other_factcheck_errors_raise():
    client = FakeClient(check_error={"message": "permission denied"})
    with patch.object(spend.db, "get_client", return_value=client):
        with pytest.raises(spend.SpendUnavailableError):
            spend.current_budget(NOON)


def test_client_that_cannot_be_built_raises():
    with patch.object(spend.db, "get_client", side_effect=ValueError("SUPABASE_URL required")):
        with pytest.raises(spend.SpendUnavailableError):
            spend.current_budget(NOON)


def test_day_boundary_follows_the_configured_timezone(monkeypatch):
    monkeypatch.setenv("SPEND_CAP_TIMEZONE", "America/New_York")
    start, resets = spend.day_bounds(datetime(2026, 10, 25, 3, 0, tzinfo=timezone.utc))
    # 03:00 UTC is 23:00 on the 24th in New York (EDT, UTC-4).
    assert start.astimezone(timezone.utc) == datetime(2026, 10, 24, 4, 0, tzinfo=timezone.utc)
    # The clocks go back on 1 Nov; the 25th is still EDT, so the day is 24 h.
    assert (resets - start).total_seconds() == 24 * 3600


def test_unknown_timezone_falls_back_to_utc(monkeypatch):
    monkeypatch.setenv("SPEND_CAP_TIMEZONE", "Mars/Olympus")
    start, _ = spend.day_bounds(NOON)
    assert start == datetime(2026, 10, 25, tzinfo=timezone.utc)
