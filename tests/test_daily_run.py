"""The daily-run settings: precedence, seeding, saving, cost (nlp_pillars/daily_run.py).

A small in-memory stand-in for PostgREST answers the queries daily_run makes.
Nothing here reaches a database.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import httpx
import pytest

from nlp_pillars import daily_run
from nlp_pillars.daily_run import (
    DailyRunConfig,
    DailyRunMigrationMissing,
    DailyRunSettingsError,
)

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (ROOT / "docs" / "migrations" / "039_daily_run_settings.sql").read_text()
SCHEMA = (ROOT / "schema.sql").read_text()
MISSING = {"code": "PGRST205", "message": "Could not find the table 'public.daily_run_settings'"}


class FakeQuery:
    def __init__(self, store, name):
        self.store, self.name, self.params = store, name, {}

    def select(self, columns="*"):
        return self

    def eq(self, column, value):
        self.params[column] = f"eq.{value}"
        return self

    def in_(self, column, values):
        self.params[column] = values
        return self

    def order(self, column, desc=False):
        return self

    def limit(self, count):
        return self

    def _rows(self):
        return self.store.tables.setdefault(self.name, [])

    def execute(self):
        if self.name in self.store.missing:
            return {"data": None, "error": MISSING}
        if self.store.read_error:
            return {"data": None, "error": {"message": self.store.read_error}}
        return {"data": [dict(r) for r in self._rows()], "error": None}

    def insert(self, row):
        if self.name in self.store.missing:
            return {"data": None, "error": MISSING}
        if self._rows():
            return {"data": None, "error": {"code": 409, "message": "Resource already exists"}}
        stored = {"enabled": False, "run_time": "08:00", "timezone": "UTC", "pillar_ids": None,
                  "scheduler_seen_at": None, "last_run": None, **row}
        self._rows().append(stored)
        self.store.writes.append(("insert", dict(row)))
        return {"data": [dict(stored)], "error": None}

    def update(self, patch_):
        if self.name in self.store.missing:
            response = httpx.Response(404, text=str(MISSING), request=httpx.Request("PATCH", "http://x"))
            raise httpx.HTTPStatusError("404", request=response.request, response=response)
        self.store.writes.append(("update", dict(patch_)))
        for row in self._rows():
            row.update(patch_)
        return {"data": [dict(r) for r in self._rows()], "error": None}


class FakeClient:
    def __init__(self, row=None, missing=False):
        self.tables = {daily_run.TABLE: [row] if row else []}
        self.missing = {daily_run.TABLE} if missing else set()
        self.read_error = None
        self.writes = []

    def table(self, name):
        return FakeQuery(self, name)

    @property
    def row(self):
        rows = self.tables[daily_run.TABLE]
        return rows[0] if rows else None


def _settings(enabled=False, time="08:00", tz="UTC"):
    return SimpleNamespace(schedule_enabled=enabled, schedule_time=time, schedule_timezone=tz)


@pytest.fixture
def env():
    """SCHEDULE_* as the env would give them; mutate the namespace to change them."""
    values = _settings()
    with patch.object(daily_run, "get_settings", return_value=values):
        yield values


def _use(client):
    return patch.object(daily_run.db, "get_client", return_value=client)


# ---------------------------------------------------------------- precedence

class TestPrecedence:
    def test_no_table_falls_back_to_env_and_says_so(self, env):
        env.schedule_enabled = True
        with _use(FakeClient(missing=True)):
            config = daily_run.load_config()
        assert config.enabled is True
        assert config.source == daily_run.SOURCE_ENV_NO_TABLE
        assert config.migration_missing
        assert config.pillar_ids is None

    def test_no_row_uses_env_defaults(self, env):
        env.schedule_time, env.schedule_timezone = "06:15", "Europe/Berlin"
        with _use(FakeClient()):
            config = daily_run.load_config()
        assert (config.enabled, config.run_time, config.timezone) == (False, "06:15", "Europe/Berlin")
        assert config.source == daily_run.SOURCE_ENV

    def test_the_row_wins_over_env(self, env):
        env.schedule_enabled = True
        row = {"enabled": False, "run_time": "21:30", "timezone": "Asia/Tokyo",
               "pillar_ids": ["a"], "last_run": {"trigger": "scheduled"}}
        with _use(FakeClient(row)):
            config = daily_run.load_config()
        assert config.enabled is False
        assert (config.run_time, config.timezone, config.pillar_ids) == ("21:30", "Asia/Tokyo", ("a",))
        assert config.source == daily_run.SOURCE_DATABASE
        assert config.last_run == {"trigger": "scheduled"}

    def test_an_unreadable_database_raises_rather_than_guessing(self, env):
        client = FakeClient()
        client.read_error = "connection refused"
        with _use(client), pytest.raises(DailyRunSettingsError):
            daily_run.load_config()

    def test_defaults_are_off(self):
        from nlp_pillars.config import Settings

        assert Settings.model_fields["schedule_enabled"].default is False
        assert "enabled           BOOLEAN NOT NULL DEFAULT false" in MIGRATION


# ---------------------------------------------------------------- seeding

class TestSeed:
    def test_first_boot_copies_env_into_the_row(self, env):
        env.schedule_enabled, env.schedule_time = True, "07:45"
        client = FakeClient()
        with _use(client):
            config = daily_run.seed_from_env()
        assert client.row["enabled"] is True and client.row["run_time"] == "07:45"
        assert client.row["pillar_ids"] is None
        assert config.source == daily_run.SOURCE_DATABASE

    def test_an_existing_row_is_left_alone(self, env):
        env.schedule_enabled = True
        client = FakeClient({"enabled": False, "run_time": "09:00", "timezone": "UTC"})
        with _use(client):
            config = daily_run.seed_from_env()
        assert client.writes == []
        assert config.enabled is False

    def test_a_broken_env_time_seeds_off_instead_of_a_broken_row(self, env):
        env.schedule_enabled, env.schedule_time = True, "25:99"
        client = FakeClient()
        with _use(client):
            daily_run.seed_from_env()
        assert client.writes == [("insert", {"id": 1, "enabled": False})]
        assert client.row["run_time"] == "08:00"

    def test_missing_table_raises_with_the_migration_name(self, env):
        with _use(FakeClient(missing=True)), pytest.raises(DailyRunMigrationMissing, match="039"):
            daily_run.seed_from_env()


# ---------------------------------------------------------------- validate / save

class TestValidateAndSave:
    def test_validate_normalises(self):
        fields = daily_run.validate(
            enabled=True, run_time="7:05", timezone_name=" Europe/Berlin ",
            pillar_ids=["a", "a", " b "], known_pillar_ids=["a", "b"],
        )
        assert fields == {"enabled": True, "run_time": "07:05", "timezone": "Europe/Berlin",
                          "pillar_ids": ["a", "b"]}

    @pytest.mark.parametrize("kwargs, message", [
        ({"run_time": "24:00"}, "out of range"),
        ({"run_time": "noon"}, "HH:MM"),
        ({"timezone_name": "Mars/Olympus_Mons"}, "IANA"),
        ({"pillar_ids": ["ghost"]}, "Unknown pillar"),
        ({"enabled": "yes"}, "true or false"),
    ])
    def test_validate_rejects(self, kwargs, message):
        base = dict(enabled=False, run_time="08:00", timezone_name="UTC",
                    pillar_ids=["a"], known_pillar_ids=["a"])
        base.update(kwargs)
        with pytest.raises(ValueError, match=message):
            daily_run.validate(**base)

    def test_save_creates_the_row_when_there_is_none(self, env):
        client = FakeClient()
        fields = daily_run.validate(enabled=True, run_time="06:00", timezone_name="UTC",
                                    pillar_ids=["a"])
        with _use(client):
            config = daily_run.save_config(fields)
        assert config.enabled is True and config.pillar_ids == ("a",)
        assert client.row["run_time"] == "06:00"

    def test_save_updates_an_existing_row(self, env):
        client = FakeClient({"enabled": False, "run_time": "08:00", "timezone": "UTC"})
        fields = daily_run.validate(enabled=False, run_time="22:10", timezone_name="UTC",
                                    pillar_ids=[])
        with _use(client):
            config = daily_run.save_config(fields)
        assert config.run_time == "22:10"
        assert config.pillar_ids == ()
        assert [w[0] for w in client.writes] == ["update"]

    def test_save_without_the_migration_raises(self, env):
        fields = daily_run.validate(enabled=False, run_time="08:00", timezone_name="UTC",
                                    pillar_ids=None)
        with _use(FakeClient(missing=True)), pytest.raises(DailyRunMigrationMissing):
            daily_run.save_config(fields)

    def test_heartbeat_and_last_run_never_raise(self, env):
        with _use(FakeClient(missing=True)):
            daily_run.touch_scheduler_seen()
            daily_run.record_last_run({"trigger": "scheduled"})
        boom = MagicMock(side_effect=RuntimeError("down"))
        with patch.object(daily_run.db, "get_client", boom):
            daily_run.touch_scheduler_seen()
            daily_run.record_last_run({})

    def test_close_unfinished_scheduled_batch(self, env):
        client = FakeClient({"enabled": True, "run_time": "08:00", "timezone": "UTC",
                             "last_run": {"trigger": "scheduled", "finished_at": None, "runs": []}})
        with _use(client):
            daily_run.close_unfinished_last_run("stopped")
        assert client.row["last_run"]["finished_at"]
        assert client.row["last_run"]["message"] == "stopped"

    def test_close_leaves_a_finished_batch_alone(self, env):
        last = {"trigger": "scheduled", "finished_at": "2026-11-24T08:05:00+00:00"}
        client = FakeClient({"enabled": True, "run_time": "08:00", "timezone": "UTC",
                             "last_run": last})
        with _use(client):
            daily_run.close_unfinished_last_run("stopped")
        assert client.writes == []


# ---------------------------------------------------------------- schedule maths

def _config(**kw):
    base = dict(enabled=True, run_time="08:00", timezone="UTC", pillar_ids=None,
                source=daily_run.SOURCE_DATABASE)
    base.update(kw)
    return DailyRunConfig(**base)


class TestNextRun:
    NOW = datetime(2026, 11, 24, 7, 0, tzinfo=timezone.utc)

    def test_later_today(self):
        assert _config().next_run_at(self.NOW) == datetime(2026, 11, 24, 8, 0, tzinfo=timezone.utc)

    def test_tomorrow_once_today_has_passed(self):
        now = datetime(2026, 11, 24, 8, 0, 1, tzinfo=timezone.utc)
        assert _config().next_run_at(now) == datetime(2026, 11, 25, 8, 0, tzinfo=timezone.utc)

    def test_in_the_configured_timezone(self):
        nxt = _config(run_time="09:30", timezone="America/New_York").next_run_at(self.NOW)
        local = nxt.astimezone(ZoneInfo("America/New_York"))
        assert (local.hour, local.minute) == (9, 30)
        assert nxt == datetime(2026, 11, 24, 14, 30, tzinfo=timezone.utc)

    def test_none_when_off_or_misconfigured(self):
        assert _config(enabled=False).next_run_at(self.NOW) is None
        assert _config(timezone="Nowhere/Land").next_run_at(self.NOW) is None

    def test_participation(self):
        pillars = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
        assert daily_run.participating(_config(), pillars) == pillars
        assert daily_run.participating(_config(pillar_ids=("b",)), pillars) == [pillars[1]]
        assert daily_run.participating(_config(pillar_ids=()), pillars) == []


# ---------------------------------------------------------------- cost

class TestCost:
    def test_documented_estimate(self):
        estimate = daily_run.estimate_daily_cost([2, 1])
        assert estimate["basis"] == "documented"
        assert estimate["papers"] == 3 and estimate["pillars"] == 2
        expected = 3 * daily_run.DOCUMENTED_COST_PER_PAPER_USD + 2 * daily_run.DOCUMENTED_DISCOVERY_COST_USD
        assert estimate["total_usd"] == round(expected, 4)

    def test_measured_estimate(self):
        estimate = daily_run.estimate_daily_cost([4], measured=(0.01, 7))
        assert estimate["basis"] == "measured" and estimate["sample_size"] == 7
        assert estimate["total_usd"] == round(0.04 + daily_run.DOCUMENTED_DISCOVERY_COST_USD, 4)

    def test_measured_cost_is_the_median_of_single_paper_runs_without_discovery(self):
        rows = [
            {"pipeline_run_stages": [
                {"name": "discovery", "detail": "haiku · $0.000100"},
                {"name": "summarize", "detail": "haiku · 1 in · 1 out · $0.010000"},
                {"name": "quiz", "detail": "haiku · $0.002000"},
            ]},
            {"pipeline_run_stages": [{"name": "summarize", "detail": "$0.020000"}]},
            {"pipeline_run_stages": [{"name": "summarize", "detail": "$0.030000"}]},
            {"pipeline_run_stages": [{"name": "persist", "detail": "no cost here"}]},
        ]
        client = MagicMock()
        query = client.table.return_value.select.return_value
        query.eq.return_value = query
        query.order.return_value = query
        query.limit.return_value = query
        query.params = {}
        query.execute.return_value = {"data": rows, "error": None}
        with patch.object(daily_run.db, "get_client", return_value=client):
            median, sample = daily_run.measured_cost_per_paper()
        assert sample == 3
        assert median == pytest.approx(0.02)
        assert query.params["kind"] == "in.(run_daily,process_selected,upload)"

    def test_measured_cost_is_none_when_unreadable(self):
        with patch.object(daily_run.db, "get_client", side_effect=RuntimeError("down")):
            assert daily_run.measured_cost_per_paper() is None


# ---------------------------------------------------------------- migration

class TestMigration:
    def test_is_idempotent_granted_reloaded_and_hand_applied(self):
        assert "CREATE TABLE IF NOT EXISTS daily_run_settings" in MIGRATION
        assert "GRANT SELECT, INSERT, UPDATE, DELETE ON daily_run_settings TO web_anon" in MIGRATION
        assert "NOTIFY pgrst, 'reload schema'" in MIGRATION
        assert "docker exec -i nlp_postgres psql" in MIGRATION
        assert "CHECK (id = 1)" in MIGRATION

    def test_seeds_no_row_so_env_stays_the_first_boot_default(self):
        assert not re.search(r"^\s*INSERT\s+INTO", MIGRATION, re.I | re.M)

    def test_schema_sql_carries_the_same_table(self):
        block = re.search(r"CREATE TABLE IF NOT EXISTS daily_run_settings \((.*?)\n\);", SCHEMA, re.S)
        assert block, "schema.sql lacks daily_run_settings"
        mig = re.search(r"CREATE TABLE IF NOT EXISTS daily_run_settings \((.*?)\n\);", MIGRATION, re.S)
        columns = lambda text: re.findall(r"^\s{4}([a-z_]+)\s+[A-Z]", text, re.M)  # noqa: E731
        assert columns(block.group(1)) == columns(mig.group(1))
        assert "enabled           BOOLEAN NOT NULL DEFAULT false" in block.group(1)

    @pytest.mark.parametrize("value, ok", [
        ("00:00", True), ("08:30", True), ("23:59", True),
        ("24:00", False), ("8:30", False), ("08:60", False), ("08:00:00", False),
    ])
    def test_run_time_check_agrees_with_the_app(self, value, ok):
        """The CHECK's regex admits exactly what normalise_run_time produces."""
        pattern = re.search(r"run_time ~ '(.+?)'", MIGRATION).group(1)
        assert bool(re.fullmatch(pattern, value)) is ok
        if ok:
            assert daily_run.normalise_run_time(value) == value
