"""The Daily run page and its API. Database, spend and dispatch are mocked."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nlp_pillars import daily_run, spend
from nlp_pillars.daily_run import DailyRunConfig
from webui.app import create_app
from webui.routers.api import daily_run as api
from webui.services.run_service import RunAlreadyActiveError

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 24, 7, 0, tzinfo=timezone.utc)
START = datetime(2026, 11, 24, tzinfo=timezone.utc)


def _config(**kw):
    base = dict(enabled=False, run_time="08:00", timezone="UTC", pillar_ids=None,
                source=daily_run.SOURCE_DATABASE)
    base.update(kw)
    return DailyRunConfig(**base)


def _pillars():
    return [
        SimpleNamespace(id="nlp-a", name="Pillar A", papers_per_day=2),
        SimpleNamespace(id="nlp-b", name="Pillar B", papers_per_day=1),
    ]


def _budget(spent=0.25, cap=2.0):
    return spend.Budget(cap_usd=cap, spent_usd=spent, day_start=START,
                        resets_at=START + timedelta(days=1))


@pytest.fixture
def world():
    """Patch every read build_status makes; yield the mocks to adjust."""
    mocks = SimpleNamespace()
    patches = {
        "load": patch.object(daily_run, "load_config", return_value=_config()),
        "pillars": patch.object(api.db, "get_pillars", return_value=_pillars()),
        "budget": patch.object(spend, "current_budget", return_value=_budget()),
        "measured": patch.object(daily_run, "measured_cost_per_paper", return_value=None),
        "runs": patch.object(api, "_fetch_runs", return_value={}),
    }
    for name, p in patches.items():
        setattr(mocks, name, p.start())
    yield mocks
    for p in patches.values():
        p.stop()


@pytest.fixture
def client():
    # Not entered as a context manager, so the lifespan (and its real
    # APScheduler) never starts; dispatch_run is mocked wherever it is reached.
    app = create_app()
    app.state.scheduler = MagicMock()
    app.state.cancel_events = {}
    return TestClient(app)


# ---------------------------------------------------------------- status

class TestStatus:
    def test_off_by_default_shows_no_next_run(self, world):
        status = api.build_status(NOW)
        assert status["config"]["enabled"] is False
        assert status["next_run_at"] is None
        assert [p["included"] for p in status["pillars"]] == [True, True]

    def test_on_shows_next_run_in_its_timezone(self, world):
        world.load.return_value = _config(enabled=True, run_time="09:30", timezone="America/New_York")
        status = api.build_status(NOW)
        assert status["next_run_at"] == "2026-11-24T09:30:00-05:00"
        assert "09:30 EST" in status["next_run_display"]

    def test_estimate_covers_only_participating_pillars(self, world):
        world.load.return_value = _config(pillar_ids=("nlp-a",))
        status = api.build_status(NOW)
        assert status["estimate"]["papers"] == 2 and status["estimate"]["pillars"] == 1
        assert status["estimate"]["basis"] == "documented"
        assert [p["included"] for p in status["pillars"]] == [True, False]

    def test_measured_estimate_is_preferred(self, world):
        world.measured.return_value = (0.01, 5)
        status = api.build_status(NOW)
        assert status["estimate"]["basis"] == "measured"
        assert status["estimate"]["total_usd"] == round(3 * 0.01 + 2 * 0.00012, 4)

    def test_cap_and_spend_are_shown_and_an_unreadable_spend_is_not_zero(self, world):
        status = api.build_status(NOW)
        assert status["budget"]["cap_usd"] == 2.0 and status["budget"]["spent_usd"] == 0.25
        world.budget.side_effect = spend.SpendUnavailableError("down")
        status = api.build_status(NOW)
        assert status["budget"]["spent_usd"] is None
        assert "down" in status["budget"]["error"]

    def test_estimate_over_cap_is_flagged(self, world):
        world.budget.return_value = _budget(cap=0.01)
        assert api.build_status(NOW)["estimate_over_cap"] is True

    def test_scheduler_heartbeat(self, world):
        world.load.return_value = _config(scheduler_seen_at=(NOW - timedelta(seconds=40)).isoformat())
        assert api.build_status(NOW)["scheduler"]["stale"] is False
        world.load.return_value = _config(scheduler_seen_at=(NOW - timedelta(minutes=10)).isoformat())
        assert api.build_status(NOW)["scheduler"]["stale"] is True
        world.load.return_value = _config()
        assert api.build_status(NOW)["scheduler"]["stale"] is True

    def test_last_run_reads_live_statuses_and_links_each_run(self, world):
        world.load.return_value = _config(last_run={
            "trigger": "scheduled", "started_at": "2026-11-24T08:00:00+00:00",
            "finished_at": "2026-11-24T08:20:00+00:00", "message": None,
            "runs": [
                {"pillar_id": "nlp-a", "run_id": "r1"},
                {"pillar_id": "nlp-b", "run_id": None, "skipped": "Daily spend cap reached"},
            ],
        })
        world.runs.return_value = {"r1": {"id": "r1", "status": "succeeded", "papers_processed": 2}}
        last = api.build_status(NOW)["last_run"]
        assert last["outcome"] == "partial"
        assert last["papers_processed"] == 2
        assert last["runs"][0]["url"] == "/pipeline/?pillar=nlp-a&run=r1"
        assert last["runs"][1] == {
            "pillar_id": "nlp-b", "run_id": None, "url": None, "status": "skipped",
            "papers_processed": None, "error": "Daily spend cap reached",
        }

    @pytest.mark.parametrize("statuses, finished, outcome", [
        (["succeeded", "succeeded"], True, "succeeded"),
        (["running", "succeeded"], True, "running"),
        (["succeeded"], False, "running"),
        (["failed", "interrupted"], True, "failed"),
    ])
    def test_last_run_outcome(self, world, statuses, finished, outcome):
        ids = [f"r{i}" for i in range(len(statuses))]
        world.load.return_value = _config(last_run={
            "trigger": "run_now", "started_at": NOW.isoformat(),
            "finished_at": NOW.isoformat() if finished else None,
            "runs": [{"pillar_id": "nlp-a", "run_id": i} for i in ids],
        })
        world.runs.return_value = {i: {"id": i, "status": s} for i, s in zip(ids, statuses, strict=True)}
        assert api.build_status(NOW)["last_run"]["outcome"] == outcome

    def test_unreadable_settings_fall_back_to_env_and_say_so(self, world):
        world.load.side_effect = daily_run.DailyRunSettingsError("db down")
        with patch.object(daily_run, "get_settings", return_value=SimpleNamespace(
                schedule_enabled=False, schedule_time="08:00", schedule_timezone="UTC")):
            status = api.build_status(NOW)
        assert "db down" in status["settings_error"]

    def test_get_endpoint(self, world, client):
        response = client.get("/api/daily-run")
        assert response.status_code == 200
        assert response.json()["config"]["enabled"] is False


# ---------------------------------------------------------------- save

def _body(**kw):
    base = {"enabled": True, "run_time": "07:30", "timezone": "Europe/Berlin",
            "pillar_ids": ["nlp-a"], "papers_per_day": {"nlp-a": 3, "nlp-b": 1}}
    base.update(kw)
    return base


class TestSave:
    def test_saves_settings_and_only_changed_papers_per_day(self, world, client):
        with patch.object(daily_run, "save_config") as save, \
             patch.object(api.db, "update_pillar", return_value=MagicMock()) as update:
            response = client.put("/api/daily-run", json=_body())
        assert response.status_code == 200, response.text
        save.assert_called_once_with({"enabled": True, "run_time": "07:30",
                                      "timezone": "Europe/Berlin", "pillar_ids": ["nlp-a"]})
        assert update.call_count == 1
        assert update.call_args.args[0] == "nlp-a"
        assert update.call_args.args[1].papers_per_day == 3

    @pytest.mark.parametrize("override, message", [
        ({"run_time": "25:00"}, "out of range"),
        ({"timezone": "Atlantis/Lost"}, "IANA"),
        ({"pillar_ids": ["ghost"]}, "Unknown pillar"),
        ({"papers_per_day": {"nlp-a": 11}}, "1 to 10"),
        ({"papers_per_day": {"nlp-a": 0}}, "1 to 10"),
        ({"papers_per_day": {"ghost": 2}}, "Unknown pillar"),
    ])
    def test_rejects_bad_input_before_writing(self, world, client, override, message):
        with patch.object(daily_run, "save_config") as save:
            response = client.put("/api/daily-run", json=_body(**override))
        assert response.status_code == 422
        assert message in response.json()["detail"]
        save.assert_not_called()

    def test_missing_migration_is_503_naming_the_file(self, world, client):
        with patch.object(daily_run, "save_config", side_effect=daily_run.DailyRunMigrationMissing()):
            response = client.put("/api/daily-run", json=_body())
        assert response.status_code == 503
        assert "039_daily_run_settings.sql" in response.json()["detail"]

    def test_unreadable_pillars_is_503(self, world, client):
        world.pillars.side_effect = api.db.PillarLookupError("down")
        response = client.put("/api/daily-run", json=_body())
        assert response.status_code == 503

    def test_a_failed_papers_update_is_reported(self, world, client):
        with patch.object(daily_run, "save_config"), \
             patch.object(api.db, "update_pillar", return_value=None):
            response = client.put("/api/daily-run", json=_body())
        assert response.status_code == 502
        assert "nlp-a" in response.json()["detail"]


# ---------------------------------------------------------------- run now

class TestRunNow:
    def test_dispatches_one_run_per_participating_pillar_and_records_it(self, world, client):
        world.load.return_value = _config(pillar_ids=("nlp-a", "nlp-b"))
        with patch.object(api, "dispatch_run", side_effect=["r-a", RunAlreadyActiveError("busy")]) as dispatch, \
             patch.object(daily_run, "record_last_run") as record:
            response = client.post("/api/daily-run/run-now")
        assert response.status_code == 202, response.text
        calls = dispatch.call_args_list
        assert [(c.kwargs["pillar_id"], c.kwargs["papers_limit"]) for c in calls] == [("nlp-a", 2), ("nlp-b", 1)]
        assert all(c.kwargs["kind"] == "run_daily" for c in calls)
        assert all(c.kwargs["trigger_source"] == "ui_pipeline" for c in calls)
        runs = response.json()["runs"]
        assert runs[0]["url"] == "/pipeline/?pillar=nlp-a&run=r-a"
        assert runs[1]["run_id"] is None and "already in progress" in runs[1]["skipped"]
        payload = record.call_args.args[0]
        assert payload["trigger"] == "run_now"
        assert [r["run_id"] for r in payload["runs"]] == ["r-a", None]

    def test_works_while_the_scheduled_run_is_off(self, world, client):
        world.load.return_value = _config(enabled=False, pillar_ids=("nlp-b",))
        with patch.object(api, "dispatch_run", return_value="r-b") as dispatch, \
             patch.object(daily_run, "record_last_run"):
            response = client.post("/api/daily-run/run-now")
        assert response.status_code == 202
        assert dispatch.call_count == 1

    def test_no_participating_pillars_is_400(self, world, client):
        world.load.return_value = _config(pillar_ids=())
        with patch.object(api, "dispatch_run") as dispatch:
            response = client.post("/api/daily-run/run-now")
        assert response.status_code == 400
        dispatch.assert_not_called()

    def test_spend_cap_reached_starts_nothing(self, world, client):
        world.budget.return_value = _budget(spent=2.0)
        with patch.object(api, "dispatch_run") as dispatch:
            response = client.post("/api/daily-run/run-now")
        assert response.status_code == 429
        dispatch.assert_not_called()

    def test_unreadable_spend_starts_nothing(self, world, client):
        world.budget.side_effect = spend.SpendUnavailableError("down")
        with patch.object(api, "dispatch_run") as dispatch:
            response = client.post("/api/daily-run/run-now")
        assert response.status_code == 503
        dispatch.assert_not_called()


# ---------------------------------------------------------------- page

class TestPage:
    def test_renders_settings_estimate_cap_and_controls(self, world, client):
        world.load.return_value = _config(pillar_ids=("nlp-a",))
        response = client.get("/daily-run/")
        assert response.status_code == 200
        html = response.text
        assert 'id="daily-run-form"' in html
        assert 'id="dr-enabled"' in html and "checked" not in html.split('id="dr-enabled"')[1].split(">")[0]
        assert 'id="daily-run-now"' in html
        assert "Pillar A" in html and "Pillar B" in html
        assert 'data-pillar-id="nlp-a"' in html
        assert "DAILY_SPEND_CAP_USD" in html
        assert "$2.00 a day" in html
        assert "None. The daily run is off." in html
        assert "/static/daily-run.js" in html

    def test_missing_migration_disables_save_and_names_the_file(self, world, client):
        world.load.return_value = _config(source=daily_run.SOURCE_ENV_NO_TABLE)
        html = client.get("/daily-run/").text
        assert "039_daily_run_settings.sql" in html
        assert 'id="dr-save" disabled' in html

    def test_last_run_links_the_pipeline_run(self, world, client):
        world.load.return_value = _config(last_run={
            "trigger": "scheduled", "started_at": NOW.isoformat(), "finished_at": NOW.isoformat(),
            "runs": [{"pillar_id": "nlp-a", "run_id": "r1"}],
        })
        world.runs.return_value = {"r1": {"id": "r1", "status": "failed", "error": "boom"}}
        html = client.get("/daily-run/").text
        assert 'href="/pipeline/?pillar=nlp-a&amp;run=r1"' in html
        assert "boom" in html

    def test_navigation_links_the_page(self):
        base = (ROOT / "webui" / "templates" / "base.html").read_text()
        assert base.count("nav_link('/daily-run/', 'Daily run'") == 2  # sidebar and drawer

    def test_spend_guard_confirms_run_now(self):
        guard = (ROOT / "webui" / "static" / "spend-guard.js").read_text()
        assert "\\/api\\/daily-run\\/run-now$" in guard
        assert "#daily-run-now" in guard
