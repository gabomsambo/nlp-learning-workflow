"""Tests for the local daily-run scheduler (nlp_pillars/scheduler.py).

Database and orchestrator calls are mocked; nothing here starts a real scheduler
or touches the network.
"""

from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from nlp_pillars.scheduler import (
    main,
    parse_schedule_time,
    resolve_timezone,
    run_all_pillars,
)


class TestParseScheduleTime:
    """SCHEDULE_TIME parsing."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("08:00", (8, 0)),
            ("00:00", (0, 0)),
            ("23:59", (23, 59)),
            ("7:05", (7, 5)),
            ("  08:30  ", (8, 30)),
        ],
    )
    def test_valid(self, value, expected):
        assert parse_schedule_time(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["", "08", "08:00:00", "8am", "24:00", "08:60", "-1:00", "aa:bb"],
    )
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError):
            parse_schedule_time(value)


class TestResolveTimezone:
    """SCHEDULE_TIMEZONE resolution."""

    def test_valid_iana_name(self):
        assert resolve_timezone("America/New_York") == ZoneInfo("America/New_York")

    def test_whitespace_is_stripped(self):
        assert resolve_timezone(" UTC ") == ZoneInfo("UTC")

    def test_unknown_zone_raises(self):
        with pytest.raises(ValueError):
            resolve_timezone("Mars/Olympus_Mons")


class TestRunAllPillars:
    """The scheduled job itself."""

    def _settings(self, papers_per_day=1):
        return MagicMock(papers_per_day=papers_per_day)

    def test_iterates_every_pillar_from_the_database(self):
        pillars = [MagicMock(id="pillar-a"), MagicMock(id="pillar-b")]
        with patch("nlp_pillars.scheduler.get_settings", return_value=self._settings(3)), \
             patch("nlp_pillars.scheduler.db.get_pillars", return_value=pillars), \
             patch("nlp_pillars.scheduler.Orchestrator") as mock_orch:
            run_all_pillars()

        run_daily = mock_orch.return_value.run_daily
        assert run_daily.call_count == 2
        assert [c.args[0] for c in run_daily.call_args_list] == ["pillar-a", "pillar-b"]
        # PAPERS_PER_DAY is the per-pillar limit, not the whole-run limit.
        assert all(c.kwargs["papers_limit"] == 3 for c in run_daily.call_args_list)

    def test_one_failing_pillar_does_not_stop_the_rest(self):
        pillars = [MagicMock(id="a"), MagicMock(id="b"), MagicMock(id="c")]
        with patch("nlp_pillars.scheduler.get_settings", return_value=self._settings()), \
             patch("nlp_pillars.scheduler.db.get_pillars", return_value=pillars), \
             patch("nlp_pillars.scheduler.Orchestrator") as mock_orch:
            mock_orch.return_value.run_daily.side_effect = [
                MagicMock(success=True),
                RuntimeError("pillar b exploded"),
                MagicMock(success=True),
            ]
            run_all_pillars()  # must not raise

        assert mock_orch.return_value.run_daily.call_count == 3

    def test_no_pillars_is_a_no_op(self):
        with patch("nlp_pillars.scheduler.get_settings", return_value=self._settings()), \
             patch("nlp_pillars.scheduler.db.get_pillars", return_value=[]), \
             patch("nlp_pillars.scheduler.Orchestrator") as mock_orch:
            run_all_pillars()

        mock_orch.assert_not_called()

    def test_database_failure_does_not_propagate(self):
        with patch("nlp_pillars.scheduler.get_settings", return_value=self._settings()), \
             patch("nlp_pillars.scheduler.db.get_pillars", side_effect=Exception("no db")), \
             patch("nlp_pillars.scheduler.Orchestrator") as mock_orch:
            run_all_pillars()  # a dead database must not kill the scheduler process

        mock_orch.assert_not_called()


class TestMain:
    """Process entry point: the off switch and settings validation."""

    def _settings(self, enabled=True, time="08:00", tz="UTC"):
        return MagicMock(
            schedule_enabled=enabled,
            schedule_time=time,
            schedule_timezone=tz,
            log_level="INFO",
        )

    def test_disabled_exits_zero_without_starting_anything(self):
        with patch("nlp_pillars.scheduler.get_settings",
                   return_value=self._settings(enabled=False)), \
             patch("nlp_pillars.scheduler.get_background_jobs_service") as mock_service:
            assert main() == 0

        mock_service.assert_not_called()

    @pytest.mark.parametrize(
        ("time", "tz"),
        [("25:00", "UTC"), ("nonsense", "UTC"), ("08:00", "Mars/Olympus_Mons")],
    )
    def test_bad_settings_exit_nonzero_rather_than_defaulting(self, time, tz):
        with patch("nlp_pillars.scheduler.get_settings",
                   return_value=self._settings(time=time, tz=tz)), \
             patch("nlp_pillars.scheduler.get_background_jobs_service") as mock_service:
            assert main() == 1

        mock_service.assert_not_called()

    def test_enabled_registers_the_daily_job_in_the_configured_timezone(self):
        with patch("nlp_pillars.scheduler.get_settings",
                   return_value=self._settings(time="08:30", tz="America/New_York")), \
             patch("nlp_pillars.scheduler.get_background_jobs_service") as mock_get, \
             patch("nlp_pillars.scheduler._shutdown") as mock_shutdown:
            # Return immediately from the blocking wait.
            mock_shutdown.wait.return_value = True
            mock_get.return_value.scheduler.get_jobs.return_value = []
            assert main() == 0

        service = mock_get.return_value
        service.add_daily_pillar_run_job.assert_called_once()
        kwargs = service.add_daily_pillar_run_job.call_args.kwargs
        assert kwargs["hour"] == 8
        assert kwargs["minute"] == 30
        assert kwargs["tzinfo"] == ZoneInfo("America/New_York")
        assert kwargs["func"] is run_all_pillars
        # start() also brings up the FSRS jobs, which nothing else ever started.
        service.start.assert_called_once()
        service.stop.assert_called_once()
