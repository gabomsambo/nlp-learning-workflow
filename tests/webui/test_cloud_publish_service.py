"""The webui side of cloud publish: debounce, retry, and "a failed publish never fails a run".

The publisher thread runs for real with millisecond waits; nothing touches the
network, a database, or Cloudflare. ``publish`` and ``watch`` are injected.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nlp_pillars.schemas import RunStatus
from webui.app import create_app
from webui.services import cloud_publish_service as service
from webui.services import run_service
from webui.services import system_status as status

PILLAR = "neural-architectures-language"
FAST = dict(debounce=0.05, max_wait=0.4, retry_base=0.1, retry_max=0.2, poll=3600)


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class Recorder:
    def __init__(self, fail_times=0, error="HTTP 503 try later"):
        self.calls = []
        self.fail_times = fail_times
        self.error = error
        self.done = threading.Event()

    def __call__(self, settings, progress):
        progress("working")
        self.calls.append(time.monotonic())
        if len(self.calls) <= self.fail_times:
            raise RuntimeError(self.error)
        self.done.set()
        return {"uploaded": 1, "uploaded_bytes": 10, "unchanged": 2, "objects": 3, "bytes": 30, "manifest_sent": True, "seconds": 0.1}


@pytest.fixture
def make():
    made = []

    def build(publish, watch=lambda: None, **kw):
        publisher = service.CloudPublisher(SimpleNamespace(), publish=publish, watch=watch, **{**FAST, **kw})
        made.append(publisher)
        return publisher

    yield build
    for publisher in made:
        publisher.stop()


def test_a_burst_of_notifications_becomes_one_publish(make):
    record = Recorder()
    publisher = make(record)
    publisher.start()
    for _ in range(5):
        publisher.notify("run:upload")
    assert wait_for(lambda: publisher.state == "idle" and record.calls)
    time.sleep(0.2)
    assert len(record.calls) == 1, "startup and the burst share one debounce window"
    assert publisher.last_summary["uploaded"] == 1 and publisher.last_success_at


def test_constant_activity_cannot_postpone_publishing_forever(make):
    record = Recorder()
    publisher = make(record, debounce=0.3, max_wait=0.25)
    publisher.start()
    stop = time.monotonic() + 1.0
    while time.monotonic() < stop and not record.calls:
        publisher.notify("review")
        time.sleep(0.02)
    assert record.calls, "the maximum wait forces a publish during an unbroken stream"


def test_a_failed_publish_is_recorded_and_retried_later(make):
    record = Recorder(fail_times=1)
    publisher = make(record)
    publisher.start()
    assert wait_for(lambda: publisher.failures == 1)
    assert publisher.state == "failed" and "503" in publisher.last_error
    assert publisher.status()["retry_in_seconds"] is not None
    assert record.done.wait(3.0), "it tries again without any new notification"
    assert wait_for(lambda: publisher.state == "idle")
    assert publisher.failures == 0 and publisher.last_error is None and len(record.calls) == 2


def test_changes_that_arrive_while_publishing_are_not_lost(make):
    gate, release = threading.Event(), threading.Event()
    calls = []

    def slow(settings, progress):
        calls.append(1)
        gate.set()
        release.wait(2)
        return {"uploaded": 0, "uploaded_bytes": 0, "unchanged": 1, "objects": 1, "bytes": 1, "manifest_sent": False, "seconds": 0}

    publisher = make(slow)
    publisher.start()
    assert gate.wait(3)
    publisher.notify("run:podcast_audio")
    release.set()
    assert wait_for(lambda: len(calls) == 2)


def test_notify_never_raises_even_if_the_publisher_is_broken(make):
    publisher = make(Recorder())
    real, publisher._cond = publisher._cond, None  # any use of the lock now raises
    publisher.notify("run:upload")
    with patch.object(service, "_instance", publisher):
        service.notify("run:upload")  # the module-level hook swallows as well
    publisher._cond = real


def test_the_module_hook_is_a_no_op_when_publishing_is_off():
    with patch.object(service, "_instance", None):
        service.notify("run:upload")


def test_another_process_finishing_a_run_triggers_a_publish(make):
    finished = ["2026-10-25T10:00:00Z"]
    record = Recorder()
    publisher = make(record, watch=lambda: finished[0], poll=0.05)
    publisher.start()
    assert wait_for(lambda: len(record.calls) == 1 and publisher.state == "idle")
    time.sleep(0.3)
    assert len(record.calls) == 1, "an unchanged pipeline_runs is not a reason to publish"
    finished[0] = "2026-10-25T10:05:00Z"
    assert wait_for(lambda: len(record.calls) == 2)


def test_an_unreadable_pipeline_runs_table_is_survived(make):
    def broken():
        raise RuntimeError("postgrest is down")

    record = Recorder()
    publisher = make(record, watch=broken, poll=0.05)
    publisher.start()
    assert wait_for(lambda: len(record.calls) >= 1)


# --- a run is never failed by the cloud ------------------------------------------


def _run(kind):
    events = {"run-1": threading.Event()}
    with (
        patch("webui.services.run_service.db") as mock_db,
        patch("webui.services.run_service.Orchestrator") as orch,
    ):
        orch.return_value.run_daily.return_value = SimpleNamespace(papers_processed=["p"], errors=[], success=True)
        orch.return_value.process_selected_papers.return_value = orch.return_value.run_daily.return_value
        orch.return_value.run_discovery_with_selection.return_value = []
        orch.return_value.infra_errors = []
        run_service.execute_run("run-1", kind, PILLAR, events["run-1"], events)
        return mock_db


def test_a_run_succeeds_when_the_publish_that_follows_it_fails(make):
    record = Recorder(fail_times=99, error="the cloud is unreachable")
    publisher = make(record, debounce=0.01)
    publisher.start()
    assert wait_for(lambda: publisher.failures >= 1)
    with patch.object(service, "_instance", publisher):
        mock_db = _run(run_service.KIND_RUN_DAILY)
    assert mock_db.finish_pipeline_run.call_args.args[1] == RunStatus.SUCCEEDED.value
    assert mock_db.finish_pipeline_run.call_args.kwargs["error"] is None
    assert wait_for(lambda: publisher.failures >= 2)
    assert "unreachable" in publisher.last_error


def test_a_run_succeeds_even_if_the_hook_itself_blows_up():
    class Broken:
        def notify(self, reason):
            raise RuntimeError("hook exploded")

    with patch.object(service, "_instance", Broken()):
        mock_db = _run(run_service.KIND_RUN_DAILY)
    assert mock_db.finish_pipeline_run.call_args.args[1] == RunStatus.SUCCEEDED.value


def test_runs_that_change_the_library_notify_and_discovery_does_not():
    class Spy:
        def __init__(self):
            self.reasons = []

        def notify(self, reason):
            self.reasons.append(reason)

    for kind, expected in (
        (run_service.KIND_RUN_DAILY, ["run:run_daily"]),
        (run_service.KIND_PROCESS_SELECTED, ["run:process_selected"]),
        (run_service.KIND_DISCOVER, []),
    ):
        spy = Spy()
        with patch.object(service, "_instance", spy):
            _run(kind)
        assert spy.reasons == expected, kind


def test_a_cancelled_or_crashed_run_still_notifies():
    class Spy:
        reasons = []

        def notify(self, reason):
            self.reasons.append(reason)

    events = {"run-1": threading.Event()}
    spy = Spy()
    with (
        patch("webui.services.run_service.db"),
        patch("webui.services.run_service.Orchestrator") as orch,
        patch.object(service, "_instance", spy),
    ):
        orch.return_value.run_daily.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            run_service.execute_run("run-1", run_service.KIND_RUN_DAILY, PILLAR, events["run-1"], events)
    assert spy.reasons == ["run:run_daily"], "a failed run can still have saved a paper"


# --- starting it -----------------------------------------------------------------


def settings(**kw):
    base = dict(
        cloud_publish_enabled=True, cloud_publish_url="https://read.example.test",
        cf_access_client_id="id.access", cf_access_client_secret="secret-value",
    )
    return SimpleNamespace(**{**base, **kw})


def test_publishing_is_off_by_default():
    with patch.object(service, "_instance", None):
        assert service.start(settings(cloud_publish_enabled=False)) is None
        assert service.current_status(settings(cloud_publish_enabled=False))["state"] == "off"


def test_enabled_but_unconfigured_does_not_start_and_names_what_is_missing():
    bad = settings(cf_access_client_secret="")
    with patch.object(service, "_instance", None):
        assert service.start(bad) is None
        current = service.current_status(bad)
    assert current["state"] == "misconfigured" and current["missing"] == ["CF_ACCESS_CLIENT_SECRET"]


def test_a_start_failure_never_stops_the_app():
    with patch.object(service, "CloudPublisher", side_effect=RuntimeError("no thread for you")), patch.object(service, "_instance", None):
        assert service.start(settings()) is None


def test_the_app_starts_and_stops_the_publisher_with_its_lifespan():
    started, stopped = [], []
    with (
        patch.object(service, "start", lambda *a, **k: started.append(1)),
        patch.object(service, "stop", lambda: stopped.append(1)),
        patch.object(run_service, "sweep_interrupted_runs_on_startup", return_value=0),
    ):
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code in (200, 503)
            assert started == [1] and stopped == []
    assert stopped == [1]


# --- Activity ---------------------------------------------------------------------

ROWS = [
    {"doc_key": "library.json", "byte_size": 2000, "published_at": "2026-10-25T16:50:11Z"},
    {"doc_key": "papers/a.json", "byte_size": 3000, "published_at": "2026-10-25T16:50:12Z"},
    {"doc_key": f"audio/{'a' * 64}.mp3", "byte_size": 5_000_000, "published_at": "2026-10-25T16:50:13Z"},
    {"doc_key": "manifest.json", "byte_size": 900, "published_at": "2026-10-25T16:51:00Z"},
]


def test_activity_row_for_each_state_is_never_fatal():
    off = status.classify_cloud_reader({"enabled": False}, None)
    assert off["state"] == "ok" and "CLOUD_PUBLISH_ENABLED" in off["fix"] and off["fatal"] is False

    unconfigured = status.classify_cloud_reader({"enabled": True, "configured": False, "missing": ["CLOUD_PUBLISH_URL"]}, None)
    assert unconfigured["state"] == "down" and "CLOUD_PUBLISH_URL" in unconfigured["fix"] and unconfigured["fatal"] is False

    failed = status.classify_cloud_reader(
        {"enabled": True, "configured": True, "state": "failed", "last_error": "HTTP 403 forbidden\nsecond line", "retry_in_seconds": 200}, ROWS
    )
    assert failed["state"] == "down" and failed["fatal"] is False
    assert "HTTP 403 forbidden" in failed["summary"] and "second line" not in failed["summary"]
    assert "Trying again in 4 min" in failed["summary"]
    assert "2 documents and 1 audio files" in failed["summary"]

    idle = status.classify_cloud_reader(
        {"enabled": True, "configured": True, "state": "idle", "last_success_at": "2026-10-25T16:51:00Z"}, ROWS
    )
    assert idle["state"] == "ok" and "2026-10-25 16:51:00 UTC" in idle["summary"]
    assert "4.8 MB" in idle["summary"] and "manifest written 2026-10-25 16:51 UTC" in idle["summary"]

    unreadable = status.classify_cloud_reader({"enabled": True, "configured": True, "state": "idle"}, None, "HTTP 404; is x applied?")
    assert "Could not read what is published" in unreadable["summary"]


def test_a_failed_publish_does_not_turn_the_page_chip_red():
    checks = [
        {"state": "ok", "label": "Web app"},
        status.classify_cloud_reader({"enabled": True, "configured": True, "state": "failed", "last_error": "x"}, ROWS),
    ]
    assert not any(status.counts_as_down(c) for c in checks)
    assert status.public_summary(checks) == "All services up"
