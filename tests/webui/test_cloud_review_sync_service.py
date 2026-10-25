"""The webui side of review sync: when the pull runs, what it does with the result.

The thread runs for real with millisecond waits; ``pull`` is injected, so nothing
touches a database or the network.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from nlp_pillars.cloud.publish import PublishError
from nlp_pillars.cloud.reviews import PullError, PullReport
from webui.services import cloud_publish_service
from webui.services import cloud_review_sync_service as service
from webui.services import system_status as status

FAST = dict(interval=0.4, settle=0.03, retry_base=0.05)


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def summary(applied=0, more=False, **extra):
    return {"fetched": applied, "applied": applied, "duplicates": 0, "skipped": 0, "cards": applied,
            "cursor": applied, "cloud_through_seq": applied, "more": more, **extra}


class Puller:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, settings, progress):
        progress("asking")
        self.calls.append(time.monotonic())
        result = self.results.pop(0) if self.results else summary()
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def make():
    made = []

    def build(pull, **kw):
        told = []
        syncer = service.ReviewSyncer(SimpleNamespace(), pull=pull, on_applied=lambda: told.append(1), **{**FAST, **kw})
        syncer.told = told
        made.append(syncer)
        return syncer

    yield build
    for syncer in made:
        syncer.stop()


def test_it_pulls_as_soon_as_it_starts_and_again_each_interval(make):
    pull = Puller()
    syncer = make(pull)
    syncer.start()
    assert wait_for(lambda: len(pull.calls) >= 2), "start, then the interval"
    assert syncer.state == "idle" and syncer.last_success_at and syncer.failures == 0


def test_a_trigger_pulls_soon_and_a_burst_of_triggers_is_one_pull(make):
    pull = Puller()
    syncer = make(pull, interval=30)
    syncer.start()
    assert wait_for(lambda: len(pull.calls) == 1)
    for _ in range(5):
        syncer.trigger("after publish")
    assert wait_for(lambda: len(pull.calls) == 2)
    time.sleep(0.2)
    assert len(pull.calls) == 2


def test_applied_reviews_tell_the_publisher_and_an_empty_pull_does_not(make):
    pull = Puller(summary(0), summary(3))
    syncer = make(pull, interval=30)
    syncer.start()
    assert wait_for(lambda: len(pull.calls) == 1)
    assert syncer.told == []
    syncer.trigger("after publish")
    assert wait_for(lambda: syncer.told == [1])
    assert syncer.last_summary["applied"] == 3


def test_more_waiting_means_it_carries_straight_on(make):
    pull = Puller(summary(5, more=True), summary(2))
    syncer = make(pull, interval=30)
    syncer.start()
    assert wait_for(lambda: len(pull.calls) == 2), "no waiting out the interval while events remain"


def test_a_failure_is_recorded_retried_sooner_than_the_interval_and_heals(make):
    pull = Puller(PullError("seq 4: connection refused"), summary(1))
    syncer = make(pull, interval=30)
    syncer.start()
    assert wait_for(lambda: syncer.failures == 1)
    assert syncer.state == "failed" and "connection refused" in syncer.last_error
    assert syncer.status()["retry_in_seconds"] is not None
    assert wait_for(lambda: syncer.state == "idle" and len(pull.calls) == 2)
    assert syncer.last_error is None and syncer.failures == 0 and syncer.told == [1]


def test_a_failed_pull_that_applied_some_reviews_still_tells_the_publisher(make):
    partial = PullReport(applied=2)
    pull = Puller(PullError("then the database went away", partial))
    syncer = make(pull, interval=30)
    syncer.start()
    assert wait_for(lambda: syncer.told == [1])
    assert syncer.state == "failed"


def test_nothing_escapes_the_thread(make):
    pull = Puller(RuntimeError("boom"), PublishError("gave up"), summary())
    syncer = make(pull, interval=30, retry_base=0.02)
    syncer.start()
    assert wait_for(lambda: len(pull.calls) == 3 and syncer.state == "idle")


def test_stop_is_prompt_and_trigger_never_raises(make):
    syncer = make(Puller(), interval=30)
    syncer.start()
    started = time.monotonic()
    syncer.stop()
    assert time.monotonic() - started < 1.0
    syncer.trigger("late")  # after stop: still harmless


def test_the_module_does_nothing_when_publishing_is_off_or_unconfigured():
    off = SimpleNamespace(cloud_publish_enabled=False)
    assert service.start(off) is None
    unconfigured = SimpleNamespace(cloud_publish_enabled=True, cloud_publish_url="", cf_access_client_id="", cf_access_client_secret="")
    assert service.start(unconfigured) is None
    service.trigger("whatever")  # no instance: a no-op
    assert service.current_status() == {"enabled": False}


def test_a_publish_triggers_a_pull():
    pull = Puller()
    syncer = service.ReviewSyncer(SimpleNamespace(), pull=pull, on_applied=lambda: None, **{**FAST, "interval": 30})
    service._instance = syncer
    try:
        syncer.start()
        assert wait_for(lambda: len(pull.calls) == 1)
        publisher = cloud_publish_service.CloudPublisher(
            SimpleNamespace(), publish=lambda s, p: {"uploaded": 0, "uploaded_bytes": 0, "unchanged": 0, "objects": 0,
                                                     "bytes": 0, "manifest_sent": False, "seconds": 0},
            watch=lambda: None, debounce=0.02, max_wait=0.1, retry_base=0.1, retry_max=0.2, poll=3600,
        )
        publisher.start()
        assert wait_for(lambda: len(pull.calls) == 2), "the publish that just finished asked for a pull"
        publisher.stop()
    finally:
        service._instance = None
        syncer.stop()


def test_a_pull_that_applies_reviews_asks_the_publisher_for_a_new_snapshot():
    heard = []
    original = cloud_publish_service._instance
    cloud_publish_service._instance = SimpleNamespace(notify=heard.append)
    try:
        service._tell_publisher()
    finally:
        cloud_publish_service._instance = original
    assert heard == ["phone reviews"]


# --- Activity --------------------------------------------------------------------

PUBLISHING = {"enabled": True, "configured": True, "state": "idle", "last_success_at": "2026-10-25T16:51:00Z"}


def test_activity_mentions_the_phone_reviews_and_stays_non_fatal():
    ok = status.classify_cloud_reader(
        PUBLISHING, None, None, {"enabled": True, "state": "idle", "last_success_at": "2026-10-25T17:00:00Z", "last_summary": {"applied": 2}}
    )
    assert ok["state"] == "ok" and "Phone reviews checked 2026-10-25 17:00:00 UTC (2 new)." in ok["summary"]

    failing = status.classify_cloud_reader(
        PUBLISHING, None, None,
        {"enabled": True, "state": "failed", "last_error": "seq 4: connection refused\nsecond", "retry_in_seconds": 90},
    )
    assert failing["state"] == "down" and failing["fatal"] is False
    assert "Phone reviews could not be pulled: seq 4: connection refused." in failing["summary"]
    assert "second" not in failing["summary"] and "Trying again in 2 min" in failing["summary"]
    assert "cloud pull-reviews" in failing["fix"]
    assert not status.counts_as_down(failing)

    off = status.classify_cloud_reader(PUBLISHING, None, None, {"enabled": False})
    assert "Phone reviews" not in off["summary"]
