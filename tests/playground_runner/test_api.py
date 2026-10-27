"""Runner HTTP API and session bookkeeping, with a fake kernel.

The fake keeps a namespace dict, so "state persists across executions" is
checked at the API level here and against a real kernel in test_kernel_live.py.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from playground_runner.app import create_app
from playground_runner.config import RunnerConfig, valid_playground_id
from playground_runner.sessions import SessionManager


class FakeKernel:
    instances = []

    def __init__(self, config, session_id, playground_id):
        self.session_id = session_id
        self.playground_id = playground_id
        self.namespace = {}
        self.count = 0
        self.interrupts = 0
        self.restarts = 0
        self.shut = False
        self.release = None
        FakeKernel.instances.append(self)

    async def start(self):
        if self.playground_id == "explodes":
            raise RuntimeError("kernel died at startup")

    async def execute(self, code, timeout_s, abort=None):
        self.count += 1
        if code == "block":
            self.release = asyncio.Event()
            waiter = asyncio.ensure_future(self.release.wait())
            stopper = asyncio.ensure_future(abort.wait())
            await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            stopper.cancel()
            return {"status": "error", "execution_ms": 1, "execution_count": self.count,
                    "outputs": [{"type": "error", "ename": "KeyboardInterrupt", "evalue": "", "traceback": ""}],
                    "truncated": False, "dropped_outputs": 0, "kernel_restarted": False}
        outputs = []
        if "=" in code:
            name, value = (part.strip() for part in code.split("=", 1))
            self.namespace[name] = value
        else:
            outputs.append({"type": "text", "text": str(self.namespace.get(code.strip(), "?"))})
        return {"status": "ok", "execution_ms": 1, "execution_count": self.count, "outputs": outputs,
                "truncated": False, "dropped_outputs": 0, "kernel_restarted": False}

    async def interrupt(self):
        self.interrupts += 1
        if self.release is not None:
            self.release.set()

    async def restart(self):
        self.restarts += 1
        self.namespace = {}
        self.count = 0

    async def shutdown(self):
        self.shut = True


def _config(**overrides):
    base = dict(max_sessions=2, idle_timeout_s=60, max_code_bytes=1_000, max_exec_timeout_s=300, exec_timeout_s=30)
    base.update(overrides)
    return RunnerConfig(**base)


@pytest.fixture
def client(tmp_path):
    FakeKernel.instances = []
    app = create_app(_config(runtime_dir=tmp_path / "rt"), FakeKernel)
    with TestClient(app) as c:
        yield c


def test_health_reports_limits_and_packages(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["max_sessions"] == 2
    assert body["limits"]["exec_timeout_s"] == 30
    assert "pandas" in body["packages"]


def test_session_lifecycle_and_state_persistence(client):
    created = client.post("/sessions", json={"playground_id": "thetaray-aml"})
    assert created.status_code == 201
    sid = created.json()["session_id"]
    assert created.json()["playground_id"] == "thetaray-aml"

    assert client.post(f"/sessions/{sid}/execute", json={"code": "x = 41"}).json()["status"] == "ok"
    second = client.post(f"/sessions/{sid}/execute", json={"code": "x"}).json()
    assert second["outputs"] == [{"type": "text", "text": "41"}]
    assert second["execution_count"] == 2
    assert second["interrupted"] is False

    listed = client.get("/sessions").json()
    assert [s["session_id"] for s in listed["sessions"]] == [sid]
    assert listed["sessions"][0]["execution_count"] == 2
    assert listed["sessions"][0]["busy"] is False

    assert client.post(f"/sessions/{sid}/restart").json() == {"session_id": sid, "restarted": True}
    assert client.post(f"/sessions/{sid}/execute", json={"code": "x"}).json()["outputs"][0]["text"] == "?"

    assert client.delete(f"/sessions/{sid}").json() == {"session_id": sid, "deleted": True}
    assert FakeKernel.instances[0].shut is True
    assert client.get("/sessions").json()["sessions"] == []
    assert client.post(f"/sessions/{sid}/execute", json={"code": "x"}).status_code == 404


def test_max_sessions_is_enforced(client):
    for _ in range(2):
        assert client.post("/sessions", json={"playground_id": "p"}).status_code == 201
    third = client.post("/sessions", json={"playground_id": "p"})
    assert third.status_code == 429
    assert "PLAYGROUND_MAX_SESSIONS" in third.json()["detail"]


@pytest.mark.parametrize("bad", ["../etc", "a/b", ".hidden", "", "x" * 200, "a..b"])
def test_playground_id_cannot_escape_the_data_directory(client, bad):
    assert client.post("/sessions", json={"playground_id": bad}).status_code == 422
    assert not valid_playground_id(bad)


def test_kernel_start_failure_is_503_and_frees_the_slot(client):
    failed = client.post("/sessions", json={"playground_id": "explodes"})
    assert failed.status_code == 503
    assert "kernel died at startup" in failed.json()["detail"]
    for _ in range(2):
        assert client.post("/sessions", json={"playground_id": "p"}).status_code == 201


def test_execute_validation(client):
    sid = client.post("/sessions", json={"playground_id": "p"}).json()["session_id"]
    assert client.post(f"/sessions/{sid}/execute", json={"code": "x" * 2_000}).status_code == 413
    too_long = client.post(f"/sessions/{sid}/execute", json={"code": "1", "timeout_s": 301})
    assert too_long.status_code == 422
    assert "PLAYGROUND_MAX_EXEC_TIMEOUT_S" in too_long.json()["detail"]
    assert client.post(f"/sessions/{sid}/execute", json={"code": "1", "timeout_s": 0}).status_code == 422
    assert client.post(f"/sessions/{sid}/execute", json={}).status_code == 422
    assert client.post("/sessions/nope/interrupt").status_code == 404
    assert client.post("/sessions/nope/restart").status_code == 404
    assert client.delete("/sessions/nope").status_code == 404


async def test_busy_session_refuses_a_second_execute_and_interrupt_reports_it(tmp_path):
    manager = SessionManager(_config(runtime_dir=tmp_path), FakeKernel)
    session = await manager.create("p")
    running = asyncio.create_task(manager.execute(session.session_id, "block", 30))
    await asyncio.sleep(0.01)
    assert session.busy
    with pytest.raises(Exception) as info:
        await manager.execute(session.session_id, "1", 30)
    assert "still running" in str(info.value)
    assert (await manager.interrupt(session.session_id))["interrupted"] is True
    result = await running
    assert result["interrupted"] is True
    assert result["status"] == "error"
    assert not session.busy


async def test_delete_stops_running_code_first(tmp_path):
    manager = SessionManager(_config(runtime_dir=tmp_path), FakeKernel)
    session = await manager.create("p")
    running = asyncio.create_task(manager.execute(session.session_id, "block", 30))
    await asyncio.sleep(0.01)
    await asyncio.wait_for(manager.delete(session.session_id), timeout=2)
    await asyncio.wait_for(running, timeout=2)
    assert session.kernel.shut is True
    assert manager.sessions == {}


async def test_idle_sessions_are_reaped_and_busy_ones_are_not(tmp_path):
    manager = SessionManager(_config(runtime_dir=tmp_path, idle_timeout_s=60), FakeKernel)
    idle = await manager.create("p")
    busy = await manager.create("p")
    running = asyncio.create_task(manager.execute(busy.session_id, "block", 30))
    await asyncio.sleep(0.01)
    later = idle.last_active_mono + 61
    assert await manager.reap_idle(now=idle.last_active_mono + 30) == []
    assert await manager.reap_idle(now=later) == [idle.session_id]
    assert idle.kernel.shut is True
    assert list(manager.sessions) == [busy.session_id]
    await manager.interrupt(busy.session_id)
    await running


def test_config_from_env_reads_overrides_and_rejects_nonsense():
    cfg = RunnerConfig.from_env(
        {
            "PLAYGROUND_EXEC_TIMEOUT_S": "60",
            "PLAYGROUND_MAX_SESSIONS": "5",
            "PLAYGROUND_IDLE_TIMEOUT_S": "120",
            "PLAYGROUND_MAX_OUTPUT_BYTES": "1000",
            "PLAYGROUND_KERNEL_SECCOMP": "0",
        }
    )
    assert (cfg.exec_timeout_s, cfg.max_sessions, cfg.idle_timeout_s, cfg.max_output_bytes) == (60, 5, 120, 1000)
    assert cfg.kernel_seccomp is False
    defaults = RunnerConfig.from_env({})
    assert (defaults.exec_timeout_s, defaults.max_sessions, defaults.idle_timeout_s) == (180, 3, 1800)
    assert defaults.kernel_seccomp is True
    with pytest.raises(ValueError):
        RunnerConfig.from_env({"PLAYGROUND_MAX_SESSIONS": "0"})
    with pytest.raises(ValueError):
        RunnerConfig.from_env({"PLAYGROUND_EXEC_TIMEOUT_S": "600", "PLAYGROUND_MAX_EXEC_TIMEOUT_S": "300"})


def test_a_kernel_failure_during_execute_is_503_naming_restart(client):
    sid = client.post("/sessions", json={"playground_id": "p"}).json()["session_id"]

    async def broken(code, timeout_s, abort=None):
        raise RuntimeError("zmq socket closed")

    FakeKernel.instances[-1].execute = broken
    response = client.post(f"/sessions/{sid}/execute", json={"code": "1"})
    assert response.status_code == 503
    assert "zmq socket closed" in response.json()["detail"]
    assert f"/sessions/{sid}/restart" in response.json()["detail"]
    # The session is not left marked busy.
    assert client.get("/sessions").json()["sessions"][0]["busy"] is False
