"""Against a real Jupyter kernel: state, timeout, interrupt, restart, outputs.

Needs the runner's own dependencies (docker/playground-runner/requirements.txt),
which the web app's environment does not carry. The "playground runner" job in
.github/workflows/tests.yml installs them and runs this file; elsewhere it skips.
"""

import asyncio
import json

import pytest

pytest.importorskip("jupyter_client")
pytest.importorskip("ipykernel")
pytest.importorskip("matplotlib")

from playground_runner.config import RunnerConfig  # noqa: E402
from playground_runner.sessions import SessionManager  # noqa: E402


def _seccomp_available() -> bool:
    try:
        import pyseccomp  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture
def config(tmp_path):
    data = tmp_path / "data" / "pg-live"
    data.mkdir(parents=True)
    (data / "tx.csv").write_text("account,amount\nA,10\nA,5\nB,25\n")
    (data / "manifest.json").write_text(json.dumps({"https://example.com/tx.csv": "tx.csv"}))
    return RunnerConfig(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runtime_dir=tmp_path / "rt",
        exec_timeout_s=30,
        max_output_bytes=200_000,
        interrupt_grace_s=3,
        kernel_seccomp=_seccomp_available(),
    )


@pytest.fixture
async def session(config):
    manager = SessionManager(config)
    created = await manager.create("pg-live")
    try:
        yield manager, created.session_id
    finally:
        await manager.shutdown_all()


async def test_state_persists_like_a_notebook(session):
    manager, sid = session
    first = await manager.execute(sid, "import pandas as pd\ntotal = 41", 30)
    assert first["status"] == "ok"
    second = await manager.execute(sid, "total + 1", 30)
    assert second["status"] == "ok"
    assert second["outputs"] == [{"type": "text", "text": "42"}]
    assert second["execution_count"] == 2


async def test_offline_dataset_hook_inside_the_kernel(session):
    manager, sid = session
    hit = await manager.execute(
        sid, "import pandas as pd\ndf = pd.read_csv('https://example.com/tx.csv')\nprint(df.groupby('account').amount.sum().to_dict())", 30
    )
    assert hit["status"] == "ok", hit
    assert hit["outputs"][0]["text"].strip() == "{'A': 15, 'B': 25}"
    miss = await manager.execute(sid, "pd.read_csv('https://example.com/other.csv')", 30)
    assert miss["status"] == "error"
    err = miss["outputs"][-1]
    assert err["ename"] == "OfflineDatasetError"
    assert "runs offline" in err["evalue"]
    # Hook leaves no helper names in the user's namespace.
    names = await manager.execute(sid, "print(sorted(n for n in dir() if 'playground' in n))", 30)
    assert names["outputs"][0]["text"].strip() == "[]"


async def test_error_traceback_has_no_ansi(session):
    manager, sid = session
    result = await manager.execute(sid, "def f():\n    return 1/0\nf()", 30)
    assert result["status"] == "error"
    err = result["outputs"][-1]
    assert err["ename"] == "ZeroDivisionError"
    assert "\x1b" not in err["traceback"]
    assert "return 1/0" in err["traceback"]


async def test_plt_show_yields_a_png(session):
    manager, sid = session
    result = await manager.execute(sid, "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.show()", 60)
    assert result["status"] == "ok", result
    images = [o for o in result["outputs"] if o["type"] == "image"]
    assert len(images) == 1
    assert images[0]["mime"] == "image/png"
    assert images[0]["data"].startswith("iVBOR")


async def test_dataframe_is_html_flagged_untrusted(session):
    manager, sid = session
    result = await manager.execute(sid, "import pandas as pd\npd.DataFrame({'a': [1, 2]})", 30)
    (out,) = result["outputs"]
    assert out["type"] == "html" and out["untrusted"] is True
    assert "<table" in out["html"]
    assert "a" in out["text"]


async def test_timeout_interrupts_and_keeps_earlier_state(session):
    manager, sid = session
    await manager.execute(sid, "kept = 'yes'", 30)
    result = await manager.execute(sid, "import time\nprint('started', flush=True)\ntime.sleep(60)", 1)
    assert result["status"] == "timeout"
    assert result["kernel_restarted"] is False
    assert result["execution_ms"] < 10_000
    assert result["outputs"][0]["text"].strip() == "started"
    assert result["outputs"][-1]["ename"] == "TimeoutError"
    assert not any(o.get("ename") == "KeyboardInterrupt" for o in result["outputs"])
    after = await manager.execute(sid, "kept", 30)
    assert after["outputs"] == [{"type": "text", "text": "'yes'"}]


async def test_timeout_restarts_a_kernel_that_ignores_the_interrupt(session):
    manager, sid = session
    await manager.execute(sid, "kept = 1", 30)
    code = "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(60)"
    result = await manager.execute(sid, code, 1)
    assert result["status"] == "timeout"
    assert result["kernel_restarted"] is True
    assert "restarted" in result["outputs"][-1]["evalue"]
    after = await manager.execute(sid, "kept", 30)
    assert after["outputs"][-1]["ename"] == "NameError"
    # The offline hook is back after the restart.
    hook = await manager.execute(sid, "import pandas as pd\nlen(pd.read_csv('https://example.com/tx.csv'))", 30)
    assert hook["outputs"] == [{"type": "text", "text": "3"}]


async def test_interrupt_stops_running_code(session):
    manager, sid = session
    running = asyncio.create_task(manager.execute(sid, "import time\nwhile True:\n    time.sleep(0.1)", 60))
    await asyncio.sleep(1.5)
    assert (await manager.interrupt(sid))["interrupted"] is True
    result = await asyncio.wait_for(running, timeout=15)
    assert result["status"] == "error"
    assert result["interrupted"] is True
    assert result["outputs"][-1]["ename"] == "KeyboardInterrupt"
    assert (await manager.execute(sid, "1 + 1", 30))["outputs"] == [{"type": "text", "text": "2"}]


async def test_a_dead_kernel_is_reported_and_restarted(session):
    manager, sid = session
    result = await manager.execute(sid, "import os\nos.kill(os.getpid(), 9)", 30)
    assert result["status"] == "error"
    assert result["kernel_restarted"] is True
    assert result["outputs"][-1]["ename"] == "KernelDied"
    assert (await manager.execute(sid, "2 * 21", 30))["outputs"] == [{"type": "text", "text": "42"}]


async def test_output_cap_truncates_a_flood(session):
    manager, sid = session
    result = await manager.execute(sid, "for i in range(50_000):\n    print('row', i)", 30)
    assert result["status"] == "ok"
    assert result["truncated"] is True
    assert result["outputs"][0]["truncated"] is True
    assert "output truncated" in result["outputs"][0]["text"]


async def test_restart_clears_state(session):
    manager, sid = session
    await manager.execute(sid, "x = 5", 30)
    await manager.restart(sid)
    after = await manager.execute(sid, "x", 30)
    assert after["outputs"][-1]["ename"] == "NameError"
    assert after["execution_count"] == 1


@pytest.mark.skipif(not _seccomp_available(), reason="pyseccomp/libseccomp not available")
async def test_kernel_cannot_open_ip_sockets(session):
    manager, sid = session
    code = (
        "import socket\n"
        "try:\n"
        "    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    print('opened')\n"
        "except PermissionError as e:\n"
        "    print('blocked', e.errno)\n"
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://127.0.0.1:1/x')\n"
        "except Exception as e:\n"
        "    print(type(e).__name__)\n"
    )
    result = await manager.execute(sid, code, 30)
    text = result["outputs"][0]["text"]
    assert "blocked 13" in text
    assert "OfflineDatasetError" in text
