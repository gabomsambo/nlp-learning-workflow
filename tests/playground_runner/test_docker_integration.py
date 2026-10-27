"""Build the runner image and prove the lockdown from inside the container.

Runs when Docker is available (skips otherwise; deselect with -m "not docker").
The container gets the exact lockdown flags from docker-compose.yml
(runner_compose.docker_run_flags) on a throwaway internal network, and a
throwaway read-only dataset directory instead of the real volume. Everything
it creates is removed afterwards except the image tag, which keeps the next
build cached.
"""

import json
import shutil
import subprocess
import textwrap
import time
import uuid

import pytest
from runner_compose import ROOT, docker_run_flags, load_compose

pytestmark = pytest.mark.docker

IMAGE = "nlp-playground-runner:pytest"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


if not _docker_available():
    pytest.skip("Docker is not available", allow_module_level=True)


def _run(args, timeout=120, check=True):
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise AssertionError(f"{' '.join(args)} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc


def _exec_python(container: str, code: str, timeout=120):
    return _run(["docker", "exec", container, "python", "-c", textwrap.dedent(code)], timeout=timeout, check=False)


@pytest.fixture(scope="module")
def container(tmp_path_factory):
    _run(["docker", "build", "-t", IMAGE, str(ROOT / "docker" / "playground-runner")], timeout=1800)

    data = tmp_path_factory.mktemp("runner-data")
    pg = data / "pg-docker"
    pg.mkdir()
    rows = ["account,month,amount"] + [f"A{i % 50},{i % 12},{(i * 37) % 1000}" for i in range(2000)]
    (pg / "tx.csv").write_text("\n".join(rows) + "\n")
    (pg / "manifest.json").write_text(json.dumps({"https://raw.githubusercontent.com/x/y/main/tx.csv": "tx.csv"}))
    for path in (data, pg, pg / "tx.csv", pg / "manifest.json"):
        path.chmod(0o755 if path.is_dir() else 0o644)

    suffix = uuid.uuid4().hex[:10]
    network = f"pgrunner-test-{suffix}"
    name = f"pgrunner-test-{suffix}"
    service = load_compose()["services"]["playground-runner"]
    _run(["docker", "network", "create", "--internal", network])
    try:
        _run(
            ["docker", "run", "-d", "--name", name, "--network", network, *docker_run_flags(service),
             "-v", f"{data}:/data:ro", IMAGE]
        )
        deadline = time.monotonic() + 60
        while True:
            probe = _exec_python(name, "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2)")
            if probe.returncode == 0:
                break
            if time.monotonic() > deadline:
                logs = _run(["docker", "logs", name], check=False)
                raise AssertionError(f"runner never became healthy:\n{logs.stdout}\n{logs.stderr}")
            time.sleep(1)
        yield name
    finally:
        _run(["docker", "rm", "-f", name], check=False)
        _run(["docker", "network", "rm", network], check=False)


def test_outbound_network_is_blocked(container):
    result = _exec_python(
        container,
        """
        import socket
        for host, port in (("1.1.1.1", 443), ("8.8.8.8", 53)):
            try:
                socket.create_connection((host, port), timeout=5).close()
                print("CONNECTED", host)
            except OSError as exc:
                print("blocked", host, type(exc).__name__)
        try:
            socket.getaddrinfo("pypi.org", 443)
            print("RESOLVED")
        except OSError:
            print("blocked dns")
        """,
    )
    assert "CONNECTED" not in result.stdout and "RESOLVED" not in result.stdout, result.stdout
    assert result.stdout.count("blocked") == 3, result.stdout + result.stderr


def test_root_filesystem_is_read_only_and_scratch_is_tmpfs(container):
    result = _exec_python(
        container,
        """
        import os
        for path in ("/usr/local/lib/x", "/opt/runner/x", "/etc/x", "/data/x", "/x"):
            try:
                open(path, "w").close()
                print("WROTE", path)
            except OSError as exc:
                print("readonly", path, exc.errno)
        for path in ("/tmp/ok", "/work/ok"):
            open(path, "w").close()
            os.remove(path)
            print("writable", path)
        mounts = open("/proc/mounts").read().splitlines()
        root = [m for m in mounts if m.split()[1] == "/"][0]
        print("rootopts", root.split()[3].split(",")[0])
        for target in ("/tmp", "/work"):
            print("fstype", target, [m.split()[2] for m in mounts if m.split()[1] == target][0])
        """,
    )
    out = result.stdout
    assert "WROTE" not in out, out
    assert out.count("readonly") == 5, out + result.stderr
    assert "writable /tmp/ok" in out and "writable /work/ok" in out
    assert "rootopts ro" in out
    assert "fstype /tmp tmpfs" in out and "fstype /work tmpfs" in out


def test_runs_unprivileged(container):
    result = _exec_python(
        container,
        """
        import os
        status = dict(line.split(":", 1) for line in open("/proc/self/status").read().splitlines() if ":" in line)
        print("uid", os.getuid())
        print("capeff", status["CapEff"].strip())
        print("nonewprivs", status["NoNewPrivs"].strip())
        """,
    )
    assert "uid 10001" in result.stdout
    assert "capeff 0000000000000000" in result.stdout
    assert "nonewprivs 1" in result.stdout


def test_end_to_end_session_inside_the_container(container):
    result = _exec_python(
        container,
        r"""
        import json, urllib.request
        BASE = "http://127.0.0.1:8765"
        def call(method, path, body=None):
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(BASE + path, data=data, method=method,
                                         headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        sid = call("POST", "/sessions", {"playground_id": "pg-docker"})["session_id"]
        def run(code, timeout_s=None):
            body = {"code": code}
            if timeout_s:
                body["timeout_s"] = timeout_s
            return call("POST", f"/sessions/{sid}/execute", body)
        out = {}
        out["load"] = run("import pandas as pd\ndf = pd.read_csv('https://raw.githubusercontent.com/x/y/main/tx.csv')\nlen(df)")
        out["features"] = run("feat = df.groupby(['account', 'month']).amount.agg(['sum', 'count']).reset_index()\nfeat.shape")
        out["forest"] = run(
            "from sklearn.ensemble import IsolationForest\n"
            "m = IsolationForest(n_estimators=50, random_state=0).fit(feat[['sum', 'count']])\n"
            "int((m.predict(feat[['sum', 'count']]) == -1).sum())"
        )
        out["miss"] = run("pd.read_csv('https://example.com/nope.csv')")
        out["socket"] = run("import socket\ntry:\n    socket.socket()\n    print('opened')\nexcept PermissionError as e:\n    print('blocked', e.errno)")
        out["seccomp"] = run("print([l for l in open('/proc/self/status') if l.startswith('Seccomp:')][0].split()[1])")
        out["plot"] = run("import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.show()")
        out["timeout"] = run("import time\ntime.sleep(30)", 2)
        out["after"] = run("len(df)")
        out["health"] = call("GET", "/health")
        call("DELETE", f"/sessions/{sid}")
        out["sessions"] = call("GET", "/sessions")["sessions"]
        print(json.dumps(out))
        """,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["load"]["outputs"] == [{"type": "text", "text": "2000"}]
    assert out["features"]["status"] == "ok"
    assert out["forest"]["status"] == "ok", out["forest"]
    assert out["miss"]["outputs"][-1]["ename"] == "OfflineDatasetError"
    assert out["socket"]["outputs"][0]["text"].strip() == "blocked 13"
    assert out["seccomp"]["outputs"][0]["text"].strip() == "2"
    assert [o["type"] for o in out["plot"]["outputs"]] == ["image"]
    assert out["timeout"]["status"] == "timeout"
    assert out["after"]["outputs"] == [{"type": "text", "text": "2000"}]
    assert out["health"]["kernel_seccomp"] is True
    assert out["health"]["limits"]["memory_limit_bytes"] == 6 * 1024**3
    assert out["health"]["packages"]["scikit-learn"]
    assert out["sessions"] == []
