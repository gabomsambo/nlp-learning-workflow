"""Launcher argument parsing and the never-remove-volumes guarantee.

Docker, curl, and xdg-open are stubbed. Nothing here talks to a daemon,
IndexTTS, or Qdrant. Run from any working directory; the script finds its
own checkout.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "workstation.sh"
LINK = ROOT / "workstation"
DESKTOP_TEMPLATE = ROOT / "scripts" / "nlp-workstation.desktop"

DOCKER_STUB = """#!/usr/bin/env bash
set -u
printf 'docker' >> "$WORKSTATION_STUB_LOG"
for arg in "$@"; do
  printf ' %q' "$arg" >> "$WORKSTATION_STUB_LOG"
done
printf '\\n' >> "$WORKSTATION_STUB_LOG"

if [[ "$1" == "info" ]]; then
  if [[ "${WORKSTATION_DOCKER_DOWN:-}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

if [[ "$1" == "image" ]]; then
  printf '%s\\n' "${WORKSTATION_IMAGE_CREATED:-2099-01-01T00:00:00Z}"
  exit 0
fi

if [[ "$1" != "compose" ]]; then
  echo "unexpected docker invocation: $*" >&2
  exit 97
fi

subcommand=""
skip_value=0
for arg in "${@:2}"; do
  if [[ "$skip_value" == "1" ]]; then
    skip_value=0
    continue
  fi
  case "$arg" in
    --project-directory|--format|-f|--file) skip_value=1 ;;
    --*) ;;
    *) subcommand="$arg"; break ;;
  esac
done

if [[ "$subcommand" == "down" ]]; then
  echo "refusing compose down" >&2
  exit 99
fi

if [[ "$subcommand" == "images" ]]; then
  if [[ -n "${WORKSTATION_IMAGE_ID:-}" ]]; then
    printf '%s\\n' "$WORKSTATION_IMAGE_ID"
  fi
  exit 0
fi

if [[ "$subcommand" == "ps" ]]; then
  svc="${@: -1}"
  case "$svc" in
    db) printf '%s\\n' "running|0|Up 1 minute (healthy)" ;;
    scheduler)
      case "${WORKSTATION_SCHEDULER:-exited0}" in
        crash) printf '%s\\n' "exited|1|Exited (1) 1 minute ago" ;;
        running) printf '%s\\n' "running|0|Up 1 minute" ;;
        *) printf '%s\\n' "exited|0|Exited (0) 2 days ago" ;;
      esac
      ;;
    playground-runner)
      case "${WORKSTATION_RUNNER:-healthy}" in
        missing) ;;
        unhealthy) printf '%s\\n' "running|0|Up 5 minutes (unhealthy)" ;;
        *) printf '%s\\n' "running|0|Up 5 minutes (healthy)" ;;
      esac
      ;;
  esac
  exit 0
fi

if [[ "$subcommand" == "logs" ]]; then
  exit 0
fi

exit 0
"""

CURL_STUB = """#!/usr/bin/env bash
set -u
printf 'curl' >> "$WORKSTATION_STUB_LOG"
for arg in "$@"; do
  printf ' %q' "$arg" >> "$WORKSTATION_STUB_LOG"
done
printf '\\n' >> "$WORKSTATION_STUB_LOG"

if [[ "${WORKSTATION_CURL_FAIL:-}" == "1" ]]; then
  exit 7
fi

url=""
for arg in "$@"; do
  case "$arg" in
    http://*|https://*) url="$arg" ;;
  esac
done

body='{"ok": true}'
case "$url" in
  *"/gradio_api/info"*)
    if [[ "${WORKSTATION_TTS_OK:-}" == "1" ]]; then
      body='{"named_endpoints":{"/gen_single":{}}}'
    else
      body=""
    fi
    ;;
  *qdrant.example.test*)
    if [[ "${WORKSTATION_QDRANT:-}" == "suspended" ]]; then
      body="404 page not found"
    else
      body='{"title":"qdrant","version":"1.11.0"}'
    fi
    ;;
  *":3000"*) body='{}' ;;
  *"/healthz"*) body='OK' ;;
  *"/health"*) body='{"ok": true}' ;;
esac
printf '%s' "$body"
exit 0
"""

XDG_STUB = """#!/usr/bin/env bash
printf 'xdg-open %s\\n' "$1" >> "$WORKSTATION_STUB_LOG"
exit 0
"""

# Shadows the real ss. A closed port prints nothing; WORKSTATION_SS_LISTEN_PORT
# is the only port reported as listening, so the suite never inspects the host.
SS_STUB = r"""#!/usr/bin/env bash
printf 'ss' >> "$WORKSTATION_STUB_LOG"
for arg in "$@"; do
  printf ' %q' "$arg" >> "$WORKSTATION_STUB_LOG"
done
printf '\n' >> "$WORKSTATION_STUB_LOG"

want="${WORKSTATION_SS_LISTEN_PORT:-}"
[[ -n "$want" ]] || exit 0
for arg in "$@"; do
  case "$arg" in
    *"sport = :${want}"*)
      printf 'LISTEN 0 1 127.0.0.1:%s 0.0.0.0:*\n' "$want"
      exit 0
      ;;
  esac
done
exit 0
"""


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
def launcher(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "stub.log"
    log.write_text("")
    state = tmp_path / "state"
    data = tmp_path / "data"
    home = tmp_path / "home"
    home.mkdir()
    marker = tmp_path / "tts-started"
    for name, body in (
        ("docker", DOCKER_STUB),
        ("curl", CURL_STUB),
        ("xdg-open", XDG_STUB),
        ("ss", SS_STUB),
    ):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)

    port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "HOME": str(home),
            "XDG_STATE_HOME": str(state),
            "XDG_DATA_HOME": str(data),
            "WORKSTATION_STUB_LOG": str(log),
            "DOCKER_HOST": "unix:///tmp/nlp-workstation-test-does-not-exist.sock",
            "VAULT_HOST_DIR": str(home / "NLP-Vault"),
            "INDEXTTS_URL": f"http://127.0.0.1:{port}",
            "INDEXTTS_START_COMMAND": f"echo started >> {marker}; sleep 60",
            "QDRANT_URL": "http://qdrant.example.test",
            "QDRANT_API_KEY": "test-key-not-real",
            "SCHEDULE_ENABLED": "false",
            "WEBUI_WAIT_SECONDS": "5",
            "TTS_WAIT_SECONDS": "5",
        }
    )
    for key in (
        "WORKSTATION_DOCKER_DOWN",
        "WORKSTATION_CURL_FAIL",
        "WORKSTATION_TTS_FAIL",
        "WORKSTATION_QDRANT",
        "WORKSTATION_IMAGE_ID",
        "WORKSTATION_IMAGE_CREATED",
        "WORKSTATION_SCHEDULER",
        "WORKSTATION_SS_LISTEN_PORT",
    ):
        env.pop(key, None)
    # The copy of the parent environment must not point mkdir at a real vault.
    # VAULT_HOST_DIR is set to the fixture home above; drop a parent value
    # only if that assignment is later removed.
    assert env["VAULT_HOST_DIR"] == str(home / "NLP-Vault")

    real_desktop = Path.home() / ".local/share/applications/nlp-workstation.desktop"
    before = real_desktop.stat().st_mtime if real_desktop.exists() else None

    def run(args, *, extra=None, cwd=None):
        merged = env.copy()
        if extra:
            merged.update(extra)
        return subprocess.run(
            [str(LINK), *args],
            cwd=cwd or tmp_path,
            env=merged,
            text=True,
            capture_output=True,
            check=False,
        )

    yield {"run": run, "log": log, "marker": marker, "state": state, "home": home, "port": port, "env": env}

    pid_file = state / "nlp-workstation" / "indextts.pid"
    if pid_file.exists():
        pid_text = pid_file.read_text().strip()
        if pid_text.isdigit():
            try:
                os.killpg(int(pid_text), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(int(pid_text), signal.SIGKILL)
                except ProcessLookupError:
                    pass
    after = real_desktop.stat().st_mtime if real_desktop.exists() else None
    assert before == after


def _log(launcher) -> str:
    return launcher["log"].read_text()


def _wait_for(path: Path, timeout: float = 3.0) -> bool:
    """Background IndexTTS start writes the marker after up has returned."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return path.exists()


def test_script_and_symlink_exist():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)
    assert LINK.is_symlink()
    assert LINK.resolve() == SCRIPT.resolve()
    text = SCRIPT.read_text()
    assert "down -v" not in text
    assert "down --volumes" not in text
    assert "compose down" not in text
    assert "--volumes" not in text


def test_shellcheck():
    shellcheck = subprocess.run(
        ["shellcheck", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
    )
    if shellcheck.returncode == 127:
        pytest.skip("shellcheck is not installed")
    assert shellcheck.returncode == 0, shellcheck.stdout + shellcheck.stderr


def test_no_args_prints_help(launcher):
    proc = launcher["run"]([])
    assert proc.returncode == 0
    assert "NLP learning workstation" in proc.stdout
    for name in ("up", "down", "status", "logs", "install-desktop"):
        assert name in proc.stdout
    assert "\x1b" not in proc.stdout
    assert _log(launcher) == ""


def test_unknown_command_and_option_do_not_call_docker(launcher):
    unknown = launcher["run"](["explode"])
    assert unknown.returncode == 2
    assert "Unknown command" in unknown.stderr
    bad = launcher["run"](["up", "--rebuild"])
    assert bad.returncode == 2
    assert "Unknown option" in bad.stderr
    assert _log(launcher) == ""


def test_up_without_tts_does_not_start_indextts_or_rebuild(launcher):
    proc = launcher["run"](["up"])
    assert proc.returncode == 0, proc.stderr
    assert (
        "IndexTTS was not started by this command. "
        "Start it with: ./workstation up --tts"
    ) in proc.stdout
    assert "IndexTTS is off" not in proc.stdout
    vault = launcher["home"] / "NLP-Vault"
    assert vault.is_dir()
    assert vault.stat().st_uid == os.getuid()
    assert "Waiting for the app ready." in proc.stdout
    assert "App: http://localhost:8000/" in proc.stdout
    assert not launcher["marker"].exists()
    log = _log(launcher)
    assert " up -d" in log
    assert "--build" not in log
    assert " stop" not in log
    assert "xdg-open http://localhost:8000/" in log


def test_up_without_tts_notices_a_running_indextts(launcher):
    proc = launcher["run"](
        ["up"],
        extra={"WORKSTATION_SS_LISTEN_PORT": str(launcher["port"])},
    )
    assert proc.returncode == 0, proc.stderr
    assert (
        f"IndexTTS is already listening on port {launcher['port']}."
        in proc.stdout
    )
    assert "IndexTTS is off" not in proc.stdout
    assert "was not started" not in proc.stdout
    assert not launcher["marker"].exists()


def test_up_creates_a_custom_vault_directory(launcher):
    target = launcher["home"] / "custom-vault"
    proc = launcher["run"](["up"], extra={"VAULT_HOST_DIR": str(target)})
    assert proc.returncode == 0, proc.stderr
    assert target.is_dir()
    assert target.stat().st_uid == os.getuid()
    assert not (launcher["home"] / "NLP-Vault").exists()


def test_vault_bind_does_not_ask_docker_to_create_the_host_path():
    text = (ROOT / "docker-compose.yml").read_text()
    assert text.count("create_host_path: false") >= 2
    assert "${VAULT_HOST_DIR:-${HOME}/NLP-Vault}:/vault" not in text


def test_up_build_and_tts_flag_order(launcher):
    proc = launcher["run"](["up", "--tts", "--build"], extra={"WORKSTATION_TTS_OK": "1"})
    assert proc.returncode == 0, proc.stderr
    log = _log(launcher)
    assert " up -d --build" in log
    assert _wait_for(launcher["marker"]), proc.stdout + proc.stderr
    pid_file = launcher["state"] / "nlp-workstation" / "indextts.pid"
    assert pid_file.is_file()
    pid = int(pid_file.read_text().strip())
    os.kill(pid, 0)


def test_up_tts_skips_start_when_port_is_taken(launcher):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        proc = launcher["run"](
            ["up", "--tts"],
            extra={
                "INDEXTTS_URL": f"http://127.0.0.1:{port}",
                "WORKSTATION_TTS_OK": "1",
                "WORKSTATION_SS_LISTEN_PORT": str(port),
            },
        )
        assert proc.returncode == 0, proc.stderr
        assert f"already listening on port {port}" in proc.stdout
        assert not launcher["marker"].exists()
        pid_file = launcher["state"] / "nlp-workstation" / "indextts.pid"
        assert not pid_file.exists()
    finally:
        sock.close()


def test_down_stops_only_the_pid_this_script_started(launcher):
    started = launcher["run"](["up", "--tts"], extra={"WORKSTATION_TTS_OK": "1"})
    assert started.returncode == 0, started.stderr
    pid_file = launcher["state"] / "nlp-workstation" / "indextts.pid"
    pid = int(pid_file.read_text().strip())
    os.kill(pid, 0)
    stopped = launcher["run"](["down"])
    assert stopped.returncode == 0, stopped.stderr
    assert "Stopping IndexTTS" in stopped.stdout
    assert "Database and uploads stay on disk." in stopped.stdout
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not pid_file.exists()
    log = _log(launcher)
    assert " stop" in log
    assert " down" not in log


def test_down_leaves_a_listener_it_did_not_start(launcher):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        proc = launcher["run"](
            ["up", "--tts"],
            extra={
                "INDEXTTS_URL": f"http://127.0.0.1:{port}",
                "WORKSTATION_TTS_OK": "1",
                "WORKSTATION_SS_LISTEN_PORT": str(port),
            },
        )
        assert proc.returncode == 0, proc.stderr
        stopped = launcher["run"](["down"])
        assert stopped.returncode == 0, stopped.stderr
        assert "Stopping IndexTTS" not in stopped.stdout
        assert sock.getsockname()[1] == port
    finally:
        sock.close()


def test_stale_image_suggests_build_unless_build_was_requested(launcher):
    stale = launcher["run"](
        ["up"],
        extra={
            "WORKSTATION_IMAGE_ID": "sha256:deadbeef",
            "WORKSTATION_IMAGE_CREATED": "2020-01-01T00:00:00Z",
        },
    )
    assert stale.returncode == 0, stale.stderr
    assert "older than the last commit" in stale.stdout
    assert "./workstation up --build" in stale.stdout
    assert "--build" not in _log(launcher)

    launcher["log"].write_text("")
    built = launcher["run"](
        ["up", "--build"],
        extra={
            "WORKSTATION_IMAGE_ID": "sha256:deadbeef",
            "WORKSTATION_IMAGE_CREATED": "2020-01-01T00:00:00Z",
        },
    )
    assert built.returncode == 0, built.stderr
    assert "older than the last commit" not in built.stdout
    assert " up -d --build" in _log(launcher)


def test_up_reports_docker_down_without_compose(launcher):
    proc = launcher["run"](["up"], extra={"WORKSTATION_DOCKER_DOWN": "1"})
    assert proc.returncode == 1
    assert "Docker is not running" in proc.stderr
    log = _log(launcher)
    assert " info" in log
    assert " up " not in log


def test_health_timeout(launcher):
    proc = launcher["run"](
        ["up"],
        extra={"WORKSTATION_CURL_FAIL": "1", "WEBUI_WAIT_SECONDS": "1"},
    )
    assert proc.returncode == 1
    assert "did not become ready" in proc.stdout
    assert "logs webui" in proc.stderr


def test_tts_probe_timeout(launcher):
    proc = launcher["run"](
        ["up", "--tts"],
        extra={"TTS_WAIT_SECONDS": "1"},
    )
    assert proc.returncode == 1
    assert "IndexTTS did not become ready" in proc.stdout
    assert _wait_for(launcher["marker"]), proc.stdout + proc.stderr


def test_status_ok_and_scheduler_exited_zero_is_normal(launcher):
    proc = launcher["run"](["status"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "Web UI" in out and "OK" in out
    assert "Postgres" in out and "healthy" in out
    assert "PostgREST" in out
    assert "SearXNG" in out
    assert "IndexTTS" in out and "DOWN" in out
    assert "./workstation up --tts" in out
    assert "Qdrant" in out and "reachable" in out
    assert "Qdrant Cloud" in out and "not configured" in out
    assert "Exited (0) is normal" in out
    assert "\x1b" not in out


def test_status_qdrant_suspended_and_scheduler_crash(launcher):
    proc = launcher["run"](
        ["status"],
        extra={
            "WORKSTATION_QDRANT": "suspended",
            "WORKSTATION_SCHEDULER": "crash",
            "QDRANT_CLOUD_URL": "http://qdrant.example.test",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "suspended free-tier cluster" in proc.stdout
    assert "Resume the cluster" in proc.stdout
    assert "Exited (1)" in proc.stdout
    assert "./workstation logs scheduler" in proc.stdout


def test_status_when_scheduling_is_on_but_scheduler_exited(launcher):
    proc = launcher["run"](
        ["status"],
        extra={"SCHEDULE_ENABLED": "true"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "while scheduling is turned on" in proc.stdout
    assert "Exited (0) is normal" not in proc.stdout


def test_status_when_docker_is_down(launcher):
    proc = launcher["run"](
        ["status"],
        extra={"WORKSTATION_DOCKER_DOWN": "1", "WORKSTATION_CURL_FAIL": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "Docker is not running" in proc.stdout
    assert proc.stdout.count("DOWN") >= 5
    assert "\x1b" not in proc.stdout


def test_logs_service_and_tts(launcher):
    webui = launcher["run"](["logs", "webui"])
    assert webui.returncode == 0, webui.stderr
    assert " logs -f webui" in _log(launcher)
    extra = launcher["run"](["logs", "webui", "db"])
    assert extra.returncode == 2

    log_file = launcher["state"] / "nlp-workstation" / "indextts.log"
    proc = subprocess.Popen(
        [str(LINK), "logs", "tts"],
        cwd=launcher["home"],
        env=launcher["env"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 3
        while not log_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert log_file.exists()
        with log_file.open("a") as handle:
            handle.write("hello-from-tts\n")
            handle.flush()
        assert proc.stdout is not None
        collected = ""
        deadline = time.time() + 3
        while "hello-from-tts" not in collected and time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            collected += line
        assert "hello-from-tts" in collected
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


def test_install_desktop_writes_this_checkout_only(launcher):
    proc = launcher["run"](["install-desktop"])
    assert proc.returncode == 0, proc.stderr
    # XDG_DATA_HOME wins over ~/.local/share.
    target = Path(launcher["env"]["XDG_DATA_HOME"]) / "applications" / "nlp-workstation.desktop"
    text = target.read_text()
    assert f"Exec={ROOT}/workstation up" in text
    assert f"Path={ROOT}" in text
    assert "Terminal=true" in text
    assert "@EXEC@" not in text
    assert "@PATH@" not in text
    assert "Installed" in proc.stdout
    assert not (launcher["home"] / ".local/share/applications/nlp-workstation.desktop").exists()


def test_help_is_coloured_on_a_tty(launcher):
    import pty

    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [str(LINK)],
        cwd=launcher["home"],
        env=launcher["env"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    chunks = []
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            chunk = os.read(master, 4096)
        except OSError:
            break
        if not chunk:
            break
        chunks.append(chunk)
        if b"install-desktop" in b"".join(chunks):
            break
    proc.wait(timeout=3)
    os.close(master)
    output = b"".join(chunks)
    assert proc.returncode == 0
    assert b"\x1b" in output


def test_get_started_points_at_workstation():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "get_started.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "superseded" in proc.stdout
    assert "./workstation up" in proc.stdout
    assert "./workstation up --tts" in proc.stdout


def test_compose_defaults_come_from_docker_compose():
    """Host-side IndexTTS settings fall back to the compose file, port 7860."""
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"source '{SCRIPT}'; "
            "compose_default INDEXTTS_URL; printf '\\n'; "
            "compose_default INDEXTTS_START_COMMAND; printf '\\n'",
        ],
        text=True,
        capture_output=True,
        check=False,
        env={"HOME": "/tmp", "PATH": os.environ.get("PATH", "")},
    )
    assert proc.returncode == 0, proc.stderr
    url, command = proc.stdout.splitlines()
    assert url == "http://host.docker.internal:7860"
    assert "--port 7860" in command
    assert "uv run webui.py" in command


def test_readme_documents_workstation_and_port_7860():
    readme = (ROOT / "README.md").read_text()
    start = readme.split("### Start it", 1)[1].split("## ", 1)[0]
    assert "./workstation up" in start
    assert "./workstation up --tts" in start
    assert "7860" in start
    assert "7861" not in start
    agents = (ROOT / "AGENTS.md").read_text()
    assert "http://host.docker.internal:7860" in agents
    assert "http://host.docker.internal:7861" not in agents
    template = DESKTOP_TEMPLATE.read_text()
    assert "Terminal=true" in template
    assert "@EXEC@" in template


def test_status_shows_the_playground_runner_from_its_healthcheck(launcher):
    proc = launcher["run"](["status"])
    assert proc.returncode == 0, proc.stderr
    line = next(l for l in proc.stdout.splitlines() if l.startswith("Playground"))
    assert "OK" in line and "(healthy)" in line
    # No host port exists, so status must not curl one.
    assert "8765" not in launcher["log"].read_text()


def test_status_playground_runner_unhealthy_or_missing(launcher):
    proc = launcher["run"](["status"], extra={"WORKSTATION_RUNNER": "unhealthy"})
    assert proc.returncode == 0, proc.stderr
    line = next(l for l in proc.stdout.splitlines() if l.startswith("Playground"))
    assert "DOWN" in line and "unhealthy" in line
    assert "./workstation logs playground-runner" in proc.stdout

    proc = launcher["run"](["status"], extra={"WORKSTATION_RUNNER": "missing"})
    line = next(l for l in proc.stdout.splitlines() if l.startswith("Playground"))
    assert "DOWN" in line and "no playground runner container" in line


def test_help_names_the_playground_runner(launcher):
    proc = launcher["run"](["help"])
    assert "playground-runner" in proc.stdout
    assert "playground runner" in proc.stdout.lower()
