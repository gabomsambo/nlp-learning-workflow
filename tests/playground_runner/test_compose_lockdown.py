"""The runner's lockdown as declared in docker-compose.yml. No Docker needed.

Each assertion is one line of docs/PLAYGROUND_RUNNER.md's lockdown list; if
one fails, the doc and the compose file disagree about what protects the host.
"""

from runner_compose import ROOT, load_compose, with_defaults


def _runner():
    return load_compose()["services"]["playground-runner"]


def test_runner_has_no_host_port_and_only_the_internal_network():
    compose = load_compose()
    runner = compose["services"]["playground-runner"]
    assert "ports" not in runner
    assert "network_mode" not in runner
    assert runner["networks"] == ["playground"]
    assert compose["networks"]["playground"]["internal"] is True
    # Only webui shares that network with it.
    on_playground = sorted(
        name for name, svc in compose["services"].items() if "playground" in (svc.get("networks") or [])
    )
    assert on_playground == ["playground-runner", "webui"]
    assert compose["services"]["webui"]["networks"] == ["default", "playground"]


def test_runner_filesystem_privileges_and_limits():
    runner = _runner()
    assert runner["read_only"] is True
    tmpfs = [with_defaults(m) for m in runner["tmpfs"]]
    assert any(m.startswith("/tmp:") for m in tmpfs)
    assert any(m.startswith("/work:") and "uid=10001" in m for m in tmpfs)
    assert runner["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in runner["security_opt"]
    assert "privileged" not in runner and "cap_add" not in runner
    assert with_defaults(runner["mem_limit"]) == "6g"
    assert with_defaults(runner["memswap_limit"]) == with_defaults(runner["mem_limit"])
    assert float(with_defaults(runner["cpus"])) > 0
    assert int(with_defaults(runner["pids_limit"])) > 0
    for key in ("mem_limit", "cpus", "pids_limit"):
        assert "${PLAYGROUND_RUNNER_" in str(runner[key]), f"{key} must be configurable from the environment"
    env = {k: with_defaults(v) for k, v in runner["environment"].items()}
    assert env["PLAYGROUND_EXEC_TIMEOUT_S"] == "180"
    assert env["PLAYGROUND_MAX_SESSIONS"] == "3"
    assert env["PLAYGROUND_IDLE_TIMEOUT_S"] == "1800"


def test_runner_reads_the_dataset_volume_read_only_and_touches_no_existing_volume():
    compose = load_compose()
    runner = compose["services"]["playground-runner"]
    assert runner["volumes"] == ["playground_data:/data:ro"]
    assert compose["volumes"]["playground_data"]["name"] == "nlp_playground_data"
    # The app's real data volumes keep their names and stay off the runner.
    assert compose["volumes"]["pg_data"]["name"] == "nlp_pg_data"
    assert compose["volumes"]["uploads_data"]["name"] == "nlp_uploads"
    assert compose["volumes"]["qdrant_data"]["name"] == "nlp_qdrant_data"


def test_runner_is_its_own_image_without_app_code_or_keys():
    runner = _runner()
    assert runner["build"]["context"] == "./docker/playground-runner"
    assert runner["image"] == "nlp-playground-runner:local"
    assert "env_file" not in runner
    dockerfile = (ROOT / "docker" / "playground-runner" / "Dockerfile").read_text()
    assert "USER runner" in dockerfile
    assert "COPY nlp_pillars" not in dockerfile and "COPY webui" not in dockerfile
    webui_env = load_compose()["services"]["webui"]["environment"]
    assert with_defaults(webui_env["PLAYGROUND_RUNNER_URL"]) == "http://playground-runner:8765"
