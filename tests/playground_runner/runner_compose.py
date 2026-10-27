"""Read the playground-runner service out of docker-compose.yml.

Shared by the static lockdown test and the Docker integration test, so the
container the integration test runs is the one compose would run.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"
_DEFAULT = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]*)\}")


def load_compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


def with_defaults(value) -> str:
    """Resolve ${VAR:-default} to its default, as compose does with VAR unset."""
    return _DEFAULT.sub(lambda m: m.group(1), str(value))


def docker_run_flags(service: dict) -> list:
    """The `docker run` flags equivalent to the service's lockdown keys."""
    flags = []
    if service.get("read_only"):
        flags.append("--read-only")
    for mount in service.get("tmpfs", []):
        flags += ["--tmpfs", with_defaults(mount)]
    for cap in service.get("cap_drop", []):
        flags += ["--cap-drop", cap]
    for opt in service.get("security_opt", []):
        flags += ["--security-opt", opt]
    flags += ["--memory", with_defaults(service["mem_limit"])]
    flags += ["--memory-swap", with_defaults(service["memswap_limit"])]
    flags += ["--cpus", with_defaults(service["cpus"])]
    flags += ["--pids-limit", with_defaults(service["pids_limit"])]
    flags += ["--shm-size", with_defaults(service["shm_size"])]
    for key, value in (service.get("environment") or {}).items():
        flags += ["-e", f"{key}={with_defaults(value)}"]
    return flags
