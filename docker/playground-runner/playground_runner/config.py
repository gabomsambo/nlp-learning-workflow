"""Runner settings, read once from the environment.

Every limit has a default sized for the target workload: pandas over about one
million rows and several scikit-learn IsolationForest fits per notebook.
docs/PLAYGROUND_RUNNER.md lists each variable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

#: A playground id names a directory under /data, so it must never contain a
#: path separator or be "." / "..".
PLAYGROUND_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def valid_playground_id(value: str) -> bool:
    return bool(PLAYGROUND_ID_RE.match(value or "")) and ".." not in value


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {raw!r}")
    return value


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {raw!r}")
    return value


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RunnerConfig:
    data_dir: Path = Path("/data")
    work_dir: Path = Path("/work")
    runtime_dir: Path = Path("/tmp/playground-runner")
    exec_timeout_s: float = 180.0
    max_exec_timeout_s: float = 900.0
    max_output_bytes: int = 2_000_000
    max_code_bytes: int = 1_000_000
    max_sessions: int = 3
    idle_timeout_s: float = 1800.0
    reap_interval_s: float = 60.0
    interrupt_grace_s: float = 10.0
    kernel_start_timeout_s: float = 60.0
    # Off only for tests run outside the image; the image always sets it on.
    kernel_seccomp: bool = True

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "RunnerConfig":
        env = os.environ if env is None else env
        base = cls()
        exec_timeout = _float(env, "PLAYGROUND_EXEC_TIMEOUT_S", base.exec_timeout_s)
        max_exec = _float(env, "PLAYGROUND_MAX_EXEC_TIMEOUT_S", max(base.max_exec_timeout_s, exec_timeout))
        if max_exec < exec_timeout:
            raise ValueError("PLAYGROUND_MAX_EXEC_TIMEOUT_S must be at least PLAYGROUND_EXEC_TIMEOUT_S")
        return cls(
            data_dir=Path(env.get("PLAYGROUND_DATA_DIR") or base.data_dir),
            work_dir=Path(env.get("PLAYGROUND_WORK_DIR") or base.work_dir),
            runtime_dir=Path(env.get("PLAYGROUND_RUNTIME_DIR") or base.runtime_dir),
            exec_timeout_s=exec_timeout,
            max_exec_timeout_s=max_exec,
            max_output_bytes=_int(env, "PLAYGROUND_MAX_OUTPUT_BYTES", base.max_output_bytes),
            max_code_bytes=_int(env, "PLAYGROUND_MAX_CODE_BYTES", base.max_code_bytes),
            max_sessions=_int(env, "PLAYGROUND_MAX_SESSIONS", base.max_sessions),
            idle_timeout_s=_float(env, "PLAYGROUND_IDLE_TIMEOUT_S", base.idle_timeout_s),
            reap_interval_s=_float(env, "PLAYGROUND_REAP_INTERVAL_S", base.reap_interval_s),
            interrupt_grace_s=_float(env, "PLAYGROUND_INTERRUPT_GRACE_S", base.interrupt_grace_s),
            kernel_start_timeout_s=_float(env, "PLAYGROUND_KERNEL_START_TIMEOUT_S", base.kernel_start_timeout_s),
            kernel_seccomp=_bool(env, "PLAYGROUND_KERNEL_SECCOMP", base.kernel_seccomp),
        )
