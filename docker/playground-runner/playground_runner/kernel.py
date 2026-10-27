"""Drive one Jupyter kernel: start, execute with a deadline, interrupt, restart.

jupyter_client is imported lazily so the rest of the package (and its tests)
loads without it.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import RunnerConfig
from .outputs import OutputCollector

KERNEL_NAME = "playground"

# Run silently after every (re)start. Leaves no names in the user's namespace.
SETUP_CODE = (
    "def __playground_setup():\n"
    "    import os\n"
    "    from playground_runner import indent, offline_data\n"
    "    indent.install(get_ipython())\n"
    "    offline_data.install(os.environ.get('PLAYGROUND_DATA_DIR', '/data'), os.environ.get('PLAYGROUND_ID'))\n"
    "    try:\n"
    "        import matplotlib\n"
    "        matplotlib.use('module://matplotlib_inline.backend_inline')\n"
    "    except ImportError:\n"
    "        pass\n"
    "__playground_setup()\n"
    "del __playground_setup\n"
)


class KernelStartError(RuntimeError):
    pass


def write_kernelspec(runtime_dir: Path) -> Path:
    """A kernelspec whose argv is our launcher rather than plain ipykernel."""
    spec_dir = runtime_dir / "kernels" / KERNEL_NAME
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "argv": [sys.executable, "-m", "playground_runner.kernel_launcher", "-f", "{connection_file}"],
        "display_name": "Playground (Python 3)",
        "language": "python",
        "interrupt_mode": "signal",
    }
    (spec_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec_dir.parent


def _python_path() -> str:
    """Make sure the kernel can import playground_runner wherever it lives."""
    here = str(Path(__file__).resolve().parent.parent)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if here not in parts:
        parts.insert(0, here)
    return os.pathsep.join(parts)


class JupyterKernel:
    """One kernel process. Not safe for concurrent execute calls; the session serialises."""

    def __init__(self, config: RunnerConfig, session_id: str, playground_id: str):
        self.config = config
        self.session_id = session_id
        self.playground_id = playground_id
        self.session_dir = config.runtime_dir / "sessions" / session_id
        self.work_dir = config.work_dir / session_id
        self.ipc_dir: Optional[Path] = None
        self.km = None
        self.kc = None
        # The playground's setup code (docs/PLAYGROUND.md, "Setup code"): run
        # after the runner's own setup on every start and restart.
        self.setup_code = ""
        self.setup_result: Dict[str, Any] = {"status": "none"}

    # -- lifecycle --------------------------------------------------------

    def _env(self) -> Dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PLAYGROUND_") or key in ("PLAYGROUND_KERNEL_SECCOMP",)
        }
        env.update(
            {
                "PLAYGROUND_ID": self.playground_id,
                "PLAYGROUND_DATA_DIR": str(self.config.data_dir),
                "PLAYGROUND_KERNEL_SECCOMP": "1" if self.config.kernel_seccomp else "0",
                "PYTHONPATH": _python_path(),
                "HOME": str(self.work_dir),
                "IPYTHONDIR": str(self.session_dir / "ipython"),
                "JUPYTER_RUNTIME_DIR": str(self.session_dir / "jupyter"),
                "JUPYTER_DATA_DIR": str(self.session_dir / "jupyter"),
                "JUPYTER_CONFIG_DIR": str(self.session_dir / "jupyter"),
                "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR") or str(self.session_dir / "matplotlib"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYDEVD_DISABLE_FILE_VALIDATION": "1",
            }
        )
        return env

    async def start(self) -> None:
        from jupyter_client.kernelspec import KernelSpecManager
        from jupyter_client.manager import AsyncKernelManager

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        kernels_dir = write_kernelspec(self.config.runtime_dir)
        ksm = KernelSpecManager(kernel_dirs=[str(kernels_dir)])
        ksm.ensure_native_kernel = False
        # ZeroMQ ipc endpoints are Unix sockets, whose paths are capped near
        # 108 bytes; a short directory of their own keeps them under it.
        self.ipc_dir = Path(tempfile.mkdtemp(prefix="pgk-"))
        self.km = AsyncKernelManager(
            kernel_name=KERNEL_NAME,
            kernel_spec_manager=ksm,
            transport="ipc",
            ip=str(self.ipc_dir / "k"),
            connection_file=str(self.session_dir / "connection.json"),
        )
        try:
            await self.km.start_kernel(env=self._env(), cwd=str(self.work_dir))
            self.kc = self.km.client()
            self.kc.start_channels()
            await self.kc.wait_for_ready(timeout=self.config.kernel_start_timeout_s)
            await self._run_setup()
        except Exception as exc:
            await self.shutdown()
            raise KernelStartError(f"The Python kernel did not start: {exc}") from exc

    async def _run_setup(self) -> None:
        msg_id = self.kc.execute(SETUP_CODE, silent=True, store_history=False, allow_stdin=False)
        reply = await self._shell_reply(msg_id, self.config.kernel_start_timeout_s)
        content = (reply or {}).get("content", {})
        if content.get("status") != "ok":
            raise KernelStartError(
                "playground setup failed: "
                f"{content.get('ename', 'no reply')}: {content.get('evalue', '')}"
            )
        await self._run_user_setup()

    async def _run_user_setup(self) -> None:
        """Run the playground's setup code. Its failure is reported, not raised:
        the session still starts, and the cell that needs a missing name says so."""
        code = self.setup_code or ""
        if not code.strip():
            self.setup_result = {"status": "none"}
            return
        timeout = self.config.exec_timeout_s
        msg_id = self.kc.execute(code, silent=True, store_history=False, allow_stdin=False)
        reply = await self._shell_reply(msg_id, timeout)
        if reply is None:
            await self.interrupt()
            await self._shell_reply(msg_id, self.config.interrupt_grace_s)
            self.setup_result = {
                "status": "error",
                "ename": "TimeoutError",
                "evalue": f"The setup code did not finish in {timeout:g} s and was interrupted.",
            }
            return
        content = reply.get("content", {})
        if content.get("status") == "ok":
            self.setup_result = {"status": "ok"}
        else:
            self.setup_result = {
                "status": "error",
                "ename": content.get("ename") or "Error",
                "evalue": content.get("evalue") or "",
            }

    async def restart(self) -> None:
        if self.km is None:
            await self.start()
            return
        await self.km.restart_kernel(now=True)
        await self.kc.wait_for_ready(timeout=self.config.kernel_start_timeout_s)
        await self._run_setup()

    async def interrupt(self) -> None:
        if self.km is not None:
            await self.km.interrupt_kernel()

    async def is_alive(self) -> bool:
        if self.km is None:
            return False
        return bool(await self.km.is_alive())

    async def shutdown(self) -> None:
        kc, km = self.kc, self.km
        self.kc = None
        self.km = None
        if kc is not None:
            try:
                kc.stop_channels()
            except Exception:
                pass
        if km is not None:
            try:
                await km.shutdown_kernel(now=True)
            except Exception:
                pass
        shutil.rmtree(self.session_dir, ignore_errors=True)
        shutil.rmtree(self.work_dir, ignore_errors=True)
        if self.ipc_dir is not None:
            shutil.rmtree(self.ipc_dir, ignore_errors=True)
            self.ipc_dir = None

    # -- execution --------------------------------------------------------

    async def _shell_reply(self, msg_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                msg = await self.kc.get_shell_msg(timeout=remaining)
            except (queue.Empty, asyncio.TimeoutError, TimeoutError):
                return None
            if msg.get("parent_header", {}).get("msg_id") == msg_id:
                return msg

    async def _drain_iopub(self) -> None:
        while True:
            try:
                await self.kc.get_iopub_msg(timeout=0.05)
            except (queue.Empty, asyncio.TimeoutError, TimeoutError):
                return

    async def execute(
        self,
        code: str,
        timeout_s: float,
        abort: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """Run code; always returns a result dict, never raises for user errors."""
        cfg = self.config
        collector = OutputCollector(cfg.max_output_bytes)
        started = time.monotonic()
        deadline = started + timeout_s
        grace_deadline: Optional[float] = None
        timed_out = False
        kernel_restarted = False
        died = False
        aborted = False
        execution_count: Optional[int] = None

        await self._drain_iopub()
        msg_id = self.kc.execute(code, store_history=True, allow_stdin=False, stop_on_error=True)

        while True:
            now = time.monotonic()
            if abort is not None and abort.is_set():
                aborted = True
                break
            if not timed_out and now >= deadline:
                timed_out = True
                grace_deadline = now + cfg.interrupt_grace_s
                await self.interrupt()
            if timed_out and grace_deadline is not None and now >= grace_deadline:
                # The interrupt did not land (e.g. stuck in C code): start over.
                await self.restart()
                kernel_restarted = True
                break
            limit = (grace_deadline if timed_out else deadline) - now
            try:
                msg = await self.kc.get_iopub_msg(timeout=max(0.05, min(1.0, limit)))
            except (queue.Empty, asyncio.TimeoutError, TimeoutError):
                if not await self.is_alive():
                    died = True
                    break
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg.get("msg_type")
            content = msg.get("content", {})
            if msg_type == "status":
                if content.get("execution_state") == "idle":
                    break
                continue
            if msg_type == "execute_input":
                execution_count = content.get("execution_count")
                continue
            collector.add_message(msg_type, content)

        status = "ok"
        reply_content: Dict[str, Any] = {}
        if died:
            await self.restart()
            kernel_restarted = True
            status = "error"
            collector.add_error(
                "KernelDied",
                "The Python process stopped while running this code, most likely because it ran out of "
                "memory. The session was restarted, so every variable is gone: run the earlier cells again.",
            )
        elif aborted:
            status = "error"
            collector.add_error("Cancelled", "The session was restarted or closed while this code was running.")
        elif not kernel_restarted:
            reply = await self._shell_reply(msg_id, 5.0)
            reply_content = (reply or {}).get("content", {})
            if reply_content.get("execution_count") is not None:
                execution_count = reply_content["execution_count"]
            if reply_content.get("status") == "error" or collector.has_error("KeyboardInterrupt"):
                status = "error"

        if timed_out:
            status = "timeout"
            collector.remove_errors("KeyboardInterrupt")
            if kernel_restarted:
                note = (
                    f"Stopped after {timeout_s:g} s. The code did not respond to the interrupt, so the session "
                    "was restarted and every variable is gone: run the earlier cells again."
                )
            else:
                note = (
                    f"Stopped after {timeout_s:g} s. Variables from earlier cells are kept; anything this cell "
                    "assigned before it was stopped may be partly done."
                )
            collector.add_error("TimeoutError", note)

        result = collector.result()
        result.update(
            {
                "status": status,
                "execution_ms": int(round((time.monotonic() - started) * 1000)),
                "execution_count": execution_count,
                "kernel_restarted": kernel_restarted,
            }
        )
        return result
