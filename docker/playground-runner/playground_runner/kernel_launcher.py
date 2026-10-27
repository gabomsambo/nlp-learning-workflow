"""Start one IPython kernel with the runner's per-process lockdown.

Usage (from the generated kernelspec):

    python -m playground_runner.kernel_launcher -f {connection_file}

Before ipykernel starts this process:

* raises its own OOM score, so when the container hits its memory limit the
  kernel that caused it is killed rather than the API server that would report
  it;
* when PLAYGROUND_KERNEL_SECCOMP is on, installs a seccomp filter that refuses
  IP sockets (AF_INET, AF_INET6, AF_PACKET) and io_uring. The container is
  already on an internal-only network; this second layer stops notebook code
  from reaching the one thing that network does connect to, the web app, and
  from calling this runner's own API. The kernel talks to the server over
  Unix-domain sockets (ipc transport), which the filter leaves alone.

The filter is inherited by every child process and cannot be removed:
no-new-privileges is set on the container and by libseccomp itself.
"""

from __future__ import annotations

import errno
import os
import socket
import sys
import warnings

BLOCKED_FAMILIES = (socket.AF_INET, socket.AF_INET6, getattr(socket, "AF_PACKET", 17))


def raise_oom_score(value: int = 1000) -> None:
    try:
        with open("/proc/self/oom_score_adj", "w") as fh:
            fh.write(str(value))
    except OSError:
        pass


def install_network_filter() -> None:
    import pyseccomp as seccomp

    filt = seccomp.SyscallFilter(seccomp.ALLOW)
    for family in BLOCKED_FAMILIES:
        filt.add_rule(seccomp.ERRNO(errno.EACCES), "socket", seccomp.Arg(0, seccomp.EQ, family))
    for name in ("io_uring_setup", "io_uring_enter", "io_uring_register"):
        try:
            filt.add_rule(seccomp.ERRNO(errno.ENOSYS), name)
        except Exception:  # syscall unknown to this libseccomp
            pass
    filt.load()


def main(argv=None) -> None:
    raise_oom_score()
    if os.environ.get("PLAYGROUND_KERNEL_SECCOMP", "1").strip().lower() in {"1", "true", "yes", "on"}:
        install_network_filter()
        # ipykernel forwards output from forked child processes over a TCP
        # loopback pipe, which the filter refuses. Cell output is unaffected;
        # only print() from inside a multiprocessing child is not shown.
        warnings.filterwarnings("ignore", message="Couldn't bind IOPub Pipe")
    from ipykernel import kernelapp

    sys.argv = [sys.argv[0]] + list(sys.argv[1:] if argv is None else argv)
    kernelapp.launch_new_instance()


if __name__ == "__main__":
    main()
