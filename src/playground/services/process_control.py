"""Shared restart / shutdown machinery for the playground Flask process.

`api/run_context.py::set_run_context` and `api/admin.py::refresh` both need
to tear down the current Python process so the listening socket on PORT is
released cleanly, then re-exec `python -m playground.server` (potentially
with a new env). Doing that from inside the Flask request handler is unsafe
(`os.execvpe` would inherit the listening socket FD and the new server
would bind-fail), so we spawn a detached helper Popen that sleeps, then
execvps the server, and `os._exit(0)` the current process after a short
grace period. The detached helper is launched with `start_new_session=True`
and `close_fds=True` so the listening FD is not inherited.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time


_GRACE_BEFORE_EXIT_S = 0.3   # let the HTTP response flush
_HELPER_DELAY_S      = 1.5   # let the OS reclaim PORT before the new bind


def _build_helper_code() -> str:
    server_argv = [sys.executable, "-m", "playground.server"]
    return (
        "import os, sys, time; "
        f"time.sleep({_HELPER_DELAY_S}); "
        f"os.execvp({sys.executable!r}, {server_argv!r})"
    )


def _restart_now(env: dict | None) -> None:
    """Spawn the detached relaunch helper, then hard-exit."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.Popen(
        [sys.executable, "-c", _build_helper_code()],
        env=full_env,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    time.sleep(_GRACE_BEFORE_EXIT_S)
    os._exit(0)


def _shutdown_now() -> None:
    time.sleep(_GRACE_BEFORE_EXIT_S)
    os._exit(0)


def spawn_replacement(env: dict | None = None) -> None:
    """Schedule a process replacement on a daemon thread; returns immediately."""
    threading.Thread(target=lambda: _restart_now(env), daemon=True).start()


def schedule_shutdown() -> None:
    """Schedule a clean shutdown on a daemon thread; returns immediately."""
    threading.Thread(target=_shutdown_now, daemon=True).start()
