# src/playground/services/cluster.py
"""SSH/SCP wrappers used by /api/cluster/*.

Everything shells out to the system `ssh` / `scp` binaries via
`subprocess.run([...])` (never `shell=True`, never string concatenation).
The user must have a working `Host gpu2` entry in ~/.ssh/config with
key-based auth — we always pass `BatchMode=yes` so password prompts fail
fast instead of hanging the request thread.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path


_HOST = "gpu2"
_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
_SAFE_RE = re.compile(r"^[A-Za-z0-9_./=+-]*\Z")


def _safe_rel(rel: str) -> str:
    """Reject anything that could escape the configured root or smuggle shell."""
    if ".." in rel:
        raise ValueError(f"invalid path (contains '..'): {rel!r}")
    if not _SAFE_RE.match(rel):
        raise ValueError(f"invalid path (forbidden chars): {rel!r}")
    return rel


def ping() -> dict:
    """Probe `ssh gpu2 true`. Returns {ok, latency_ms, error}."""
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["ssh", *_SSH_OPTS, _HOST, "true"],
            capture_output=True, text=True, timeout=8,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "latency_ms": 8000, "error": "ssh timeout"}
    latency_ms = int((time.monotonic() - t0) * 1000)
    if proc.returncode == 0:
        return {"ok": True, "latency_ms": latency_ms, "error": None}
    return {"ok": False, "latency_ms": latency_ms,
            "error": (proc.stderr or proc.stdout or "ssh failed").strip()}


from common.run_context import PARTICLE, TARGET_MATERIAL


# Mirrors RUNS_DIR's cluster-side value in common/paths.py.
_CLUSTER_TB_ROOT_TEMPLATE = "/storage/agrp/rotemdo/runs/{p}/{m}"


def cluster_tb_root() -> str:
    return _CLUSTER_TB_ROOT_TEMPLATE.format(
        p=PARTICLE.name.lower(), m=TARGET_MATERIAL.name.lower(),
    )


def _parse_ls_line(line: str) -> dict | None:
    """Parse one `ls -lApL --time-style=+%s` row into an entry dict, or None."""
    if not line or line.startswith("total "):
        return None
    parts = line.split(maxsplit=6)
    if len(parts) < 7:
        return None
    perms, _links, _owner, _group, size_s, mtime_s, name = parts
    is_dir = name.endswith("/") or perms.startswith("d")
    name = name.rstrip("/").rstrip()
    try:
        mtime = int(mtime_s)
    except ValueError:
        mtime = None
    if is_dir:
        size_mb = None
    else:
        try:
            size_mb = round(int(size_s) / (1024 * 1024), 2)
        except ValueError:
            size_mb = None
    return {"name": name, "is_dir": is_dir, "size_mb": size_mb, "mtime": mtime}


def list_dir(rel: str) -> list[dict]:
    """List entries at <cluster_tb_root>/<rel>. Returns [{name,is_dir,size_mb,mtime}]."""
    rel = _safe_rel(rel)
    remote = cluster_tb_root() + (f"/{rel}" if rel else "")
    proc = subprocess.run(
        ["ssh", *_SSH_OPTS, _HOST, "--",
         "ls", "-lApL", "--time-style=+%s", "--", remote],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            (proc.stderr or proc.stdout or "ssh ls failed").strip()
        )
    entries = []
    for line in proc.stdout.splitlines():
        entry = _parse_ls_line(line)
        if entry is not None:
            entries.append(entry)
    return entries


from common.paths import RUNS_DIR
from playground.services.checkpoints import STAGE_TO_LOGGER


_LOGGER_TO_STAGE = {v: k for k, v in STAGE_TO_LOGGER.items()}


def stage_for_logger_dir(logger_dir: str) -> str | None:
    return _LOGGER_TO_STAGE.get(logger_dir)


def _scp(remote_rel: str, local: Path) -> subprocess.CompletedProcess:
    """One scp call. `remote_rel` is appended to the cluster TB root verbatim
    (already validated by the caller)."""
    return subprocess.run(
        ["scp", "-B", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         f"{_HOST}:{cluster_tb_root()}/{remote_rel}", str(local)],
        capture_output=True, text=True, timeout=300,
    )


def fetch_checkpoint(rel_ckpt: str) -> dict:
    """Copy a single .ckpt + its sibling meta.json from gpu2 into the
    mirrored local RUNS_DIR tree. Returns metadata about the import.

    Expected shape of `rel_ckpt`:
        <logger_dir>/<RUN_ID>/checkpoints/<file>.ckpt
    """
    rel_ckpt = _safe_rel(rel_ckpt)
    if not rel_ckpt.endswith(".ckpt"):
        raise ValueError(f"not a .ckpt: {rel_ckpt!r}")
    parts = rel_ckpt.split("/")
    if len(parts) != 4 or parts[2] != "checkpoints":
        raise ValueError(
            f"expected <logger>/<run>/checkpoints/<file>.ckpt, got {rel_ckpt!r}"
        )
    logger_dir, run_id, _, _file = parts

    local_ckpt = RUNS_DIR / rel_ckpt
    local_ckpt.parent.mkdir(parents=True, exist_ok=True)

    proc = _scp(rel_ckpt, local_ckpt)
    if proc.returncode != 0:
        local_ckpt.unlink(missing_ok=True)
        raise RuntimeError(
            (proc.stderr or proc.stdout or "scp failed").strip()
        )

    # Best-effort meta.json — non-fatal.
    meta_rel = f"{logger_dir}/{run_id}/meta.json"
    meta_local = RUNS_DIR / meta_rel
    meta_proc = _scp(meta_rel, meta_local)
    if meta_proc.returncode != 0:
        meta_local.unlink(missing_ok=True)

    size_mb = round(local_ckpt.stat().st_size / (1024 * 1024), 2)
    return {
        "local_path": str(local_ckpt),
        "stage":      stage_for_logger_dir(logger_dir),
        "run_id":     run_id,
        "size_mb":    size_mb,
    }
