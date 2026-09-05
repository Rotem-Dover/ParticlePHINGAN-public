#!/usr/bin/env python3
"""PreToolUse guard: block writes to storage/, .venv/, checkpoints, and scientific artifacts.

Reads the tool-use JSON payload on stdin. Exits 2 (with a message on stderr) to block,
0 to allow. The harness surfaces the stderr message back to Claude.
"""
import json
import sys
from pathlib import Path

PROTECTED_PREFIXES = (
    "storage/",
    ".venv/",
)
PROTECTED_SUFFIXES = (
    ".ckpt",
    ".root",
    ".pkl",
    ".npz",
    ".bin",
)

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)  # malformed payload — don't block

tool_name = payload.get("tool_name", "")
if tool_name not in ("Edit", "Write", "NotebookEdit"):
    sys.exit(0)

tool_input = payload.get("tool_input", {}) or {}
path_str = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
if not path_str:
    sys.exit(0)

try:
    project_root = Path(__file__).resolve().parents[2]
    rel = Path(path_str).resolve().relative_to(project_root).as_posix()
except Exception:
    rel = path_str

rel_lower = rel.lower()

if any(rel.startswith(p) for p in PROTECTED_PREFIXES) or any(rel_lower.endswith(s) for s in PROTECTED_SUFFIXES):
    print(
        f"BLOCKED: '{rel}' matches a protected path or file type -- "
        f"storage/, .venv/, or a checkpoint/ROOT/pickle/npz/bin file "
        f"(.ckpt, .root, .pkl, .npz, .bin). Data and checkpoints under "
        f"storage/ are gitignored and expensive to regenerate: rebuild "
        f"them via the appropriate build script or training pipeline "
        f"instead (see scripts/ and "
        f"physics/g4h_ionisation/generators/train_networks/). The "
        f"virtualenv is recreated with `pip install -r requirements.txt`. "
        f"Ask the user to override if this write is intentional.",
        file=sys.stderr,
    )
    sys.exit(2)

sys.exit(0)
