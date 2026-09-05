"""Resulting-table endpoint for the Table tab.

Serves one of two tables from the most-recently-run simulation — whichever
tab (Trajectory / Pull / Pairwise) triggered it:

  - ``which=sys`` (default): the system's merged states + step-features
    DataFrame (`TrackSimulator.states_and_steps_features_df`), kept on the
    server in the cached TrackSimulator;
  - ``which=g4``: the GEANT4 truth tracks the run was compared against,
    lazily loaded from the run's ``g4_file`` via the shared tracks cache.
    Only available after a Pull / Pairwise run (a plain Trajectory run has
    no G4 reference). The two tables have *different* column schemas.

The full DataFrames stay on the server; the HTTP response carries only a
paginated slice plus the total row count.

Also hosts a tiny notebook-style "kernel": a persistent namespace
(`df`, `df_sys`, `df_g4`, `pd`, `np`, `plt`, `ts`) against which the Table
tab's cells run arbitrary Python via `exec()`, capturing stdout/stderr, the
last expression's repr, and any matplotlib figures (rendered to inline PNGs).
`df` aliases `df_sys`; `df_g4` is None when the last run had no G4 reference.
This is arbitrary code execution by design — acceptable only because the
playground binds to 127.0.0.1 and is a single-user local research tool.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import math
import traceback
from typing import Any

from flask import Blueprint, jsonify, request

from common.paths import TRACKS_DIR
from playground.services.sim_runner import get_last_run


bp = Blueprint("table", __name__)

_MAX_LIMIT = 5000

# Persistent kernel namespace, shared across cell runs (single-process server),
# mirroring how progress / last-run state already live as module globals. None
# until first initialised; rebound on /kernel/reset.
_kernel_ns: dict[str, Any] | None = None


def _clean(v):
    """JSON-safe scalar: NaN/Inf → None (invalid JSON otherwise)."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _g4_df(run: dict[str, Any]):
    """Raw GEANT4 tracks DataFrame for the last run, or None.

    Loaded lazily from the run's ``g4_file`` through the shared (cached)
    tracks loader, so it costs nothing until the G4 table is requested.
    """
    g4_file = run.get("g4_file")
    if not g4_file:
        return None
    from playground.services.g4_slices import load_tracks
    try:
        return load_tracks(g4_file)
    except FileNotFoundError:
        return None


@bp.route("/last", methods=["GET"])
def last():
    run = get_last_run()
    if run is None or run.get("ts") is None:
        return jsonify({"available": False})

    ts = run["ts"]
    g4_file = run.get("g4_file")
    # `which` selects the table: 'sys' (system output) or 'g4' (truth tracks).
    which = (request.args.get("which") or "sys").strip().lower()
    if which not in ("sys", "g4"):
        which = "sys"

    # `g4_available` lets the frontend enable/disable the G4 option without a
    # separate probe (loading the file just to know it exists would be wasteful).
    g4_available = bool(g4_file) and any(
        (TRACKS_DIR / f"{g4_file}{ext}").exists() for ext in (".pkl", ".root")
    )

    if which == "g4":
        from playground.services.g4_slices import TracksFileTooLarge
        try:
            df = _g4_df(run)
        except TracksFileTooLarge as exc:
            return jsonify({"available": False, "error": str(exc)})
    else:
        df = ts.states_and_steps_features_df

    # `columns` is always the full schema so the frontend column chooser stays
    # populated even when a query filters every row out or errors. Column
    # *selection* is done client-side (cheap: pages are capped); only the row
    # *query* is applied here, since it filters across rows the client doesn't
    # hold.
    meta = {
        "available": True,
        "which": which,
        "source": run.get("source"),
        "E0_eV": run.get("E0_eV"),
        "n_events": run.get("n_events"),
        "n_steps": run.get("n_steps"),
        "g4_available": g4_available,
        "g4_file": g4_file,
        "columns": [str(c) for c in df.columns] if df is not None else [],
    }

    # G4 requested but no reference file for this run (e.g. a Trajectory run).
    if which == "g4" and df is None:
        return jsonify({**meta, "total_rows": 0, "offset": 0, "limit": 0,
                        "rows": [],
                        "query_error": "no GEANT4 reference for the last run — "
                                       "run a Pull or Pairwise comparison first."})

    query = (request.args.get("query") or "").strip()
    if query:
        try:
            # All column names are valid Python identifiers, so df.query reads
            # naturally (e.g. "KineticEnergy > 5e7 and StepLength < 1e-5").
            df = df.query(query)
        except Exception as e:  # noqa: BLE001 — surface any pandas/parse error
            return jsonify({**meta, "total_rows": 0, "offset": 0,
                            "limit": 0, "rows": [],
                            "query_error": f"{type(e).__name__}: {e}"})

    total = int(len(df))
    offset = max(0, int(request.args.get("offset", 0)))
    limit = min(_MAX_LIMIT, max(1, int(request.args.get("limit", 200))))
    page = df.iloc[offset:offset + limit]

    # itertuples preserves per-column dtypes (EventNum/StepNum stay ints),
    # unlike a to_numpy(float) cast that would float-ify the index columns.
    rows = [[_clean(v) for v in row]
            for row in page.itertuples(index=False, name=None)]

    return jsonify({**meta, "total_rows": total, "offset": offset,
                    "limit": limit, "rows": rows})


# ── notebook kernel ─────────────────────────────────────────────────────────

def _init_kernel(ns: dict, run: dict[str, Any]) -> None:
    """(Re)bind the standard names into a fresh kernel namespace.

    `df_sys` / `df_g4` are COPIES of the system and GEANT4 tables so cell-side
    mutations can't corrupt the cached simulator's DataFrame or the shared
    tracks cache. `df` aliases `df_sys`; `df_g4` is None when the run has no
    G4 reference.
    """
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")          # headless: render figures to PNG buffers
    import matplotlib.pyplot as plt

    ts = run["ts"]
    g4 = _g4_df(run)

    ns.clear()
    ns["pd"] = pd
    ns["np"] = np
    ns["plt"] = plt
    ns["ts"] = ts
    ns["df_sys"] = ts.states_and_steps_features_df.copy()
    ns["df_g4"] = g4.copy() if g4 is not None else None
    ns["df"] = ns["df_sys"]        # backwards-compatible default handle


def _ensure_kernel():
    """Return (namespace, error). Lazily initialises on first use."""
    global _kernel_ns
    run = get_last_run()
    if run is None or run.get("ts") is None:
        return None, "no simulation has been run yet — run one in the Trajectory, Pull, or Pairwise tab first."
    if _kernel_ns is None:
        _kernel_ns = {}
        _init_kernel(_kernel_ns, run)
    return _kernel_ns, None


def _run_code(ns: dict, code: str) -> dict:
    """Exec a cell against `ns`, Jupyter-style.

    The trailing expression (if any) is echoed via repr; stdout/stderr are
    captured; every open matplotlib figure is rendered to an inline PNG and
    then closed.
    """
    import matplotlib.pyplot as plt

    out, err = io.StringIO(), io.StringIO()
    result_repr = None
    error = None

    try:
        parsed = ast.parse(code, mode="exec")
    except SyntaxError:
        return {"ok": False, "stdout": "", "stderr": "",
                "error": traceback.format_exc(), "images": [], "result": None}

    # Pop a trailing bare expression so we can echo its value (like a REPL).
    last_expr = None
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        last_expr = ast.Expression(parsed.body.pop().value)

    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            if parsed.body:
                exec(compile(parsed, "<cell>", "exec"), ns)
            if last_expr is not None:
                val = eval(compile(last_expr, "<cell>", "eval"), ns)
                if val is not None:
                    result_repr = repr(val)
        except Exception:        # noqa: BLE001 — surface the traceback to the cell
            error = traceback.format_exc()

    images = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                    facecolor=fig.get_facecolor())
        images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    plt.close("all")

    return {"ok": error is None, "stdout": out.getvalue(),
            "stderr": err.getvalue(), "error": error,
            "images": images, "result": result_repr}


def _kernel_info(ns: dict) -> dict:
    df = ns.get("df")
    g4 = ns.get("df_g4")
    info = {"n_rows": None, "n_cols": None,
            "g4_rows": None, "g4_cols": None}
    if df is not None:
        info["n_rows"], info["n_cols"] = int(len(df)), int(df.shape[1])
    if g4 is not None:
        info["g4_rows"], info["g4_cols"] = int(len(g4)), int(g4.shape[1])
    return info


@bp.route("/exec", methods=["POST"])
def exec_cell():
    payload = request.get_json(force=True) or {}
    code = payload.get("code", "") or ""
    ns, kerr = _ensure_kernel()
    if ns is None:
        return jsonify({"ok": False, "error": kerr, "stdout": "", "stderr": "",
                        "images": [], "result": None})
    return jsonify(_run_code(ns, code))


@bp.route("/kernel/reset", methods=["POST"])
def kernel_reset():
    global _kernel_ns
    run = get_last_run()
    if run is None or run.get("ts") is None:
        _kernel_ns = None
        return jsonify({"ok": False,
                        "error": "no simulation has been run yet."})
    _kernel_ns = {}
    _init_kernel(_kernel_ns, run)
    return jsonify({"ok": True, **_kernel_info(_kernel_ns)})
