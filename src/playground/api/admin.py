"""Admin endpoints: health probe, full-restart, and clean shutdown.

Restart and shutdown delegate to `services.process_control`. The /health
endpoint returns this process's startup timestamp so the frontend can
detect "I'm talking to a fresh process now" after triggering a restart.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from flask import Blueprint, jsonify

from playground.services.process_control import (
    spawn_replacement, schedule_shutdown,
)


bp = Blueprint("admin", __name__)

_STARTED_AT = datetime.now(timezone.utc).isoformat()


@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "pid": os.getpid(), "started_at": _STARTED_AT})


@bp.route("/refresh", methods=["POST"])
def refresh():
    spawn_replacement()
    return jsonify({"restarting": True, "port": os.environ.get("PORT", "5050")})


@bp.route("/shutdown", methods=["POST"])
def shutdown():
    schedule_shutdown()
    return jsonify({"stopping": True})
