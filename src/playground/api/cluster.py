"""HTTP surface for the Cluster tab.

Thin pass-through over `playground.services.cluster`:
- GET  /ping        — `ping()`
- GET  /ls          — `list_dir(path)` ; ValueError → 400, RuntimeError → 502
- POST /import      — `fetch_checkpoint(path)` ; ValueError → 400, RuntimeError → 502
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from playground.services import cluster as cluster_svc


bp = Blueprint("cluster", __name__)


@bp.route("/ping", methods=["GET"])
def ping():
    return jsonify(cluster_svc.ping())


@bp.route("/ls", methods=["GET"])
def ls():
    path = request.args.get("path", "")
    try:
        entries = cluster_svc.list_dir(path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"entries": entries})


@bp.route("/import", methods=["POST"])
def import_ckpt():
    payload = request.get_json(silent=True) or {}
    path = payload.get("path")
    if not path:
        return jsonify({"error": "missing 'path' in body"}), 400
    try:
        result = cluster_svc.fetch_checkpoint(path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify(result)
