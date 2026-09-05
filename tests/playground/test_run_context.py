"""The run-context inventory must count the actual artifact formats: scaler
state is portable ``<ClassName>.state.pt`` tensor state, not a pickle, and
training datasets are ``*.npy``, not a pickle. These tests pin that
discovery and that a non-matching file in either directory is not counted.
"""
from __future__ import annotations

import json

import pytest
from flask import Flask

import playground.api.run_context as run_context_mod


@pytest.fixture
def inventory(monkeypatch, tmp_path):
    """Build the endpoint against a synthetic storage layout and return its
    parsed inventory + paths blocks."""
    scalers = tmp_path / "scalers"
    datasets = tmp_path / "datasets"
    for d in (scalers, datasets):
        d.mkdir(parents=True)

    # A stray pickle that must NOT be counted as a scaler.
    (scalers / "ContinuousXScaler.state.pt").touch()
    (scalers / "SecondaryXScaler.state.pt").touch()
    (scalers / "SecondaryYScaler.state.pt").touch()
    (scalers / "ContinuousYScaler.state.pt").touch()
    (scalers / "other_scaler.pkl").touch()

    # Training datasets are .npy (manifest.json is not a dataset).
    (datasets / "continuous_X.npy").touch()
    (datasets / "continuous_Y.npy").touch()
    (datasets / "manifest.json").touch()

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(run_context_mod, "SCALERS_DIR", scalers)
    monkeypatch.setattr(run_context_mod, "DATASETS_DIR", datasets)
    monkeypatch.setattr(run_context_mod, "TRACKS_DIR", empty)
    monkeypatch.setattr(run_context_mod, "CDFS_DIR", empty)

    app = Flask(__name__)
    app.register_blueprint(run_context_mod.bp, url_prefix="/api/run_context")
    resp = app.test_client().get("/api/run_context/")
    assert resp.status_code == 200
    return json.loads(resp.data)


def test_scalers_inventory_counts_state_pt(inventory):
    scalers = inventory["inventory"]["scalers"]
    assert "ContinuousXScaler.state.pt" in scalers
    assert "SecondaryXScaler.state.pt" in scalers
    assert "SecondaryYScaler.state.pt" in scalers
    assert "ContinuousYScaler.state.pt" in scalers
    # A pickled file is not a scaler -- only *.state.pt is counted.
    assert "other_scaler.pkl" not in scalers


def test_datasets_inventory_counts_npy(inventory):
    datasets = inventory["inventory"]["datasets"]
    assert "continuous_X.npy" in datasets
    assert "continuous_Y.npy" in datasets
    assert "manifest.json" not in datasets


def test_paths_block_names_the_scalers_and_datasets_roots(inventory):
    paths = inventory["paths"]
    assert paths["scalers"].endswith("scalers")
    assert paths["datasets"].endswith("datasets")


def _client():
    app = Flask(__name__)
    app.register_blueprint(run_context_mod.bp, url_prefix="/api/run_context")
    return app.test_client()


def test_get_reports_active_beam_and_registry(monkeypatch, tmp_path):
    """The read endpoint must surface the active beam preset name and the
    full PRESETS registry so the frontend can build a beam dropdown."""
    from common.run_context import ACTIVE_NAME
    from physics.definitions.beam import PRESETS

    body = _client().get("/api/run_context/").get_json()
    assert body["beam"] == ACTIVE_NAME
    assert set(body["available_beams"]) == set(PRESETS)


def test_set_rejects_unknown_beam(monkeypatch):
    import playground.services.process_control as pc

    calls = []
    monkeypatch.setattr(pc, "spawn_replacement",
                        lambda env=None: calls.append(env))
    resp = _client().post("/api/run_context/set",
                          json={"beam": "proton_in_unobtainium_100MeV"})
    assert resp.status_code == 400
    assert "beam" in resp.get_json()["error"]
    assert calls == []


def test_set_is_a_noop_for_the_active_beam(monkeypatch):
    import playground.services.process_control as pc
    from common.run_context import ACTIVE_NAME

    calls = []
    monkeypatch.setattr(pc, "spawn_replacement",
                        lambda env=None: calls.append(env))
    resp = _client().post("/api/run_context/set", json={"beam": ACTIVE_NAME})
    assert resp.status_code == 200
    assert resp.get_json()["restarting"] is False
    assert calls == []


def test_set_restarts_with_phingan_beam_env(monkeypatch):
    """Switching to a different registered beam must re-exec the server
    with PHINGAN_BEAM in the environment -- the only env var run_context
    reads for beam selection."""
    import playground.services.process_control as pc
    from common.run_context import ACTIVE_NAME
    from physics.definitions.beam import PRESETS

    target = next(n for n in PRESETS if n != ACTIVE_NAME)
    calls = []
    monkeypatch.setattr(pc, "spawn_replacement",
                        lambda env=None: calls.append(env))
    resp = _client().post("/api/run_context/set", json={"beam": target})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["restarting"] is True
    assert body["beam"] == target
    assert calls == [{"PHINGAN_BEAM": target}]


def test_g4_hist2d_reports_oversized_file_cleanly(monkeypatch):
    """Selecting a root too large for a full load (no pkl cache yet) must
    surface as a clean 4xx message, not an unhandled 500 — only the Table
    tab handled TracksFileTooLarge before this."""
    from flask import Flask
    import playground.api.g4_data as g4_mod
    from playground.services.g4_slices import TracksFileTooLarge

    def boom(name):
        raise TracksFileTooLarge("100MeV_10k.root is 8.6 GB — too large")
    monkeypatch.setattr(g4_mod, "load_tracks", boom)

    app = Flask(__name__)
    app.register_blueprint(g4_mod.bp, url_prefix="/api/g4")
    resp = app.test_client().get("/api/g4/hist2d?file=100MeV_10k")
    assert resp.status_code == 413
    assert "too large" in resp.get_json()["error"]
