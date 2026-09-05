"""The Trajectory tab POSTs to ``/api/stepping/run`` and polls
``/api/stepping/progress``; both must be served by the ``stepping``
blueprint the server registers. The simulator itself is stubbed: these tests
pin the request/response contract the frontend (``static/js/tabs/tracking.js``)
relies on, not the physics.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from flask import Flask

from common.enums import G4Columns


def _client(monkeypatch, fake_ts):
    import playground.api.stepping as stepping_mod

    calls = []

    def fake_run(processes, E0_eV, n_events, n_steps, **kw):
        calls.append(dict(processes=processes, E0_eV=E0_eV,
                          n_events=n_events, n_steps=n_steps, **kw))
        return fake_ts

    monkeypatch.setattr(stepping_mod, "run_simulation", fake_run)
    app = Flask(__name__)
    app.register_blueprint(stepping_mod.bp, url_prefix="/api/stepping")
    return app.test_client(), calls


class _FakeTS:
    """Two events of three states each; event 1 dies after its second state
    (NaN-padded, as ``TrackSimulator`` pads dead lanes)."""

    @property
    def states_along_propagation_df(self):
        return pd.DataFrame({
            G4Columns.EventNum: [0, 0, 0, 1, 1, 1],
            G4Columns.StepNum:  [0, 1, 2, 0, 1, 2],
            G4Columns.X: [0.0, 1.0, 2.0, 0.0, 1.0, np.nan],
            G4Columns.Y: [0.0, 0.1, 0.2, 0.0, -0.1, np.nan],
            G4Columns.Z: [0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            G4Columns.KineticEnergy: [1e8, 9e7, 8e7, 1e8, 9e7, np.nan],
        })

    @property
    def steps_features_df(self):
        return pd.DataFrame({
            G4Columns.EventNum: [0, 0, 1, 1],
            G4Columns.StepNum:  [0, 1, 0, 1],
            G4Columns.StepLength: [1.0, 1.0, 1.0, np.nan],
            G4Columns.ContinuousLoss: [1e7, 1e7, 1e7, np.nan],
        })


def test_server_registers_the_stepping_blueprint():
    from playground.server import create_app
    rules = {r.rule for r in create_app().url_map.iter_rules()}
    assert "/api/stepping/run" in rules
    assert "/api/stepping/progress" in rules


def test_progress_returns_the_registry_snapshot(monkeypatch):
    client, _ = _client(monkeypatch, _FakeTS())
    body = json.loads(client.get("/api/stepping/progress").data)
    assert {"phase", "step", "total"} <= set(body)


def test_run_requires_processes(monkeypatch):
    client, _ = _client(monkeypatch, _FakeTS())
    resp = client.post("/api/stepping/run", json={"n_events": 2})
    assert resp.status_code == 400


def test_run_returns_subsampled_trajectories_and_feature_summary(monkeypatch):
    client, calls = _client(monkeypatch, _FakeTS())
    resp = client.post("/api/stepping/run", json={
        "processes": [{"name": "ion"}], "E0_eV": 1e8,
        "n_events": "2", "n_steps": "3", "trajectory_subsample": "1",
    })
    assert resp.status_code == 200, resp.data
    body = json.loads(resp.data)

    # Numeric fields arrive as strings from the form inputs; they are coerced.
    assert calls[0]["n_events"] == 2 and calls[0]["n_steps"] == 3
    assert calls[0]["processes"] == [{"name": "ion"}]

    assert body["n_events"] == 2 and body["n_steps"] == 3
    assert body["n_trajectories_returned"] == 1
    t = body["trajectories"][0]
    assert t["event"] == 0
    assert t["x"] == [0.0, 1.0, 2.0] and len(t["ke"]) == 3

    fs = body["feature_summary"]
    assert set(fs) == {G4Columns.StepLength, G4Columns.ContinuousLoss}
    assert fs[G4Columns.StepLength]["mean"] == pytest.approx(1.0)


def test_run_drops_nan_padded_rows(monkeypatch):
    client, _ = _client(monkeypatch, _FakeTS())
    body = json.loads(client.post("/api/stepping/run", json={
        "processes": [{"name": "ion"}], "n_events": 2, "n_steps": 3,
        "trajectory_subsample": 5,
    }).data)
    ev1 = [t for t in body["trajectories"] if t["event"] == 1][0]
    assert ev1["x"] == [0.0, 1.0]
    assert all(v == v for v in ev1["ke"])  # no NaN leaks into JSON
