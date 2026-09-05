"""The cluster tunnel must browse and mirror the runs tree.

Trainings write under ``RUNS_DIR`` on the cluster, and
``fetch_checkpoint`` mirrors downloads into the local ``RUNS_DIR`` tree, so
checkpoint discovery must scan that same tree — or a fetched checkpoint
lands where the dropdowns never look.
"""
from __future__ import annotations

import subprocess

import playground.services.checkpoints as checkpoints_mod
import playground.services.cluster as cluster_mod


def test_cluster_tb_root_points_at_the_runs_tree():
    root = cluster_mod.cluster_tb_root()
    assert root.startswith("/storage/agrp/rotemdo/runs/")
    assert root.endswith(
        f"/{cluster_mod.PARTICLE.name.lower()}"
        f"/{cluster_mod.TARGET_MATERIAL.name.lower()}"
    )


def test_fetch_checkpoint_mirrors_into_the_local_runs_tree(
        monkeypatch, tmp_path):
    runs = tmp_path / "runs" / "proton" / "aluminum"
    monkeypatch.setattr(cluster_mod, "RUNS_DIR", runs)

    def fake_scp(remote_rel, local):
        local.write_bytes(b"ckpt-bytes")
        return subprocess.CompletedProcess(args=[], returncode=0)

    monkeypatch.setattr(cluster_mod, "_scp", fake_scp)

    rel = "logger_cnt/2031-03-14-090000_continuous_iron_abc_def/checkpoints/last.ckpt"
    result = cluster_mod.fetch_checkpoint(rel)

    assert result["local_path"] == str(runs / rel)
    assert (runs / rel).read_bytes() == b"ckpt-bytes"
    assert result["stage"] == "continuous"


def test_fetch_checkpoint_writes_only_under_runs_dir(monkeypatch, tmp_path):
    """Negative control: every file fetch_checkpoint creates lands under the
    runs tree -- none beside it in tmp_path."""
    runs = tmp_path / "runs" / "proton" / "aluminum"
    monkeypatch.setattr(cluster_mod, "RUNS_DIR", runs)

    def fake_scp(remote_rel, local):
        local.write_bytes(b"x")
        return subprocess.CompletedProcess(args=[], returncode=0)

    monkeypatch.setattr(cluster_mod, "_scp", fake_scp)
    cluster_mod.fetch_checkpoint("logger_sec/run_1/checkpoints/last.ckpt")

    created_files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert created_files, "fetch_checkpoint wrote nothing; test is vacuous"
    for p in created_files:
        assert p.is_relative_to(runs), f"{p} landed outside the runs tree"


def test_discovery_scans_the_runs_tree(monkeypatch, tmp_path):
    runs = tmp_path / "runs" / "proton" / "aluminum"
    ckpt_dir = runs / "logger_cnt" / "2031-03-14_continuous_iron_abc_def" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "last.ckpt").write_bytes(b"w")
    monkeypatch.setattr(checkpoints_mod, "RUNS_DIR", runs)

    items = checkpoints_mod.list_checkpoints("continuous")
    assert [i["run_id"] for i in items] == ["2031-03-14_continuous_iron_abc_def"]
    assert {i["source"] for i in items} == {"runs"}


def test_discovery_survives_a_missing_runs_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoints_mod, "RUNS_DIR", tmp_path / "nonexistent")
    items = checkpoints_mod.list_checkpoints("continuous")
    assert items == []
