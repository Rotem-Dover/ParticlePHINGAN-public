"""The two training submitters must build a correct qsub line and, above all,
must REFUSE when an artifact the job would open is absent.

The refusal is the point. `deploy_and_train.sh` syncs `src/` only, so a
cluster that has never had `storage/` staged accepts the job, waits in the
queue, allocates a GPU and dies inside `DataModule.setup()`. Two of the three
assertions here are about that path, and each is verified to fail when the
preflight is removed, not merely to pass while it is present.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
SUBMITTERS = ("continuous", "secondary")


def _run(stage, *args, env=None):
    return subprocess.run(
        [sys.executable, f"cluster_scripts/submit_{stage}_job.py", *args],
        cwd=SRC, capture_output=True, text=True,
        env={**{"PATH": "/usr/bin:/bin", "PYTHONPATH": "."}, **(env or {})},
    )


@pytest.mark.parametrize("stage", SUBMITTERS)
def test_dry_run_builds_a_qsub_line_with_the_pinned_resources(stage):
    r = _run(stage, "--dry-run")
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("qsub "))

    assert f"-N train_{stage}" in line
    assert "walltime=72:00:00" in line
    # io=3: enough storage-write bandwidth for a trainer's checkpoint
    # cadence, without claiming a share sized for a write-heavy
    # measurement job.
    assert ",io=3" in line
    assert "-q gpu" in line and "ngpus=1" in line
    assert line.rstrip().endswith(f"/src/cluster_scripts/run_on_node_{stage}.sh")
    # Cluster paths, not the submitting machine's.
    assert 'RUN_DIR="/srv01/agrp/rotemdo/src/"' in line


@pytest.mark.parametrize("stage", SUBMITTERS)
def test_resume_is_refused_when_the_checkpoint_is_absent(stage):
    """A resume that silently starts from scratch would waste a 72 h slot and
    look like a run that simply never converged."""
    r = _run(stage, "--dry-run", "--resume", "version_does_not_exist")
    assert r.returncode == 1
    assert "resume checkpoint" in r.stderr
    assert "qsub" not in r.stdout


def test_preflight_reports_every_missing_artifact_not_just_the_first(tmp_path):
    """Unit-level, so it does not depend on a storage override hook existing."""
    sys.path.insert(0, str(SRC / "cluster_scripts"))
    sys.path.insert(0, str(SRC))
    from _training_job import preflight

    absent = {f"artifact {i}": tmp_path / f"nope_{i}.npy" for i in range(3)}
    with pytest.raises(SystemExit) as e:
        preflight("continuous", absent, None)
    assert e.value.code == 1


def test_target_width_mismatch_is_caught_at_submit_time(tmp_path):
    """`N_SECONDARY_FEATURES` IS the generator's output width, and the .npy on
    disk is whatever was built last. A disagreement surfaces ~10 minutes into
    the job as a reshape error naming neither cause; catch it in milliseconds."""
    sys.path.insert(0, str(SRC / "cluster_scripts"))
    sys.path.insert(0, str(SRC))
    from common.config import N_SECONDARY_FEATURES
    import submit_secondary_job as ssj

    wrong = tmp_path / "secondary_Y.npy"
    np.save(wrong, np.zeros((8, N_SECONDARY_FEATURES + 1), dtype=np.float64))
    with pytest.raises(SystemExit) as e:
        ssj._check_target_width(wrong)
    assert "width mismatch" in str(e.value)

    right = tmp_path / "ok_Y.npy"
    np.save(right, np.zeros((8, N_SECONDARY_FEATURES), dtype=np.float64))
    ssj._check_target_width(right)  # must not raise


def test_checkpoint_cadence_is_bounded_against_the_inode_quota(monkeypatch):
    """`save_top_k=-1` retains EVERY saved checkpoint, so the cadence alone
    sets the file count: MAX_EPOCHS / every_n_epochs per stage. A dense
    cadence run risks exhausting the cluster's inode quota mid-run across
    two concurrent stages, which fails a 72 h job at an arbitrary point
    rather than at submit time.

    The cadence is `select_ckpt_every()` (PHINGAN_CKPT_EVERY, set by
    `deploy_and_train.sh --ckpt-every=N`); this pins its DEFAULT -- what a
    run that does not opt in gets -- not the per-run override, which is the
    operator's call. Source-pinned that build_trainer actually calls it, so
    a hardcoded literal cannot creep back in.
    """
    import importlib
    from common.config import MAX_EPOCHS

    monkeypatch.delenv("PHINGAN_CKPT_EVERY", raising=False)
    for stage in SUBMITTERS:
        src = (SRC / "physics/g4h_ionisation/generators/train_networks"
               / f"{stage}_generator" / "train.py").read_text()
        assert "every_n_epochs=select_ckpt_every()" in src, (
            f"{stage}: build_trainer() must take its cadence from select_ckpt_every()")
        mod = importlib.import_module(
            f"physics.g4h_ionisation.generators.train_networks.{stage}_generator.train")
        cadence = mod.select_ckpt_every()
        retained = MAX_EPOCHS / cadence
        assert retained <= 1_000, (
            f"{stage}: default every_n_epochs={cadence} retains {retained:.0f} "
            f"checkpoints at MAX_EPOCHS={MAX_EPOCHS}; two stages must stay well "
            f"inside the inode quota")
