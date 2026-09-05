"""`get_run_version()` names every TensorBoard run directory for the two
WGAN-GP trainers. The resolution order (resume beats fresh-cluster beats
local-mint) is the contract `deploy_and_train.sh` and the submitters rely on:
a resumed leg must land in the SAME directory as the run it continues, and a
fresh deploy must land in the directory the deploy script announced."""
import json
import re

import pytest

from physics.g4h_ionisation.generators.train_networks import run_version


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch):
    """Each test starts with no PHINGAN_* env, an empty local-id cache, and
    the config constant pinned False (its import-time value would otherwise
    force the resume branch for every test on a machine with RESUME_RUN_ID
    exported)."""
    for var in ("PHINGAN_RESUME", "PHINGAN_RESUME_VERSION", "PHINGAN_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(run_version, "_GENERATED_RUN_ID", None)
    monkeypatch.setattr(run_version, "RESUME_FROM_CHECKPOINT", False)


def test_resume_version_wins_over_run_id(monkeypatch):
    monkeypatch.setenv("PHINGAN_RESUME", "1")
    monkeypatch.setenv("PHINGAN_RESUME_VERSION",
                       "2031-03-14-090000_continuous_abcdef1234_00c0ffee")
    monkeypatch.setenv("PHINGAN_RUN_ID", "must_not_be_used")
    assert run_version.get_run_version() == \
        "2031-03-14-090000_continuous_abcdef1234_00c0ffee"


def test_resume_without_version_defaults_to_version_0(monkeypatch):
    """Must match each stage's `_resume_ckpt_path()` default so the logger
    and the checkpoint lookup agree on the directory."""
    monkeypatch.setenv("PHINGAN_RESUME", "1")
    assert run_version.get_run_version() == "version_0"


def test_fresh_cluster_run_uses_phingan_run_id(monkeypatch):
    monkeypatch.setenv("PHINGAN_RUN_ID",
                       "2031-03-14-090000_secondary_abcdef1234_00c0ffee")
    assert run_version.get_run_version() == \
        "2031-03-14-090000_secondary_abcdef1234_00c0ffee"


def test_local_run_mints_a_timestamped_id_and_caches_it():
    first = run_version.get_run_version()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{6}_local_[0-9a-f]{8}", first)
    # Cached: repeated calls in one process must agree, or the logger and
    # meta.json would name two different directories.
    assert run_version.get_run_version() == first


def test_config_constant_alone_selects_the_resume_branch(monkeypatch):
    """Flipping RESUME_FROM_CHECKPOINT in config (which also drops
    LEARNING_RATE 10x at import) must behave like PHINGAN_RESUME=1 with no
    version override."""
    monkeypatch.setattr(run_version, "RESUME_FROM_CHECKPOINT", True)
    assert run_version.get_run_version() == "version_0"


def test_write_run_meta_round_trip(tmp_path):
    run_dir = tmp_path / "logger_cnt" / "some_run"  # does not exist yet
    meta = {"run_id": "some_run", "git_sha": "abcdef1234", "stage": "continuous"}
    run_version.write_run_meta(run_dir, meta)
    assert json.loads((run_dir / "meta.json").read_text()) == {**meta, "legs": [meta]}


def test_write_run_meta_appends_a_second_leg(tmp_path):
    """A resumed leg must not clobber the first leg's git_sha -- the whole
    point of the append-only `legs` history."""
    run_dir = tmp_path / "logger_cnt" / "some_run"
    first = {"run_id": "some_run", "git_sha": "abcdef1234", "stage": "continuous"}
    second = {"run_id": "some_run", "git_sha": "0011223344", "stage": "continuous"}
    run_version.write_run_meta(run_dir, first)
    run_version.write_run_meta(run_dir, second)
    on_disk = json.loads((run_dir / "meta.json").read_text())
    assert {k: v for k, v in on_disk.items() if k != "legs"} == second
    assert on_disk["legs"] == [first, second]
