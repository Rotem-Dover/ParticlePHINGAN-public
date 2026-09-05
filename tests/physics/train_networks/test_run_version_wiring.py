"""Both trainers must log under the resolved run version (never Lightning's
auto-incremented version_N) and drop a meta.json provenance record into the
run directory. Behavioral, not a source pin: build_trainer() is called with
RUNS_DIR monkeypatched to tmp_path, so nothing touches storage."""
import json

import pytest

from physics.g4h_ionisation.generators.train_networks import run_version
from physics.g4h_ionisation.generators.train_networks.continuous_generator import (
    train as cnt_train)
from physics.g4h_ionisation.generators.train_networks.secondary_generator import (
    train as sec_train)
from common.run_context import ACTIVE_NAME

RUN_ID = "2031-03-14-090000_test_abcdef1234_00c0ffee"

CASES = [
    pytest.param(cnt_train, "logger_cnt", "continuous", id="continuous"),
    pytest.param(sec_train, "logger_sec", "secondary", id="secondary"),
]


@pytest.mark.parametrize("train_mod,logger_name,stage", CASES)
def test_build_trainer_logs_under_the_run_id_and_writes_meta(
        tmp_path, monkeypatch, train_mod, logger_name, stage):
    monkeypatch.delenv("PHINGAN_RESUME", raising=False)
    monkeypatch.delenv("PHINGAN_SEED", raising=False)
    monkeypatch.delenv("PHINGAN_SEC_OUTPUT_ACT", raising=False)
    monkeypatch.delenv("PHINGAN_SEC_WIDTH", raising=False)
    monkeypatch.setenv("PHINGAN_RUN_ID", RUN_ID)
    monkeypatch.setenv("PHINGAN_GIT_SHA", "abcdef1234")
    monkeypatch.setattr(run_version, "RESUME_FROM_CHECKPOINT", False)
    monkeypatch.setattr(run_version, "_GENERATED_RUN_ID", None)
    monkeypatch.setattr(train_mod, "RUNS_DIR", tmp_path)

    train_mod.build_trainer()

    run_dir = tmp_path / logger_name / RUN_ID
    assert run_dir.is_dir(), (
        f"expected the run dir named by PHINGAN_RUN_ID, found: "
        f"{[p.name for p in (tmp_path / logger_name).iterdir()] if (tmp_path / logger_name).is_dir() else 'nothing'}")
    meta = json.loads((run_dir / "meta.json").read_text())
    # `seed` records the per-run RNG seed: PHINGAN_SEED unset =>
    # select_seed() default 42.
    expected = {"run_id": RUN_ID, "git_sha": "abcdef1234", "stage": stage,
                "beam": ACTIVE_NAME, "seed": 42,
                # ckpt_every: PHINGAN_CKPT_EVERY unset =>
                # select_ckpt_every() default 100.
                "ckpt_every": 100}
    if stage == "secondary":
        # PHINGAN_SEC_OUTPUT_ACT unset => config.SECONDARY_OUTPUT_ACTIVATION.
        from common.config import SECONDARY_OUTPUT_ACTIVATION
        expected["secondary_output_activation"] = SECONDARY_OUTPUT_ACTIVATION
        # The stretch is the config constant when the activation is
        # "stretched_sigmoid", else 0.0.
        from common.config import SECONDARY_STRETCH_EPS
        expected["secondary_stretch_eps"] = (
            SECONDARY_STRETCH_EPS if SECONDARY_OUTPUT_ACTIVATION == "stretched_sigmoid" else 0.0)
        from common.config import N_NEURON_MULTIPLIER_SEC
        expected["secondary_width"] = N_NEURON_MULTIPLIER_SEC
    assert {k: v for k, v in meta.items() if k != "legs"} == expected
    assert meta["legs"] == [expected]
