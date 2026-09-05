"""--ckpt-every / PHINGAN_CKPT_EVERY plumbing: the submitter forwards the
checkpoint cadence into the qsub env only when given, refuses a
disagreement with an already-set env value, and both trainers'
select_ckpt_every() read it, defaulting to 100 when unset."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "cluster_scripts"))
from _training_job import build_parser, _assemble_env  # noqa: E402


def test_env_carries_ckpt_every_only_when_given(monkeypatch):
    monkeypatch.delenv("PHINGAN_CKPT_EVERY", raising=False)
    p = build_parser("continuous", "32gb")
    on = _assemble_env(p.parse_args(["--dry-run", "--ckpt-every", "10"]))
    off = _assemble_env(p.parse_args(["--dry-run"]))
    assert 'PHINGAN_CKPT_EVERY="10"' in on
    assert "PHINGAN_CKPT_EVERY" not in off


def test_ckpt_every_disagreeing_with_env_is_refused(monkeypatch):
    monkeypatch.setenv("PHINGAN_CKPT_EVERY", "100")
    p = build_parser("secondary", "16gb")
    with pytest.raises(SystemExit):
        _assemble_env(p.parse_args(["--dry-run", "--ckpt-every", "10"]))


def test_ckpt_every_must_be_positive():
    p = build_parser("secondary", "16gb")
    with pytest.raises(SystemExit):
        _assemble_env(p.parse_args(["--dry-run", "--ckpt-every", "0"]))


@pytest.mark.parametrize("stage", ["continuous", "secondary"])
def test_trainers_read_ckpt_every(monkeypatch, stage):
    import importlib
    mod = importlib.import_module(
        f"physics.g4h_ionisation.generators.train_networks.{stage}_generator.train")
    monkeypatch.delenv("PHINGAN_CKPT_EVERY", raising=False)
    assert mod.select_ckpt_every() == 100
    monkeypatch.setenv("PHINGAN_CKPT_EVERY", "10")
    assert mod.select_ckpt_every() == 10
    monkeypatch.setenv("PHINGAN_CKPT_EVERY", "0")
    with pytest.raises(ValueError):
        mod.select_ckpt_every()
