"""--seed threads from the submitter parser into the qsub env string as
PHINGAN_SEED, and the trainers read it (default 42, the value that was
hardcoded before the flag existed).

Submitter-side plumbing here; the trainer-side read is tested alongside.
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC / "cluster_scripts"))
sys.path.insert(0, str(SRC))

from _training_job import build_parser, _assemble_env  # noqa: E402


def test_parser_accepts_seed():
    p = build_parser("secondary", default_mem="16gb")
    assert p.parse_args(["--dry-run", "--seed", "7"]).seed == 7
    assert p.parse_args(["--dry-run", "--seed=7"]).seed == 7
    assert p.parse_args(["--dry-run"]).seed is None


def test_env_string_carries_seed_only_when_given():
    p = build_parser("secondary", default_mem="16gb")
    on = _assemble_env(p.parse_args(["--dry-run", "--seed", "7"]))
    off = _assemble_env(p.parse_args(["--dry-run"]))
    assert 'PHINGAN_SEED="7"' in on
    assert "PHINGAN_SEED" not in off
    assert on.replace(',PHINGAN_SEED="7"', "") == off


def test_seed_disagreeing_with_env_is_refused(monkeypatch):
    monkeypatch.setenv("PHINGAN_SEED", "3")
    p = build_parser("secondary", default_mem="16gb")
    with pytest.raises(SystemExit) as e:
        _assemble_env(p.parse_args(["--dry-run", "--seed", "7"]))
    assert e.value.code != 0 and "PHINGAN_SEED" in str(e.value.code)


def test_seed_agreeing_with_env_is_not_refused(monkeypatch):
    monkeypatch.setenv("PHINGAN_SEED", "7")
    p = build_parser("secondary", default_mem="16gb")
    assert 'PHINGAN_SEED="7"' in _assemble_env(p.parse_args(["--dry-run", "--seed", "7"]))


def test_trainer_reads_seed_from_env_default_42(monkeypatch):
    from physics.g4h_ionisation.generators.train_networks.secondary_generator import train as sec_train
    from physics.g4h_ionisation.generators.train_networks.continuous_generator import train as cnt_train
    for mod in (sec_train, cnt_train):
        monkeypatch.delenv("PHINGAN_SEED", raising=False)
        assert mod.select_seed() == 42
        monkeypatch.setenv("PHINGAN_SEED", "7")
        assert mod.select_seed() == 7
        monkeypatch.setenv("PHINGAN_SEED", "seven")
        with pytest.raises(ValueError):
            mod.select_seed()
