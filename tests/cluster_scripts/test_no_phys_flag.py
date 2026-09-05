"""--no-phys threads from the submitter parser into the qsub env string.

Consumes Task 3's PHINGAN_NO_PHYS contract (train.py's select_arm()). This
file only tests the submitter-side plumbing -- the parser flag and the env
string it produces -- not the trainer.
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC / "cluster_scripts"))
sys.path.insert(0, str(SRC))

from _training_job import build_parser, _assemble_env  # noqa: E402


def test_parser_accepts_no_phys():
    p = build_parser("continuous", default_mem="32gb")
    args = p.parse_args(["--dry-run", "--no-phys"])
    assert args.no_phys is True


def test_env_string_carries_no_phys():
    p = build_parser("continuous", default_mem="32gb")
    on = _assemble_env(p.parse_args(["--dry-run", "--no-phys"]))
    off = _assemble_env(p.parse_args(["--dry-run"]))
    assert 'PHINGAN_NO_PHYS="1"' in on
    assert "PHINGAN_NO_PHYS" not in off
    # the refactor must not perturb the phys env string in any other way
    assert on.replace(',PHINGAN_NO_PHYS="1"', "") == off


def test_no_phys_flag_agrees_with_env_1_and_is_not_refused(monkeypatch):
    """Mirrors the --beam/PHINGAN_BEAM agreement path: the submitter process
    itself already ran under PHINGAN_NO_PHYS="1" (e.g. deploy_and_train.sh's
    ssh invocation exported it) and --no-phys was also passed -- agreement,
    no refusal."""
    monkeypatch.setenv("PHINGAN_NO_PHYS", "1")
    p = build_parser("continuous", default_mem="32gb")
    env = _assemble_env(p.parse_args(["--dry-run", "--no-phys"]))
    assert 'PHINGAN_NO_PHYS="1"' in env


def test_no_phys_env_1_without_flag_is_refused(monkeypatch, capsys):
    """The env var is already set truthy but --no-phys was not passed --
    disagreement, refuse with a nonzero exit, same style as the beam
    refusal."""
    monkeypatch.setenv("PHINGAN_NO_PHYS", "1")
    p = build_parser("continuous", default_mem="32gb")
    with pytest.raises(SystemExit) as e:
        _assemble_env(p.parse_args(["--dry-run"]))
    assert e.value.code != 0
    assert "disagrees" in str(e.value.code)
    assert "PHINGAN_NO_PHYS" in str(e.value.code)


def test_no_phys_flag_with_unset_env_is_not_refused(monkeypatch):
    """The deploy script never exports PHINGAN_NO_PHYS itself -- only the job
    env carries it. When the submitter process's own environment has no
    PHINGAN_NO_PHYS at all, the flag alone is the source of truth and must
    not be refused."""
    monkeypatch.delenv("PHINGAN_NO_PHYS", raising=False)
    p = build_parser("continuous", default_mem="32gb")
    env = _assemble_env(p.parse_args(["--dry-run", "--no-phys"]))
    assert 'PHINGAN_NO_PHYS="1"' in env
