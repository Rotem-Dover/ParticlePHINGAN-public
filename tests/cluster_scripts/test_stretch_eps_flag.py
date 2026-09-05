"""--sec-stretch-eps / PHINGAN_SEC_STRETCH_EPS plumbing: the submitter
accepts --sec-output-act stretched_sigmoid with its eps, forwards the flag
into the qsub env, and refuses a disagreement with an already-set env
value."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "cluster_scripts"))
from _training_job import build_parser, _assemble_env  # noqa: E402


def test_parser_accepts_stretched_sigmoid_and_eps():
    p = build_parser("secondary", "16gb")
    a = p.parse_args(["--dry-run", "--sec-output-act", "stretched_sigmoid",
                      "--sec-stretch-eps", "0.02"])
    assert a.sec_output_act == "stretched_sigmoid" and a.sec_stretch_eps == 0.02


def test_env_carries_eps_only_when_given(monkeypatch):
    monkeypatch.delenv("PHINGAN_SEC_STRETCH_EPS", raising=False)
    monkeypatch.delenv("PHINGAN_SEC_OUTPUT_ACT", raising=False)
    p = build_parser("secondary", "16gb")
    on = _assemble_env(p.parse_args(["--dry-run", "--sec-output-act", "stretched_sigmoid",
                                     "--sec-stretch-eps", "0.005"]))
    off = _assemble_env(p.parse_args(["--dry-run"]))
    assert 'PHINGAN_SEC_STRETCH_EPS="0.005"' in on
    assert 'PHINGAN_SEC_OUTPUT_ACT="stretched_sigmoid"' in on
    assert "PHINGAN_SEC_STRETCH_EPS" not in off


def test_eps_disagreeing_with_env_is_refused(monkeypatch):
    monkeypatch.setenv("PHINGAN_SEC_STRETCH_EPS", "0.01")
    p = build_parser("secondary", "16gb")
    with pytest.raises(SystemExit):
        _assemble_env(p.parse_args(["--dry-run", "--sec-stretch-eps", "0.02"]))
    monkeypatch.setenv("PHINGAN_SEC_STRETCH_EPS", "0.02")
    assert 'PHINGAN_SEC_STRETCH_EPS="0.02"' in _assemble_env(
        p.parse_args(["--dry-run", "--sec-stretch-eps", "0.02"]))


def test_nonpositive_eps_is_refused(monkeypatch):
    monkeypatch.delenv("PHINGAN_SEC_STRETCH_EPS", raising=False)
    p = build_parser("secondary", "16gb")
    with pytest.raises(SystemExit):
        _assemble_env(p.parse_args(["--dry-run", "--sec-stretch-eps", "0"]))


def test_parser_accepts_hardtanh_open_top():
    p = build_parser("secondary", "16gb")
    a = p.parse_args(["--dry-run", "--sec-output-act", "hardtanh_open_top"])
    assert a.sec_output_act == "hardtanh_open_top"
