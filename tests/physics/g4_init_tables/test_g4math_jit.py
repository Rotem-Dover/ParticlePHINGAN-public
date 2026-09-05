"""
The runtime spline kernel (g4pv_eval) needs G4Log inside @torch.jit.script.
torch 2.12 TorchScript can't do the f64<->i64 bitcast, so g4_log_jit uses
torch.frexp (exact exponent extraction) instead. These tests pin bitwise
equality with the eager bit-manipulation implementation.
"""
import pytest
import torch

from physics.g4_init_tables._g4math import g4_log, g4_log_jit


def test_g4_log_jit_is_scripted():
    # Letting the compiler inline this kernel introduces ULP-level codegen
    # drift against the bit-exact G4 contract, so it stays a scripted
    # function that graph-breaks out of the compiled phases rather than
    # being inlined.
    assert isinstance(g4_log_jit, torch.jit.ScriptFunction)


def test_g4_log_jit_bitwise_equals_eager_over_table_range():
    # 10 eV .. 10 TeV — the full init-table energy span.
    x = torch.exp(torch.linspace(2.302585, 29.933606, 500_001, dtype=torch.float64))
    assert torch.equal(g4_log_jit(x), g4_log(x))


def test_g4_log_jit_bitwise_equals_eager_near_one():
    x = torch.linspace(0.1, 3.0, 100_001, dtype=torch.float64)
    assert torch.equal(g4_log_jit(x), g4_log(x))


def test_g4_log_jit_nonpositive_gives_nan():
    x = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    assert g4_log_jit(x).isnan().all()


def test_g4_log_jit_rejects_float32():
    with pytest.raises(Exception):  # TorchScript surfaces the assert as a RuntimeError
        g4_log_jit(torch.tensor([1.0], dtype=torch.float32))
