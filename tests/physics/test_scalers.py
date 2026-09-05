"""Scaler behavior pins independent of any frozen reference capture."""
import ast
import inspect

import torch

from physics.g4h_ionisation.generators.continuous_generator.scaler import (
    ContinuousXScaler)
from physics.g4h_ionisation.generators.secondary_generator import scaler as sec_scaler


def _cnt_input():
    ke = torch.logspace(4, 8, steps=64, dtype=torch.float32)
    length = torch.logspace(-10, -4, steps=64, dtype=torch.float32)
    return torch.stack([ke, length], dim=1)


def test_scale_does_not_mutate_its_input():
    # scale() writes in place on column 1 in an earlier implementation;
    # callers rely on the caller's tensor surviving.
    x = _cnt_input()
    before = x.clone()
    ContinuousXScaler.load().scale(x)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def test_secondary_scaler_enum_indices_are_int_wrapped():
    """torch 2.5 FX codegen cannot serialize IntEnum tensor indices; any
    subscript using SecondaryFeats must go through int(). See CLAUDE.md."""
    tree = ast.parse(inspect.getsource(sec_scaler))
    bare = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        for sub in ast.walk(node.slice):
            if (isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "SecondaryFeats"):
                # acceptable only when the enclosing expression is int(...)
                parents = [n for n in ast.walk(node.slice)
                           if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Name)
                           and n.func.id == "int"
                           and sub in list(ast.walk(n))]
                if not parents:
                    bare.append(ast.unparse(node))
    assert not bare, f"un-wrapped SecondaryFeats indices: {bare}"
