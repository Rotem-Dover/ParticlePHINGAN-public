"""
Shared per-particle path/fixture helpers for g4_init_tables tests.

The production code resolves dataset paths from `common.run_context.PARTICLE`
(currently `proton`). The tests need to cover BOTH protons and electrons,
so we resolve the per-particle subtree explicitly here rather than reading
the global `INIT_TABLES_DIR` / `TABLES_DIR` constants.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from common.paths import RESOURCES_ROOT
from physics.definitions.material import aluminum, beryllium, iron
from physics.definitions.particle import electron, proton


_PARTICLES = {"proton": proton, "electron": electron}
_MATERIALS = {"aluminum": aluminum, "iron": iron, "beryllium": beryllium}


def particle_param() -> pytest.MarkDecorator:
    """Use as `@particle_param()` to parametrize a test over both particles.
    The test should accept a `particle_name: str` argument; resolve concrete
    `Particle`/`Material` and dataset paths via `paths_for(particle_name)`.
    """
    return pytest.mark.parametrize("particle_name", list(_PARTICLES.keys()))


def material_cases() -> pytest.MarkDecorator:
    """Parametrize a test over every (particle, material) pair that has a
    g4_reference dump tree. Electrons exist for aluminum only.
    """
    return pytest.mark.parametrize("particle_name,material_name", [
        ("proton", "aluminum"),
        ("proton", "iron"),
        ("proton", "beryllium"),
        ("electron", "aluminum"),
    ])


def particle_material(particle_name: str, material_name: str = "aluminum"):
    return _PARTICLES[particle_name], _MATERIALS[material_name]


def paths_for(particle_name: str, material_name: str = "aluminum") -> dict[str, Path]:
    p = particle_name
    m = _MATERIALS[material_name].name.lower()
    root = RESOURCES_ROOT / p / m
    validation = root / "validation_references"
    return {
        "init_tables": root / "init_tables",
        "tables": root / "tables",
        # Test/validation-only data lives under validation_references/: the
        # GEANT4 .bin dumps (dEdx.bin, frange.bin, ...) and the g4_reference
        # parity files, kept separate from the runtime tables/ tree.
        "validation": validation,
        "g4_reference": validation / "g4_reference",
    }
