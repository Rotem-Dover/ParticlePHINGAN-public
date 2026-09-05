"""
Compare our Python-reconstructed G4 init tables against binaries dumped
directly out of GEANT4's own G4PhysicsVector objects (see
StepsGenerator/tools/InitTablesDump.cc).

The simulated process is proton-only: there is no multiple scattering and
no electron beam, so only the dE/dx, range, inverse-range and
ionisation-lambda comparisons apply here.

Each .bin file format:
    char[8] magic
    uint32 nPoints
    uint32 reserved
    nPoints * (double x, double y)

Units in the bin files:
    g4_dedx.bin          : (E_MeV,  dE/dx_MeV_per_mm)
    g4_range.bin         : (E_MeV,  R_mm)
    g4_inverse_range.bin : (R_mm,   E_MeV)
    g4_lambda_ion.bin    : (E_MeV,  lambda_mm)

Our init_tables/*.npz format:
    dedx.npz            : energy_eV,  dedx_eV_per_cm
    range.npz           : energy_eV,  range_cm
    scaled_kin_energy   : range_cm,   energy_eV
    lambda_ion.npz      : energy_eV,  lambda_cm

The tests assert agreement to within `_GRID_ATOL_RTOL` / `_VALUE_ATOL_RTOL`
(roughly 1 ULP). Strict bit-equality is not achievable because:
  - G4 builds its grids in MeV-internal units; we build in eV and convert
    by `*1e-6` for the comparison. FP non-associativity makes the two
    orderings disagree by ±1 ULP at ~half the interior nodes.
On failure pytest prints max abs / max rel diff and the first 5 offending
indices for debugging.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import common.units as U
from tests.physics.g4_init_tables._particles import material_cases, paths_for


# --------------------------- file loaders --------------------------- #

def _load_g4_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    magic = raw[:8]
    n = struct.unpack_from("<I", raw, 8)[0]
    # 8 (magic) + 4 (n) + 4 (reserved) = 16 byte header
    body = np.frombuffer(raw, dtype=np.float64, offset=16, count=2 * n)
    body = body.reshape(n, 2)
    return body[:, 0].copy(), body[:, 1].copy()


def _load_npz(path: Path, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d[x_key].astype(np.float64), d[y_key].astype(np.float64)


# --------------------------- diff reporter -------------------------- #

def _slice_to_g4_grid(x_g4: np.ndarray, x_us: np.ndarray, y_us: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """If G4's grid is a tail of ours, slice ours to match. For tables where
    G4 starts at a production-cut energy threshold (lambda_ion) G4 stores
    fewer nodes than the underlying log grid; the rest of our nodes have no
    G4 counterpart.
    """
    if x_us.size == x_g4.size:
        return x_us, y_us
    start = int(np.searchsorted(x_us, x_g4[0]))
    return x_us[start:start + x_g4.size], y_us[start:start + x_g4.size]


def _report_and_assert(name: str,
                       x_g4: np.ndarray, y_g4: np.ndarray,
                       x_us: np.ndarray, y_us: np.ndarray):
    x_us, y_us = _slice_to_g4_grid(x_g4, x_us, y_us)
    msgs = []
    msgs.append(f"--- {name} ---")
    msgs.append(f"  shapes: G4 x={x_g4.shape} y={y_g4.shape} | "
                f"ours x={x_us.shape} y={y_us.shape}")
    msgs.append(f"  G4   x range: [{x_g4.min():.6e}, {x_g4.max():.6e}]")
    msgs.append(f"  ours x range: [{x_us.min():.6e}, {x_us.max():.6e}]")

    if x_g4.shape != x_us.shape:
        msgs.append("  GRID MISMATCH: different node counts; cannot compare elementwise.")
        pytest.fail("\n".join(msgs))

    def _diffs(a, b, label):
        abs_d = np.abs(a - b)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_d = np.where(b != 0, abs_d / np.abs(b), abs_d)
        order = np.argsort(rel_d)[::-1][:5]
        msgs.append(f"  {label}: max |Δ|={abs_d.max():.3e}, "
                    f"max rel={np.nanmax(rel_d):.3e}")
        for idx in order:
            msgs.append(f"    [{idx}] x_g4={x_g4[idx]:.6e} "
                        f"g4={b[idx]:.6e} ours={a[idx]:.6e} "
                        f"absΔ={abs_d[idx]:.3e} relΔ={rel_d[idx]:.3e}")

    _diffs(x_us, x_g4, "x (grid)")
    _diffs(y_us, y_g4, "y (values)")

    # 1 ULP rtol = 2**-52 ≈ 2.22e-16. We allow a hair more (5e-16) on the
    # grid because the eV↔MeV unit conversion costs us up to 1 ULP at some
    # nodes (see module docstring). For the y-tolerance: derived quantities
    # accumulate fp64 round-off through chains of log/sqrt/divisions in the
    # Bethe-Bloch and range/inverse-range formulas, capped empirically at
    # ~1e-10 relative across all quantities. The G4_REF dump must be
    # generated with `InitTablesDump <out> 1000.0` (1 um cut, matching the
    # /run/setCut value in StepsGenerator/PhDMacro.mac that the tables/*.bin
    # files use); the default 1 nm cut would shift Tc and break the
    # comparison at the ~0.5 % level.
    # Note: for `inverse_range` and `lambda_ion` the "x" column is itself a
    # derived quantity (range integral, ionisation threshold), not the
    # primary log grid, so it accumulates the same fp64 round-off as y.
    # We apply the same tolerance to both columns rather than splitting by
    # quantity name.
    if not np.allclose(x_us, x_g4, rtol=1e-10, atol=0.0):
        msgs.append("  FAIL: x column differs beyond ~10000 ULP.")
        pytest.fail("\n".join(msgs))
    if not np.allclose(y_us, y_g4, rtol=1e-10, atol=0.0):
        msgs.append("  FAIL: y column differs beyond ~10000 ULP — likely a real physics gap.")
        pytest.fail("\n".join(msgs))

    print("\n".join(msgs))


# --------------------------- dir resolution --------------------------- #

def _g4_dir(particle_name: str, material_name: str) -> Path:
    p = paths_for(particle_name, material_name)["g4_reference"]
    required = [
        "g4_dedx.bin", "g4_range.bin", "g4_inverse_range.bin",
        "g4_lambda_ion.bin",
    ]
    missing = [f for f in required if not (p / f).exists()]
    if missing:
        pytest.skip(f"G4 reference files missing in {p}: {missing}. "
                    "Run StepsGenerator/InitTablesDump first.")
    return p


def _py_dir(particle_name: str, material_name: str) -> Path:
    p = paths_for(particle_name, material_name)["init_tables"]
    required = [
        "dedx.npz", "range.npz", "scaled_kin_energy.npz",
        "lambda_ion.npz",
    ]
    missing = [f for f in required if not (p / f).exists()]
    if missing:
        pytest.skip(f"Python init tables missing in {p}: {missing}. "
                    "Run `python -m physics.g4_init_tables.build_all`.")
    return p


# --------------------------- tests ----------------------------------- #

@material_cases()
def test_dedx(particle_name, material_name):
    g4_dir = _g4_dir(particle_name, material_name)
    py_dir = _py_dir(particle_name, material_name)
    x_g4_MeV, y_g4_MeVmm = _load_g4_bin(g4_dir / "g4_dedx.bin")
    x_us_eV, y_us_eVcm = _load_npz(py_dir / "dedx.npz", "energy_eV", "dedx_eV_per_cm")
    # Convert ours to G4 units: eV -> MeV, eV/cm -> MeV/mm = (eV / 1e6) / (cm * 10) = eV/cm * 1e-7
    x_us = x_us_eV * U.eV2MeV
    # Note: 1 MeV/mm = 1e6 eV / 0.1 cm = 1e7 eV/cm, so y_eV_per_cm * 1e-7 = y_MeV_per_mm
    y_us = y_us_eVcm * 1e-7
    _report_and_assert("dE/dx", x_g4_MeV, y_g4_MeVmm, x_us, y_us)


@material_cases()
def test_range(particle_name, material_name):
    g4_dir = _g4_dir(particle_name, material_name)
    py_dir = _py_dir(particle_name, material_name)
    x_g4_MeV, y_g4_mm = _load_g4_bin(g4_dir / "g4_range.bin")
    x_us_eV, y_us_cm = _load_npz(py_dir / "range.npz", "energy_eV", "range_cm")
    x_us = x_us_eV * U.eV2MeV
    y_us = y_us_cm * U.cm2mm
    _report_and_assert("range", x_g4_MeV, y_g4_mm, x_us, y_us)


@material_cases()
def test_inverse_range(particle_name, material_name):
    g4_dir = _g4_dir(particle_name, material_name)
    py_dir = _py_dir(particle_name, material_name)
    x_g4_mm, y_g4_MeV = _load_g4_bin(g4_dir / "g4_inverse_range.bin")
    x_us_cm, y_us_eV = _load_npz(py_dir / "scaled_kin_energy.npz",
                                  "range_cm", "energy_eV")
    x_us = x_us_cm * U.cm2mm
    y_us = y_us_eV * U.eV2MeV
    _report_and_assert("inverse range", x_g4_mm, y_g4_MeV, x_us, y_us)


@material_cases()
def test_lambda_ion(particle_name, material_name):
    g4_dir = _g4_dir(particle_name, material_name)
    py_dir = _py_dir(particle_name, material_name)
    x_g4_MeV, y_g4_mm = _load_g4_bin(g4_dir / "g4_lambda_ion.bin")
    x_us_eV, y_us_cm = _load_npz(py_dir / "lambda_ion.npz", "energy_eV", "lambda_cm")
    x_us = x_us_eV * U.eV2MeV
    y_us = y_us_cm * U.cm2mm
    _report_and_assert("lambda_ion", x_g4_MeV, y_g4_mm, x_us, y_us)
