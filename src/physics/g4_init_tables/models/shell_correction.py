"""
Port of G4EmCorrections::ShellCorrection (K + L + M + N + O/P shells).

Reference: geant4-11.2.0/source/processes/electromagnetic/utils/src/
G4EmCorrections.cc:
  ShellCorrection (line 507)
  KShell          (line 377)
  LShell          (line 425)
  Initialise      (line 884)  — builds CK, CL, ZK, VL from base tables

Numerical tables are auto-extracted from the G4 source into _shell_tables.py.

G4AtomicShells per-element electron counts are inlined here for aluminum,
silicon, beryllium and iron. Adding a new element means appending to
ATOMIC_SHELL_ELECTRONS.
"""
from __future__ import annotations

import math

import torch

import physics.definitions.physical_constants as C
from physics.g4_init_tables._g4math import g4_log_scalar
from physics.definitions.material import Material
from physics.definitions.particle import Particle
from physics.g4_init_tables.models.bethe_bloch import TWOPI_MC2_RCL2, _kinematics
from physics.g4_init_tables.models import _shell_tables as T

# G4AtomicShells::GetNumberOfElectrons(Z, j) — per-shell electron counts.
# G4 indexes shells from j=0 (K), j=1 (L1), j=2 (L2), j=3 (L3), j=4 (M1)...
# Only the values needed for the current run are inlined.
ATOMIC_SHELL_ELECTRONS = {
    13: [2, 2, 2, 4, 2, 1],   # Aluminum: K=2, L1=2, L2=2, L3=4, M1=2, M2=1
    14: [2, 2, 2, 4, 2, 2],   # Silicon
    # Extracted by scripts/extract_g4_material_constants.py from
    # G4AtomicShells.cc.
    4: [2, 2],   # Beryllium
    26: [2, 2, 2, 4, 2, 2, 4, 6, 2],   # Iron
}
ATOMIC_NUMBER_OF_SHELLS = {Z: len(s) for Z, s in ATOMIC_SHELL_ELECTRONS.items()}

# CLHEP/Geant4 derives `fine_structure_const = elm_coupling/hbarc` from SI,
# not from the rounded literal `1/137.035999084`. Use C.alpha so the value
# is bit-identical to G4's runtime alpha; the small offset from the literal
# would otherwise propagate through the shell correction into dE/dx.
ALPHA2 = C.alpha ** 2

# G4PhysicsFreeVector-style linear interpolation for sThetaK, sThetaL.
def _free_linear(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """G4PhysicsFreeVector::Value with linear interpolation, clamped at edges."""
    x = torch.as_tensor(x, dtype=torch.float64)
    idx = torch.bucketize(x.contiguous(), xs, right=False).clamp(1, len(xs) - 1) - 1
    x1, x2 = xs[idx], xs[idx + 1]
    y1, y2 = ys[idx], ys[idx + 1]
    out = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    out = torch.where(x <= xs[0], ys[0], out)
    out = torch.where(x >= xs[-1], ys[-1], out)
    return out


# ---------- Precompute the K-shell and L-shell coefficient tables ----------
# G4EmCorrections::Initialise builds CK[20][29] and CL[26][28] from the base
# bk1, bk2, bls1-3, bll1-3, SK, TK, SL, TL arrays. ZK[20] and VL[26] are the
# asymptotic-form trailing coefficients computed from the last eta row.
_nK = 20
_nL = 26
_nEtaK = 29
_nEtaL = 28


def _build_CK_ZK() -> tuple[torch.Tensor, torch.Tensor]:
    CK = torch.zeros((_nK, _nEtaK), dtype=torch.float64)
    ZK = torch.zeros(_nK, dtype=torch.float64)
    for i in range(_nEtaK):
        et = float(T.Eta[i])
        loget = g4_log_scalar(et)
        for j in range(_nK):
            b = float(T.bk2[i][10 - j]) if j < 10 else float(T.bk1[i][20 - j])
            CK[j, i] = float(T.SK[j]) * loget + float(T.TK[j]) - b
            if i == _nEtaK - 1:
                ZK[j] = et * (et * et * CK[j, i] - et * float(T.UK[j]) - float(T.VK[j]))
    return CK, ZK


def _build_CL_VL() -> tuple[torch.Tensor, torch.Tensor]:
    CL = torch.zeros((_nL, _nEtaL), dtype=torch.float64)
    VL = torch.zeros(_nL, dtype=torch.float64)
    for i in range(_nEtaL):
        et = float(T.Eta[i])
        loget = g4_log_scalar(et)
        for j in range(_nL):
            if j < 8:
                bs = float(T.bls3[i][8 - j]); b = float(T.bll3[i][8 - j])
            elif j < 17:
                bs = float(T.bls2[i][17 - j]); b = float(T.bll2[i][17 - j])
            else:
                bs = float(T.bls1[i][26 - j]); b = float(T.bll1[i][26 - j])
            c = float(T.SL[j]) * loget + float(T.TL[j])
            CL[j, i] = c - bs - 3.0 * b
            if i == _nEtaL - 1:
                VL[j] = et * (et * CL[j, i] - float(T.UL[j]))
    return CL, VL


_CK, _ZK = _build_CK_ZK()
_CL, _VL = _build_CL_VL()

_THE_K = torch.tensor([0.64, 0.65, 0.66, 0.68, 0.70, 0.72, 0.74, 0.75, 0.76, 0.78,
                       0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95],
                      dtype=torch.float64)
_THE_L = torch.tensor([0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40,
                       0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52, 0.54, 0.55, 0.56,
                       0.58, 0.60, 0.62, 0.64, 0.65, 0.66],
                      dtype=torch.float64)


def _g4_index(x: float, xs: torch.Tensor) -> int:
    """Port of G4EmCorrections::Index: largest i<=n-2 s.t. x >= xs[i]."""
    n = xs.numel()
    iddd = n - 1
    while True:
        iddd -= 1
        if iddd <= 0 or x >= float(xs[iddd]):
            break
    return iddd


def _value1(xv: float, x1: float, x2: float, y1: float, y2: float) -> float:
    return y1 + (y2 - y1) * (xv - x1) / (x2 - x1)


def _value2(xv, yv, x1, x2, y1, y2, z11, z21, z12, z22):
    return (z11 * (x2 - xv) * (y2 - yv) + z22 * (xv - x1) * (yv - y1) +
            z12 * (x2 - xv) * (yv - y1) + z21 * (xv - x1) * (y2 - yv)) / \
           ((x2 - x1) * (y2 - y1))


def _k_shell_scalar(tet: float, eta: float) -> float:
    x = tet
    itet = 0
    if tet < float(_THE_K[0]):
        x = float(_THE_K[0])
    elif tet > float(_THE_K[_nK - 1]):
        x = float(_THE_K[_nK - 1]); itet = _nK - 2
    else:
        itet = _g4_index(x, _THE_K)
    if eta >= float(T.Eta[_nEtaK - 1]):
        u = _value1(x, float(_THE_K[itet]), float(_THE_K[itet + 1]),
                    float(T.UK[itet]), float(T.UK[itet + 1]))
        v = _value1(x, float(_THE_K[itet]), float(_THE_K[itet + 1]),
                    float(T.VK[itet]), float(T.VK[itet + 1]))
        z = _value1(x, float(_THE_K[itet]), float(_THE_K[itet + 1]),
                    float(_ZK[itet]), float(_ZK[itet + 1]))
        return (u + v / eta + z / (eta * eta)) / eta
    y = eta
    ieta = 0
    if eta < float(T.Eta[0]):
        y = float(T.Eta[0])
    else:
        ieta = _g4_index(y, T.Eta[:_nEtaK])
    return _value2(x, y, float(_THE_K[itet]), float(_THE_K[itet + 1]),
                   float(T.Eta[ieta]), float(T.Eta[ieta + 1]),
                   float(_CK[itet, ieta]), float(_CK[itet + 1, ieta]),
                   float(_CK[itet, ieta + 1]), float(_CK[itet + 1, ieta + 1]))


def _l_shell_scalar(tet: float, eta: float) -> float:
    x = tet
    itet = 0
    if tet < float(_THE_L[0]):
        x = float(_THE_L[0])
    elif tet > float(_THE_L[_nL - 1]):
        x = float(_THE_L[_nL - 1]); itet = _nL - 2
    else:
        itet = _g4_index(x, _THE_L)
    if eta >= float(T.Eta[_nEtaL - 1]):
        u = _value1(x, float(_THE_L[itet]), float(_THE_L[itet + 1]),
                    float(T.UL[itet]), float(T.UL[itet + 1]))
        v = _value1(x, float(_THE_L[itet]), float(_THE_L[itet + 1]),
                    float(_VL[itet]), float(_VL[itet + 1]))
        return (u + v / eta) / eta
    y = eta
    ieta = 0
    if eta < float(T.Eta[0]):
        y = float(T.Eta[0])
    else:
        ieta = _g4_index(y, T.Eta[:_nEtaL])
    return _value2(x, y, float(_THE_L[itet]), float(_THE_L[itet + 1]),
                   float(T.Eta[ieta]), float(T.Eta[ieta + 1]),
                   float(_CL[itet, ieta]), float(_CL[itet + 1, ieta]),
                   float(_CL[itet, ieta + 1]), float(_CL[itet + 1, ieta + 1]))


def _shell_correction_scalar(Z: int, ba2: float) -> float:
    """ShellCorrection for one element, evaluated at scalar ba2 (β²/α²).

    Mirrors the G4 loop over elements but for a mono-element material.
    Returns the dimensionless `term` that G4 finally divides by
    TotNbOfAtomsPerVolume — here that ratio is just atomDensity/N = 1.
    """
    iz = Z
    Zf = float(Z)
    Z2 = (Zf - 0.3) ** 2 if iz != 1 else 1.0
    f = 0.5 if iz == 1 else 1.0
    eta = ba2 / Z2
    tet_k = float(_free_linear(torch.tensor([Zf], dtype=torch.float64),
                               T.xzk, T.yzk)[0]) if iz > 11 else Z2 * (1.0 + Z2 * 0.25 * ALPHA2)
    res = f * _k_shell_scalar(tet_k, eta)

    if iz > 2:
        ZD = T.ZD
        Zeff = Zf - float(ZD[iz]) if iz < 10 else Zf - float(ZD[10])
        Z2 = Zeff * Zeff
        eta = ba2 / Z2
        tet_l = float(_free_linear(torch.tensor([Zf], dtype=torch.float64),
                                   T.xzl, T.yzl)[0])
        f_l = 0.125
        ntot = ATOMIC_NUMBER_OF_SHELLS[iz]
        nmax = min(4, ntot)
        electrons = ATOMIC_SHELL_ELECTRONS[iz]
        norm = 0.0
        eshell = 0.0
        for j in range(1, nmax):
            ne = electrons[j]
            if iz <= 15:
                tet_lj = 0.25 * Z2 * (1.0 + 5 * Z2 * ALPHA2 / 16.0) if j < 3 \
                    else 0.25 * Z2 * (1.0 + Z2 * ALPHA2 / 16.0)
            else:
                tet_lj = tet_l
            norm += ne
            eshell += tet_lj * ne
            res += f_l * ne * _l_shell_scalar(tet_lj, eta)
        if ntot > nmax:
            eshell /= norm
            HM = T.HM
            if iz < 28:
                res += f_l * (iz - 10) * _l_shell_scalar(eshell, float(HM[iz - 11]) * eta)
            elif iz < 63:
                res += f_l * 18 * _l_shell_scalar(eshell, float(HM[iz - 11]) * eta)
            else:
                res += f_l * 18 * _l_shell_scalar(eshell, float(HM[52]) * eta)
            if iz > 32:
                HN = T.HN
                if iz < 60:
                    res += f_l * (iz - 28) * _l_shell_scalar(eshell, float(HN[iz - 33]) * eta)
                elif iz < 63:
                    res += 4 * _l_shell_scalar(eshell, float(HN[iz - 33]) * eta)
                else:
                    res += 4 * _l_shell_scalar(eshell, float(HN[30]) * eta)
                if iz > 60:
                    res += f_l * (iz - 60) * _l_shell_scalar(eshell, 150.0 * eta)
    return res / Zf


def shell_correction(particle: Particle, material: Material, kin_E_eV: torch.Tensor) -> torch.Tensor:
    """G4EmCorrections::ShellCorrection for a mono-element material, dimensionless.

    Returns a tensor with the same shape as kin_E_eV. The Bethe-Bloch caller
    subtracts 2·shell from dE/dx after dividing by chunk-coefficient.
    """
    _, _, beta2, _ = _kinematics(particle, kin_E_eV)
    ba2 = beta2 / ALPHA2  # G4 SetupKinematics
    Z = int(round(material.Z))
    if Z not in ATOMIC_SHELL_ELECTRONS:
        raise NotImplementedError(f"Atomic shell data not inlined for Z={Z}")
    flat = ba2.flatten().tolist()
    vals = torch.tensor([_shell_correction_scalar(Z, b) for b in flat], dtype=torch.float64)
    return vals.view_as(kin_E_eV)
