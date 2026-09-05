"""
Port of G4BraggModel for low-energy proton ionisation (T <= 2 MeV).

Reference: geant4-11.2.0/source/processes/electromagnetic/standard/src/
G4BraggModel.cc, specifically:
  - G4BraggModel::ComputeDEDXPerVolume      (lines 256-284)
  - G4BraggModel::DEDX                      (lines 620-704)
  - G4BraggModel::ElectronicStoppingPower   (lines 473-617)

DEDX dispatch follows G4 exactly: for NIST-named materials (G4_Al, G4_Si, …)
G4 consults the PSTAR tabulated data via G4PSTARStopping; only when no PSTAR
match is found does it fall back to the ICRU49 Ziegler-type analytic fit in
ElectronicStoppingPower. The PSTAR data is auto-extracted from
G4PSTARStopping.cc / G4NISTStoppingData.hh into _pstar_tables.py.
"""
from __future__ import annotations

import math

import torch

import physics.definitions.physical_constants as C
import common.units as U
from physics.g4_init_tables._g4math import g4_log
from physics.definitions.material import Material
from physics.definitions.particle import Particle
from physics.g4_init_tables.interpolator import G4PhysicsVector
from physics.g4_init_tables.models.bethe_bloch import TWOPI_MC2_RCL2, t_max, _kinematics
from physics.g4_init_tables.models import _pstar_tables as _PSTAR


LOWEST_KIN_ENERGY_eV = 0.25e3   # 0.25 keV (proton scale)
HIGH_ENERGY_LIMIT_eV = 2e6      # 2 MeV (proton scale)
_PROTON_MASS_AMU = C.proton_mass_amu
_ZIEGLER_FACTOR_eV_cm2 = U.eV2MeV * 0  # see comment below
# Ziegler factor: in CLHEP it is eV*cm²*1e-15. We work in (eV, cm), so
# multiplying ESP (dimensionless × 1e-15) by N_atoms (1/cm³) × Ziegler factor
# yields eV/cm. In our units that's simply ESP * N_atoms * 1e-15 [eV·cm²].
_ZIEGLER_FACTOR_eV_cm2 = 1.0e-15  # eV · cm²


# ICRU49 coefficients a[Z-1][0..4] from G4BraggModel.cc.
# Only the elements this project's materials use are inlined here; the
# full Z=1..92 table can be pasted in when needed. Extracted by
# scripts/extract_g4_material_constants.py.
_ICRU49 = {
    13: (4.154e0, 4.739e0, 2.766e3, 1.645e2, 2.023e-2),   # Aluminum
    14: (4.914e0, 5.598e0, 3.193e3, 2.327e2, 1.419e-2),   # Silicon
    4: (2.248, 2.59, 966.0, 153.8, 0.03475),   # Beryllium
    26: (3.519, 3.963, 6065.0, 1243.0, 0.007782),   # Iron
}

# PSTAR data by material name. Built once with G4PhysicsVector spline (matches
# G4PSTARStopping::AddData where the FreeVector is constructed with spline=true).
_T0_eV = _PSTAR.T0_MeV * 1e6
_PSTAR_EMIN_eV = float(_T0_eV[0])  # = 1 keV: lowest tabulated PSTAR node

def _f32_roundtrip(t: torch.Tensor) -> torch.Tensor:
    # G4NISTStoppingData.hh stores the PSTAR stopping powers as `G4float`
    # literals (e.g. `171.9f`); G4PSTARStopping::FindData casts them to
    # G4double via `(G4double)stop[i]*fac`, which does NOT recover the
    # precision lost in the float32 representation. To match G4's runtime
    # table bit-for-bit, round-trip through float32 here.
    return t.to(torch.float64).to(torch.float32).to(torch.float64)


_PSTAR_DATA = {
    "Aluminum":  _f32_roundtrip(_PSTAR.DEDX_Al_MeV_cm2_per_g),
    "Silicon":   _f32_roundtrip(_PSTAR.DEDX_Si_MeV_cm2_per_g),
    "Iron":      _f32_roundtrip(_PSTAR.DEDX_Fe_MeV_cm2_per_g),
    "Beryllium": _f32_roundtrip(_PSTAR.DEDX_Be_MeV_cm2_per_g),
}
_PSTAR_VECTORS = {
    name: G4PhysicsVector(_T0_eV, vals, log_grid=False, use_spline=True)
    for name, vals in _PSTAR_DATA.items()
}


def _pstar_dedx_per_unit_mass(material_name: str, kin_E_eV: torch.Tensor) -> torch.Tensor:
    """Mirror G4PSTARStopping::GetElectronicDEDX exactly.

    Above the lowest tabulated PSTAR node (1 keV): spline value.
    Below: PSTAR[0] · sqrt(E / 1keV), matching the sqrt extrapolation G4
    hardcodes in G4PSTARStopping.hh:132. Letting `G4PhysicsVector` clamp to
    PSTAR[0] instead would flatten dE/dx below the first node rather than
    tapering it to zero, so the extrapolation is applied explicitly here.
    """
    pv = _PSTAR_VECTORS[material_name]
    first_val = float(_PSTAR_DATA[material_name][0])
    below = kin_E_eV < _PSTAR_EMIN_eV
    extrap = first_val * torch.sqrt(torch.clamp(kin_E_eV, min=0.0) / _PSTAR_EMIN_eV)
    spline = pv(kin_E_eV)
    return torch.where(below, extrap, spline)


def _electronic_stopping_power_per_atom(Z: int, kin_E_eV: torch.Tensor) -> torch.Tensor:
    """G4BraggModel::ElectronicStoppingPower in 1e-15 eV·cm² per atom.

    Argument energy is the *projectile* (proton) kin energy in eV;
    the ICRU49 fit takes T = kinEnergy / (keV * protonMassAMU) in keV/amu.
    """
    if Z not in _ICRU49:
        raise NotImplementedError(f"ICRU49 coefficients not tabulated for Z={Z}")
    a0, x1, x2, x3, x4 = _ICRU49[Z]  # a0 == a[i][0] is unused in the fit

    # T in keV/amu (proton kineticEnergy is per-proton; mass in amu == 1)
    T_keV_per_amu = kin_E_eV * U.eV2keV / _PROTON_MASS_AMU

    # G4 source: fac=1; if T<10: fac = sqrt(T*0.1), T=10
    fac = torch.where(T_keV_per_amu < 10.0,
                      torch.sqrt(torch.clamp(T_keV_per_amu, min=0.0) * 0.1),
                      torch.ones_like(T_keV_per_amu))
    T = torch.where(T_keV_per_amu < 10.0,
                    torch.full_like(T_keV_per_amu, 10.0),
                    T_keV_per_amu)

    slow = x1 * torch.pow(T, 0.45)
    shigh = g4_log(1.0 + x3 / T + x4 * T) * x2 / T
    ionloss = slow * shigh * fac / (slow + shigh)
    return torch.clamp(ionloss, min=0.0)


def _dedx_pure_element(material: Material, kin_E_eV: torch.Tensor) -> torch.Tensor:
    """G4BraggModel::DEDX, in eV/cm.

    Mirrors G4 exactly: try PSTAR first (line 642-647 of G4BraggModel.cc),
    fall back to the ICRU49 ElectronicStoppingPower for elemental materials.
    """
    if material.name in _PSTAR_VECTORS:
        # G4: return fPSTAR->GetElectronicDEDX(iPSTAR, kinEnergy) * material->GetDensity()
        # PSTAR vector value is in MeV·cm²/g; × density (g/cm³) → MeV/cm.
        dedx_MeV_per_cm = _pstar_dedx_per_unit_mass(material.name, kin_E_eV) * material.rho
        return dedx_MeV_per_cm * 1e6  # eV/cm

    Z = int(round(material.Z))
    n_atoms_per_cm3 = material.number_of_atoms_per_volume
    esp = _electronic_stopping_power_per_atom(Z, kin_E_eV)  # 1e-15 eV·cm²
    return esp * n_atoms_per_cm3 * _ZIEGLER_FACTOR_eV_cm2   # eV/cm


def compute_dedx_per_volume(
    particle: Particle, material: Material, kin_E_eV: torch.Tensor, cut_eV: float
) -> torch.Tensor:
    """G4BraggModel::ComputeDEDXPerVolume. Returns eV/cm."""
    mass = particle.mass_eV
    mass_rate = mass / C.proton_mass_eV  # G4: massRate

    tmax = t_max(particle, kin_E_eV)
    tlim = LOWEST_KIN_ENERGY_eV * mass_rate
    tmin = torch.maximum(torch.minimum(torch.full_like(kin_E_eV, cut_eV), tmax),
                         torch.full_like(kin_E_eV, tlim))

    # Branch: kinEnergy < tlim → sqrt extrapolation from DEDX(lowestKinEnergy)
    e_for_dedx = torch.where(kin_E_eV < tlim, torch.full_like(kin_E_eV, LOWEST_KIN_ENERGY_eV), kin_E_eV)
    dedx_base = _dedx_pure_element(material, e_for_dedx)
    dedx = torch.where(
        kin_E_eV < tlim,
        dedx_base * torch.sqrt(torch.clamp(kin_E_eV, min=0.0) / tlim),
        dedx_base,
    )

    # delta-ray correction for tmin < tmax
    tau, _, _, _ = _kinematics(particle, kin_E_eV)
    x = tmin / tmax
    del_corr = (g4_log(x) * (tau + 1.0) ** 2 / (tau * (tau + 2.0)) + 1.0 - x) \
                * TWOPI_MC2_RCL2 * material.electron_density
    apply_corr = (tmin < tmax) & (kin_E_eV >= tlim)
    dedx = torch.where(apply_corr, dedx + del_corr, dedx)

    dedx = torch.clamp(dedx, min=0.0) * (particle.charge ** 2)
    return dedx
