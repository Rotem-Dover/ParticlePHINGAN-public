"""
Fundamental physical constants in the project's internal unit system (eV, cm).

Values match CLHEP/Geant4 (geant4-11.2.0/source/externals/clhep/include/CLHEP/
Units/PhysicalConstants.h) rather than current CODATA, because every G4-derived
table this project reproduces was built against the CLHEP numbers: the electron
radius r_e (via alpha and hbarc) and the atomic mass A both enter the ionisation
cross section directly, so a CODATA/CLHEP mismatch in either shows up as a
systematic offset in every cross section computed from them. Keep these
synchronised with CLHEP rather than scipy.
"""
import math

import common.units as U

# CLHEP (PhysicalConstants.h, lines 80-83): particle masses.
electron_mass_eV   = 0.510998910 * U.MeV2eV       # 510998.910 eV
proton_mass_eV     = 938.272013 * U.MeV2eV        # 938272013.0 eV
neutron_mass_eV    = 939.56536  * U.MeV2eV
amu_eV             = 931.494028 * U.MeV2eV

# Avogadro: CLHEP uses the post-2019 SI exact value, same as scipy.
avogadro           = 6.02214076e+23

# CLHEP/Geant4 *derives* hbarc and alpha from SI inputs (the comment in
# CLHEP/Units/PhysicalConstants.h:60-62 quoting `hbarc = 197.32705e-12 MeV·mm`
# is informational; the actual code is `hbarc = hbar_Planck * c_light` and
# `alpha = elm_coupling / hbarc`). Reproduce that chain so we get the exact
# same r_e CLHEP uses at runtime.
_h_SI       = 6.62607015e-34            # J·s   (CLHEP h_Planck)
_c_SI       = 2.99792458e+8             # m/s   (CLHEP c_light)
_e_SI       = 1.602176634e-19           # C     (CLHEP e_SI)
_mu0_SI     = 4.0 * math.pi * 1.0e-7    # H/m   (CLHEP mu0)
_hbar_SI    = _h_SI / (2.0 * math.pi)
hbarc_eV_cm = (_hbar_SI * _c_SI / _e_SI) * U.m2cm   # ≈ 1.973269804593e-5 eV·cm
alpha       = _mu0_SI * _c_SI * _e_SI ** 2 / (2.0 * _h_SI)   # ≈ 1/137.0359992
electron_radius_cm = alpha * hbarc_eV_cm / electron_mass_eV  # ≈ 2.8179405e-13 cm

# Mass in atomic mass units — used for mass-density bookkeeping only. CLHEP
# itself doesn't expose these directly; use the unrounded scipy values
# (consistent with how G4's NistElementBuilder picks atomic masses from PDG).
from scipy.constants import physical_constants as _pc  # noqa: E402
electron_mass_amu  = _pc["electron mass in u"][0]
proton_mass_amu    = _pc["proton mass in u"][0]
electron_mag_mom   = _pc["electron mag. mom. to Bohr magneton ratio"][0]
proton_mag_mom     = _pc["proton mag. mom. to nuclear magneton ratio"][0]

hbar_Planck        = _hbar_SI / _e_SI                       # eV·s
speed_of_light     = _c_SI * U.m2cm                         # cm/s
