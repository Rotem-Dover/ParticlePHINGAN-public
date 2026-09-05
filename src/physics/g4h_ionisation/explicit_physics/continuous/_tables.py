"""
Numpy-facing adapters over the locally computed init tables (InitTableSet).
Expose the same property API and units the explicit-physics callers use
(energies in eV, ranges in mm, dE/dx in eV/cm) on top of the shared
fp64 G4PhysicsVector splines. The derived artifacts built on these
(continuous mean/std lookup, CDFs, qprob table) regenerate from this table
source whenever their caches are deleted.
"""
import numpy as np
import torch

from physics.g4_init_tables.interpolator import g4pv_inverse_range_eval
from physics.g4_init_tables.runtime import get_init_table_set


class _PVFunc:
    """Scalar/ndarray adapter around a torch-tensor function."""

    def __init__(self, fn, scale_in: float = 1.0, scale_out: float = 1.0):
        self._fn = fn
        self._si = scale_in
        self._so = scale_out

    def __call__(self, x):
        arr = np.asarray(x, dtype=np.float64)
        t = torch.from_numpy(np.atleast_1d(arr) * self._si)
        y = self._fn(t).cpu().numpy() * self._so
        return float(y[0]) if arr.ndim == 0 else y.reshape(arr.shape)


class Geant4Tables:
    """Init-table-backed lookups with the units the explicit-physics model expects.

    The constructor takes no paths; tables come from the active
    (particle, material) via common.run_context / common.paths.
    """

    def __init__(self):
        ts = get_init_table_set()
        from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import GeantConstants
        min_kin = GeantConstants().min_kin_energy
        inv = ts.inverse_range

        self.__energy_to_dE_dx = _PVFunc(ts.dedx)                          # eV -> eV/cm
        self.__energy_to_range = _PVFunc(ts.range, scale_out=10.0)         # eV -> mm
        self.__range_to_energy = _PVFunc(ts.inverse_range, scale_in=0.1)   # mm -> eV
        self.__range_to_scaled_energy = _PVFunc(                            # mm -> eV
            lambda r: g4pv_inverse_range_eval(
                inv.bin, inv.data, inv.sec_deriv, min_kin, r),
            scale_in=0.1,
        )

    @property
    def energy_to_dE_dx(self):
        return self.__energy_to_dE_dx

    @property
    def energy_to_range(self):
        return self.__energy_to_range

    @property
    def range_to_energy(self):
        return self.__range_to_energy

    @property
    def range_to_scaled_energy(self):
        return self.__range_to_scaled_energy
