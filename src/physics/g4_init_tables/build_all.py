"""
Driver: build the initialisation tables for the active beam
(`common.run_context.ACTIVE`, selected via `PHINGAN_BEAM`) and dump them
under `common.paths.INIT_TABLES_DIR`.

Run from the repo root with the venv active:

    python -m physics.g4_init_tables.build_all
"""
from __future__ import annotations

import logging

from common.paths import INIT_TABLES_DIR
from common.run_context import ACTIVE
from physics.g4_init_tables.build_dedx import build_dedx_table, make_g4_vector, save_dedx
from physics.g4_init_tables.build_lambda_ion import build_lambda_table, save_lambda
from physics.g4_init_tables.build_range import (
    build_range_table, build_scaled_kin_energy, save_range, save_scaled_kin_energy,
)
from physics.g4_init_tables.grid import G4_BINS_PER_DECADE
from physics.definitions.beam import BeamConfig

log = logging.getLogger(__name__)

# Must match the G4 simulation that produced the reference dumps:
# PhysListEmStandard sets MinEnergy=10 eV, MaxEnergy=10 TeV, 10 bins/decade.
# See StepsGenerator/PhDMacro.mac and StepsGenerator/src/PhysListEmStandard.cc.
E_MIN_eV = 1e1          # 10 eV
E_MAX_eV = 1e13         # 10 TeV


def build_for(beam: BeamConfig) -> None:
    particle = beam.particle
    material = beam.material
    out_dir = INIT_TABLES_DIR
    log.info("Building init tables for %s in %s -> %s",
             particle.name, material.name, out_dir)

    cut = material.Tc  # eV

    # dE/dx
    e_grid, dedx = build_dedx_table(particle, material, cut_eV=cut,
                                    e_min=E_MIN_eV, e_max=E_MAX_eV)
    log.info("dE/dx range: %.3e .. %.3e eV/cm", float(dedx.min()), float(dedx.max()))
    save_dedx(e_grid, dedx, out_dir)

    # Range + scaled kinetic energy (use spline-backed vector for the integrand)
    dedx_pv = make_g4_vector(e_grid, dedx)
    e_grid_r, range_cm = build_range_table(dedx_pv)
    save_range(e_grid_r, range_cm, out_dir)
    r_grid, e_grid_inv = build_scaled_kin_energy(e_grid_r, range_cm)
    save_scaled_kin_energy(r_grid, e_grid_inv, out_dir)

    # Ionisation lambda(E)
    e_grid_l, sigma_l, lam_l = build_lambda_table(particle, material, cut_eV=cut,
                                                   e_min=E_MIN_eV, e_max=E_MAX_eV)
    save_lambda(e_grid_l, sigma_l, lam_l, out_dir)

    log.info("done %s in %s. %d bins/decade.",
             particle.name, material.name, G4_BINS_PER_DECADE)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_for(ACTIVE)


if __name__ == "__main__":
    main()
