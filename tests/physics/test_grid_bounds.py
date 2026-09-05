"""Grid-bounds regression for the registered proton beams.

Three proton beams (aluminum, iron, beryllium) are registered via
`PHINGAN_BEAM`, with aluminum as the default; the sections below check each
preset's own grid bounds plus the default env's re-exports.
"""
import common.units as U
from physics.definitions.beam import proton_in_aluminum_100MeV


def test_proton_grid_bounds_unchanged():
    gb = proton_in_aluminum_100MeV.grid_bounds
    assert gb.energy_min_eV == 0.4 * U.MeV2eV
    assert gb.step_len_min_m == 1e-11
    assert gb.step_len_max_m == 1e-4


def test_run_context_default_proton_reexports():
    import common.run_context as rc
    assert rc.GRID_ENERGY_MIN_eV == 0.4 * U.MeV2eV
    assert rc.GRID_STEP_LEN_MIN_m == 1e-11
    assert rc.GRID_STEP_LEN_MAX_m == 1e-4
    assert rc.GRID_BOUNDS is rc.ACTIVE.grid_bounds


def test_config_reexports_proton():
    import common.config as cfg
    assert cfg.GRID_ENERGY_MIN_eV == 0.4 * U.MeV2eV
    assert cfg.GRID_STEP_LEN_MIN_m == 1e-11
    assert cfg.GRID_STEP_LEN_MAX_m == 1e-4


def test_iron_grid_bounds():
    from physics.definitions.beam import proton_in_iron_100MeV
    gb = proton_in_iron_100MeV.grid_bounds
    assert gb.energy_min_eV == 0.4 * U.MeV2eV
    assert gb.step_len_min_m == 1e-11
    assert gb.step_len_max_m == 3e-4


def test_beryllium_grid_bounds():
    from physics.definitions.beam import proton_in_beryllium_100MeV
    gb = proton_in_beryllium_100MeV.grid_bounds
    assert gb.energy_min_eV == 0.4 * U.MeV2eV
    assert gb.step_len_min_m == 1e-11
    assert gb.step_len_max_m == 2e-4
