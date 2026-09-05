"""The iron/beryllium Material floats are pasted from the extended
InitTablesDump MATPROPS printout (Geant4 11.2.0, /run/setCut 1 um). This test
re-parses that committed printout and pins the constants to it, so a drifted
edit fails against the recorded G4 doubles."""
import re
from pathlib import Path

import pytest

from physics.definitions.material import beryllium, iron

REPO = Path(__file__).resolve().parents[2]
DUMP = REPO / "measurements/g4_material_constants_dump.txt"


def _props(g4_name: str) -> dict[str, float]:
    for line in DUMP.read_text().splitlines():
        if line.startswith("MATPROPS") and f"name={g4_name} " in line:
            return {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.e+]+)", line)}
    raise AssertionError(f"no MATPROPS line for {g4_name} in {DUMP}")


@pytest.mark.parametrize("mat,g4_name", [(iron, "G4_Fe"), (beryllium, "G4_Be")])
def test_material_floats_equal_the_g4_dump(mat, g4_name):
    p = _props(g4_name)
    assert mat.Tc == p["Tc_eV"]
    assert mat.I == p["Imean_eV"]
    assert mat.A == p["A_g_mole"]
    assert mat.Z == p["Z"]
    assert mat.rho == p["density_g_cm3"]
    assert mat.radiation_length_cm == p["radlen_cm"]


def test_derived_quantities_are_positive():
    for m in (iron, beryllium):
        assert m.number_of_atoms_per_volume > 0
        assert m.electron_density > 0
