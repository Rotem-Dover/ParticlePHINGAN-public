"""Fe/Be entries exist in all four per-material model-constant tables, and a
material with no Sternheimer entry now fails loud instead of silently
skipping the density correction."""
import pytest
import torch

from physics.definitions.material import aluminum, beryllium, iron
from physics.definitions.particle import proton
from physics.g4_init_tables.models import _pstar_tables as P
from physics.g4_init_tables.models.bethe_bloch import _STERNHEIMER, compute_dedx_per_volume
from physics.g4_init_tables.models.bragg import _ICRU49, _PSTAR_DATA
from physics.g4_init_tables.models.shell_correction import (
    ATOMIC_SHELL_ELECTRONS,
    ATOMIC_NUMBER_OF_SHELLS,
)


def test_sternheimer_has_fe_be():
    assert "Iron" in _STERNHEIMER and "Beryllium" in _STERNHEIMER


def test_missing_sternheimer_entry_raises():
    import dataclasses
    ghost = dataclasses.replace(aluminum, name="Unobtainium")
    with pytest.raises(KeyError):
        compute_dedx_per_volume(proton, ghost, torch.tensor([1e7], dtype=torch.float64), cut_eV=1e3)


def test_pstar_has_fe_be():
    for name in ("Iron", "Beryllium"):
        assert name in _PSTAR_DATA
        assert _PSTAR_DATA[name].shape == P.DEDX_Al_MeV_cm2_per_g.shape


def test_icru49_has_z4_and_z26():
    assert 4 in _ICRU49 and 26 in _ICRU49
    assert len(_ICRU49[4]) == 5 and len(_ICRU49[26]) == 5


def test_atomic_shells_z4_z26():
    assert ATOMIC_SHELL_ELECTRONS[4] == [2, 2]
    assert sum(ATOMIC_SHELL_ELECTRONS[26]) == 26
    assert ATOMIC_NUMBER_OF_SHELLS[26] == len(ATOMIC_SHELL_ELECTRONS[26]) == 9
