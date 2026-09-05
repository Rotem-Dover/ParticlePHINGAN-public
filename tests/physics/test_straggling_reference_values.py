"""The three straggling models reproduce a committed set of reference values.

`tests/data/straggling_reference.npz` holds, for 121 (kinetic energy, step
length) points, every scalar of the continuous model's state, its regime
label, and 512-point samples of the four distributions. The values were
computed by `scripts/capture_straggling_reference.py` on this code and are
compared at rtol=1e-12, so any numerical drift in `explicit_physics`, the
init tables it reads, or the beam constants fails here.
"""
from pathlib import Path

import numpy as np
import pytest

import common.units as U
from common.utils import PointInPhaseSpace
from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import (
    ContinuousStragglingModel,
)
from physics.g4h_ionisation.explicit_physics.discrete.straggling_function import (
    DiscreteStragglingModel,
)
from physics.g4h_ionisation.explicit_physics.step_length.straggling_function import (
    StepLengthModel,
)

REFERENCE_VALUES = Path(__file__).resolve().parents[1] / "data" / "straggling_reference.npz"
RTOL = 1e-12
N_PDF = 512

pytestmark = pytest.mark.skipif(not REFERENCE_VALUES.exists(), reason="reference values not captured")


@pytest.fixture(scope="module")
def g():
    return np.load(REFERENCE_VALUES, allow_pickle=False)


@pytest.fixture(scope="module")
def recomputed(g):
    cnt, sec, length = ContinuousStragglingModel(), DiscreteStragglingModel(), StepLengthModel()
    names = [str(n) for n in g["scalar_names"]]
    n = g["ke_ev"].shape[0]
    scalars = np.full((n, len(names)), np.nan)
    types, pdf_cnt, cdf_cnt, pdf_sec, pdf_len = [], [], [], [], []

    for i in range(n):
        ke, l_m = float(g["ke_ev"][i]), float(g["l_m"][i])
        point = PointInPhaseSpace(step_length=l_m * U.m2cm, primary_energy=ke)
        cnt.set_state(point)
        sec.set_state(point)
        length.set_state(point)

        state = {k: float(v) for k, v in cnt.__dict__.items()
                 if not k.startswith("_") and isinstance(v, (int, float))
                 and not isinstance(v, bool)}
        for j, name in enumerate(names):
            if name in state:
                scalars[i, j] = state[name]

        types.append("|".join(sorted(t.name for t in cnt.straggling_type)))
        # Leading 0.0 is required, not cosmetic: few_ionization_collisions_pdf's
        # own normalization asserts x[0] == 0 ("should be called only with a
        # full range of x"); a pure logspace(0.0, ...) starts at 10**0 = 1 and
        # trips that assertion in the few-collisions regime. Must match the
        # abscissa the capture script used.
        xs = np.concatenate(([0.0], np.logspace(0.0, np.log10(ke), N_PDF - 1)))
        pdf_cnt.append(np.asarray(cnt.straggling_function(xs), dtype=np.float64))
        cdf_cnt.append(np.asarray(cnt.straggling_cdf(xs), dtype=np.float64))
        pdf_sec.append(np.asarray(sec.straggling_function(xs), dtype=np.float64))
        pdf_len.append(np.asarray(
            length.straggling_function(np.logspace(-9.0, 1.0, N_PDF)), dtype=np.float64))

    return dict(scalars=scalars, straggling_type=np.array(types),
                pdf_cnt=np.array(pdf_cnt), cdf_cnt=np.array(cdf_cnt),
                pdf_sec=np.array(pdf_sec), pdf_len=np.array(pdf_len))


def test_the_reference_values_are_the_right_beam(g):
    assert str(g["beam"]) == "proton_in_Aluminum"


def test_the_captured_scalar_set_is_not_empty(g):
    # Negative control: a capture that recorded nothing would make every
    # comparison below vacuously true.
    assert g["scalars"].shape[0] >= 100
    assert g["scalars"].shape[1] >= 10


def test_the_straggling_regime_matches_at_every_point(g, recomputed):
    # The regime is a discrete branch choice. If it differs anywhere, the
    # continuous comparisons below are comparing different functions.
    mismatched = [
        (float(g["ke_ev"][i]), float(g["l_m"][i]), str(g["straggling_type"][i]), recomputed["straggling_type"][i])
        for i in range(g["ke_ev"].shape[0])
        if str(g["straggling_type"][i]) != recomputed["straggling_type"][i]
    ]
    assert mismatched == []


def test_the_scalar_state_matches_reference(g, recomputed):
    np.testing.assert_allclose(recomputed["scalars"], g["scalars"], rtol=RTOL, equal_nan=True)


@pytest.mark.parametrize("key", ["pdf_cnt", "cdf_cnt", "pdf_sec", "pdf_len"])
def test_the_distributions_match_reference(g, recomputed, key):
    np.testing.assert_allclose(recomputed[key], g[key], rtol=RTOL, atol=0.0)
