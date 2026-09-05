"""Gate T's statistics must discriminate in both directions and must not be
fooled by the zero-atom spike in E_SEC / THETA."""
import numpy as np
import pytest

from benchmark.gate_t_stats import compare_distributions


def _sample(n=20000, seed=0, atom_frac=0.2, scale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(mean=8.0, sigma=1.0, size=n) * scale
    x[rng.random(n) < atom_frac] = 0.0
    return x


def test_identical_samples_score_near_zero():
    a = {"E_SEC": _sample(seed=1)}
    out = compare_distributions(a, a)
    assert out["ks"]["E_SEC"] == pytest.approx(0.0, abs=1e-12)
    assert out["atom_mass"]["E_SEC"] == pytest.approx(0.0, abs=1e-12)


def test_independent_draws_score_small_but_nonzero():
    out = compare_distributions({"E_SEC": _sample(seed=1)},
                                {"E_SEC": _sample(seed=2)})
    assert 0.0 < out["ks"]["E_SEC"] < 0.05


def test_a_shifted_continuum_is_detected():
    # NEGATIVE CONTROL. A gate that cannot fail is not a gate.
    out = compare_distributions({"E_SEC": _sample(seed=1, scale=1.5)},
                                {"E_SEC": _sample(seed=2)})
    assert out["ks"]["E_SEC"] > 0.1


def test_atom_mass_is_measured_separately_from_the_continuum():
    # Same continuum, different atom fraction: the continuum KS must stay
    # small while the atom mass moves. If the atom leaked into the KS, this
    # would read as a large shape difference that is not one.
    out = compare_distributions({"E_SEC": _sample(seed=1, atom_frac=0.40)},
                                {"E_SEC": _sample(seed=2, atom_frac=0.20)})
    assert out["ks"]["E_SEC"] < 0.05, "atom leaked into the continuum KS"
    assert out["atom_mass"]["E_SEC"] == pytest.approx(0.20, abs=0.02)


@pytest.mark.slow
def test_order_statistics_survive_large_inputs():
    # torch.Tensor.quantile raises above ~16M elements and truth files are
    # far larger, so the implementation must use numpy. 17M floats is ~136 MB
    # per array -- large but tractable, and this is the only place the limit
    # is actually crossed.
    n = 17_000_000
    rng = np.random.default_rng(0)
    a = {"E_CNT": rng.random(n)}
    b = {"E_CNT": rng.random(n)}
    out = compare_distributions(a, b)
    assert np.isfinite(out["ks"]["E_CNT"])
