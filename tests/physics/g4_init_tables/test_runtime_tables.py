"""
Runtime-path tests: the scripted g4pv_eval kernel and the InitTableSet
loader that the stepping/tracking processes consume. These pin that the
*scripted* kernel matches the eager spline bit for bit and that the table
wiring (units, spline targets, edge semantics) is correct.
"""
import numpy as np
import pytest
import torch

from physics.g4_init_tables.grid import log_energy_grid
from physics.g4_init_tables.interpolator import (
    G4PhysicsVector,
    g4pv_eval,
    g4pv_inverse_range_eval,
)


def _smooth_log_pv(n: int = 121) -> G4PhysicsVector:
    e = log_energy_grid(10.0, 1e13, n)
    y = torch.log(e) * torch.sqrt(e)          # smooth, spans many decades
    return G4PhysicsVector(e, y, log_grid=True, use_spline=True)


def _free_pv() -> G4PhysicsVector:
    gen = torch.Generator().manual_seed(42)
    x = torch.cumsum(torch.rand(60, dtype=torch.float64, generator=gen) + 0.1, dim=0)
    y = x ** 1.5
    return G4PhysicsVector(x, y, log_grid=False, use_spline=True)


class TestG4pvEvalKernel:
    def test_is_scripted(self):
        # A scripted function is bit-exact against the G4-validated eager
        # path; letting the compiler inline this kernel introduces
        # ULP-level codegen drift.
        assert isinstance(g4pv_eval, torch.jit.ScriptFunction)

    def test_class_call_equals_kernel_log_grid(self):
        pv = _smooth_log_pv()
        q = torch.exp(torch.linspace(2.4, 29.8, 50_001, dtype=torch.float64))
        via_class = pv(q)
        via_kernel = g4pv_eval(pv.bin, pv.data, pv.sec_deriv,
                               pv.log_emin, pv.inv_dbin_log,
                               True, True, q)
        assert torch.equal(via_class, via_kernel)

    def test_class_call_equals_kernel_free_grid(self):
        pv = _free_pv()
        q = torch.linspace(float(pv.bin[0]) - 1.0, float(pv.bin[-1]) + 1.0,
                           10_001, dtype=torch.float64)
        assert torch.equal(pv(q), g4pv_eval(pv.bin, pv.data, pv.sec_deriv,
                                            0.0, 0.0, False, True, q))

    def test_edge_clamping(self):
        pv = _smooth_log_pv()
        q = torch.tensor([1.0, 10.0, 1e13, 1e14], dtype=torch.float64)
        out = pv(q)
        assert out[0] == pv.data[0] and out[1] == pv.data[0]
        assert out[2] == pv.data[-1] and out[3] == pv.data[-1]

        fv = _free_pv()
        fq = torch.stack([fv.bin[0], fv.bin[-1]])
        fout = fv(fq)
        assert fout[0] == fv.data[0] and fout[1] == fv.data[-1]

    def test_node_values_exact(self):
        # At a node the cubic term vanishes (b*(b-1) = 0 for b in {0, 1}),
        # but G4's log-formula bin lookup may land a node query in either
        # adjacent bin: b=0 returns y_i verbatim, while b=1 computes
        # y_{i-1} + (y_i - y_{i-1}), whose rounding may differ from y_i by
        # 1 ULP. Allow that ULP.
        pv = _smooth_log_pv()
        interior = pv.bin[1:-1]
        assert torch.allclose(pv(interior), pv.data[1:-1], rtol=1e-14, atol=0.0)

    def test_float32_query_upcast(self):
        pv = _smooth_log_pv()
        q32 = torch.tensor([1e6, 1e8], dtype=torch.float32)
        out = pv(q32)
        assert out.dtype == torch.float64


class TestInverseRangeEval:
    def test_below_first_node_quadratic_and_zero(self):
        pv = _free_pv()
        min_kin = 1000.0
        rmin = float(pv.bin[0])
        r = torch.tensor([-1.0, 0.0, 0.5 * rmin, rmin], dtype=torch.float64)
        e = g4pv_inverse_range_eval(pv.bin, pv.data, pv.sec_deriv, min_kin, r)
        assert e[0] == 0.0 and e[1] == 0.0
        assert torch.isclose(e[2], torch.tensor(min_kin * 0.25, dtype=torch.float64))
        # at rmin the table takes over; b=0 so the result is exact
        assert torch.equal(e[3], pv.data[0])


from common.paths import PATH_INIT_DEDX
from physics.g4_init_tables.runtime import InitTableSet, get_init_table_set


def _require_init_tables():
    if not PATH_INIT_DEDX.exists():
        pytest.skip("init tables missing — run physics.g4_init_tables.build_all")


class TestInitTableSet:
    @pytest.fixture(scope="class")
    def ts(self):
        _require_init_tables()
        return InitTableSet.load()

    def test_sigma_ion_node_values(self, ts):
        from common.paths import PATH_INIT_LAMBDA_ION
        d = np.load(PATH_INIT_LAMBDA_ION)
        e = torch.from_numpy(d["energy_eV"][1:-1])
        sig = torch.from_numpy(d["sigma_per_cm"][1:-1])
        # node queries can land in either adjacent bin (1-ULP tolerance,
        # see test_node_values_exact).
        assert torch.allclose(ts.sigma_ion(e), sig, rtol=1e-12, atol=0.0)

    def test_dedx_and_range_units(self, ts):
        # 100 MeV protons in Al: dE/dx ~ 9.6 MeV/cm = 9.6e6 eV/cm (GEANT4/NIST),
        # CSDA range ~ 3.7 cm.
        e = torch.tensor([100e6], dtype=torch.float64)
        dedx = float(ts.dedx(e)[0])
        rng = float(ts.range(e)[0])
        assert 5e6 < dedx < 1e8, f"dE/dx unit suspicion: {dedx} (want eV/cm)"
        assert 1.0 < rng < 10.0, f"range unit suspicion: {rng} (want cm)"

    def test_inverse_range_roundtrip(self, ts):
        e = torch.tensor([1e6, 1e7, 1e8], dtype=torch.float64)
        r = ts.range(e)
        e_back = ts.inverse_range(r)
        assert torch.allclose(e_back, e, rtol=1e-8)

    def test_to_is_chainable(self, ts):
        assert ts.to("cpu") is ts

    def test_nodes_stay_float64(self, ts):
        for pv in (ts.dedx, ts.range, ts.inverse_range, ts.sigma_ion):
            assert pv.bin.dtype == torch.float64
            assert pv.data.dtype == torch.float64

    def test_sigma_ion_biased_pivots_at_epeak(self, ts):
        epeak = ts.sigma_ion_epeak
        # below the peak: identical to fresh sigma
        e_lo = torch.tensor([0.5 * epeak], dtype=torch.float64)
        assert torch.equal(ts.sigma_ion_biased(e_lo), ts.sigma_ion(e_lo))
        # above the peak: evaluated at max(epeak, 0.8 E)
        e_hi = torch.tensor([50e6], dtype=torch.float64)
        assert torch.equal(ts.sigma_ion_biased(e_hi),
                           ts.sigma_ion(e_hi * 0.8))


class TestInitTableSetErrors:
    def test_missing_file_names_build_command(self, tmp_path, monkeypatch):
        import physics.g4_init_tables.runtime as rt
        monkeypatch.setattr(rt, "PATH_INIT_DEDX", tmp_path / "nope.npz")
        with pytest.raises(FileNotFoundError, match="build_all"):
            rt.InitTableSet.load()

    def test_shared_instance_refuses_device_move(self):
        _require_init_tables()
        ts = get_init_table_set()
        with pytest.raises(RuntimeError, match="shared"):
            ts.to("cuda")
        assert ts.to("cpu") is ts  # cpu/no-op moves stay allowed


def test_get_init_table_set_is_cached():
    _require_init_tables()
    assert get_init_table_set() is get_init_table_set()


def test_init_table_set_has_no_multiple_scattering_vector():
    # Four vectors: dE/dx, range, inverse-range and ionisation cross
    # section. There is no multiple-scattering stage, so there is no
    # lambda0_msc vector to load and no path constant to load it from.
    from physics.g4_init_tables.runtime import InitTableSet

    ts = InitTableSet.load("cpu")
    assert not hasattr(ts, "e2_over_lambda0")
    assert not hasattr(ts, "lambda0_msc")
    assert all(hasattr(ts, n) for n in
               ("dedx", "range", "inverse_range", "sigma_ion"))


def test_shared_instance_refuses_to_move_off_cpu():
    # get_init_table_set() hands out ONE shared CPU instance to offline
    # consumers; moving it would corrupt every other holder. Runtime code
    # must own a fresh InitTableSet.load(device).
    from physics.g4_init_tables.runtime import get_init_table_set

    with pytest.raises(RuntimeError, match="shared CPU instance"):
        get_init_table_set().to("cuda")
