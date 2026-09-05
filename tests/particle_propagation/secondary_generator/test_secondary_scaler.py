"""Tests for SecondaryXScaler / SecondaryYScaler.

X bounds are DATA-FITTED (restored by `load()` from `<ClassName>.state.pt`),
not derived from BeamConfig; E_SEC uses the physics bounds (t_max, Tc). The
Y scaler operates on the single-column `SecondaryFeats` layout
(E_SEC_IDX = 0). Numerical values are pinned by `tests/physics/test_scalers.py`.
"""
import pytest
import torch

from common.enums import SecondaryFeats
from physics.g4h_ionisation.generators.secondary_generator.scaler import SecondaryXScaler, SecondaryYScaler


class TestSecondaryXScaler:

    @pytest.fixture
    def scaler(self):
        return SecondaryXScaler.load()

    @pytest.fixture
    def x_data(self, scaler):
        torch.manual_seed(42)
        n = 100
        lo, hi = scaler.min_post_log, scaler.max_post_log
        return torch.pow(10, torch.rand(n) * (hi - lo) + lo)

    def test_bounds_come_from_the_fitted_state(self, scaler):
        assert scaler.min_post_log.ndim == 0
        assert scaler.max_post_log.ndim == 0
        assert torch.isfinite(scaler.min_post_log)
        assert torch.isfinite(scaler.max_post_log)
        assert scaler.max_post_log > scaler.min_post_log

    def test_scale_rescale_roundtrip(self, scaler, x_data):
        scaled = scaler.scale(x_data)
        assert torch.all(scaled >= -1e-5)
        assert torch.all(scaled <= 1.0 + 1e-5)
        rescaled = scaler.rescale(scaled)
        assert torch.allclose(x_data, rescaled, atol=1e-3, rtol=1e-3)

    def test_device_move(self, scaler):
        scaler.to('cpu')
        assert scaler.min_post_log.device.type == 'cpu'


class TestSecondaryYScaler:
    """The Y scaler is single-column, `[e_sec]`: it has no `theta_min` /
    `theta_max` and carries no fitted state at all -- every bound it uses
    is class-level physics (t_max, Tc)."""

    @pytest.fixture
    def scaler(self):
        return SecondaryYScaler.load()

    @pytest.fixture
    def y_data(self):
        torch.manual_seed(42)
        n = 100
        e_sec = torch.pow(10, torch.rand(n) * 3 + 2)
        return e_sec.reshape(n, 1)

    @pytest.fixture
    def phase_space(self):
        torch.manual_seed(7)
        return torch.pow(10, torch.rand(100) * 2 + 7)

    def test_the_scaler_carries_no_fitted_state(self):
        """`_fit_y_scaler` sets neither `theta_min` nor `theta_max` on a
        fresh instance -- this scaler has no data-fitted state.

        Asserted on a FRESH instance, not a loaded one: `load()` `setattr`s
        every key present in a state file without validating it, so
        loading a state file that still carries `theta_min`/`theta_max`
        would silently re-attach both as dead attributes even though
        nothing reads them, which would make a loaded instance a false
        negative for this assertion.
        """
        fresh = SecondaryYScaler()
        assert not hasattr(fresh, "theta_min")
        assert not hasattr(fresh, "theta_max")

    def test_output_is_one_column(self, scaler, y_data, phase_space):
        assert scaler.scale(y_data, phase_space=phase_space).shape == (100, 1)

    def test_scale_rescale_roundtrip(self, scaler, y_data, phase_space):
        scaled = scaler.scale(y_data, phase_space=phase_space)
        rescaled = scaler.rescale(scaled, phase_space=phase_space)
        torch.testing.assert_close(
            y_data[:, SecondaryFeats.E_SEC_IDX],
            rescaled[:, SecondaryFeats.E_SEC_IDX], rtol=1e-3, atol=1e-3)


class TestSecondaryYScalerZeroAtom:
    """The z == 0 Dirac atom must rescale through a pinned `Tc` buffer,
    not through the generic log-space exponential formula.

    Every lane with an exact-zero z maps, under the generic formula, to the
    same constant `10 ** log10(Tc)`. Because that constant would be
    produced by an exponential, an eager and a compiled backend can round
    it to different adjacent float32 values on GPU, which would displace
    the whole atom together and read as a mismatch on any statistic
    comparing the two arms. Routing z == 0 through a pinned buffer instead
    of the exponential removes that sensitivity.
    """

    @pytest.fixture
    def scaler(self):
        return SecondaryYScaler.load()

    def test_zero_z_rescales_to_the_buffer_bit_for_bit_at_every_ke(self, scaler):
        ke = torch.pow(10, torch.linspace(5.6, 8.0, 64))  # 0.4 MeV .. 100 MeV
        z = torch.zeros(ke.numel(), 1)

        got = scaler.rescale(z, phase_space=ke)[:, int(SecondaryFeats.E_SEC_IDX)]
        want = scaler._Tc.expand_as(got)

        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (
            f"z==0 lanes did not land on the pinned Tc buffer: "
            f"{sorted(set(got.tolist()))[:4]} vs {float(scaler._Tc)}")

    def test_the_buffer_is_float32_of_the_active_material_Tc(self, scaler):
        from common.run_context import TARGET_MATERIAL

        want = torch.tensor(TARGET_MATERIAL.Tc, dtype=torch.float32)
        assert scaler._Tc.dtype is torch.float32
        assert scaler._Tc.ndim == 0
        assert torch.equal(scaler._Tc.view(torch.int32), want.view(torch.int32))
        # And it is not the value an exponential round-trip through
        # log10(Tc) would produce -- that lands a few ulp below float32(Tc).
        roundtrip = 10 ** torch.log10(torch.tensor(TARGET_MATERIAL.Tc))
        assert not torch.equal(scaler._Tc.view(torch.int32),
                               roundtrip.view(torch.int32))

    def test_nonzero_z_is_untouched_by_the_where(self, scaler):
        ke = torch.full((5,), 1.0e8)
        z = torch.tensor([[1e-6], [0.25], [0.5], [0.75], [1.0]])
        got = scaler.rescale(z, phase_space=ke)[:, int(SecondaryFeats.E_SEC_IDX)]

        _, max_ = scaler.min_max_e_secondary(ke)
        log10_min = scaler._log10_Tc
        want = 10 ** (z[:, 0] * (torch.log10(max_) - log10_min) + log10_min)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))

    def test_source_pin_zero_branch_bypasses_the_exponential(self):
        """Structural pin, in the repo's established source-pin style: the
        z == 0 branch must be selected with `torch.where` against the buffer,
        so no exponential is evaluated for the atom's value."""
        import ast
        import inspect
        import textwrap

        from physics.g4h_ionisation.generators.secondary_generator import scaler as m

        src = textwrap.dedent(
            inspect.getsource(m.SecondaryYScaler.rescale_e_secondary))
        tree = ast.parse(src)
        wheres = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "where"]
        assert wheres, "rescale_e_secondary must route the atom via torch.where"
        assert "_Tc" in ast.unparse(wheres[0].args[1])

        fn = tree.body[0]
        pow_idx = min(i for i, s in enumerate(fn.body)
                      if isinstance(s, ast.Assign) and "10 **" in ast.unparse(s.value))

        # Resolve the condition through a single local alias if there is one.
        # An UNRESOLVABLE condition (inlined, or bound some other way) defaults
        # to `pow_idx + 1`, i.e. it FAILS the ordering assert below rather than
        # passing by default -- otherwise an inlined mask written after the
        # exponential would slip through.
        assigns = {t.id: (i, ast.unparse(s.value))
                   for i, s in enumerate(fn.body)
                   if isinstance(s, ast.Assign)
                   for t in s.targets if isinstance(t, ast.Name)}
        cond = ast.unparse(wheres[0].args[0])
        cond_idx, cond_src = assigns.get(cond, (pow_idx + 1, cond))
        assert "== 0" in cond_src, cond_src

        # ...and it must be taken BEFORE the log-space transform, i.e. against
        # the network's raw z, not against the exponentiated value.
        assert cond_idx < pow_idx, (
            f"z == 0 mask is not provably taken before the exponential: {cond_src}")

    def test_rescale_traces_into_one_graph_with_no_breaks(self, scaler):
        # post_step_do_it's compiled scope covers predict/rescale; the
        # `torch.where` against a buffer must stay break-free.
        import torch._dynamo as dynamo

        ke = torch.full((256,), 1.0e8)
        z = torch.zeros(256, 1)

        dynamo.reset()
        explanation = dynamo.explain(scaler.rescale)(z, phase_space=ke)
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_the_buffer_follows_a_device_move(self, scaler):
        # `_Tc` is a CLASS attribute, so asserting only `.device.type == 'cpu'`
        # is vacuous on a CPU box -- it holds whether or not `to()` touched it.
        # The operative evidence is the instance shadow: `to()` assigns
        # `self._Tc`, so the name must appear in the instance __dict__.
        assert "_Tc" not in vars(scaler)
        scaler.to('cpu')
        assert "_Tc" in vars(scaler), "to() does not move the _Tc buffer"
        assert scaler._Tc.device.type == 'cpu'
