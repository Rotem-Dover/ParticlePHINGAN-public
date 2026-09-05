"""ContinuousYScaler must consume the analytic mean/std lookup table built
by build_analytic_mean_std_table.py: a table with NaN over the
deterministic-loss region (~29% of cells) and a step grid starting at
GRID_STEP_LEN_MIN_m.
"""
import inspect

import torch
import pytest

import common.paths as paths
from physics.g4h_ionisation.generators.continuous_generator.scaler import ContinuousYScaler

CNT_Y_SCALER_STATE = paths.SCALERS_DIR / "ContinuousYScaler.state.pt"

# Deliberately NOT a skipif. `ContinuousYScaler.load()` hard-fails without this
# artifact, so it is a REQUIRED input, not an optional one -- and
# tests/physics/test_scalers.py already hard-fails on the same condition.
# Skipping here would make the same missing file green in one file and red in
# another. Build it with scripts/build_continuous_scaler_state.py.
assert CNT_Y_SCALER_STATE.exists(), (
    f"{CNT_Y_SCALER_STATE} is missing. It is a required artifact: build the "
    f"analytic lookup with `python -m physics.g4h_ionisation.generators."
    f"continuous_generator.build_analytic_mean_std_table` (~2 h), then "
    f"`python scripts/build_continuous_scaler_state.py`."
)


def _strip_comments(src: str) -> str:
    """Source with `#` comments removed, so a prose mention of a forbidden
    identifier does not read as its use."""
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


def test_the_live_lookup_stores_nan_over_the_deterministic_region():
    # The builder stores NaN for cells in the deterministic-loss region
    # instead of a fabricated sigma; ~29% of cells fall in that region.
    s = ContinuousYScaler.load()
    nan_frac = torch.isnan(s.lookup_table[..., 0]).float().mean().item()
    assert nan_frac > 0.25, f"lookup has {nan_frac:.4f} NaN; expected > 0.25"


def test_the_live_lookup_covers_the_configured_step_grid():
    # The step grid starts at GRID_STEP_LEN_MIN_m = 1e-11 m.
    s = ContinuousYScaler.load()
    assert float(s.step_table[0]) == pytest.approx(-11.0, abs=1e-6)


def test_scale_raises_rather_than_emitting_nan():
    # _compute_lookup_indices clips both axes with no diagnostic. An
    # out-of-grid short step clamps into a fully-NaN column of the lookup
    # table, so a silently-wrong-but-finite z would become a NaN gradient.
    # Guard it.
    s = ContinuousYScaler.load()
    # A step length far below the grid, which the clip sends into the
    # degenerate prefix.
    phase = torch.tensor([[1.0e8, 1.0e-30]], dtype=torch.float32)
    e_cnt = torch.tensor([[1.0e3]], dtype=torch.float32)
    with pytest.raises(AssertionError, match="NaN"):
        s.scale(e_cnt, phase_space=phase)


def test_rescale_is_deliberately_unguarded_and_returns_nan():
    # The MIRROR of the test above, and the reason the guard is asymmetric.
    # The NaN region is not only reachable by an out-of-grid clamp -- it is
    # INTERIOR: 29.2% of cells, every kinetic energy at L <~ 1e-8 m -- so on
    # the inference path an ordinary short step gathers NaN on ~30% of
    # lanes. Asserting here would refuse every run.
    # G4hIonisation.along_step_do_it discards those lanes with
    # torch.where(is_deterministic, average_loss, e_cnt), and the escape is
    # guarded at that real site (test below).
    s = ContinuousYScaler.load()
    phase = torch.tensor([[1.0e8, 1.0e-10]], dtype=torch.float32)
    z = torch.tensor([[0.0]], dtype=torch.float32)
    out = s.rescale(z, phase_space=phase)
    assert torch.isnan(out).all(), out


def _poison_one_lane(gen):
    """Make `gen.predict` emit NaN on lane 0 -- a lane the
    is_deterministic/full_deposit masks do not cover. Returns the
    unpoisoned predict method."""
    real_predict = gen.predict

    def poisoned(x):
        y = real_predict(x).clone()
        y[0] = float("nan")
        return y

    gen.predict = poisoned
    return real_predict


def test_along_step_guards_the_escape_that_rescale_does_not():
    # The guard that replaces a rescale-side assertion: if a NaN survives the
    # is_deterministic/full_deposit where-chain, along_step_do_it LATCHES it
    # into a device-resident flag and check_nan_guard() refuses. The two
    # predicates are not the same expression the table builder used, so their
    # agreement is empirical and worth pinning.
    #
    # Direct callers (this test) own the latch explicitly: it is zeroed in
    # G4hIonisation.__init__, so no run() wrapper is needed -- reset and read
    # are just called by hand. TrackSimulator.run() does the same thing around
    # its loop (tests below).
    from benchmark.track_simulator_config import build_stepping_manager

    proc = build_stepping_manager("phin_gan", "cpu").processes[0]

    # Seeded: the poisoned lane is lane 0, and with ~0.1% probability a
    # random draw puts lane 0's own (E, L) inside the deterministic /
    # full-deposit set, where the where-chain legitimately overwrites the
    # injected NaN and the guard correctly stays silent. That is a
    # pre-existing flake in the CONTROL, not in the mechanism, and a fixed
    # seed removes it without weakening the control.
    torch.manual_seed(0)

    ke = torch.full((64,), 1.0e8)
    step = torch.full((64,), 1.0e-5)
    # Ordinary physics: no NaN escapes, and the guard stays silent.
    proc.reset_nan_guard()
    out = proc.along_step_do_it(kinetic_energy=ke, step_length=step)
    assert torch.isfinite(out["e_cnt"]).all()
    proc.check_nan_guard()

    # Negative control: force the network to emit NaN and confirm the guard
    # actually fires (a guard that cannot fail is not a guard).
    real_predict = _poison_one_lane(proc.continuous_generator)
    try:
        proc.along_step_do_it(kinetic_energy=ke, step_length=step)
    finally:
        proc.continuous_generator.predict = real_predict
    with pytest.raises(RuntimeError, match="NaN e_cnt escaped"):
        proc.check_nan_guard()


def test_the_along_step_detection_is_unconditional_by_construction():
    # SOURCE pin, deliberately: there is no GPU in this environment, so the
    # CUDA arm cannot be exercised here at all. What it defends against is
    # someone reintroducing a device/tracing condition on the DETECTION,
    # which would silently restore CPU-only coverage while the CPU tests
    # stay green.
    #
    # The accumulate is a pure device-side op (`|=` into a 0-d bool), so it
    # costs no sync on any device; the only host read is check_nan_guard(),
    # which the run loop calls once per run, outside every compiled region.
    import physics.g4h_ionisation.g4h_ionisation as gi

    src = inspect.getsource(gi.G4hIonisation.along_step_do_it)
    assert "self._nan_seen |= torch.isnan(e_cnt).any()" in src
    # No escape hatch on the detection path. Comments are stripped first: the
    # block above the accumulate NAMES `is_cuda`/`is_compiling()` in prose,
    # explaining why neither appears in the code.
    code = _strip_comments(src)
    assert "is_cuda" not in code
    assert "is_compiling" not in code
    # The read lives elsewhere, and it is a real host read.
    read_src = inspect.getsource(gi.G4hIonisation.check_nan_guard)
    assert "bool(self._nan_seen)" in read_src
    assert "NaN e_cnt escaped" in read_src
    assert "is_deterministic ORs" in read_src


def test_the_nan_accumulator_lives_outside_the_compiled_core():
    # SOURCE pin. The guard contract is unconditional, sync-free, and read
    # once per run; WHERE the accumulate sits is itself pinned, because
    # compiling the `.any()` reduction together with the whole along-step
    # body lets the compiler fold it into one kernel whose geometry is
    # baked from the compile-time size hint and never revisited, which
    # dominates the phase's runtime and makes it sensitive to the warm-up
    # shape. The phase math is therefore compiled on its own
    # (`_along_step_core`) and the accumulate runs in the UNcompiled public
    # entry `along_step_do_it`.
    import physics.g4h_ionisation.g4h_ionisation as gi

    core_src = inspect.getsource(gi.G4hIonisation._along_step_core)
    assert "_nan_seen" not in _strip_comments(core_src), \
        "the accumulator must not be inside the compiled core"

    wrapper_src = inspect.getsource(gi.G4hIonisation.along_step_do_it)
    assert "self._nan_seen |= torch.isnan(e_cnt).any()" in wrapper_src

    # The compile latch must attach to the core, never to the public entry --
    # otherwise the hoist is undone silently and the reduction is fused back in.
    to_code = _strip_comments(inspect.getsource(gi.G4hIonisation.to))
    assert '"_along_step_core"' in to_code
    assert '"along_step_do_it"' not in to_code


def test_the_compiled_core_still_traces_into_one_graph_with_no_breaks():
    # The hoist must not have cost the phase its single fused graph: the core
    # is what `to()` compiles on CUDA, so a break here would be a real
    # regression rather than a cosmetic one. (CPU trace; the device latch in
    # `to()` is inert here, so the core is the plain bound method.)
    import torch._dynamo as dynamo
    from benchmark.track_simulator_config import build_stepping_manager

    proc = build_stepping_manager("phin_gan", "cpu").processes[0]
    ke = torch.full((256,), 1.0e8)
    step = torch.full((256,), 1.0e-5)

    dynamo.reset()
    explanation = dynamo.explain(proc._along_step_core)(ke, step)
    reasons = "\n".join(
        f"  {r.reason} @ {r.user_stack[-1].filename}:{r.user_stack[-1].lineno}"
        for r in explanation.break_reasons)
    assert explanation.graph_break_count == 0, (
        f"_along_step_core graph-broke {explanation.graph_break_count} "
        f"time(s):\n{reasons}")
    assert explanation.graph_count == 1, (
        f"expected a single fused graph, got {explanation.graph_count}")


def _poisoned_run(simulator, poison_at_step):
    """Run 3 events x `n_steps` with the continuous net emitting NaN on
    exactly one lane at exactly one step index."""
    proc = simulator.stepping_manager.processes[0]
    real_predict = proc.continuous_generator.predict
    calls = {"n": 0}

    def counted(x):
        calls["n"] += 1
        y = real_predict(x).clone()
        if calls["n"] == poison_at_step:
            y[0] = float("nan")
        return y

    proc.continuous_generator.predict = counted
    try:
        simulator.run(E_0=1.0e8, n_events=3, n_steps=20)
    finally:
        proc.continuous_generator.predict = real_predict


def test_a_mid_run_nan_is_still_caught_at_run_end():
    # Losslessness (the whole point of OR-ing into a latch rather than reading
    # per step): a single bad lane at an arbitrary interior step must still
    # surface at the end of the run, many steps later.
    from benchmark.track_simulator_config import build_track_simulator

    sim = build_track_simulator("phin_gan", device="cpu")

    # Seeded: the poisoned lane is lane 0, and with ~0.1% probability a
    # random draw puts lane 0's own (E, L) inside the deterministic /
    # full-deposit set, where the where-chain legitimately overwrites the
    # injected NaN and the guard correctly stays silent. That is a
    # pre-existing flake in the CONTROL, not in the mechanism, and a fixed
    # seed removes it without weakening the control.
    torch.manual_seed(0)
    with pytest.raises(RuntimeError, match="NaN e_cnt escaped"):
        _poisoned_run(sim, poison_at_step=7)


def test_the_latch_is_reset_between_runs():
    # A fresh run() must not inherit the previous run's flag, or one poisoned
    # run would red every later one.
    from benchmark.track_simulator_config import build_track_simulator

    sim = build_track_simulator("phin_gan", device="cpu")

    # Seeded: the poisoned lane is lane 0, and with ~0.1% probability a
    # random draw puts lane 0's own (E, L) inside the deterministic /
    # full-deposit set, where the where-chain legitimately overwrites the
    # injected NaN and the guard correctly stays silent. That is a
    # pre-existing flake in the CONTROL, not in the mechanism, and a fixed
    # seed removes it without weakening the control.
    torch.manual_seed(0)
    with pytest.raises(RuntimeError, match="NaN e_cnt escaped"):
        _poisoned_run(sim, poison_at_step=3)
    assert bool(sim.stepping_manager.processes[0]._nan_seen), \
        "latch should still be set immediately after the failing run"

    # Clean run on the SAME simulator instance: run() resets first.
    sim.run(E_0=1.0e8, n_events=3, n_steps=20)
    assert not bool(sim.stepping_manager.processes[0]._nan_seen)


def test_an_in_grid_point_still_scales_finitely():
    # Negative control: the guard must not fire on ordinary physical inputs, or
    # it would be trivially satisfied by an always-raising scaler.
    s = ContinuousYScaler.load()
    phase = torch.tensor([[1.0e8, 1.0e-5]], dtype=torch.float32)
    e_cnt = torch.tensor([[1.0e3]], dtype=torch.float32)
    out = s.scale(e_cnt, phase_space=phase)
    assert torch.isfinite(out).all(), out


# --- the escape sliver -------------------------------------------------
# The table's NaN region and the runtime's `is_deterministic` predicate are
# computed by DIFFERENT code: the builder labels a coarse cell deterministic
# from `ContinuousStragglingModel.average_loss` (explicit_physics) and then
# NaN-erodes the fine grid, while `_along_step_core` evaluates
# `_average_loss_fp64` (the built init tables) at the lane's exact (E, L) and
# compares against `min_energy_loss`. Nearest-cell rounding closes the last
# gap: a lane sitting in the upper half of the LAST NaN column gathers NaN
# while its own average_loss has already crossed 10 eV, so the where-chain
# leaves the NaN in place. `is_deterministic` therefore ORs in the lookup
# table's own NaN mask directly, rather than re-deriving the regime a
# second time, which is what closes this gap.


def _scan_escape_points(scaler, proc, max_hits: int = 8):
    """(E, L) points that gather NaN yet fail the runtime's absorb predicate.

    Vectorized over the 5000 energy rows: for each row take the LAST NaN
    column and place L just inside that column's upper rounding edge (the
    most escape-favourable L the cell can hold) and E at the row's lower cell
    edge (dE/dx falls with E above ~80 keV, so the lower edge maximises
    average_loss inside the cell). Returns a list of (E, L) float pairs.
    """
    from physics.g4h_ionisation.g4h_ionisation import G4CONSTS

    nan = torch.isnan(scaler.lookup_table[..., 0])
    kin_log = scaler.kin_table.to(torch.float64)
    step_log = scaler.step_table.to(torch.float64)
    dk = (kin_log[-1] - kin_log[0]) / (kin_log.numel() - 1)
    ds = (step_log[-1] - step_log[0]) / (step_log.numel() - 1)

    # last NaN column per row (every row's NaN region is a contiguous L-prefix)
    last = nan.to(torch.int64) * torch.arange(nan.shape[1]).unsqueeze(0)
    last = last.max(dim=1).values

    ke = (10.0 ** (kin_log - 0.5 * dk)).to(torch.float32)
    ln = (10.0 ** (step_log[last] + 0.499 * ds)).to(torch.float32)

    ki, si = scaler._compute_lookup_indices(ke, ln)
    gathered_nan = nan[ki, si]
    avg = proc.compute_average_loss(ke, ln)
    absorbed = (avg > ke) | (avg < G4CONSTS.min_energy_loss)
    escape = gathered_nan & ~absorbed
    idx = torch.nonzero(escape).flatten()[:max_hits]
    return [(float(ke[i]), float(ln[i])) for i in idx]


def test_the_table_nan_region_and_the_runtime_predicate_disagree_on_a_sliver():
    # NON-VACUITY control for the test below, and the geometry finding itself.
    # If a future table rebuild removes the disagreement entirely this reds --
    # which is the correct outcome: it means the pinned artifact changed shape
    # and the absorption contract needs re-measuring, not that the fix is
    # unnecessary.
    from benchmark.track_simulator_config import build_stepping_manager

    proc = build_stepping_manager("phin_gan", "cpu").processes[0]
    pts = _scan_escape_points(proc.continuous_generator.y_scaler, proc)
    assert pts, "no escape point found; the sliver's geometry has changed"


def test_a_lookup_nan_lane_is_absorbed_even_when_the_predicate_calls_it_stochastic():
    # THE FIX. A lane whose mean/std gather returned NaN must leave
    # along_step_do_it with a FINITE e_cnt, regardless of what
    # `average_loss < min_energy_loss` says about it: the table's NaN IS the
    # builder's declaration that the cell is deterministic, and average_loss
    # is exactly the physics G4 uses there.
    #
    # Many lanes at the same point on purpose: the escape is further filtered
    # by the zero-loss Bernoulli draw (zero_prob ~= 0.84 at these ~10 eV mean
    # losses), so a single lane reproduces it only ~16% of the time.
    from benchmark.track_simulator_config import build_stepping_manager

    proc = build_stepping_manager("phin_gan", "cpu").processes[0]
    pts = _scan_escape_points(proc.continuous_generator.y_scaler, proc)
    assert pts

    torch.manual_seed(0)
    for e_val, l_val in pts:
        ke = torch.full((4096,), e_val, dtype=torch.float32)
        step = torch.full((4096,), l_val, dtype=torch.float32)
        proc.reset_nan_guard()
        out = proc.along_step_do_it(kinetic_energy=ke, step_length=step)
        assert torch.isfinite(out["e_cnt"]).all(), (
            f"NaN e_cnt at the escape point E={e_val:.6e} eV, L={l_val:.6e} m "
            f"({int(torch.isnan(out['e_cnt']).sum())}/4096 lanes)")
        proc.check_nan_guard()
