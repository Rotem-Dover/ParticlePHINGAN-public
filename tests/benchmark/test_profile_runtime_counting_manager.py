"""Regression test for the CountingManager NaN-guard passthrough.

TrackSimulator.run() calls self.stepping_manager.reset_nan_guard() before its
step loop and self.stepping_manager.check_nan_guard() after it (the NaN
accumulator contract described in CLAUDE.md's "The continuous mean/std
lookup"). profile_runtime.py's CountingManager wraps the real
SteppingManager and is swapped in as simulator.stepping_manager for every
warm-up/sweep run; before this test was written it did not proxy either
method, so every profile_runtime invocation crashed with
`AttributeError: 'CountingManager' object has no attribute 'reset_nan_guard'`
the first time run() was called -- CPU included, so this test needs no GPU.
"""
import benchmark.runtime_profiling.profile_runtime as pr
from benchmark.runtime_profiling.simulator_builders import build_eager_simulator


def test_counting_manager_proxies_nan_guard_and_run_completes():
    simulator, sm = build_eager_simulator("phin_gan", device="cpu")
    counter = pr.CountingManager(sm)
    simulator.stepping_manager = counter

    # Pre-fix this raises AttributeError inside run() at reset_nan_guard();
    # tiny sizes keep this fast on CPU.
    simulator.run(E_0=1e8, n_events=4, n_steps=8, save_steps_states=False)

    # The proxy's own bookkeeping should still work post-fix.
    assert len(counter.batch_sizes) > 0


def test_counting_manager_has_both_nan_guard_methods():
    """Structural pin: reset_nan_guard/check_nan_guard must stay proxied even
    if CountingManager grows other passthroughs later."""
    assert hasattr(pr.CountingManager, "reset_nan_guard")
    assert hasattr(pr.CountingManager, "check_nan_guard")
