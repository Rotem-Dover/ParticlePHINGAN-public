import inspect

import numpy as np
import pytest

import benchmark.runtime_profiling.verify_physics_gate as vpg


def test_module_does_not_import_unrelated_stages():
    """verify_physics_gate drives only the length/continuous/secondary
    stages of the built simulator; it must not reach into multiple
    scattering, the training package, or the continuous stage's offline
    Monte Carlo sampler."""
    src = inspect.getsource(vpg)
    for dead in ("g4h_multiple_scattering", "train_networks",
                 "continuous_generator.physics_mc"):
        assert dead not in src, f"{dead} must not be imported here"


def test_gate_b_is_eager_anchored():
    """Only the optimized-minus-eager delta gates; absolutes are reported,
    not gated -- the eager arm on this device/run size is the reference,
    not any fixed reference value."""
    assert "eager-anchored" in (vpg.run_gate_b.__doc__ or "")


def test_gate_b_reports_both_statistics_and_the_device():
    r = vpg.run_gate_b(device="cpu", n_events=2, n_steps=200, seed=1)
    assert set(("e2e", "pull", "passed", "device", "thresholds")) <= set(r)
    assert r["device"] == "cpu"


def test_gate_b_on_cpu_is_identically_zero():
    """On cpu the compile and fusion branches are both dead, so optimized and
    eager are the same program: every delta must be exactly zero. A nonzero
    here means the two builders diverged for a reason that is not
    optimization."""
    r = vpg.run_gate_b(device="cpu", n_events=2, n_steps=200, seed=1)
    assert r["e2e"]["max_ks"] == 0.0
    assert r["passed"] is True


def test_thresholds_are_not_inherited_from_the_electron_gate():
    src = inspect.getsource(vpg)
    assert "electron" not in src.lower() or "not inherited" in src.lower()


def test_calibration_helper_exists():
    assert callable(vpg.gate_b_null)
    assert "seeds" in inspect.signature(vpg.gate_b_null).parameters


# ----- positive controls -----------------------------------------------------
# test_gate_b_on_cpu_is_identically_zero cannot tell a working gate from one
# hard-wired to zero. These show both statistics are CAPABLE of a nonzero
# answer, which is what makes the zero above falsifiable.
def _synthetic(n: int = 2000, *, scale: float = 1.0, shift: float = 0.0,
               seed: int = 0) -> np.ndarray:
    """(n, 4) array in GATE_B_FEATURES order: L, THETA, E_CNT, E_SEC.
    THETA/E_SEC carry a zero atom on the same rows, as the real frame does,
    with a continuum comfortably above the _MIN_CONTINUUM guard."""
    rng = np.random.default_rng(seed)
    has_sec = rng.random(n) < 0.8
    theta = np.where(has_sec, rng.random(n) * 1e-3 * scale + shift, 0.0)
    e_sec = np.where(has_sec, rng.random(n) * 1e4 * scale + shift, 0.0)
    L = rng.random(n) * 1e-5 * scale + shift
    e_cnt = rng.random(n) * 1e4 * scale + shift
    return np.stack([L, theta, e_cnt, e_sec], axis=1)


def test_e2e_statistic_is_zero_on_identical_inputs():
    a = _synthetic(seed=1)
    assert vpg._e2e_statistic(a, a.copy())["max_ks"] == 0.0


def test_e2e_statistic_is_positive_on_different_inputs():
    a, b = _synthetic(seed=1), _synthetic(seed=1, scale=2.0)
    assert vpg._e2e_statistic(a, b)["max_ks"] > 0.0


def test_pull_statistic_is_zero_on_identical_inputs():
    a = _synthetic(seed=2)
    assert vpg._pull_statistic(a, a.copy())["max_abs_mean_shift"] == 0.0


def test_pull_statistic_is_positive_on_a_shifted_mean():
    a, b = _synthetic(seed=2), _synthetic(seed=2, shift=1e-3)
    assert vpg._pull_statistic(a, b)["max_abs_mean_shift"] > 0.0


# ----- guards ----------------------------------------------------------------
def test_e2e_statistic_refuses_a_too_small_continuum():
    """An empty/near-empty E_SEC continuum must be a loud failure, not a 0.0
    that would read as perfect agreement between the two arms."""
    a = _synthetic(seed=3)
    b = a.copy()
    b[:, vpg.GATE_B_FEATURES.index("E_SEC")] = 0.0
    with pytest.raises(RuntimeError, match="nonzero entries"):
        vpg._e2e_statistic(a, b)


def test_vacuity_refusal_is_cuda_only():
    """On cpu both optimization branches are legitimately dead, so identical
    kernel modes are expected and must not be refused."""
    modes = {"continuous": {"fused": False}, "secondary": {"fused": False}}
    assert vpg._vacuity_reason("cpu", modes, modes) is None


def test_unfused_cuda_optimized_arm_is_refused_not_passed():
    """try_enable_fused_forward only log.warns when it refuses, so a cuda run
    with nothing fused compares eager against eager."""
    modes = {"continuous": {"fused": False}, "secondary": {"fused": False}}
    reason = vpg._vacuity_reason("cuda", modes, modes)
    assert reason is not None and "vacuous" in reason


def test_identical_cuda_kernel_modes_are_refused():
    modes = {"continuous": {"fused": True}, "secondary": {"fused": True}}
    assert vpg._vacuity_reason("cuda", modes, dict(modes)) is not None


def test_an_indexed_cuda_device_string_is_still_checked():
    """`cuda:0` is accepted by torch and by `.to()`. An exact-string `==
    "cuda"` test would let such a caller past the vacuity check entirely, i.e.
    report `passed: True` on an eager-vs-eager run."""
    modes = {"continuous": {"fused": False}, "secondary": {"fused": False}}
    reason = vpg._vacuity_reason("cuda:0", modes, modes)
    assert reason is not None and "vacuous" in reason


def test_an_indexed_cuda_device_string_still_binds_the_run_size_pin():
    with pytest.raises(ValueError, match="calibration working point"):
        vpg.run_gate_b(device="cuda:1", n_events=201, n_steps=12_000, seed=1)


def test_a_genuinely_fused_cuda_run_is_not_refused():
    opt = {"continuous": {"fused": True}, "secondary": {"fused": True}}
    eager = {"continuous": {"fused": False}, "secondary": {"fused": False}}
    assert vpg._vacuity_reason("cuda", opt, eager) is None


# ----- the calibration contract ---------------------------------------------
# GATE_B_THRESHOLDS are exactly 2x a GPU eager-vs-eager null measured on an
# RTX 6000 Ada, seeds 1-4, 200 events x 12000 steps
# (measurements/gate_b/report.json) -- the null is pasted into the module
# comment above GATE_B_THRESHOLDS so the calibration stays auditable
# against that measurement.
_MEASURED_NULL = {"e2e_max_ks": 0.0018312158233761977,
                  "pull_max_abs_mean_shift": 2.8352793270005705,
                  "atom_mass_delta": 0.0007535443505701789}


def test_every_gated_statistic_has_a_threshold_key():
    """The three keys are the gate's contract. A missing one does not mean
    'default'; it means a gate silently vanished."""
    assert set(vpg.GATE_B_THRESHOLDS) == set(vpg.GATE_B_STATISTICS)
    assert set(vpg.GATE_B_STATISTICS) == {
        "e2e_max_ks", "pull_max_abs_mean_shift", "atom_mass_delta"}


def test_thresholds_are_exactly_twice_the_pasted_null():
    """Pin the constants to the audit block they claim to come from, so a
    hand-edited threshold and its stated provenance cannot drift apart. A
    widened threshold with an unchanged comment is the failure mode this
    catches, and it is the one the project explicitly forbids."""
    for key, null in _MEASURED_NULL.items():
        assert vpg.GATE_B_THRESHOLDS[key] == 2 * null, key


def test_calibration_working_point_is_the_one_the_null_was_measured_at():
    assert vpg.GATE_B_CALIBRATION_RUN_SIZE == {"n_events": 200, "n_steps": 12_000}
    assert vpg.GATE_B_CALIBRATION_DEVICE == "cuda"


# ----- a malformed thresholds dict must be loud, never quietly ungating ------
def test_thresholds_missing_a_key_are_refused():
    """Every pre-Task-15 caller passes a two-key dict. Read with `.get` that
    would drop the atom-mass gate -- i.e. stop gating the secondary-generation
    probability -- while still reporting `passed: true`."""
    with pytest.raises(ValueError, match="atom_mass_delta"):
        vpg._validate_thresholds({"e2e_max_ks": 1.0,
                                  "pull_max_abs_mean_shift": 1.0})


def test_thresholds_with_an_unknown_key_are_refused():
    with pytest.raises(ValueError, match="unknown"):
        vpg._validate_thresholds(dict(vpg.GATE_B_THRESHOLDS, e2e_maxks=1.0))


def test_explicit_none_is_accepted_as_report_do_not_gate():
    thr = {k: None for k in vpg.GATE_B_STATISTICS}
    assert vpg._validate_thresholds(thr) is thr


def test_run_gate_b_refuses_a_malformed_thresholds_dict():
    with pytest.raises(ValueError, match="atom_mass_delta"):
        vpg.run_gate_b(device="cpu", n_events=2, n_steps=200, seed=1,
                       thresholds={"e2e_max_ks": 1.0,
                                   "pull_max_abs_mean_shift": 1.0})


# ----- the working point is n-dependent and therefore pinned -----------------
def test_gating_at_the_wrong_run_size_is_refused_on_the_calibration_device():
    """The pull statistic is n-dependent: a systematic fp bias grows ~sqrt(n)
    in pull units while the eager-vs-eager null does not, so the calibrated
    thresholds do not transfer to another run size."""
    with pytest.raises(ValueError, match="calibration working point"):
        vpg.run_gate_b(device="cuda", n_events=201, n_steps=12_000, seed=1)


def test_the_run_size_pin_does_not_bind_on_cpu():
    """On cpu both optimization branches are dead, so the two builders are the
    same program and every delta is exactly 0.0 regardless of run size. Making
    the pin bind there would break the gate's own plumbing check."""
    r = vpg.run_gate_b(device="cpu", n_events=2, n_steps=200, seed=1)
    assert r["passed"] is True


def test_gate_b_null_reports_whether_it_matches_the_calibration_point(monkeypatch):
    """`gate_b_null` REPORTS the match rather than asserting it, on purpose:
    asserting would make recalibrating at a new working point impossible
    without editing the constant first. The caller decides.

    Behavioural in BOTH directions: a source-string match would still pass if
    the flag were hard-wired True. The working point is moved by
    monkeypatching the calibration constant rather than by running the real
    200x12000 point, which is a GPU-scale run.
    """
    sig = inspect.signature(vpg.gate_b_null)
    assert {"n_events", "n_steps"} <= set(sig.parameters)

    small = {"n_events": 2, "n_steps": 200}
    monkeypatch.setattr(vpg, "GATE_B_CALIBRATION_RUN_SIZE", dict(small))
    r = vpg.gate_b_null(device="cpu", n_events=2, n_steps=200, seeds=(1, 2))
    assert r["calibration_run_size"] == small
    assert r["run_size_matches_calibration"] is True

    monkeypatch.setattr(vpg, "GATE_B_CALIBRATION_RUN_SIZE",
                        {"n_events": 3, "n_steps": 200})
    r = vpg.gate_b_null(device="cpu", n_events=2, n_steps=200, seeds=(1, 2))
    assert r["run_size_matches_calibration"] is False


# ----- the atom-mass gate (Task 15) -----------------------------------------
def test_max_atom_mass_delta_is_the_max_of_the_per_feature_deltas():
    a, b = _synthetic(seed=5), _synthetic(seed=6)
    e2e = vpg._e2e_statistic(a, b)
    assert e2e["max_atom_mass_delta"] == max(e2e["atom_mass_delta"].values())


def _atom_shifted(n: int = 20_000, *, frac: float, seed: int = 7) -> np.ndarray:
    """Same continuum SHAPE, different zero-atom mass.

    The nonzero values are a uniform grid over a fixed range rather than a
    random sample, so the two arms' continua agree to ~1/k under KS however
    many rows carry a secondary; L and E_CNT are identical between arms. Only
    the FRACTION of rows carrying a secondary changes. That isolates the
    statistic under test: a secondary-generation-probability drift with an
    unchanged secondary spectrum, which is the shape a fused kernel's fp drift
    on the generation probability would have.
    """
    rng = np.random.default_rng(seed)
    L = rng.random(n) * 1e-5
    e_cnt = rng.random(n) * 1e4
    k = int(n * frac)
    theta, e_sec = np.zeros(n), np.zeros(n)
    theta[:k] = np.linspace(1e-9, 1e-3, k)
    e_sec[:k] = np.linspace(1.0, 1e4, k)
    return np.stack([L, theta, e_cnt, e_sec], axis=1)


def test_an_atom_mass_only_shift_fails_the_gate():
    """POSITIVE CONTROL for the new gate: at this working point the continuum
    KS and the pull both stay well inside their thresholds while the zero-atom
    mass moves past its own, so before Task 15 this run passed. The atom mass
    IS the secondary-generation probability -- the observable a fused kernel's
    fp drift would most plausibly move.

    The mass shift is deliberately small (0.003). A large one WOULD also move
    the pull, since the pull is taken over all rows including the zeros; the
    atom gate earns its place on small shifts, where the other two statistics
    are still clean.
    """
    a = _atom_shifted(frac=0.800)
    b = _atom_shifted(frac=0.803)
    thr = dict(vpg.GATE_B_THRESHOLDS)

    e2e, pull = vpg._e2e_statistic(a, b), vpg._pull_statistic(a, b)
    # The shift is invisible to the other two statistics ...
    assert e2e["max_ks"] <= thr["e2e_max_ks"]
    assert pull["max_abs_mean_shift"] <= thr["pull_max_abs_mean_shift"]
    # ... and unmistakable to this one.
    assert e2e["max_atom_mass_delta"] > thr["atom_mass_delta"]

    passed = (e2e["max_ks"] <= thr["e2e_max_ks"]
              and pull["max_abs_mean_shift"] <= thr["pull_max_abs_mean_shift"]
              and e2e["max_atom_mass_delta"] <= thr["atom_mass_delta"])
    assert passed is False


def _atom_only_breach(delta: float):
    """A stand-in `_e2e_statistic` whose ONLY out-of-threshold statistic is the
    zero-atom mass. `max_ks` is pinned at 0.0 and the pull is left to the real
    (identically-zero, on cpu) computation, so the boolean this drives can only
    come from the atom-mass clause."""
    def _stub(got, ref):
        return {"max_ks": 0.0,
                "ks": {f: 0.0 for f in vpg.GATE_B_FEATURES},
                "atom_mass_delta": {f: delta for f in vpg._ATOM_FEATURES},
                "max_atom_mass_delta": delta}
    return _stub


def test_run_gate_b_passed_consumes_the_atom_mass_threshold(monkeypatch):
    """REGRESSION PIN on the consumer, not the statistic.

    `test_an_atom_mass_only_shift_fails_the_gate` recomputes `passed` locally,
    so deleting `run_gate_b`'s `atom_mass_delta` clause leaves it green. This
    calls `run_gate_b` itself and asserts its OWN `passed` flips with the
    atom-mass statistic alone. Mutating that clause to a no-op must turn this
    red -- that is the whole point of the test.
    """
    thr = vpg.GATE_B_THRESHOLDS["atom_mass_delta"]
    kw = dict(device="cpu", n_events=2, n_steps=200, seed=1,
              thresholds=dict(vpg.GATE_B_THRESHOLDS))

    monkeypatch.setattr(vpg, "_e2e_statistic", _atom_only_breach(10 * thr))
    breached = vpg.run_gate_b(**kw)
    assert breached["vacuous"] is None, "must fail on the statistic, not vacuity"
    assert breached["e2e"]["max_ks"] <= vpg.GATE_B_THRESHOLDS["e2e_max_ks"]
    assert breached["pull"]["max_abs_mean_shift"] <= vpg.GATE_B_THRESHOLDS[
        "pull_max_abs_mean_shift"]
    assert breached["passed"] is False

    # Same run, same stub, atom mass inside its threshold -> green. Without
    # this arm a `passed = False` constant would satisfy the assertion above.
    monkeypatch.setattr(vpg, "_e2e_statistic", _atom_only_breach(0.5 * thr))
    assert vpg.run_gate_b(**kw)["passed"] is True


def test_run_gate_b_atom_mass_threshold_of_none_reports_instead_of_gating(
        monkeypatch):
    """The `None` == report-do-not-gate contract, also read through
    `run_gate_b`'s own `passed`."""
    thr = dict(vpg.GATE_B_THRESHOLDS, atom_mass_delta=None)
    monkeypatch.setattr(
        vpg, "_e2e_statistic",
        _atom_only_breach(10 * vpg.GATE_B_THRESHOLDS["atom_mass_delta"]))
    r = vpg.run_gate_b(device="cpu", n_events=2, n_steps=200, seed=1,
                       thresholds=thr)
    assert r["passed"] is True
    assert r["e2e"]["max_atom_mass_delta"] > 0.0


def test_gate_b_reports_both_arms_kernel_modes():
    r = vpg.run_gate_b(device="cpu", n_events=2, n_steps=200, seed=1)
    assert set(r["kernel_mode"]) == {"optimized", "eager"}
    for arm in r["kernel_mode"].values():
        assert set(arm) == {"length", "continuous", "secondary"}
        assert all("fused" in info for info in arm.values())
