"""The stage-0 instrument must discriminate in BOTH directions: report ~0 on
kinematically consistent angles and a large residual on inconsistent ones.
A validator that only ever reports agreement proves nothing."""
import math

import numpy as np
import pandas as pd

from benchmark.validate_theta_determinism import validate

M_PROTON_eV = 938272013.0
M_ELECTRON_eV = 510998.91


def _exact_theta(t0, t):
    """Independent fp64 scalar reference -- deliberately not the kernel."""
    p0 = math.sqrt(t0 * (t0 + 2 * M_PROTON_eV))
    pe = math.sqrt(t * (t + 2 * M_ELECTRON_eV))
    if pe == 0.0:
        return 0.0
    cos_e = min(1.0, t * (t0 + M_PROTON_eV + M_ELECTRON_eV) / (pe * p0))
    sin_e = math.sqrt(max(0.0, 1 - cos_e * cos_e))
    return math.atan2(pe * sin_e, p0 - pe * cos_e)


def _frame(n=2000, seed=3, consistent=True):
    rng = np.random.default_rng(seed)
    ke = rng.uniform(2.0e7, 1.0e8, n)            # pre-step KE [eV]
    cnt = rng.uniform(0.0, 1.0e4, n)             # continuous loss [eV]
    sec = rng.uniform(1.0e3, 2.0e5, n)           # delta-ray KE [eV]
    post = ke - cnt
    if consistent:
        theta = np.array([_exact_theta(a, b) for a, b in zip(post, sec)])
    else:
        theta = rng.uniform(1e-6, 1e-3, n)
    return pd.DataFrame({
        "KineticEnergy": ke,
        "ContinuousLoss": cnt,
        "SecondaryLoss": sec,
        "angleDiscrete": theta,
    })


def test_reports_near_zero_residual_on_consistent_angles():
    out = validate(_frame(consistent=True))
    assert out["n_rows"] == 2000
    # fp64 kernel against an fp64 scalar reference: agreement is at the
    # rounding floor, far below stage 0's ~1e-8 physics target.
    assert out["median_rel"] < 1e-12
    assert out["max_rel"] < 1e-9


def test_reports_a_large_residual_on_inconsistent_angles():
    # NEGATIVE CONTROL. Without this, a validator that always returns 0 --
    # e.g. one that accidentally compares the reconstruction to itself --
    # would pass the test above and prove nothing.
    out = validate(_frame(consistent=False))
    assert out["median_rel"] > 0.1


def test_uses_post_continuous_energy_not_pre_step_energy():
    # G4 produces the delta-ray AFTER the along-step continuous loss. Using
    # the pre-step energy mismatches angleDiscrete by ~1e-4 relative instead
    # of ~1e-8 -- four orders of magnitude, and exactly the kind of quiet
    # error that would look like a physics failure. Build a frame with a
    # LARGE continuous loss so the two choices are cleanly separable.
    df = _frame(consistent=True)
    df["ContinuousLoss"] = df["KineticEnergy"] * 0.5
    df["angleDiscrete"] = [
        _exact_theta(a, b)
        for a, b in zip(df["KineticEnergy"] - df["ContinuousLoss"], df["SecondaryLoss"])
    ]
    out = validate(df)
    assert out["median_rel"] < 1e-12, (
        "validator is not using the post-continuous energy")


def test_ignores_rows_with_no_secondary():
    # theta is identically 0 where no delta ray was produced, so the relative
    # residual is 0/0. Those rows must be dropped, not counted as agreement
    # (which would dilute the statistic) nor as NaN (which would poison it).
    df = _frame(consistent=True)
    df.loc[:499, "SecondaryLoss"] = 0.0
    df.loc[:499, "angleDiscrete"] = 0.0
    out = validate(df)
    assert out["n_rows"] == 1500
    assert not math.isnan(out["median_rel"])
