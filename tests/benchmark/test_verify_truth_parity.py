"""verify_truth_parity's gate: its thresholds sit strictly above the
calibrated seed-to-seed null, it detects an injected continuous-loss shift,
it refuses a zero-row comparison, its report is well-formed, its two
frames resolve by column name to a fixed index map, and its default run
size is the gate's own calibration size, not any other gate's.

Run-size note: the gate's thresholds are calibrated at 500 events x 12000
steps (~4.9M rows) -- a KS threshold this tight (~1e-3) is meaningless at a
much smaller sample, where sampling scatter alone would exceed it (KS
critical value ~ 1.36*sqrt(2/n): a few thousand rows already gives ~0.02,
well past the threshold). An injected 5% shift and a zero-event run are
both detectable at any size, so those two tests use small explicit sizes
to stay fast; the null-margin, well-formed-report and column-map tests use
`_SMALL_N_EVENTS` / `_SMALL_N_STEPS` for the same reason, since none of
them depends on the calibration-scale threshold being tight.
"""
import pytest

import benchmark.verify_truth_parity as vtp

_SMALL_N_EVENTS = 30
_SMALL_N_STEPS = 3000


def test_thresholds_exceed_the_measured_null():
    # A threshold at or below the calibration null fails for RNG scatter
    # rather than physics. This pins the 2x convention.
    for feat, thr in vtp.KS_THRESHOLDS.items():
        null = vtp.MEASURED_NULL["ks"][feat]
        assert null > 0.0, f"{feat}: null is exactly 0 -- splits were not independent"
        assert thr > null, f"{feat}: threshold {thr} <= null {null}"
    for feat, thr in vtp.ATOM_MASS_THRESHOLDS.items():
        null = vtp.MEASURED_NULL["atom_mass"][feat]
        assert null > 0.0, f"{feat} atom: null is exactly 0"
        assert thr > null, f"{feat} atom: threshold {thr} <= null {null}"
    assert vtp.ENERGY_DISTANCE_THRESHOLD > vtp.MEASURED_NULL["energy_distance"]


def test_gate_detects_an_injected_physics_shift():
    # A gate that cannot fail is not a gate. Small explicit size: a 5%
    # continuous-loss shift dwarfs sampling noise even at a few thousand
    # rows, so this does not need calibration-scale agreement to be a valid
    # sensitivity check.
    result = vtp.run_truth_gate(n_events=_SMALL_N_EVENTS, n_steps=_SMALL_N_STEPS,
                                seed=1, _inject_e_cnt_scale=1.05)
    assert not result["passed"], "gate did not detect a 5% continuous-loss shift"


def test_gate_refuses_a_vacuous_comparison():
    with pytest.raises(ValueError, match="rows"):
        vtp.run_truth_gate(n_events=0)


def test_column_maps_are_the_frames_actual_layouts():
    # The two frames are NOT the same width and neither is the raw
    # StepFeaturesIdxes feature tensor -- resolution is by column name, per
    # side (see module docstring).
    from benchmark.track_simulator_config import build_track_simulator

    ts = build_track_simulator("phin_gan", device="cpu")
    ts.run(E_0=1e8, n_events=4, n_steps=8, save_steps_states=True)
    assert [str(c) for c in ts.steps_features_df.columns] == list(vtp.SIM_COLUMNS)
    assert vtp.SIM_IDX == {"L": 2, "THETA": 3, "E_CNT": 4, "E_SEC": 5}


def test_gate_produces_a_well_formed_report():
    import math

    result = vtp.run_truth_gate(n_events=_SMALL_N_EVENTS, n_steps=_SMALL_N_STEPS, seed=1)
    for feat, value in result["ks"].items():
        assert math.isfinite(value), f"ks[{feat}] is not finite: {value}"
        assert 0.0 <= value <= 1.0, f"ks[{feat}] out of range: {value}"
    for feat, value in result["atom_mass"].items():
        assert math.isfinite(value), f"atom_mass[{feat}] not finite: {value}"
        assert 0.0 <= value <= 1.0, f"atom_mass[{feat}] out of range: {value}"
    assert math.isfinite(result["energy_distance"])
    assert result["energy_distance"] >= 0.0
    assert isinstance(result["passed"], bool)
    assert result["n_rows"]["L"]["sim"] > 0 and result["n_rows"]["L"]["ref"] > 0


def test_default_run_size_is_gate_ts_own():
    assert (vtp.DEFAULT_N_EVENTS, vtp.DEFAULT_N_STEPS) == (500, 12000)


def test_cli_defaults_match_the_module_constants():
    args = vtp._build_parser().parse_args([])
    assert args.n_events == vtp.DEFAULT_N_EVENTS
    assert args.n_steps == vtp.DEFAULT_N_STEPS
    assert args.device == "cpu"
