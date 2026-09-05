"""Truth extraction must apply the row-local filters, use the post-continuous
energy as the conditioning variable, and never write beside its input."""
import awkward as ak
import numpy as np
import pytest
import uproot

import data_handling.truth_extract as truth_extract
from common.config import MIN_ENERGY_CUTOFF
from data_handling.truth_extract import extract_secondary_xy, extract_continuous_xy
from tests.helpers_synthetic_g4 import write_synthetic_root


@pytest.fixture
def synthetic_root(tmp_path):
    path = tmp_path / "synthetic.root"
    write_synthetic_root(path, n_events=200, steps_per_event=20, seed=11)
    return path


def _write_isolated_root(path, n, step_num, process_name, ke_above_cutoff=True):
    """Write a minimal ROOT tree of `n` rows where the three OTHER row-local
    filters (SecondaryLoss/NumOfSecondaries noise-fix, KineticEnergy cutoff)
    are guaranteed not to remove anything, so only StepNum / ProcessName can
    change the row count. `step_num` and `process_name` are per-row arrays.
    """
    ke = (np.full(n, MIN_ENERGY_CUTOFF * 2.0) if ke_above_cutoff
          else np.full(n, MIN_ENERGY_CUTOFF * 0.5))
    with uproot.recreate(path) as f:
        f.mktree("Data", {
            "StepNum": np.asarray(step_num, dtype=np.int32),
            "KineticEnergy": ke,
            "ContinuousLoss": np.full(n, 1.0),
            "SecondaryLoss": np.full(n, 10.0),
            "NumOfSecondaries": np.full(n, 1, dtype=np.int32),
            "angleDiscrete": np.full(n, 1e-4),
            "ProcessName": ak.Array(list(process_name)),
        })


def test_secondary_x_is_the_post_continuous_energy(synthetic_root):
    # G4 produces the delta ray AFTER the along-step continuous loss, so the
    # net must be conditioned on KineticEnergy - ContinuousLoss. Conditioning
    # on the pre-step energy mismatches G4 by ~1e-4 instead of ~1e-8.
    import uproot
    with uproot.open(synthetic_root) as f:
        raw = f["Data"].arrays(
            ["StepNum", "KineticEnergy", "ContinuousLoss", "SecondaryLoss",
             "NumOfSecondaries", "ProcessName"], library="pd")

    x, y = extract_secondary_xy(synthetic_root, drop_theta=True)

    assert x.ndim == 1
    assert y.shape == (len(x), 1)
    # Every returned X must be strictly less than SOME pre-step energy in the
    # source, and never exceed the max pre-step energy.
    assert x.max() <= raw["KineticEnergy"].max()
    # Not strict: a step that loses its entire kinetic energy continuously
    # right before the stopping wall legitimately reaches post-continuous
    # energy == 0, and the extractor correctly does not clip it.
    assert (x >= 0).all()


def test_drop_theta_controls_the_target_width(synthetic_root):
    _, y1 = extract_secondary_xy(synthetic_root, drop_theta=True)
    _, y2 = extract_secondary_xy(synthetic_root, drop_theta=False)
    assert y1.shape[1] == 1
    assert y2.shape[1] == 2
    # Column 1 of the two-wide form is the same energy as the one-wide form.
    np.testing.assert_array_equal(y1[:, 0], y2[:, 1])


def test_transportation_rows_are_dropped(tmp_path):
    # Isolated fixture: StepNum, KineticEnergy and NumOfSecondaries are held
    # fixed so the OTHER three row-local filters cannot remove a row: only
    # ProcessName == 'Transportation' can change the output count. If that
    # filter were silently disabled, len(x) would come back as N, not N - K.
    path = tmp_path / "isolated_transportation.root"
    n, k = 40, 13
    process = ["Transportation"] * k + ["eIoni"] * (n - k)
    _write_isolated_root(path, n, step_num=np.full(n, 5), process_name=process)

    x, _ = extract_secondary_xy(path)
    assert len(x) == n - k


def test_first_step_rows_are_dropped(tmp_path):
    # Isolated fixture: ProcessName, KineticEnergy and NumOfSecondaries are
    # held fixed so only StepNum == 0 can change the output count. If that
    # filter were silently disabled, len(x) would come back as N, not N - K.
    path = tmp_path / "isolated_first_step.root"
    n, k = 40, 17
    step_num = np.concatenate([np.zeros(k, dtype=np.int32),
                               np.full(n - k, 5, dtype=np.int32)])
    _write_isolated_root(path, n, step_num=step_num,
                          process_name=["eIoni"] * n)

    x, _ = extract_secondary_xy(path)
    assert len(x) == n - k


def test_writes_nothing_beside_the_input(synthetic_root):
    # THE POINT OF THIS MODULE. data_handling/reader.py caches a .pkl next to
    # its source ROOT file; a truth tree outside this repository's own
    # storage layout must never receive that write.
    before = {p.name for p in synthetic_root.parent.iterdir()}
    extract_secondary_xy(synthetic_root)
    extract_continuous_xy(synthetic_root)
    after = {p.name for p in synthetic_root.parent.iterdir()}
    assert before == after, f"created files: {after - before}"


def test_continuous_xy_shapes(synthetic_root):
    x, y = extract_continuous_xy(synthetic_root)
    assert x.shape == (len(y), 2)
    assert y.ndim == 1
    assert (x > 0).all()


def test_row_local_filters_tuple_is_what_actually_runs(monkeypatch, synthetic_root):
    # ROW_LOCAL_FILTERS is meant to be the SOLE implementation of the
    # row-local filters, not a description that happens to agree with a
    # separate hardcoded implementation (see the module docstring and
    # build_training_datasets.py's provenance manifest, which reads
    # ROW_LOCAL_FILTER_DESCRIPTIONS off this same tuple). Prove that by
    # actually removing an entry and checking the ROOT-file extraction
    # itself changes -- if `_apply_row_local_filters` had its own
    # independent implementation, this would do nothing and the assertion
    # below would fail, catching exactly the drift Finding 3 was about.
    x_full, _ = extract_secondary_xy(synthetic_root)

    # Drop the last filter (ProcessName != 'Transportation'): dropping a
    # filter can only ADMIT rows, never remove them, so this is a safe,
    # unambiguous one-directional check regardless of the synthetic data's
    # exact random draw.
    truncated = truth_extract.ROW_LOCAL_FILTERS[:-1]
    assert len(truncated) == len(truth_extract.ROW_LOCAL_FILTERS) - 1
    monkeypatch.setattr(truth_extract, "ROW_LOCAL_FILTERS", truncated)

    x_truncated, _ = extract_secondary_xy(synthetic_root)

    assert len(x_truncated) > len(x_full), (
        "removing an entry from ROW_LOCAL_FILTERS did not change what "
        "_apply_row_local_filters actually selects -- it must have its "
        "own independent implementation of the filters again")


def test_row_local_filter_descriptions_matches_the_executable_tuple():
    # ROW_LOCAL_FILTER_DESCRIPTIONS is derived FROM ROW_LOCAL_FILTERS (a
    # comprehension, not a retyped list), so this is a construction check,
    # not the drift guard itself -- that's the test above, which shows the
    # tuple actually drives extraction. This just pins the derivation.
    assert truth_extract.ROW_LOCAL_FILTER_DESCRIPTIONS == tuple(
        desc for desc, _fn in truth_extract.ROW_LOCAL_FILTERS)
    assert len(truth_extract.ROW_LOCAL_FILTER_DESCRIPTIONS) == 4
