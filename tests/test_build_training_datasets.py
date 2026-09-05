"""build_training_datasets writes the per-stage X/Y arrays plus a provenance
manifest, so a claim about what data a training run used is checkable
against a record rather than taken on faith."""
import json

import numpy as np
import pytest

import build_training_datasets as brd
from data_handling.truth_extract import (
    ROW_LOCAL_FILTER_DESCRIPTIONS,
    extract_secondary_xy,
)
from tests.helpers_synthetic_g4 import write_synthetic_root


def test_manifest_records_provenance(tmp_path):
    # A claim about what data a training run used is unfalsifiable without
    # a record of WHAT it built on. The manifest is what makes the
    # datasets auditable.
    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=100, steps_per_event=10, seed=3)
    out = tmp_path / "out"
    brd.build(root, out)

    manifest = json.loads((out / "manifest.json").read_text())
    for key in ("source_root", "source_bytes", "git_sha", "filters",
                "n_rows_secondary", "n_rows_continuous"):
        assert key in manifest, f"manifest is missing {key}"
    assert manifest["n_rows_secondary"] > 0


def test_manifest_source_bytes_matches_the_real_file_size(tmp_path):
    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=100, steps_per_event=10, seed=3)
    out = tmp_path / "out"
    manifest = brd.build(root, out)

    assert manifest["source_bytes"] == root.stat().st_size
    assert manifest["source_root"] == str(root.resolve())


def test_manifest_filters_come_from_truth_extracts_own_constant(tmp_path):
    # Finding 3: a hand-copied prose list in this module could silently
    # drift from data_handling.truth_extract's actual filters. Guard against
    # a future regression that re-introduces a second hardcoded copy here.
    # ROW_LOCAL_FILTER_DESCRIPTIONS is itself derived from ROW_LOCAL_FILTERS
    # -- the executable (description, callable) list _apply_row_local_
    # filters actually runs (see tests/data_handling/test_truth_extract.py
    # ::test_row_local_filters_tuple_is_what_actually_runs for the
    # non-tautological half: proof the tuple drives real behaviour, not
    # just this comparison).
    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=50, steps_per_event=10, seed=3)
    out = tmp_path / "out"
    manifest = brd.build(root, out)

    assert manifest["filters"] == list(ROW_LOCAL_FILTER_DESCRIPTIONS)


def test_build_writes_both_secondary_targets_by_default(tmp_path):
    # secondary_generator.datamodule.DataModule can load either a 2-column
    # [theta, e_sec] target or a 1-column e_sec-only target, depending on
    # its own drop_theta setting. build() must not choose between them --
    # by default it writes both, to distinct filenames, and records which
    # ones in the manifest.
    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=100, steps_per_event=10, seed=3)
    out = tmp_path / "out"
    manifest = brd.build(root, out)

    assert manifest["secondary_target"] == "both"
    assert set(manifest["arrays"]) == {
        "secondary_X.npy", "secondary_Y.npy", "secondary_Y_drop_theta.npy",
        "continuous_X.npy", "continuous_Y.npy",
    }

    x_sec = np.load(out / "secondary_X.npy")
    y_sec_2col = np.load(out / "secondary_Y.npy")
    y_sec_1col = np.load(out / "secondary_Y_drop_theta.npy")
    x_cnt = np.load(out / "continuous_X.npy")
    y_cnt = np.load(out / "continuous_Y.npy")

    n = manifest["n_rows_secondary"]
    assert x_sec.shape == (n,)
    assert y_sec_2col.shape == (n, 2)      # [theta, e_sec]
    assert y_sec_1col.shape == (n, 1)      # e_sec only
    assert manifest["arrays"]["secondary_Y.npy"] == [n, 2]
    assert manifest["arrays"]["secondary_Y_drop_theta.npy"] == [n, 1]

    # The 1-column form must be the SecondaryLoss column, verified against an
    # INDEPENDENT oracle: extract_secondary_xy(drop_theta=True), which selects
    # the column by NAME from the ROOT file. This deliberately does NOT index
    # the 2-column array with SecondaryFeats.E_SEC_IDX: the 2-column layout is
    # [angleDiscrete, SecondaryLoss] (the extractor's fixed order), while
    # SecondaryFeats is a single-column enum with E_SEC_IDX = 0, so indexing
    # the 2-column array with that enum value would silently select theta
    # instead of the secondary energy.
    _, y_oracle = extract_secondary_xy(root, drop_theta=True)
    np.testing.assert_array_equal(y_sec_1col, y_oracle)
    # And the physical tripwire that would have caught it even without the
    # oracle: a delta-ray energy is bounded below by the production threshold
    # (keV scale); theta is ~1e-4 rad. Any nonzero value below 1 eV means the
    # wrong column or the wrong units.
    nz = y_sec_1col[y_sec_1col > 0]
    assert nz.size and nz.min() > 1.0, "drop_theta target is not an energy in eV"

    assert x_cnt.shape[0] == y_cnt.shape[0] == manifest["n_rows_continuous"]
    assert x_cnt.shape[1] == 2


@pytest.mark.parametrize("target,expected_files,forbidden_files", [
    (brd.SECONDARY_TARGET_WITH_THETA, {"secondary_Y.npy"}, {"secondary_Y_drop_theta.npy"}),
    (brd.SECONDARY_TARGET_DROP_THETA, {"secondary_Y_drop_theta.npy"}, {"secondary_Y.npy"}),
])
def test_secondary_target_can_be_narrowed_explicitly(
        tmp_path, target, expected_files, forbidden_files):
    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=50, steps_per_event=10, seed=7)
    out = tmp_path / "out"
    manifest = brd.build(root, out, secondary_target=target)

    assert manifest["secondary_target"] == target
    for name in expected_files:
        assert (out / name).exists()
        assert name in manifest["arrays"]
    for name in forbidden_files:
        assert not (out / name).exists()
        assert name not in manifest["arrays"]


def test_build_rejects_an_unknown_secondary_target(tmp_path):
    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=50, steps_per_event=10, seed=7)
    with pytest.raises(ValueError):
        brd.build(root, tmp_path / "out", secondary_target="bogus")


def test_datamodule_setup_succeeds_against_the_built_secondary_target(tmp_path):
    # The real integration point. A shape mismatch between what build()
    # writes and what DataModule expects -- constructed the way
    # secondary_generator/train.py::train_gan() constructs it, i.e. at the
    # default drop_theta=True -- would raise IndexError on the first line
    # of setup() (_filter_samples indexes Y_real[:, SecondaryFeats.E_SEC_IDX],
    # column 0). Exercise that path for real, not just np.load + shape
    # arithmetic against a manifest the same run produced.
    from physics.g4h_ionisation.generators.secondary_generator.scaler import (
        SecondaryXScaler, SecondaryYScaler,
    )
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.datamodule import DataModule

    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=300, steps_per_event=20, seed=9)
    out = tmp_path / "out"
    brd.build(root, out)  # default: writes both targets

    dm = DataModule(
        SecondaryXScaler, SecondaryYScaler,
        data_dir=out, scaler_dir=tmp_path / "scalers",
    )  # drop_theta left at its default (True), exactly as train_gan() does
    dm.setup()

    assert dm.X is not None and dm.Y is not None
    assert dm.Y.shape[1] == 1
    assert dm.scaler_x is not None and dm.scaler_y is not None
    # setup() is memoized; a second call must be a no-op, not refit its
    # scalers again.
    scaler_x_before = dm.scaler_x
    dm.setup()
    assert dm.scaler_x is scaler_x_before


def test_a_2column_target_is_silently_accepted_and_is_NOT_caught_here(tmp_path):
    """A width mismatch does NOT raise in setup(). Pinning that it doesn't.

    Feed the 2-column [theta, e_sec] array to the 1-column default wiring
    and everything "works" --

      * `_filter_samples` indexes `Y_real[:, SecondaryFeats.E_SEC_IDX]`, which
        is column 0, and column 0 of the 2-column array is THETA;
      * `_prepare_data_for_training` reshapes to
        `(-1, batch, N_SECONDARY_FEATURES)` = `(..., 1)`, which succeeds
        because the element count still divides.

    So the run would train on scattering angles labelled as energies, for
    ~76 h, and nothing in the data path would object. The ONLY thing that
    catches it is `submit_secondary_job.py`'s preflight, which reads the .npy
    header and compares against `N_SECONDARY_FEATURES` before submitting --
    which is why that check exists and must not be removed.

    Pinned as an accepted-and-documented hazard rather than fixed here:
    hardening `setup()` itself is a change to the training path, and the
    submitter's own preflight check is the layer responsible for catching
    this instead.
    """
    from physics.g4h_ionisation.generators.secondary_generator.scaler import (
        SecondaryXScaler, SecondaryYScaler,
    )
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.datamodule import DataModule

    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=300, steps_per_event=20, seed=9)
    out = tmp_path / "out"
    brd.build(root, out)
    (out / "secondary_Y.npy").rename(out / "secondary_Y_drop_theta.npy")

    dm = DataModule(
        SecondaryXScaler, SecondaryYScaler,
        data_dir=out, scaler_dir=tmp_path / "scalers",
    )
    dm.setup()          # does not raise -- that is the point
    assert dm.Y is not None

def test_build_does_not_write_beside_the_source_root(tmp_path):
    # truth_extract streams; nothing should ever appear next to the source
    # ROOT file (no .pkl cache, no stray output).
    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=50, steps_per_event=10, seed=1)
    before = {p.name for p in tmp_path.iterdir()}
    out = tmp_path / "out"
    brd.build(root, out)
    after = {p.name for p in tmp_path.iterdir() if p.name != "out"}
    assert after == before


def test_build_is_idempotent_and_deterministic(tmp_path):
    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=50, steps_per_event=10, seed=5)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    m1 = brd.build(root, out1)
    m2 = brd.build(root, out2)

    assert m1["n_rows_secondary"] == m2["n_rows_secondary"]
    assert m1["n_rows_continuous"] == m2["n_rows_continuous"]
    x1 = np.load(out1 / "secondary_X.npy")
    x2 = np.load(out2 / "secondary_X.npy")
    np.testing.assert_array_equal(x1, x2)


def test_root_defaults_to_the_configured_g4_truth_path():
    # The builder's default source is the paths constant, not a copied
    # string, so a bare invocation always reads the currently configured
    # G4 truth file and a beam/material switch retargets it automatically.
    from common.paths import PATH_TO_GEANT4_DATA
    args = brd._build_parser().parse_args([])
    assert args.root == PATH_TO_GEANT4_DATA
