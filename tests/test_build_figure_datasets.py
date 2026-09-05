"""build_figure_datasets reproduces the paper's figure-input pkl sets from a
truth ROOT file, with one recorded divergence: secondary_Y_filtered is
(M, 1) e_sec-only (fig 10's __main__ and spec decision 1)."""
import json
import pickle

import numpy as np
import pytest

import build_figure_datasets as bfd
from tests.helpers_synthetic_g4 import write_synthetic_root


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("figpkl")
    root = tmp / "synthetic.root"
    write_synthetic_root(root, n_events=100, steps_per_event=10, seed=3)
    out = tmp / "out"
    manifest = bfd.build(root, out)
    return root, out, manifest


def _load(out, name):
    with open(out / name, "rb") as f:
        return pickle.load(f)


def test_step_length_pair_is_raw_float64_columns(built):
    root, out, _ = built
    x = _load(out, f"{root.stem}_step_length_X.pkl")
    y = _load(out, f"{root.stem}_step_length_Y.pkl")
    assert x.ndim == 2 and x.shape[1] == 1 and x.dtype == np.float64
    assert y.shape == x.shape and y.dtype == np.float64
    assert (x > 0).all() and (y > 0).all()


def test_continuous_filtered_drops_zero_and_full_loss(built):
    root, out, _ = built
    x = _load(out, f"{root.stem}_continuous_X_filtered.pkl")
    y = _load(out, f"{root.stem}_continuous_Y_filtered.pkl")
    assert x.ndim == 2 and x.shape[1] == 2 and x.dtype == np.float32
    assert y.shape == (x.shape[0], 1) and y.dtype == np.float32
    assert (y[:, 0] > 0).all()
    assert (y[:, 0] != x[:, 0]).all()


def test_secondary_filtered_is_one_column_positive(built):
    root, out, _ = built
    x = _load(out, f"{root.stem}_secondary_X_filtered.pkl")
    y = _load(out, f"{root.stem}_secondary_Y_filtered.pkl")
    assert x.ndim == 1 and x.dtype == np.float32
    assert y.ndim == 2 and y.shape == (x.shape[0], 1) and y.dtype == np.float32
    assert (y > 0).all()
    # delta-ray energies sit above the production threshold, keV scale
    assert y.min() > 1.0


def test_manifest_records_provenance_and_counts(built):
    root, out, manifest = built
    on_disk = json.loads(
        (out / f"{root.stem}_figure_datasets_manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["source_bytes"] == root.stat().st_size
    for key in ("n_rows_raw", "n_rows_continuous_filtered",
                "n_rows_secondary_filtered", "git_sha", "files"):
        assert key in manifest


def test_shuffle_is_deterministic(built, tmp_path):
    root, out, _ = built
    out2 = tmp_path / "out2"
    bfd.build(root, out2)
    a = _load(out, f"{root.stem}_continuous_X_filtered.pkl")
    b = _load(out2, f"{root.stem}_continuous_X_filtered.pkl")
    np.testing.assert_array_equal(a, b)
