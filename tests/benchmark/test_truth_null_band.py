"""The truth-vs-truth pull band: pairing, summary statistics and the text
the figure prints. Pure-function tests on synthetic histograms; the ten-file
study itself is a measurement, recorded under measurements/pull_truth_null/."""
import json

import numpy as np
import pytest

from benchmark.system_vs_g4.pull_analysis import truth_null_band as tnb


def _gauss_hists(rng, n_events=2000, grid=40, p_touch=0.05):
    """Per-event energy depositions on a grid, in the sparse regime the pull
    assumes: each event touches a bin with probability `p_touch`, depositing
    an exponential amount. The library's pull denominator is the sum of
    squared per-event deposits, which equals the variance of the bin total
    only when few events touch each bin (as with real tracks)."""
    touched = rng.random((n_events, grid * grid)) < p_touch
    per_event = np.where(touched, rng.exponential(1.0, size=touched.shape), 0.0)
    return tnb.HistStats(
        sum=per_event.sum(axis=0).reshape(grid, grid),
        sum_sq=(per_event ** 2).sum(axis=0).reshape(grid, grid),
        counts=touched.sum(axis=0).reshape(grid, grid).astype(np.int64),
    )


def test_pair_pull_of_two_iid_draws_is_standard_normal():
    rng = np.random.default_rng(0)
    h1, h2 = _gauss_hists(rng), _gauss_hists(rng)
    r = tnb.pair_pull(h1, h2, min_avg_tracks=3, pull_range=5.0)
    assert abs(r["mu"]) < 0.1
    assert abs(r["sigma"] - 1.0) < 0.1
    assert r["n_bins"] == 40 * 40


def test_pair_pull_drops_sparse_bins():
    rng = np.random.default_rng(1)
    h1, h2 = _gauss_hists(rng), _gauss_hists(rng)
    h1.counts[:5, :] = 1
    h2.counts[:5, :] = 1
    r = tnb.pair_pull(h1, h2, min_avg_tracks=3, pull_range=5.0)
    assert r["n_bins"] == 35 * 40


def test_all_pairs_scores_every_combination_once():
    rng = np.random.default_rng(2)
    hists = [_gauss_hists(rng, n_events=200, grid=10) for _ in range(4)]
    pairs = tnb.all_pairs(hists, seeds=[42, 43, 44, 45], min_avg_tracks=3,
                          pull_range=5.0)
    assert [(p["i"], p["j"]) for p in pairs] == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert pairs[-1]["seeds"] == [44, 45]


def test_summary_matches_hand_computation():
    pairs = [
        {"i": 0, "j": 1, "mu": 0.02, "sigma": 1.01},
        {"i": 0, "j": 2, "mu": -0.03, "sigma": 0.99},
        {"i": 1, "j": 2, "mu": 0.01, "sigma": 1.00},
    ]
    s = tnb.summarize(pairs, n_batches=3)
    assert s["mu_max_abs"] == pytest.approx(0.03)
    assert s["sigma_min"] == pytest.approx(0.99)
    assert s["sigma_max"] == pytest.approx(1.01)
    assert s["mu_mean"] == pytest.approx(0.0)
    assert s["sigma_mean"] == pytest.approx(1.0)
    # Leave-one-out with three batches leaves a single pair per replicate,
    # so the jackknife standard error is the spread of the three pairs.
    mu_jk = np.array([0.01, -0.03, 0.02])
    assert s["mu_jackknife_se"] == pytest.approx(
        np.sqrt(2 / 3 * np.sum((mu_jk - mu_jk.mean()) ** 2)))


def test_tolerance_text_prints_the_four_paper_lines():
    s = {"mu_max_abs": 0.0341, "sigma_min": 0.9912, "sigma_max": 1.0088,
         "mu_mean": -0.0206, "sigma_mean": 1.0021}
    text = tnb.tolerance_text(s)
    lines = text.split("\n")
    assert lines[0] == r"$|\mu_t| \leq 0.034$"
    # Two decimals, rounded outward: 0.9912 -> 0.99, 1.0088 -> 1.01.
    assert lines[1] == r"$\sigma_t \in [0.99, 1.01]$"
    assert lines[2] == r"$\langle\mu\rangle$: $-0.021$"
    assert lines[3] == r"$\langle\sigma\rangle$: $1.002$"


def test_committed_measurement_renders(tmp_path):
    """The JSON the figure reads is committed; it must parse and carry the
    ten-seed study at the figure's binning."""
    if not tnb.DEFAULT_OUT.exists():
        pytest.skip("truth-vs-truth study not recorded")
    doc = json.loads(tnb.DEFAULT_OUT.read_text())
    assert doc["meta"]["grid"] == 1500
    assert doc["meta"]["min_avg_tracks"] == 3
    assert len(doc["meta"]["seeds"]) == 10
    assert len(doc["pairs"]) == 45
    assert tnb.tolerance_text(doc["summary"]).count("\n") == 3
