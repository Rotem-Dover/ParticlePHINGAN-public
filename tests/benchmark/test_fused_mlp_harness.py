"""CPU-checkable pieces of the GPU fused-MLP harness: the terminal oracle,
the column labels, and the clamp-probe search logic. The kernel arms
(_launch, _clamp_probe's kernel-vs-_apply_terminal check) are GPU-only by
design; _clamp_search is pure torch (_emulate_packed_forward only) and is
deliberately CPU-testable -- see its docstring in
benchmark/runtime_profiling/test_fused_mlp.py."""
import torch

from benchmark.runtime_profiling.test_fused_mlp import (
    _COL_NAMES, _apply_terminal, _clamp_missing, _clamp_search)
from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
    SecondaryGenerator)
from physics.interfaces.fused_mlp import build_spec


def test_secondary_column_label_is_e_sec_only():
    """SecondaryGenerator emits one column; the parity harness must label
    it "e_sec", not fall back to a two-column ["theta", "e_sec"] naming
    that would misname the single output."""
    assert _COL_NAMES["SecondaryGenerator"] == ["e_sec"]


def test_terminal_oracle_matches_each_kind():
    raw = torch.tensor([-0.5, 0.3, 1.4, 2.0, 2.7])
    torch.testing.assert_close(_apply_terminal(raw, 1),
                               raw.clamp(0.0, 1.0))
    torch.testing.assert_close(
        _apply_terminal(raw, 2),
        torch.where(raw > 1.0, 2.0 - raw, raw).clamp(0.0, 1.0))
    # kind 2 at raw > 2 must floor at 0 (reflect-then-clamp order)
    assert _apply_terminal(torch.tensor([2.7]), 2).item() == 0.0


# ---------------------------------------------------------------------------
# The secondary architecture's clamp probe cannot reach raw > 2 by scaling
# the input alone: LayerNorm sits before every Linear layer but the
# readout, so it renormalizes away the input's scale and the readout's raw
# range stays essentially fixed regardless of how far the input is scaled.
# The probe must reach the deep regime by amplifying the readout weights
# instead.
# ---------------------------------------------------------------------------

def test_clamp_search_input_scaling_alone_cannot_reach_the_deep_regime():
    """On the secondary architecture, scaling the conditioning/latent input
    over 3 decades never pushes the readout's raw output past 2 (a fresh
    random init stays within roughly [-1.1, 0.6] at every scale tried),
    because every Linear but the readout is preceded by a LayerNorm that
    renormalizes the input's scale away. If a future change let input
    scaling alone reach raw > 2, this test's `seen_deep` assertion and its
    flat-range assertions would catch it."""
    torch.manual_seed(0)
    spec = build_spec(SecondaryGenerator().eval())
    import dataclasses
    from physics.interfaces.fused_mlp import _emulate_packed_forward
    raw_spec = dataclasses.replace(spec, clamp_kind=0)
    seen_deep = False
    lo_bound, hi_bound = [], []
    torch.manual_seed(99)
    for scale in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        labels = (torch.rand(4096, spec.n_cond) - 0.5) * 2.0 * scale
        latent = torch.randn(4096, spec.n_u) * scale
        x = torch.cat([labels, latent], dim=-1).contiguous().float()
        raw = _emulate_packed_forward(raw_spec, x)
        lo_bound.append(raw.min().item())
        hi_bound.append(raw.max().item())
        seen_deep |= bool((raw > 2.0).any().item())
    assert not seen_deep
    # the range stays essentially flat across 3 decades of input scale --
    # the scale-invariance the docstring above describes, not noise.
    assert max(hi_bound) - min(hi_bound) < 1.0
    assert max(lo_bound) - min(lo_bound) < 1.0


def test_clamp_search_reaches_the_deep_regime_via_amplification():
    """On the secondary spec, _clamp_search must find a batch covering
    raw > 1, raw < 0 AND raw > 2, and report that it needed readout-weight
    amplification to reach the deep (raw > 2) condition."""
    torch.manual_seed(0)
    spec = build_spec(SecondaryGenerator().eval())
    weights_before = spec.weights.clone()
    bias_before = spec.bias.clone()
    chosen, seen, used_spec, amplified = _clamp_search(
        spec, m=4096, device="cpu", need_deep=True)
    assert chosen is not None
    assert amplified is True
    assert seen == {"hi": True, "lo": True, "deep": True}
    assert used_spec is not spec               # a copy, not the live spec
    assert used_spec.weights is not spec.weights
    scale, x, raw, hi, lo = chosen
    assert bool((raw > 2.0).any())
    # spec itself (and its weights/bias) must be untouched -- still needed
    # by whatever runs after the clamp probe in the real harness.
    torch.testing.assert_close(spec.weights, weights_before)
    torch.testing.assert_close(spec.bias, bias_before)
    assert _clamp_missing(seen, need_deep=True) == []


def test_clamp_search_shallow_only_net_needs_no_amplification():
    """clamp_kind == 1 (plain Hardtanh) nets don't require the deep regime
    at all, so _clamp_search must not touch (let alone clone) the spec.

    The readout is scaled 5x up front (a deliberate TEST fixture, not
    _clamp_search's own amplification -- verified empirically to clear both
    thresholds at every one of the swept scales) so this net's shallow
    regime is reliably reachable regardless of random seed, isolating the
    behavior under test: does _clamp_search skip amplification when
    need_deep is False, even though the un-amplified `_try` call already
    succeeds."""
    torch.manual_seed(0)
    sec = SecondaryGenerator(n_embedding=25, n_noise=25, n_trunk=50,
                             n_half=25, n_features=1,
                             output_activation="hardtanh").eval()
    with torch.no_grad():
        sec.model[2][3].weight *= 5.0
        sec.model[2][3].bias *= 5.0
    spec = build_spec(sec)
    assert spec.clamp_kind == 1
    weights_before = spec.weights.clone()
    chosen, seen, used_spec, amplified = _clamp_search(
        spec, m=4096, device="cpu", need_deep=False)
    assert chosen is not None
    assert amplified is False
    assert used_spec is spec
    torch.testing.assert_close(spec.weights, weights_before)
    assert _clamp_missing(seen, need_deep=False) == []


def test_clamp_missing_names_only_the_unreached_conditions():
    all_seen = {"hi": True, "lo": True, "deep": True}
    assert _clamp_missing(all_seen, need_deep=True) == []
    assert _clamp_missing(all_seen, need_deep=False) == []

    deep_missing = {"hi": True, "lo": True, "deep": False}
    assert _clamp_missing(deep_missing, need_deep=True) == [
        "raw > 2 (reflect's deep branch)"]
    # deep is irrelevant when the net's own clamp_kind doesn't need it
    assert _clamp_missing(deep_missing, need_deep=False) == []

    nothing_seen = {"hi": False, "lo": False, "deep": False}
    assert _clamp_missing(nothing_seen, need_deep=True) == [
        "raw > 1 (upper bound)", "raw < 0 (lower bound)",
        "raw > 2 (reflect's deep branch)"]
