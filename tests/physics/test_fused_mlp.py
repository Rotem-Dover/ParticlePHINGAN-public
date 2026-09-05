"""CPU-side unit tests for the fused-MLP packing and its pure-torch emulator.

No GPU and no triton: build_spec and _emulate_packed_forward are pure torch,
and they are where packing/masking bugs actually live. The kernel body is
covered by the GPU parity harness instead.
"""
import sys
import types

import pytest
import torch
from torch import nn

from common.config import N_EMBEDDING_CNT, N_NOISE_CNT
from common.enums import SecondaryFeats
from physics.g4h_ionisation.generators.continuous_generator.neural_net import (
    ContinuousGenerator)
from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
    SecondaryGenerator)
from physics.interfaces.fused_mlp import (
    _ACT_KINDS, _TILE_LADDER, _emulate_packed_forward,
    _leaf_modules, _shape_like_eager, build_spec, resolve_activation,
    try_enable_fused_forward)

# build_spec reads these from the generator's defining module.
_N_INPUT_FEATURES = 2
_N_U_FEATURES = 1


class _FakeGen(nn.Module):
    """Minimal module matching the kernel's structural template:
    embed(Linear+LN+act, Linear) -> concat u -> trunk(...) -> Linear(.,1)."""

    @staticmethod
    def _u_features(u: torch.Tensor) -> torch.Tensor:
        return u.unsqueeze(-1)

    def __init__(self, n_cond: int = 2, n_u: int = 1, emb: int = 25):
        super().__init__()
        self.n_embedding = emb
        self.n_noise = n_u
        self.embed = nn.Sequential(
            nn.Linear(n_cond, emb), nn.LayerNorm(emb), nn.SiLU(),
            nn.Linear(emb, emb), nn.Flatten())
        self.model = nn.Sequential(
            nn.Linear(emb + n_u, 2 * emb), nn.LayerNorm(2 * emb), nn.SiLU(),
            nn.Linear(2 * emb, emb), nn.LayerNorm(emb), nn.SiLU(),
            nn.Linear(emb, 1))

    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Layer-by-layer replay on an explicit [B, n_cond + n_u] input.

        Deliberately NOT the module's own forward: no RNG is drawn, so the
        comparison against the emulator is exact rather than distributional.
        """
        n_cond = self.embed[0].in_features
        h = self.embed(x[:, :n_cond])
        h = torch.cat([h, x[:, n_cond:]], dim=-1)
        return self.model(h).flatten()


# A net wide enough to force the (64, 128) tile pair: emb 50 + n_u 50 = 100,
# so the trunk is 2E = 100 and only the 128 rung fits. It lives in its OWN
# module namespace because build_spec reads _N_INPUT_FEATURES/_N_U_FEATURES
# off the generator's defining module, and this file's values (2, 1) describe
# _FakeGen. With neither constant present build_spec falls back to
# structural derivation -- the same path both stage generators take.
_WIDE_MOD = types.ModuleType("_fused_wide_tile_fixture")
sys.modules[_WIDE_MOD.__name__] = _WIDE_MOD


class _WideFakeGen(nn.Module):
    """Same structural template as _FakeGen, twice as wide."""

    def __init__(self, n_cond: int = 2, n_u: int = 50, emb: int = 50):
        super().__init__()
        self.n_embedding = emb
        self.n_noise = n_u
        self.embed = nn.Sequential(
            nn.Linear(n_cond, emb), nn.LayerNorm(emb), nn.SiLU(),
            nn.Linear(emb, emb), nn.Flatten())
        self.model = nn.Sequential(
            nn.Linear(emb + n_u, 2 * emb), nn.LayerNorm(2 * emb), nn.SiLU(),
            nn.Linear(2 * emb, emb), nn.LayerNorm(emb), nn.SiLU(),
            nn.Linear(emb, 1))

    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        n_cond = self.embed[0].in_features
        h = self.embed(x[:, :n_cond])
        return self.model(torch.cat([h, x[:, n_cond:]], dim=-1)).flatten()


_WideFakeGen.__module__ = _WIDE_MOD.__name__
_WIDE_MOD._WideFakeGen = _WideFakeGen


# A net with THREE distinct LayerNorm multipliers (1E, 4E, 2E) -- the same
# LN shape as SecondaryGenerator's trunk, but with a scalar unclamped
# readout.
# It exists so ln_w_mask's 8-bits-per-layer raw-width encoding is pinned
# by a raw literal against a single-output net, independently of the
# secondary's multi-output epilogue, and so a shift/stride error cannot
# hide behind packer/emulator agreement.
_MIXED_MOD = types.ModuleType("_fused_mixed_ln_fixture")
sys.modules[_MIXED_MOD.__name__] = _MIXED_MOD


class _MixedLNGen(nn.Module):
    """emb 25, n_u 25; LN widths 25 / 100 / 50 = 1E / 4E / 2E."""

    def __init__(self, n_cond: int = 2, n_u: int = 25, emb: int = 25):
        super().__init__()
        self.n_embedding = emb
        self.n_noise = n_u
        self.embed = nn.Sequential(
            nn.Linear(n_cond, emb), nn.LayerNorm(emb), nn.SiLU(),
            nn.Linear(emb, emb), nn.Flatten())
        self.model = nn.Sequential(
            nn.Linear(emb + n_u, 4 * emb), nn.LayerNorm(4 * emb), nn.SiLU(),
            nn.Linear(4 * emb, 2 * emb), nn.LayerNorm(2 * emb), nn.SiLU(),
            nn.Linear(2 * emb, 1))

    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        n_cond = self.embed[0].in_features
        h = self.embed(x[:, :n_cond])
        return self.model(torch.cat([h, x[:, n_cond:]], dim=-1)).flatten()


_MixedLNGen.__module__ = _MIXED_MOD.__name__
_MIXED_MOD._MixedLNGen = _MixedLNGen


@pytest.fixture
def gen():
    torch.manual_seed(0)
    return _FakeGen().eval()


def test_spec_records_the_structural_shape(gen):
    spec = build_spec(gen)
    assert spec.n_cond == 2
    assert spec.n_u == 1
    assert spec.emb_w == 25
    assert spec.n_layers == 5
    assert spec.trunk_at == 2


def test_spec_layer_norm_and_activation_masks(gen):
    spec = build_spec(gen)
    # LN after layers 0, 2, 3; activation after the same three
    assert spec.ln_mask == 0b01101
    assert spec.act_mask == 0b01101


def test_emulator_matches_a_layer_by_layer_replay(gen):
    spec = build_spec(gen)
    x = torch.randn(64, spec.n_cond + spec.n_u)
    got = _emulate_packed_forward(spec, x)
    want = gen.reference_forward(x)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_padding_columns_do_not_leak_into_the_result(gen):
    """Everything outside the real widths must stay exactly zero; if padding
    leaked, a wider batch would change the answer."""
    spec = build_spec(gen)
    x = torch.randn(8, spec.n_cond + spec.n_u)
    base = _emulate_packed_forward(spec, x)
    spec.weights[:, 60:, :] = 1e6
    spec.bias[:, 60:] = 1e6
    torch.testing.assert_close(_emulate_packed_forward(spec, x), base)


def test_unsupported_module_is_refused_not_approximated():
    torch.manual_seed(0)
    gen = _FakeGen()
    gen.model = nn.Sequential(nn.Linear(26, 25), nn.Tanh(), nn.Linear(25, 1))
    with pytest.raises(ValueError, match="unsupported module"):
        build_spec(gen)


def test_output_wider_than_the_narrow_tile_is_refused():
    """The readout may have up to tile_a columns -- the epilogue gathers
    one weight COLUMN per output -- but not more; a wider readout is a
    different kernel shape and must be refused, not approximated."""
    torch.manual_seed(0)
    gen = _FakeGen()
    gen.model[-1] = nn.Linear(25, 40)   # tile_a is 32 for this net
    with pytest.raises(ValueError, match=r"final Linear out_features 40 "
                                         r"outside \[1, 32\]"):
        build_spec(gen)


# ---------------------------------------------------------------------------
# Activation kinds + the continuous generator's own architecture
# ---------------------------------------------------------------------------

def test_activation_resolver_covers_both_kinds():
    assert _ACT_KINDS == {"silu": 0, "relu": 1}
    assert resolve_activation("silu") == 0
    assert resolve_activation("relu") == 1


def test_activation_resolver_refuses_the_unknown():
    with pytest.raises(ValueError, match="unsupported activation"):
        resolve_activation("gelu")


def test_silu_net_records_act_kind_zero(gen):
    assert build_spec(gen).act_kind == 0


def test_continuous_generator_packs_with_relu():
    torch.manual_seed(0)
    cnt = ContinuousGenerator().eval()
    spec = build_spec(cnt)
    assert spec.act_kind == 1
    assert spec.n_cond == 2
    assert spec.n_u == N_NOISE_CNT == 25
    assert spec.emb_w == N_EMBEDDING_CNT == 25
    assert spec.n_layers == 6
    assert spec.trunk_at == 2


def test_continuous_generator_splices_on_the_wide_tile():
    """emb_w + n_u = 50 does not fit the 32 tile, so the embed readout's
    OUTPUT tile must be the wide one -- the existing _tile_masks rule."""
    torch.manual_seed(0)
    spec = build_spec(ContinuousGenerator().eval())
    assert not (spec.in_a_mask >> spec.trunk_at) & 1
    assert not (spec.out_a_mask >> (spec.trunk_at - 1)) & 1


def test_continuous_emulator_matches_the_module():
    """Paired: an explicit gen_input, so no RNG is involved on either side."""
    torch.manual_seed(0)
    cnt = ContinuousGenerator().eval()
    spec = build_spec(cnt)
    x = torch.randn(64, spec.n_cond + spec.n_u)
    with torch.no_grad():
        embedded = cnt.embed(x[:, :spec.n_cond].unsqueeze(1))
        want = cnt.model(torch.cat([embedded, x[:, spec.n_cond:]], dim=-1)).flatten()
    torch.testing.assert_close(_emulate_packed_forward(spec, x), want,
                               rtol=1e-5, atol=1e-6)


def test_mixed_activations_are_refused():
    torch.manual_seed(0)
    cnt = ContinuousGenerator().eval()
    cnt.model[0][2] = nn.SiLU()   # trunk block 0 activation
    with pytest.raises(ValueError, match="mixes activation kinds"):
        build_spec(cnt)


# ---------------------------------------------------------------------------
# The tile pair (WA, WB) is chosen per generator, not hardcoded.
# ---------------------------------------------------------------------------

def test_tile_ladder_is_the_four_supported_widths():
    """16 is tl.dot's minimum dimension, so the ladder never grows an
    8-entry."""
    assert _TILE_LADDER == (16, 32, 64, 128)


def test_continuous_net_lands_on_the_32_64_pair():
    torch.manual_seed(0)
    spec = build_spec(ContinuousGenerator().eval())
    assert (spec.tile_a, spec.tile_b) == (32, 64)


def test_live_secondary_lands_on_the_16_16_pair():
    torch.manual_seed(0)
    spec = build_spec(SecondaryGenerator().eval())
    assert (spec.tile_a, spec.tile_b) == (16, 16)
    assert spec.weights.shape == (6, 16, 16)
    # degenerate pair: every layer fits the narrow tile, masks all-ones
    assert spec.in_a_mask == 0b111111
    assert spec.out_a_mask == 0b111111


def test_min_tile_forces_the_wider_pair():
    """The sweep's tile-32 arm and the shipped _MIN_TILE path: same spec,
    wider padding, identical math."""
    torch.manual_seed(0)
    sec = SecondaryGenerator().eval()
    spec16 = build_spec(sec)
    spec32 = build_spec(sec, min_tile=32)
    assert (spec32.tile_a, spec32.tile_b) == (32, 32)
    x = torch.randn(256, spec16.n_cond + spec16.n_u) * 3.0
    torch.testing.assert_close(_emulate_packed_forward(spec16, x),
                               _emulate_packed_forward(spec32, x),
                               rtol=1e-5, atol=1e-6)


def test_min_tile_table_is_read_for_the_class():
    """_MIN_TILE is a per-class floor build_spec must read, so electing a
    different minimum tile for a class needs only a one-line table entry."""
    from physics.interfaces import fused_mlp as fm
    torch.manual_seed(0)
    sec = SecondaryGenerator().eval()
    fm._MIN_TILE["SecondaryGenerator"] = 32
    try:
        assert (build_spec(sec).tile_a, build_spec(sec).tile_b) == (32, 32)
    finally:
        del fm._MIN_TILE["SecondaryGenerator"]
    assert (build_spec(sec).tile_a, build_spec(sec).tile_b) == (16, 16)


def test_live_secondary_arch_fuses():
    """The 5/5/16 hardtanh_reflect_top secondary's LayerNorm widths pack
    into build_spec's raw-width encoding, so it fuses rather than falling
    back to eager."""
    torch.manual_seed(0)
    spec = build_spec(SecondaryGenerator().eval())
    assert (spec.n_layers, spec.trunk_at, spec.emb_w, spec.n_u) == (6, 2, 5, 5)
    assert spec.n_out == 1
    assert spec.clamp_kind == 2
    assert _ln_widths(spec) == [5, 0, 16, 16, 8, 0]


def test_live_secondary_emulator_matches_the_module():
    """Emulator vs a layer-by-layer replay of the LIVE default shape, in eval
    mode — the reflect rule applied on both sides.

    UNFLATTENED on purpose: SecondaryGenerator.forward does not call
    .flatten() (unlike ContinuousGenerator.forward), so its real eager
    return is [B, 1], not [B] -- spec.squeeze_out (measured structurally by
    build_spec's _probe_squeeze_out) must be False here, and the emulator's
    _shape_like_eager output must match that, not a blanket
    n_out==1 => squeeze assumption. See
    test_secondary_fused_output_shape_matches_eager for the dedicated
    shape-contract pin."""
    torch.manual_seed(0)
    sec = SecondaryGenerator().eval()
    spec = build_spec(sec)
    assert spec.squeeze_out is False
    x = torch.randn(256, spec.n_cond + spec.n_u) * 3.0   # push into the tails
    with torch.no_grad():
        embedded = sec.embed(x[:, :spec.n_cond].unsqueeze(1))
        want = sec.model(torch.cat([embedded, x[:, spec.n_cond:]], dim=-1))
    got = _emulate_packed_forward(spec, x)
    assert got.shape == want.shape == (256, 1)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_reflect_epilogue_order_is_reflect_then_clamp():
    """Order pin at the extremes: for raw > 2 the reflected value goes
    negative and the trailing clamp floors it at 0. A clamp-then-reflect
    implementation agrees everywhere in the bulk and only disagrees here."""
    import dataclasses
    torch.manual_seed(0)
    sec = SecondaryGenerator().eval()
    spec = build_spec(sec)
    # The live net has LayerNorm immediately before every Linear but the
    # readout, so an untrained (random-init) forward's pre-terminal output is
    # bounded to a narrow band by construction -- verified empirically:
    # scaling the raw input by up to 1e8, or drawing up to 5e7 rows, never
    # pushes the readout past ~1.3. That is a property of random init plus
    # LayerNorm, not of the reflect-then-clamp order under test, so the
    # readout is amplified post-hoc to reach every regime deterministically;
    # clamp_kind (== 2, asserted by test_live_secondary_arch_fuses) and
    # every other structural field still come from the real live module.
    spec.weights[spec.n_layers - 1] *= 20.0
    spec.bias[spec.n_layers - 1] *= 20.0
    raw_spec = dataclasses.replace(spec, clamp_kind=0)
    x = torch.randn(4096, spec.n_cond + spec.n_u) * 30.0
    raw = _emulate_packed_forward(raw_spec, x)
    # the probe must actually cover every regime or it is vacuous
    assert (raw < 0).any() and ((raw > 1) & (raw < 2)).any() and (raw > 2).any()
    got = _emulate_packed_forward(spec, x)
    want = torch.where(raw > 1.0, 2.0 - raw, raw).clamp(0.0, 1.0)
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("act", ["sigmoid", "stretched_sigmoid",
                                 "hardtanh_open_top"])
def test_other_terminals_still_refuse(act):
    torch.manual_seed(0)
    sec = SecondaryGenerator(output_activation=act).eval()
    with pytest.raises(ValueError, match="not implemented in the fused"):
        build_spec(sec)


def test_reflect_mid_network_is_refused():
    torch.manual_seed(0)
    sec = SecondaryGenerator().eval()
    from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
        HardtanhReflectTop)
    sec.model[0][2] = HardtanhReflectTop()
    with pytest.raises(ValueError, match="must be terminal"):
        build_spec(sec)


def test_new_secondary_falls_back_to_eager_not_crash():
    """The runtime path: try_enable_fused_forward refuses and logs, never
    raises — the simulator must keep working with the secondary eager. On
    CPU try_enable_fused_forward refuses on the device check (fusion is
    CUDA-only) before build_spec ever runs, regardless of architecture."""
    torch.manual_seed(0)
    sec = SecondaryGenerator().eval()
    try_enable_fused_forward(sec)   # must not raise
    x = torch.full((8,), 0.5)
    with torch.inference_mode():
        assert sec.forward(x).shape == (8, 1)


def test_weight_storage_stride_follows_the_wide_tile():
    torch.manual_seed(0)
    spec = build_spec(ContinuousGenerator().eval())
    assert spec.weights.shape == (spec.n_layers, spec.tile_b, spec.tile_b)
    assert spec.bias.shape == (spec.n_layers, spec.tile_b)


def test_a_layer_wider_than_the_ladder_is_refused():
    torch.manual_seed(0)
    cnt = ContinuousGenerator().eval()
    cnt.model[1][0] = nn.Linear(50, 200, bias=False)
    cnt.model[1][1] = nn.LayerNorm(200)
    with pytest.raises(ValueError, match="exceeds the 128 tile"):
        build_spec(cnt)


def test_wide_tile_net_packs_and_emulates_on_the_64_128_pair():
    """End-to-end proof that the tile pair is really a parameter.

    Everything else in this file runs on (32, 64), where a tile_b-stride
    packing bug is invisible because tile_b still happens to be 64. This net
    -- emb 50, n_u 50, 100-wide trunk -- is the smallest thing that forces
    (64, 128): the 128 storage stride, the wide splice, and a rectangular
    64->128 layer, all through _emulate_packed_forward -- exercising a
    genuinely different tile pair rather than duplicating
    test_continuous_emulator_matches_the_module's (32, 64) case.
    """
    torch.manual_seed(0)
    gen = _WideFakeGen().eval()
    spec = build_spec(gen)

    assert (spec.tile_a, spec.tile_b) == (64, 128)
    assert spec.weights.shape == (spec.n_layers, 128, 128)
    assert spec.bias.shape == (spec.n_layers, 128)
    assert (spec.n_layers, spec.trunk_at, spec.emb_w, spec.n_u) == (5, 2, 50, 50)
    # layers 0, 1, 4 on the 64 tile in; 0, 3, 4 out -- except the readout's
    # output, which the splice rule (50 + 50 > 64) forces onto the 128 tile.
    assert spec.in_a_mask == 0b10011
    assert spec.out_a_mask == 0b11001
    assert not (spec.out_a_mask >> (spec.trunk_at - 1)) & 1   # splice is wide
    assert not (spec.in_a_mask >> spec.trunk_at) & 1

    x = torch.randn(64, spec.n_cond + spec.n_u)
    with torch.no_grad():
        want = gen.reference_forward(x)
    torch.testing.assert_close(_emulate_packed_forward(spec, x), want,
                               rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# ln_w_mask: raw-width encoding for each layer's LayerNorm width, 8 bits per
# layer, with no assumption that the width is emb_w times a power of two.
# ---------------------------------------------------------------------------

def _ln_widths(spec):
    return [(spec.ln_w_mask >> (8 * l)) & 255 for l in range(spec.n_layers)]


def test_continuous_ln_widths_are_25_50_50_25():
    """LN widths on layers 0, 2, 3, 4 -- raw values, no emb_w coupling."""
    torch.manual_seed(0)
    spec = build_spec(ContinuousGenerator().eval())
    w = _ln_widths(spec)
    assert [w[l] for l in (0, 2, 3, 4)] == [25, 50, 50, 25]


def test_50_wide_secondary_shape_ln_widths_are_25_50_50_25():
    """A 50-wide shape keeps exercising the packing math."""
    torch.manual_seed(0)
    sec = SecondaryGenerator(n_embedding=25, n_noise=25,
                             n_trunk=50, n_half=25, n_features=1,
                             output_activation="hardtanh").eval()
    spec = build_spec(sec)
    w = _ln_widths(spec)
    assert [w[l] for l in (0, 2, 3, 4)] == [25, 50, 50, 25]


def test_arbitrary_ln_width_packs():
    """A width that is NOT emb_w times a power of two (30 with emb_w=25)
    packs and emulates -- the whole point of the raw encoding. Replaces
    test_non_power_of_two_ln_width_is_refused, whose premise is retired.
    The next Linear must move with the LN or the width-mismatch check
    fires instead and this proves nothing."""
    torch.manual_seed(0)
    cnt = ContinuousGenerator().eval()
    cnt.model[0][0] = nn.Linear(50, 30, bias=False)
    cnt.model[0][1] = nn.LayerNorm(30)
    cnt.model[1][0] = nn.Linear(30, 50, bias=False)
    spec = build_spec(cnt)
    assert _ln_widths(spec)[2] == 30


def test_ln_width_unequal_to_linear_out_is_refused():
    """The exact width check survives the encoding change."""
    torch.manual_seed(0)
    cnt = ContinuousGenerator().eval()
    cnt.model[0][1] = nn.LayerNorm(30)   # Linear stays 50-wide
    with pytest.raises(ValueError, match="LN width 30 != Linear out 50"):
        build_spec(cnt)


def test_ln_w_mask_fits_the_unroll_budget():
    """12 layers x 8 bits = 96 bits; a Python-int constexpr, but bounded."""
    torch.manual_seed(0)
    sec = SecondaryGenerator(n_embedding=25, n_noise=25,
                             n_trunk=50, n_half=25, n_features=1,
                             output_activation="hardtanh").eval()
    spec = build_spec(sec)
    assert 0 <= spec.ln_w_mask < (1 << 96)


def test_mixed_ln_widths_pack_to_the_literal_byte_mask():
    """Raw-value pin on the three-width fixture (25 / 100 / 50 on layers
    0 / 2 / 3). Written as a literal on purpose: an off-by-one in the 8*l
    stride lands on a different number, so packer/emulator agreement
    cannot hide it."""
    torch.manual_seed(0)
    spec = build_spec(_MixedLNGen().eval())
    assert (spec.tile_a, spec.tile_b) == (64, 128)
    assert spec.ln_mask == 0b01101
    assert spec.ln_w_mask == 845414425 == 25 | (100 << 16) | (50 << 24)
    assert _ln_widths(spec) == [25, 0, 100, 50, 0]


def test_emulator_actually_reads_the_width_bytes():
    """Mutation check: demote layer 2's LN width 100 -> 50 in the spec only.
    If the emulator ignored ln_w_mask the output would be unchanged."""
    torch.manual_seed(0)
    gen = _MixedLNGen().eval()
    spec = build_spec(gen)
    x = torch.randn(64, spec.n_cond + spec.n_u)
    base = _emulate_packed_forward(spec, x)
    spec.ln_w_mask -= (50 << 16)   # layer 2: 100 -> 50
    assert not torch.equal(_emulate_packed_forward(spec, x), base)


def test_w50_mask_is_gone():
    torch.manual_seed(0)
    spec = build_spec(ContinuousGenerator().eval())
    assert not hasattr(spec, "w50_mask")


def test_mixed_ln_widths_emulate_the_module():
    """The 4E LayerNorm must normalize over 100 columns, not 50."""
    torch.manual_seed(0)
    gen = _MixedLNGen().eval()
    spec = build_spec(gen)
    x = torch.randn(64, spec.n_cond + spec.n_u)
    with torch.no_grad():
        want = gen.reference_forward(x)
    torch.testing.assert_close(_emulate_packed_forward(spec, x), want,
                               rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Task 13: multi-output epilogue (n_out) + the terminal Hardtanh clamp
# ---------------------------------------------------------------------------

def test_continuous_is_single_output_and_unclamped():
    torch.manual_seed(0)
    spec = build_spec(ContinuousGenerator().eval())
    assert spec.n_out == 1
    assert spec.clamp_kind == 0


def test_secondary_is_single_output_and_clamped():
    """Hardtanh(0, 1) is the terminal op of the secondary's last block;
    `n_out` is 1, one column, e_sec."""
    torch.manual_seed(0)
    # explicit widths: a 50-wide (hardtanh) net keeps exercising the
    # clamp_kind == 1 path; the live default is clamp_kind == 2, covered by
    # test_live_secondary_arch_fuses and the reflect-epilogue tests
    sec = SecondaryGenerator(n_embedding=25, n_noise=25,
                             n_trunk=50, n_half=25, n_features=1,
                             output_activation="hardtanh").eval()
    spec = build_spec(sec)
    assert spec.n_out == 1
    assert spec.clamp_kind == 1
    assert spec.n_layers == 6
    assert spec.trunk_at == 2


def test_the_clamp_bounds_are_read_from_the_generator_module():
    """The clamp must reproduce the module's own Hardtanh, not an assumed
    +/-1. Pinned against the live leaf so a checkpoint-era edit to the bounds
    can never be silently absorbed by a hardcoded (0, 1) in the packer."""
    torch.manual_seed(0)
    sec = SecondaryGenerator(output_activation="hardtanh").eval()
    (ht,) = [m for m in sec.model.modules() if isinstance(m, nn.Hardtanh)]
    assert (ht.min_val, ht.max_val) == (0, 1)
    assert ht is (_leaf_modules(sec.embed) + _leaf_modules(sec.model))[-1]


def test_secondary_emulator_matches_the_module():
    torch.manual_seed(0)
    # explicit widths: a 50-wide (hardtanh) shape keeps exercising the
    # clamp_kind == 1 path; the live default is clamp_kind == 2, covered by
    # test_live_secondary_arch_fuses and the reflect-epilogue tests
    sec = SecondaryGenerator(n_embedding=25, n_noise=25,
                             n_trunk=50, n_half=25, n_features=1,
                             output_activation="hardtanh").eval()
    spec = build_spec(sec)
    x = torch.randn(64, spec.n_cond + spec.n_u)
    with torch.no_grad():
        embedded = sec.embed(x[:, :spec.n_cond].unsqueeze(1))
        want = sec.model(torch.cat([embedded, x[:, spec.n_cond:]], dim=-1))
    got = _emulate_packed_forward(spec, x)
    # n_out == 1 (SecondaryFeats has one column), but SecondaryGenerator.
    # forward does NOT flatten its own output (unlike ContinuousGenerator's,
    # see test_single_output_path_is_unchanged_by_the_generalization) -- the
    # emulator's shape follows the real eager module's own convention
    # (spec.squeeze_out, structurally probed), not n_out alone.
    assert spec.squeeze_out is False
    assert got.ndim == 2 and got.shape == (64, 1)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


# The secondary net has one output column, e_sec, so there is no column
# ordering for the fused epilogue to get wrong; the per-column value check
# is test_secondary_emulator_matches_the_module.


def test_secondary_output_respects_the_clamp():
    torch.manual_seed(0)
    # explicit widths: a 50-wide shape keeps exercising the packing
    # math; the current default (5/5/16, hardtanh_reflect_top) is covered by
    # test_reflect_epilogue_order_is_reflect_then_clamp instead
    sec = SecondaryGenerator(n_embedding=25, n_noise=25,
                             n_trunk=50, n_half=25, n_features=1,
                             output_activation="hardtanh").eval()
    spec = build_spec(sec)
    got = _emulate_packed_forward(spec, torch.randn(256, spec.n_cond + spec.n_u))
    assert got.min() >= 0.0 and got.max() <= 1.0


def test_the_clamp_is_load_bearing_on_the_single_column():
    """Mutation check: clearing spec.clamp_kind must change the output. A
    dropped clamp shows up here.

    This covers the single output column (e_sec); the mutation it guards
    against -- a clamp that is present in the spec but not applied -- would
    be invisible without it.
    """
    torch.manual_seed(0)
    # explicit widths: a 50-wide shape keeps exercising the packing
    # math; the current default (5/5/16, hardtanh_reflect_top) is covered by
    # test_reflect_epilogue_order_is_reflect_then_clamp instead
    sec = SecondaryGenerator(n_embedding=25, n_noise=25,
                             n_trunk=50, n_half=25, n_features=1,
                             output_activation="hardtanh").eval()
    spec = build_spec(sec)
    x = torch.randn(1024, spec.n_cond + spec.n_u)
    clamped = _emulate_packed_forward(spec, x)
    spec.clamp_kind = 0
    raw = _emulate_packed_forward(spec, x)
    assert not torch.equal(clamped, raw)
    torch.testing.assert_close(clamped, raw.clamp(0.0, 1.0))


def test_single_output_path_is_unchanged_by_the_generalization():
    """N_OUT=1 must reduce to the plain scalar path exactly -- same gather, same
    summation order -- so the continuous net is bit-identical to before."""
    torch.manual_seed(0)
    spec = build_spec(ContinuousGenerator().eval())
    x = torch.randn(64, spec.n_cond + spec.n_u)
    got = _emulate_packed_forward(spec, x)
    assert got.ndim == 1 and got.shape == (64,)


# ---------------------------------------------------------------------------
# fused_forward's output shape must match the REAL eager module's own
# convention, per net -- not an n_out==1 assumption. ContinuousGenerator.
# forward ends `.flatten()`; SecondaryGenerator.forward does not, so its
# real eager return is [B, 1] even though n_out == 1 too. If fused_forward
# did not restore this per-net shape, the fused secondary would crash
# post_step_do_it with IndexError, since SecondaryYScaler.rescale indexes
# `tensor[:, E_SEC_IDX]`, which only works on a 2-D tensor.
#
# The kernel cannot run on CPU, so these tests exercise the shared,
# testable shape-restoring function (`_shape_like_eager`) that both
# `_emulate_packed_forward` and `fused_forward` call -- re-checking only the
# emulator would not pin `fused_forward`'s own contract, since the two can
# disagree independently.
# ---------------------------------------------------------------------------

def test_squeeze_out_is_measured_per_module_not_assumed_from_n_out():
    """Two n_out==1 nets can still disagree on their eager return rank, so
    squeeze_out must be measured on each module rather than inferred from
    n_out alone."""
    torch.manual_seed(0)
    cnt_spec = build_spec(ContinuousGenerator().eval())
    sec_spec = build_spec(SecondaryGenerator().eval())
    assert cnt_spec.n_out == sec_spec.n_out == 1
    assert cnt_spec.squeeze_out is True
    assert sec_spec.squeeze_out is False


def test_shape_like_eager_restores_the_continuous_convention():
    """[B, 1] kernel-native output -> [B], matching
    ContinuousGenerator.forward's own `.flatten()`."""
    torch.manual_seed(0)
    spec = build_spec(ContinuousGenerator().eval())
    kernel_out = torch.randn(8, spec.n_out)
    shaped = _shape_like_eager(kernel_out, spec)
    assert shaped.shape == (8,)
    torch.testing.assert_close(shaped, kernel_out[:, 0])


def test_shape_like_eager_restores_the_secondary_convention():
    """[B, 1] kernel-native output stays [B, 1], matching
    SecondaryGenerator.forward's own (unflattened) return -- this is the
    exact contract `fused_forward` must honor for `post_step_do_it` (via
    SecondaryYScaler.rescale's `tensor[:, E_SEC_IDX]` indexing) not to
    IndexError on the fused path."""
    torch.manual_seed(0)
    spec = build_spec(SecondaryGenerator().eval())
    kernel_out = torch.randn(8, spec.n_out)
    shaped = _shape_like_eager(kernel_out, spec)
    assert shaped.shape == (8, 1)
    torch.testing.assert_close(shaped, kernel_out)


def test_fused_forward_shape_contract_on_cpu():
    """`fused_forward`'s user-visible shape, pinned end to end without CUDA.

    `try_enable_fused_forward` refuses on CPU (fusion is CUDA-only) before
    `fused_forward` is ever installed, so this cannot call the real
    `gen.forward()` post-enable on this machine -- it instead reproduces
    exactly what `fused_forward`'s own final line does
    (`_shape_like_eager(out, spec)` on the kernel's native `[B, n_out]`
    output), which is the actual function under test, factored out for
    exactly this reason. A test that only re-checked the emulator would not
    cover this, since the emulator and `fused_forward` can disagree
    independently on which shape they restore.
    """
    torch.manual_seed(0)
    for gen_cls, want_shape in ((ContinuousGenerator, (8,)),
                                (SecondaryGenerator, (8, 1))):
        spec = build_spec(gen_cls().eval())
        kernel_native = torch.randn(8, spec.n_out)   # torch.ops.phingan.fused_mlp's own shape
        assert _shape_like_eager(kernel_native, spec).shape == want_shape


def _strip_comments(src: str) -> str:
    """Source with `#` comments removed, so a prose mention of a forbidden
    identifier does not read as its use."""
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


def test_fused_forward_restores_shape_via_shared_helper_not_inline_indexing():
    """Source pin for the shape-restoration fix: `fused_forward`'s final line
    must route through `_shape_like_eager` (the single source of truth shared
    with `_emulate_packed_forward`), not an inline `out[:, 0]` pattern that
    assumes n_out == 1 always means a squeezed [B] output -- `squeeze_out`
    is measured per module instead (see
    test_squeeze_out_is_measured_per_module_not_assumed_from_n_out and
    test_shape_like_eager_restores_the_secondary_convention above)."""
    import inspect

    from physics.interfaces import fused_mlp

    src = inspect.getsource(fused_mlp.try_enable_fused_forward)
    code = _strip_comments(src)
    assert "_shape_like_eager" in code
    assert "out[:, 0]" not in code


def test_clamp_before_the_final_linear_is_refused():
    """Hardtanh must be terminal; anywhere else is a different network."""
    torch.manual_seed(0)
    sec = SecondaryGenerator().eval()
    sec.model[0][2] = nn.Hardtanh(min_val=0, max_val=1)
    with pytest.raises(ValueError, match="must be terminal"):
        build_spec(sec)


def test_a_differently_bounded_clamp_is_refused():
    """Only Hardtanh(0, 1) is implemented; other bounds must raise rather
    than be approximated by the (0, 1) the kernel hardcodes."""
    torch.manual_seed(0)
    sec = SecondaryGenerator().eval()
    sec.model[2][-1] = nn.Hardtanh(min_val=-1, max_val=1)
    with pytest.raises(ValueError, match="only Hardtanh"):
        build_spec(sec)
