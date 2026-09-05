"""
Triton kernel for the whole-net fused MLP forward pass.

Lives in its own module (imported lazily by fused_mlp) because
@triton.jit resolves names from MODULE globals — `tl` must be a module-level
import — and because importing triton at all must stay optional on CPU-only
machines. See fused_mlp.py for the design contract.

Rectangular tiles: two constexpr bitmasks, IN_A_MASK / OUT_A_MASK, declare
each layer's padded input/output tile width — the narrow tile WA or the
wide tile WB, a per-net pair chosen from `_TILE_LADDER` by
`fused_mlp._choose_tiles` (16 is `tl.dot`'s minimum M/N/K, so it is the
ladder floor, never an 8-tile). The kernel holds dual activation tiles
(a_a / a_b); the unrolled `static_range` loop picks the dot shape per layer
and every dead branch folds away at trace time, so each net still compiles
to straight-line code. Parameters are stored with the WB stride regardless
of which tile a layer sits on. Padding slots stay exact zeros — parity vs
eager remains at fp32 summation-reorder level.

Multi-output epilogue: the final layer emits N_OUT columns into an
[n_rows, N_OUT] row-major buffer, one weighted sum per column, then an
optional terminal epilogue selected by CLAMP_KIND: 0 = none, 1 =
Hardtanh(0, 1), 2 = HardtanhReflectTop's eval rule (reflect z>1 spill as
2-z, THEN clamp to [0, 1] -- that order is load-bearing, since raw > 2 must
floor at 0 rather than go negative and stay there). N_OUT == 1 traces to a
plain scalar epilogue. Column j is the module's own output column j, and
the row-major `rows * N_OUT + j` store preserves that order in the output
buffer.

LayerNorm width: each layer's LayerNorm width is stored raw, eight bits per
layer (bits [8l, 8l+8) of LN_W_MASK). Storing the width directly, rather
than as a multiple of the embedding width, admits any per-layer LayerNorm
width. EMB_W is a separate kernel constexpr the u-splice needs, independent
of any layer's LayerNorm width.
"""
import triton
import triton.language as tl


@triton.jit
def _bias_ln_act(a, cols, l, b_ptr, g_ptr, bb_ptr,
                 STRIDE: tl.constexpr, NW: tl.constexpr,
                 LN: tl.constexpr, ACT: tl.constexpr,
                 ACT_KIND: tl.constexpr):
    """Bias add + optional masked LayerNorm(NW) + optional activation.

    `cols` is the tile's column index vector (width WA or WB); parameter rows
    are stored with STRIDE (= the WB-wide storage layout).
    """
    a = a + tl.load(b_ptr + l * STRIDE + cols)[None, :]
    if LN:
        m = cols[None, :] < NW
        am = tl.where(m, a, 0.0)
        mean = tl.sum(am, axis=1) / NW
        d = tl.where(m, a - mean[:, None], 0.0)
        var = tl.sum(d * d, axis=1) / NW
        rstd = 1.0 / tl.sqrt(var + 1e-5)
        gam = tl.load(g_ptr + l * STRIDE + cols)[None, :]
        bet = tl.load(bb_ptr + l * STRIDE + cols)[None, :]
        a = d * rstd[:, None] * gam + bet
    if ACT:
        if ACT_KIND == 0:
            a = a * tl.sigmoid(a)     # SiLU
        else:
            a = tl.maximum(a, 0.0)    # ReLU
    return a


@triton.jit
def fused_mlp_kernel(
    x_ptr, out_ptr, w_ptr, b_ptr, g_ptr, bb_ptr,
    n_rows, x_stride,
    NC: tl.constexpr, NU: tl.constexpr,
    N_LAYERS: tl.constexpr, TRUNK_AT: tl.constexpr,
    LN_MASK: tl.constexpr, ACT_MASK: tl.constexpr, LN_W_MASK: tl.constexpr,
    ACT_KIND: tl.constexpr,
    IN_A_MASK: tl.constexpr, OUT_A_MASK: tl.constexpr,
    EMB_W: tl.constexpr,
    BLOCK: tl.constexpr, WA: tl.constexpr, WB: tl.constexpr,
    TF32: tl.constexpr,
    N_OUT: tl.constexpr, CLAMP_KIND: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    rmask = rows < n_rows
    c_a = tl.arange(0, WA)
    c_b = tl.arange(0, WB)

    # Dual activation tiles: exactly one is live per layer (per the width
    # masks); the dead one's ops fold away in the unrolled trace.
    a_a = tl.zeros((BLOCK, WA), dtype=tl.float32)
    a_b = tl.zeros((BLOCK, WB), dtype=tl.float32)
    # conditioning features into cols < NC; padding stays exactly 0
    if (IN_A_MASK >> 0) & 1:
        a_a = tl.load(
            x_ptr + rows[:, None] * x_stride + c_a[None, :],
            mask=rmask[:, None] & (c_a[None, :] < NC), other=0.0,
        )
    else:
        a_b = tl.load(
            x_ptr + rows[:, None] * x_stride + c_b[None, :],
            mask=rmask[:, None] & (c_b[None, :] < NC), other=0.0,
        )

    for l in tl.static_range(N_LAYERS):
        if l == TRUNK_AT:
            # splice u-features into cols [EMB_W, EMB_W + NU) of whichever
            # tile carries the embed readout output
            if (IN_A_MASK >> l) & 1:
                u = tl.load(
                    x_ptr + rows[:, None] * x_stride + (c_a[None, :] - EMB_W + NC),
                    mask=rmask[:, None]
                    & (c_a[None, :] >= EMB_W) & (c_a[None, :] < EMB_W + NU),
                    other=0.0,
                )
                a_a = tl.where(c_a[None, :] < EMB_W, a_a, 0.0) + u
            else:
                u = tl.load(
                    x_ptr + rows[:, None] * x_stride + (c_b[None, :] - EMB_W + NC),
                    mask=rmask[:, None]
                    & (c_b[None, :] >= EMB_W) & (c_b[None, :] < EMB_W + NU),
                    other=0.0,
                )
                a_b = tl.where(c_b[None, :] < EMB_W, a_b, 0.0) + u
        if l == N_LAYERS - 1:
            # final Linear(., N_OUT): one weighted sum per output column --
            # no dot, no column extraction. Each weight column is a stride-WB
            # gather. For N_OUT == 1 this reduces to a plain scalar path.
            # Column j is the module's own output column j, and the
            # row-major `rows * N_OUT + j` store preserves that order in the
            # [n, N_OUT] output buffer.
            for j in tl.static_range(N_OUT):
                bj = tl.load(b_ptr + l * WB + j)
                if (IN_A_MASK >> l) & 1:
                    wv = tl.load(w_ptr + l * WB * WB + c_a * WB + j)
                    o = tl.sum(a_a * wv[None, :], axis=1) + bj
                else:
                    wv = tl.load(w_ptr + l * WB * WB + c_b * WB + j)
                    o = tl.sum(a_b * wv[None, :], axis=1) + bj
                if CLAMP_KIND == 1:
                    o = tl.minimum(tl.maximum(o, 0.0), 1.0)
                if CLAMP_KIND == 2:
                    # reflect-then-clamp; order is load-bearing (raw > 2 -> 0)
                    o = tl.where(o > 1.0, 2.0 - o, o)
                    o = tl.minimum(tl.maximum(o, 0.0), 1.0)
                tl.store(out_ptr + rows * N_OUT + j, o, mask=rmask)
        else:
            # LN width: raw value, eight bits per layer -- independent of
            # the embedding width, so any per-layer LayerNorm width fits.
            nw = (LN_W_MASK >> (8 * l)) & 255
            # IEEE fp32 dot by default (parity with eager to fp32 rounding).
            # TF32=True routes the dot through the tensor cores instead.
            if (IN_A_MASK >> l) & 1:
                if (OUT_A_MASK >> l) & 1:
                    wmat = tl.load(w_ptr + l * WB * WB + c_a[:, None] * WB + c_a[None, :])
                    a_a = tl.dot(a_a, wmat, allow_tf32=TF32)
                    a_a = _bias_ln_act(a_a, c_a, l, b_ptr, g_ptr, bb_ptr, WB, nw,
                                       (LN_MASK >> l) & 1, (ACT_MASK >> l) & 1,
                                       ACT_KIND)
                else:
                    wmat = tl.load(w_ptr + l * WB * WB + c_a[:, None] * WB + c_b[None, :])
                    a_b = tl.dot(a_a, wmat, allow_tf32=TF32)
                    a_b = _bias_ln_act(a_b, c_b, l, b_ptr, g_ptr, bb_ptr, WB, nw,
                                       (LN_MASK >> l) & 1, (ACT_MASK >> l) & 1,
                                       ACT_KIND)
            else:
                if (OUT_A_MASK >> l) & 1:
                    wmat = tl.load(w_ptr + l * WB * WB + c_b[:, None] * WB + c_a[None, :])
                    a_a = tl.dot(a_b, wmat, allow_tf32=TF32)
                    a_a = _bias_ln_act(a_a, c_a, l, b_ptr, g_ptr, bb_ptr, WB, nw,
                                       (LN_MASK >> l) & 1, (ACT_MASK >> l) & 1,
                                       ACT_KIND)
                else:
                    wmat = tl.load(w_ptr + l * WB * WB + c_b[:, None] * WB + c_b[None, :])
                    a_b = tl.dot(a_b, wmat, allow_tf32=TF32)
                    a_b = _bias_ln_act(a_b, c_b, l, b_ptr, g_ptr, bb_ptr, WB, nw,
                                       (LN_MASK >> l) & 1, (ACT_MASK >> l) & 1,
                                       ACT_KIND)
