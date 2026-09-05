"""
Whole-net fused CUDA kernel for the generator MLPs.

Motivation: each generator's forward pass is a short chain of narrow
Linear/LayerNorm/activation blocks operating on a small per-lane feature
vector. Run eagerly at large batch sizes, each block streams its activation
tensor to main memory and back, so the cost is dominated by memory traffic
even though every net's weights together are small enough to stay
cache-resident. This module runs each generator's ENTIRE forward pass in a
single Triton kernel: activations live in registers for the whole layer chain,
weights are read through L2, and main-memory traffic collapses to one read of
the input features and one write of the (1- or n_out-column) output per lane.

Scope and contract:

- Inference-only, opt-in via PHINGAN_FUSED_MLP=1 (read by G4hIonisation.to()
  on CUDA, mirroring the phase-compile latch idiom).
  Checkpoint-compatible: weights are packed from the loaded nn.Module; nothing
  about training or the checkpoints changes.
- The kernel reproduces forward() to fp32 rounding (tl.dot in IEEE fp32, no
  TF32): summation order differs from cuBLAS, so outputs match to a small
  relative tolerance, not bitwise. Physics is unaffected (a shift at that
  scale is far below the WGAN-GP training residual); the parity harness
  (benchmark/runtime_profiling/test_fused_mlp.py) gates this on real
  checkpoints before any production run.
- PHINGAN_FUSED_MLP_TF32 (read once at enable time, only meaningful with
  the fused path on) additionally routes the dots through the tensor cores
  (TF32: 10-bit-mantissa inputs, fp32 accumulate). Value "1" = all stages;
  a ':'-separated stage list (e.g. "secondary") scopes it. TF32 is REJECTED
  as a default: on the continuous net it buys negligible throughput at real
  accuracy risk, and on the secondary net, while it does speed up its phase
  and passes the optimization gate, it pushes one physics statistic (THETA's
  KS) further from its calibrated null than the IEEE path does. IEEE ships.
  TF32 abandons the tight eager parity that IEEE mode holds; acceptance
  shifts onto the optimization gate
  (`benchmark/runtime_profiling/verify_physics_gate.py --gate b`). The
  actual per-generator mode is recorded as `_fused_mlp_tf32` for truthful
  report latching.
- Both stages share one structural template, introspected from the
  module rather than hard-coded:
      embed:  Linear(nc, E)+LN+act, [Linear(E, E)+LN+act] x k, Linear(E, E)
      concat: [embed_out(E) | u_features(nu)]
      trunk:  [Linear(., mE)+LN+act] x t, Linear(mE, E)+LN+act,
              Linear(E, n_out) [+ terminal epilogue]
  with a single activation kind per net (SiLU or ReLU, see _ACT_KINDS) and
  E = 25 for the continuous net and E = 5 for the secondary net. The
  readout width n_out and the optional terminal epilogue
  (`clamp_kind`) are read off the module: `ContinuousGenerator` is
  scalar and unclamped (clamp_kind=0); `SecondaryGenerator` is
  single-column and terminates in HardtanhReflectTop (clamp_kind=2 — reflect
  z>1 spill as 2-z, then clamp to [0, 1]); a plain Hardtanh(0, 1) terminal is
  clamp_kind=1.
  try_enable_fused_forward() refuses
  (returns False, logs why) on any structural mismatch — new architectures
  fall back to the eager forward instead of silently computing wrong numbers.
- Exposed to torch.compile as a custom op (phingan::fused_mlp) with a
  registered fake, so the compiled process phases trace straight through it
  and CUDA graphs can capture the launch.

The pure-torch emulator (_emulate_packed_forward) mirrors the kernel's padded
math exactly and runs on CPU — it is the unit-test surface for the packing
and the padding/masking logic (tests/physics/test_fused_mlp.py).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import torch
from torch import nn

log = logging.getLogger(__name__)

# Padded on-chip tile widths the kernel supports. Each net gets a PAIR: the
# narrow tile `tile_a` and the wide tile `tile_b`, and every layer is on one
# or the other. Two tiles, not three, so the kernel body stays a 2x2 in/out
# cascade instead of a 3x3 one.
# 16 is tl.dot's minimum M/N/K, i.e. the ladder floor -- never an 8-tile.
_TILE_LADDER = (16, 32, 64, 128)
_LN_EPS = 1e-5  # torch.nn.LayerNorm default
_BLOCK = 128
_NUM_WARPS = 8

# Generators eligible for fusion. Physics-MC samplers and anything else fall
# through try_enable_fused_forward unchanged -- which is why the analytic
# PhysicsLengthGenerator is absent rather than refused.
#
# This is a NAME check, so membership alone proves nothing about the
# architecture: only build_spec's
# structural validation distinguishes them, and it refuses (logs, falls back to
# eager) on any mismatch. Never relax that to "the class name is known".
# Kept mutable: gate B's vacuity control empties this set to force the
# unfusable-class refusal path.
_FUSABLE_CLASS_NAMES = {
    "ContinuousGenerator", "SecondaryGenerator",
}

# Shipped per-class tile floor, keyed like _LAUNCH_CONFIG. Empty unless a
# sweep elects a wider-than-minimal tile for a net; the sweep's own arms
# force a tile via build_spec(min_tile=...) directly and never ship that way.
_MIN_TILE: dict[str, int] = {}

# Activation kinds the fused kernel implements. The kernel selects on the
# constexpr ACT_KIND; a net whose activation is not here is refused, never
# approximated with a different one.
_ACT_KINDS = {"silu": 0, "relu": 1}


def resolve_activation(name: str) -> int:
    try:
        return _ACT_KINDS[name]
    except KeyError:
        raise ValueError(
            f"unsupported activation {name!r}; fused kernel implements "
            f"{sorted(_ACT_KINDS)}") from None


@dataclass
class _FusedSpec:
    """Per-network compile-time configuration + packed parameters."""
    n_cond: int              # conditioning features (x_scaler output width)
    n_u: int                 # u-derived features
    n_layers: int            # total Linear layers (embed + trunk)
    trunk_at: int            # index of the first trunk Linear (= #embed Linears)
    emb_w: int               # embedding width E (used for the u-splice)
    ln_mask: int             # bit l set -> LayerNorm after Linear l
    act_mask: int            # bit l set -> activation after Linear l
    # Eight bits per layer, bits [8l, 8l+8): layer l's raw LayerNorm width
    # (0 where no LN), stored directly rather than as a multiple of the
    # embedding width so any per-layer width is expressible. Widths are
    # bounded by the 128 tile ceiling, so a byte always fits.
    ln_w_mask: int
    weights: torch.Tensor = field(repr=False)   # [L, tb, tb] fp32, W[l,:in,:out]
    bias: torch.Tensor = field(repr=False)      # [L, tb] fp32 (zeros where absent)
    gamma: torch.Tensor = field(repr=False)     # [L, tb] fp32 (zeros where no LN)
    beta: torch.Tensor = field(repr=False)      # [L, tb] fp32
    act_kind: int = 0        # _ACT_KINDS value; one activation kind per net
    # Readout width: the final Linear emits n_out columns, taken in the
    # module's own column order (`SecondaryFeats`' single E_SEC_IDX column
    # for the secondary net). n_out=1 reduces to a plain scalar path
    # exactly.
    n_out: int = 1
    # Terminal epilogue kind: 0 = none, 1 = Hardtanh(0, 1) clamp,
    # 2 = HardtanhReflectTop's eval rule (reflect z>1 as 2-z, then clamp).
    # Any other terminal is refused by _extract_layers rather than approximated.
    clamp_kind: int = 0
    tile_a: int = 32         # narrow tile width (kernel constexpr WA)
    tile_b: int = 64         # wide tile width; also the parameter storage stride
    in_a_mask: int = 0       # bit l set -> layer l's input tile is tile_a (else tile_b)
    out_a_mask: int = 0      # bit l set -> layer l's output tile is tile_a (else tile_b)
    tf32: bool = False       # tensor-core dots (PHINGAN_FUSED_MLP_TF32 latch)
    launch_block: int = _BLOCK   # per-stage swept config; v1 default fallback
    launch_warps: int = _NUM_WARPS
    # Whether the REAL eager module squeezes a scalar (n_out==1) readout to
    # [B] itself (e.g. ContinuousGenerator.forward ends `.flatten()`) or
    # returns the trailing Linear's own [B, 1] unchanged (e.g.
    # SecondaryGenerator.forward — no flatten in its own body). Measured
    # structurally by build_spec (a tiny real eager forward, see
    # `_probe_squeeze_out`), NEVER inferred from n_out or the class name —
    # two n_out==1 nets can disagree, which is exactly why fused_forward
    # cannot simply assume n_out==1 always means [B].
    # Default True (squeeze) for specs built without going through
    # build_spec's probe (synthetic dataclasses in tests).
    squeeze_out: bool = True


# Registry keyed by an int the custom op can carry as a plain argument.
_SPEC_REGISTRY: dict[int, _FusedSpec] = {}


# ---------------------------------------------------------------------------
# Structure introspection & packing
# ---------------------------------------------------------------------------

def _leaf_modules(seq: nn.Module) -> list[nn.Module]:
    """Leaf modules of (possibly nested) nn.Sequential, in execution order."""
    out = []
    for m in seq.children():
        if isinstance(m, nn.Sequential):
            out.extend(_leaf_modules(m))
        else:
            out.append(m)
    return out


_ACT_MODULES = {nn.SiLU: "silu", nn.ReLU: "relu"}


def _extract_layers(gen) -> tuple[list[dict], str, int]:
    """([{linear, ln|None, act:bool}] in execution order, activation name,
    clamp_kind).

    `clamp_kind` is 1 iff the net ends in `nn.Hardtanh(0, 1)`, which the
    kernel folds into its epilogue as a plain clamp, or 2 iff it ends in
    `HardtanhReflectTop`: its eval rule reflects z>1 spill (z -> 2-z), then
    clamps to [0, 1]. A Hardtanh/HardtanhReflectTop anywhere but the very
    last leaf, or a Hardtanh with any other bounds, is refused: both would be
    a different network, and silently dropping or re-bounding it would move
    the physics.
    """
    leaves = _leaf_modules(gen.embed) + _leaf_modules(gen.model)
    layers: list[dict] = []
    kinds: set[str] = set()
    clamp = 0
    for m in leaves:
        if isinstance(m, nn.Linear):
            layers.append({"linear": m, "ln": None, "act": False})
        elif isinstance(m, nn.LayerNorm):
            if not layers or layers[-1]["ln"] is not None:
                raise ValueError("LayerNorm not directly after a Linear")
            layers[-1]["ln"] = m
        elif type(m) in _ACT_MODULES:
            if not layers:
                raise ValueError("activation before any Linear")
            layers[-1]["act"] = True
            kinds.add(_ACT_MODULES[type(m)])
        elif isinstance(m, nn.Hardtanh):
            if m.min_val != 0 or m.max_val != 1:
                raise ValueError(f"only Hardtanh(0, 1) is implemented; got "
                                 f"({m.min_val}, {m.max_val})")
            if m is not leaves[-1]:
                raise ValueError("Hardtanh must be terminal (it is the output "
                                 "clamp); found one mid-network")
            clamp = 1
        elif isinstance(m, nn.Flatten):
            continue  # embed readout Flatten — a no-op on [B, E]
        elif type(m).__name__ == "HardtanhReflectTop":
            # Inference-only kernel => the EVAL rule: reflect z>1 spill
            # (z -> 2-z), then clamp to [0, 1]. Reflect-before-clamp order
            # is load-bearing (raw > 2 must floor at 0).
            if m is not leaves[-1]:
                raise ValueError("HardtanhReflectTop must be terminal (it is "
                                 "the output activation); found one mid-network")
            clamp = 2
        elif type(m).__name__ in ("StretchedSigmoid", "HardtanhOpenTop"):
            raise ValueError(f"terminal {type(m).__name__} is not implemented in "
                             "the fused kernel epilogue; falling back to eager")
        elif isinstance(m, nn.Sigmoid):
            # SecondaryGenerator(output_activation="sigmoid"). The kernel
            # epilogue implements the Hardtanh(0, 1) clamp and the
            # HardtanhReflectTop reflect rule; a sigmoid epilogue is not one
            # of those. Refuse -> eager.
            raise ValueError("terminal Sigmoid is not implemented in the fused "
                             "kernel epilogue (only the Hardtanh(0, 1) clamp and "
                             "the HardtanhReflectTop reflect rule); falling back "
                             "to eager")
        else:
            raise ValueError(
                f"unsupported module in fused path: {type(m).__name__} "
                f"(only Linear/LayerNorm/{'/'.join(sorted(_ACT_MODULES.values()))}"
                "/Flatten)")
    if len(kinds) > 1:
        raise ValueError(f"net mixes activation kinds {sorted(kinds)}; "
                         "the fused kernel has one ACT_KIND per net")
    return layers, (kinds.pop() if kinds else "silu"), clamp


def _choose_tiles(layers: list[dict], emb_w: int, n_u: int,
                  min_tile: int | None = None) -> tuple[int, int]:
    """Smallest (narrow, wide) pair from _TILE_LADDER fitting every layer
    width and the u/latent splice width, floored at `min_tile` if given."""
    widths = {emb_w + n_u}
    for layer in layers:
        widths.add(layer["linear"].in_features)
        widths.add(layer["linear"].out_features)
    need = max(widths)
    if min_tile is not None:
        need = max(need, min_tile)
    tile_b = next((t for t in _TILE_LADDER if t >= need), None)
    if tile_b is None:
        raise ValueError(f"widest layer {need} exceeds the "
                         f"{_TILE_LADDER[-1]} tile")
    narrower = [t for t in _TILE_LADDER
               if t < tile_b and (min_tile is None or t >= min_tile)]
    return (narrower[-1] if narrower else tile_b), tile_b


def _tile_masks(layers: list[dict], trunk_at: int, emb_w: int, n_u: int,
                tile_a: int, tile_b: int) -> tuple[int, int]:
    """Per-layer padded tile-width masks (bit l set -> tile_a, else tile_b).

    Width-driven: each side gets the smaller of the {tile_a, tile_b} pair that
    fits the real layer shape. One structural exception: the u-splice happens
    on the embed readout's OUTPUT tile, so if emb_w + n_u doesn't fit tile_a
    the readout (and hence the trunk-in layer's input) must stay on the wide
    tile. A net whose embedding width and u-feature count together stay under
    tile_a splices on the narrow tile; one whose combined width exceeds it
    (e.g. the continuous net, whose n_u equals its embedding width) lands on
    the wide tile instead.
    """
    in_a = out_a = 0
    for l, layer in enumerate(layers):
        if layer["linear"].in_features <= tile_a:
            in_a |= 1 << l
        if layer["linear"].out_features <= tile_a:
            out_a |= 1 << l
    if emb_w + n_u > tile_a:
        out_a &= ~(1 << (trunk_at - 1))
        # always redundant given trunk_at detection (fin == emb_w + n_u >
        # tile_a never sets the bit); kept for symmetry with the continuity rule
        in_a &= ~(1 << trunk_at)
    return in_a, out_a


def _validate_tile_continuity(layers: list[dict], in_a_mask: int,
                              out_a_mask: int, emb_w: int, n_u: int,
                              trunk_at: int, tile_a: int, tile_b: int) -> None:
    """Every real layer shape must fit its declared tile, consecutive tiles
    must agree (layer l's input tile is layer l-1's output tile), and the
    u-splice must fit the tile it lands on."""
    for l, layer in enumerate(layers):
        fin, fout = layer["linear"].in_features, layer["linear"].out_features
        in_pad = tile_a if (in_a_mask >> l) & 1 else tile_b
        out_pad = tile_a if (out_a_mask >> l) & 1 else tile_b
        if fin > in_pad or fout > out_pad:
            raise ValueError(f"layer {l}: {fin}->{fout} exceeds tile "
                             f"{in_pad}x{out_pad}")
        if l > 0:
            prev_out = tile_a if (out_a_mask >> (l - 1)) & 1 else tile_b
            if in_pad != prev_out:
                raise ValueError(f"layer {l}: tile continuity mismatch "
                                 f"({prev_out} out -> {in_pad} in)")
    if (out_a_mask >> (trunk_at - 1)) & 1 and emb_w + n_u > tile_a:
        raise ValueError(f"u-splice does not fit the {tile_a} tile "
                         f"(emb_w {emb_w} + n_u {n_u} > {tile_a})")


def _probe_squeeze_out(gen, n_cond: int, dev: torch.device) -> bool:
    """Measure, structurally, whether THIS module's own eager forward
    squeezes a scalar readout to [B] or leaves it [B, 1].

    Not a class-name check (`ContinuousGenerator` and `SecondaryGenerator`
    disagree on this despite both being scalar-output WGAN nets: the
    continuous forward ends `.flatten()`, the secondary's does not) — the
    only thing that can never disagree with what `fused_forward` has to
    reproduce is running the module's REAL `forward` once. Two rows, same
    `[m, n_cond]` convention the GPU parity harness (`test_fused_mlp.py`)
    itself feeds both eager and fused paths, so this measurement and the
    harness's own comparison can never talk past each other.

    Modules that do not override `forward` at all (`nn.Module`'s abstract
    stub raises `NotImplementedError`) are structural test fixtures with no
    real eager convention to measure — falls back to the dataclass default
    (`True`, i.e. squeeze) rather than raising, so build_spec keeps working
    on those. Any other forward-time failure (e.g. a fixture whose forward
    assumes inputs this probe does not construct) falls back the same way:
    this is a best-effort measurement, and refusing to build a spec over it
    would be a worse failure mode than defaulting.
    """
    if type(gen).forward is nn.Module.forward:
        return True
    try:
        # fork_rng: this is a build-time probe, not a physics draw. Without
        # forking, any randn/rand the real forward consumes (e.g. a
        # generator's internal noise draw) advances the seeded global RNG
        # stream, so a fused-vs-eager comparison run under one seed would no
        # longer see the same per-lane draws as an equivalent eager-only run.
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            with torch.no_grad():
                probe = torch.zeros(2, n_cond, dtype=torch.float32, device=dev)
                out = type(gen).forward(gen, probe)
    except Exception:
        return True
    if out.ndim not in (1, 2):
        raise ValueError(f"{type(gen).__name__}: eager forward returned "
                         f"{out.ndim}-D output on a [2, {n_cond}] probe; "
                         "only 1-D or 2-D readouts are supported")
    return out.ndim == 1


def _shape_like_eager(kernel_out: torch.Tensor, spec: _FusedSpec) -> torch.Tensor:
    """Reshape a native [B, n_out] kernel/emulator output to match the real
    eager module's own return convention (`spec.squeeze_out`, measured
    structurally by `_probe_squeeze_out` — never assumed from n_out). Shared
    by `_emulate_packed_forward` and `fused_forward` so both stay coupled to
    the same single source of truth."""
    return kernel_out[:, 0] if spec.squeeze_out else kernel_out


def build_spec(gen, min_tile: int | None = None) -> _FusedSpec:
    """Introspect a generator and pack its parameters for the fused kernel.

    `min_tile` floors both tiles of the chosen pair -- used by the launch
    sweep to force a wider-than-minimal pair, and by the shipped `_MIN_TILE`
    per-class floor when a sweep elects one. Raises ValueError on any
    structural assumption violation — callers treat that as "not fusable",
    never as "fuse approximately".
    """
    module_cls = type(gen).__name__
    net_module = type(gen).__module__
    # Conditioning / u-feature widths: a net's module may declare
    # `_N_INPUT_FEATURES` / `_N_U_FEATURES` as module constants; if it does
    # not, both fall back to the module's own structure (embed's first
    # Linear, and the net's latent width). Structural derivation is safe
    # because it reads the SAME objects the packing reads — it cannot
    # disagree with the weights.
    import importlib
    mod = importlib.import_module(net_module)
    n_cond = getattr(mod, "_N_INPUT_FEATURES", None)
    if n_cond is None:
        n_cond = _leaf_modules(gen.embed)[0].in_features
    n_u = getattr(mod, "_N_U_FEATURES", None)
    if n_u is None:
        n_u = int(gen.n_noise)
    n_cond, n_u = int(n_cond), int(n_u)

    layers, act_name, clamp = _extract_layers(gen)
    n_layers = len(layers)
    if n_layers > 12:
        raise ValueError(f"{module_cls}: {n_layers} Linear layers > fused unroll budget")

    emb_w = int(gen.n_embedding)
    if min_tile is None:
        min_tile = _MIN_TILE.get(module_cls)
    # Tile pair first: the widest real layer decides the storage stride, and
    # every later width check is against it.
    tile_a, tile_b = _choose_tiles(layers, emb_w, n_u, min_tile)
    # Conservative headroom floor, not derived from any LayerNorm-width
    # encoding: `ln_w_mask` stores each layer's LayerNorm width directly (8
    # bits per layer, so up to 255), independent of emb_w. The binding
    # LayerNorm bound is elsewhere and exact — an LN width equals its
    # Linear's out_features (checked below), and every out_features is checked
    # against tile_b here and against its own declared tile in
    # _validate_tile_continuity, so `nw` can never exceed the tile it is
    # masked over. What survives here is a floor on E itself: a net with
    # emb_w > tile_b / 2 is refused even when all its LayerNorms are E-wide
    # and would in fact fit. That over-refusal is safe (the caller falls back
    # to the eager forward), and relaxing it is a separate decision from how
    # LN widths are encoded.
    if emb_w * 2 > tile_b:
        raise ValueError(f"n_embedding {emb_w} too wide for the {tile_b}-padded kernel")

    # trunk starts at the Linear whose in_features == emb_w + n_u
    trunk_at = None
    for i, l in enumerate(layers):
        if l["linear"].in_features == emb_w + n_u and i > 0:
            trunk_at = i
            break
    if trunk_at is None:
        raise ValueError(f"{module_cls}: could not locate trunk start "
                         f"(no Linear with in_features == {emb_w + n_u})")
    # embed readout (layer trunk_at - 1) must be a bare Linear E -> E
    readout = layers[trunk_at - 1]
    if (readout["linear"].out_features != emb_w or readout["ln"] is not None
            or readout["act"]):
        raise ValueError(f"{module_cls}: embed readout is not a bare Linear({emb_w})")
    # Readout: n_out columns, emitted in the module's own column order. The
    # epilogue gathers ONE weight column per output (no dot), so the readout
    # must fit the narrow tile — a wider one is a different kernel shape and
    # is refused rather than silently truncated.
    n_out = layers[-1]["linear"].out_features
    if n_out < 1 or n_out > tile_a:
        raise ValueError(f"{module_cls}: final Linear out_features {n_out} "
                         f"outside [1, {tile_a}]")
    if layers[-1]["ln"] is not None or layers[-1]["act"]:
        raise ValueError(f"{module_cls}: final layer must be a bare "
                         f"Linear(., {n_out}) plus an optional clamp")

    dev = next(gen.parameters()).device
    weights = torch.zeros(n_layers, tile_b, tile_b, dtype=torch.float32, device=dev)
    bias = torch.zeros(n_layers, tile_b, dtype=torch.float32, device=dev)
    gamma = torch.zeros(n_layers, tile_b, dtype=torch.float32, device=dev)
    beta = torch.zeros(n_layers, tile_b, dtype=torch.float32, device=dev)
    ln_mask = act_mask = ln_w_mask = 0

    for l, layer in enumerate(layers):
        lin: nn.Linear = layer["linear"]
        fin, fout = lin.in_features, lin.out_features
        if fin > tile_b or fout > tile_b:
            raise ValueError(f"layer {l}: {fin}->{fout} exceeds padded width {tile_b}")
        weights[l, :fin, :fout] = lin.weight.detach().t().to(torch.float32)
        if lin.bias is not None:
            bias[l, :fout] = lin.bias.detach().to(torch.float32)
        if layer["ln"] is not None:
            ln: nn.LayerNorm = layer["ln"]
            (n,) = ln.normalized_shape
            if n != fout:
                raise ValueError(f"layer {l}: LN width {n} != Linear out {fout}")
            ln_w_mask |= n << (8 * l)
            gamma[l, :n] = ln.weight.detach().to(torch.float32)
            beta[l, :n] = ln.bias.detach().to(torch.float32)
            ln_mask |= 1 << l
        if layer["act"]:
            act_mask |= 1 << l

    in_a_mask, out_a_mask = _tile_masks(layers, trunk_at, emb_w, n_u,
                                        tile_a, tile_b)
    _validate_tile_continuity(layers, in_a_mask, out_a_mask, emb_w, n_u,
                              trunk_at, tile_a, tile_b)
    blk, warps = _LAUNCH_CONFIG.get(module_cls, (_BLOCK, _NUM_WARPS))
    squeeze_out = _probe_squeeze_out(gen, n_cond, dev)
    return _FusedSpec(
        n_cond=n_cond, n_u=n_u, n_layers=n_layers, trunk_at=trunk_at,
        emb_w=emb_w, ln_mask=ln_mask, act_mask=act_mask,
        ln_w_mask=ln_w_mask,
        weights=weights, bias=bias, gamma=gamma, beta=beta,
        act_kind=resolve_activation(act_name),
        n_out=n_out, clamp_kind=clamp,
        tile_a=tile_a, tile_b=tile_b,
        in_a_mask=in_a_mask, out_a_mask=out_a_mask,
        launch_block=blk, launch_warps=warps,
        squeeze_out=squeeze_out,
    )


# ---------------------------------------------------------------------------
# Pure-torch emulator — the kernel's math, testable on CPU
# ---------------------------------------------------------------------------

def _emulate_packed_forward(spec: _FusedSpec, x: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the kernel over the PACKED parameters.

    x: [B, n_cond + n_u] fp32 (conditioning features then u-features).
    Returns [B] fp32 when spec.n_out == 1, else [B, n_out] in the module's own
    column order. Mirrors the per-layer tile widths (in_a/out_a masks),
    padding, splice and the optional output clamp exactly; used by the unit
    tests to validate packing + masks, and by the GPU parity harness as a
    cross-check.
    """
    B = x.shape[0]

    def _pad(l: int, mask: int) -> int:
        return spec.tile_a if (mask >> l) & 1 else spec.tile_b

    a = torch.zeros(B, _pad(0, spec.in_a_mask), dtype=torch.float32,
                    device=x.device)
    a[:, :spec.n_cond] = x[:, :spec.n_cond]
    for l in range(spec.n_layers):
        if l == spec.trunk_at:
            a[:, spec.emb_w:] = 0.0
            a[:, spec.emb_w:spec.emb_w + spec.n_u] = x[:, spec.n_cond:]
        wi, wo = _pad(l, spec.in_a_mask), _pad(l, spec.out_a_mask)
        a = a @ spec.weights[l, :wi, :wo] + spec.bias[l, :wo]
        if (spec.ln_mask >> l) & 1:
            n = (spec.ln_w_mask >> (8 * l)) & 255
            v = a[:, :n]
            mean = v.mean(dim=1, keepdim=True)
            var = ((v - mean) ** 2).mean(dim=1, keepdim=True)
            a = torch.zeros_like(a)
            a[:, :n] = ((v - mean) / torch.sqrt(var + _LN_EPS)
                        * spec.gamma[l, :n] + spec.beta[l, :n])
        if (spec.act_mask >> l) & 1:
            # Mirrors the kernel's constexpr ACT_KIND switch exactly.
            if spec.act_kind == _ACT_KINDS["silu"]:
                a = a * torch.sigmoid(a)
            else:
                a = torch.clamp(a, min=0.0)
    out = a[:, :spec.n_out]
    if spec.clamp_kind == 1:
        out = out.clamp(0.0, 1.0)
    elif spec.clamp_kind == 2:
        out = torch.where(out > 1.0, 2.0 - out, out).clamp(0.0, 1.0)
    return _shape_like_eager(out, spec)


# ---------------------------------------------------------------------------
# Triton kernel + custom op (CUDA only; imported lazily)
# ---------------------------------------------------------------------------

def _build_kernel():
    # Lazy import: the kernel module imports triton at module level (a
    # @triton.jit function resolves names from its module globals, so `tl`
    # cannot live in a closure), and CPU-only machines must not need triton.
    from physics.interfaces._fused_mlp_kernel import fused_mlp_kernel
    return fused_mlp_kernel


# Per-net launch configs (BLOCK, num_warps), measured by the --sweep mode of
# benchmark/runtime_profiling/test_fused_mlp.py. Classes not listed fall back
# to (_BLOCK, _NUM_WARPS). Re-sweep after any kernel-structure change — the
# optimum follows the instruction mix AND the net's tile pair.
#
# `SecondaryGenerator`'s (BLOCK=128, num_warps=2) is the elected config for
# its (16, 16) tile pair, measured 0.134 ms per launch at 1,048,576 lanes on
# an RTX 6000 Ada, against 0.518 ms for the (BLOCK=32, num_warps=16) anchor
# config swept in the same (16, 16) tile arm -- see
# measurements/fused_mlp_sweep/sweep_r1.log and sweep_r2.log for the full
# grid and the repeat pass. `_MIN_TILE` stays empty for this net: no floor
# override is needed for the shipped nets. `ContinuousGenerator`'s
# (BLOCK=64, num_warps=8) is not covered by the sweep in
# measurements/fused_mlp_sweep/; a wider sweep is a candidate for a future
# measurement, not a known regression.
#
# Configs are swept at one lane count (1,048,576 lanes). Production decays
# the alive lane count through power-of-two re-compaction, so a config
# elected here is the optimum at that scale and not necessarily at smaller
# ones.
_LAUNCH_CONFIG: dict[str, tuple[int, int]] = {
    "ContinuousGenerator":  (64, 8),
    "SecondaryGenerator":   (128, 2),
}


def _launch(spec: _FusedSpec, x: torch.Tensor, block: int | None = None,
            num_warps: int | None = None) -> torch.Tensor:
    import triton
    kernel = _build_kernel()
    n = x.shape[0]
    out = torch.empty(n, spec.n_out, dtype=x.dtype, device=x.device)
    if n == 0:
        return out
    block = spec.launch_block if block is None else block
    warps = spec.launch_warps if num_warps is None else num_warps
    grid = (triton.cdiv(n, block),)
    kernel[grid](
        x, out, spec.weights, spec.bias, spec.gamma, spec.beta,
        n, x.stride(0),
        NC=spec.n_cond, NU=spec.n_u,
        N_LAYERS=spec.n_layers, TRUNK_AT=spec.trunk_at,
        LN_MASK=spec.ln_mask, ACT_MASK=spec.act_mask,
        LN_W_MASK=spec.ln_w_mask,
        ACT_KIND=spec.act_kind,
        IN_A_MASK=spec.in_a_mask, OUT_A_MASK=spec.out_a_mask,
        EMB_W=spec.emb_w,
        BLOCK=block, WA=spec.tile_a, WB=spec.tile_b, TF32=spec.tf32,
        N_OUT=spec.n_out, CLAMP_KIND=spec.clamp_kind,
        num_warps=warps,
    )
    return out


# Custom op so the compiled process phases trace through the launch (dynamo
# sees one opaque node with a registered fake; CUDA graphs capture the kernel).
_OPS_REGISTERED = False


def _register_ops():
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return
    from torch.library import custom_op

    # n_out rides on the signature (rather than being looked up from the
    # registry) so the registered fake can shape the output without
    # dereferencing spec_id during tracing.
    @custom_op("phingan::fused_mlp", mutates_args=())
    def fused_mlp(x: torch.Tensor, spec_id: int,
                  n_out: int) -> torch.Tensor:
        return _launch(_SPEC_REGISTRY[spec_id], x)

    @fused_mlp.register_fake
    def _(x: torch.Tensor, spec_id: int, n_out: int) -> torch.Tensor:
        return x.new_empty(x.shape[0], n_out)

    _OPS_REGISTERED = True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def try_enable_fused_forward(gen) -> bool:
    """Shadow gen.forward with the fused-kernel version (instance attribute,
    same idiom as the phase compiles). Returns True iff enabled. Refuses —
    with a log line, falling back to the eager forward — on physics-MC
    samplers, unknown architectures, non-CUDA devices, or missing triton."""
    if getattr(gen, "_fused_mlp", False):
        return True
    if type(gen).__name__ not in _FUSABLE_CLASS_NAMES:
        return False
    try:
        dev = next(gen.parameters()).device
    except StopIteration:
        return False
    if dev.type != "cuda":
        log.warning("fused MLP requested but %s is on %s — skipping",
                    type(gen).__name__, dev)
        return False
    try:
        import triton  # noqa: F401
        spec = build_spec(gen)
        _register_ops()
    except Exception as err:
        log.warning("fused MLP disabled for %s: %s", type(gen).__name__, err)
        return False

    # TF32 latch, read once at enable time (same idiom as PHINGAN_FUSED_MLP
    # itself): tensor-core dots trade bitwise-vs-eager parity for speed —
    # acceptance moves to the physics gate. Per-stage: the env value may
    # scope TF32 to a stage subset (see fused_mlp_tf32_requested). Recorded
    # on the generator so the profiling/gate reports state the ACTUAL kernel
    # mode, not the request.
    spec.tf32 = fused_mlp_tf32_requested(type(gen).__name__)
    gen._fused_mlp_tf32 = spec.tf32

    spec_id = id(gen)
    _SPEC_REGISTRY[spec_id] = spec
    n_cond = spec.n_cond
    u_features = getattr(type(gen), "_u_features", None)

    def fused_forward(labels: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        labels = labels.view(-1, n_cond)
        if u_features is None:
            # WGAN generators: the "u-features" ARE the latent, drawn the
            # same way and in the same stream position as the eager forward
            # (randn before embed; embed consumes no RNG). That keeps a
            # seeded fused-vs-eager comparison paired per lane.
            tail = torch.randn(labels.shape[0], spec.n_u,
                               device=labels.device, dtype=labels.dtype)
        else:
            if u is None:
                u = torch.rand(labels.shape[0], device=labels.device,
                               dtype=labels.dtype)
            tail = u_features(u)
        x = torch.cat([labels, tail], dim=-1).contiguous().float()
        out = torch.ops.phingan.fused_mlp(x, spec_id, spec.n_out)
        # Match the eager forward's OWN shape convention exactly --
        # spec.squeeze_out was measured structurally off a real eager
        # forward in build_spec (_probe_squeeze_out), never assumed from
        # n_out == 1 (ContinuousGenerator and SecondaryGenerator are both
        # scalar-output but disagree on this: see _shape_like_eager).
        return _shape_like_eager(out, spec)

    gen.forward = fused_forward  # instance-dict shadow; class forward untouched
    gen._fused_mlp = True
    log.info("fused MLP enabled for %s (%d layers, %d params packed)",
             type(gen).__name__, spec.n_layers,
             sum(p.numel() for p in gen.parameters()))
    return True


def fused_mlp_requested() -> bool:
    return os.environ.get("PHINGAN_FUSED_MLP", "0") == "1"


# Stage names for PHINGAN_FUSED_MLP_TF32 stage lists, by generator class.
_TF32_STAGE_BY_CLASS = {
    "ContinuousGenerator": "continuous",
    "SecondaryGenerator": "secondary",
}


def fused_mlp_tf32_requested(class_name: str | None = None) -> bool:
    """PHINGAN_FUSED_MLP_TF32 is "1" (all stages) or a ':'/','-separated
    stage list, e.g. "secondary" (':' is the PBS-safe separator); the valid
    stage names are the values of _TF32_STAGE_BY_CLASS, i.e. "continuous"
    and "secondary". TF32 is REJECTED as a default and stage scoping is what
    keeps it opt-in: see the module docstring for why each stage is
    rejected. IEEE ships.

    With class_name=None, reports whether TF32 is requested for ANY stage.
    """
    val = os.environ.get("PHINGAN_FUSED_MLP_TF32", "0").strip()
    if val in ("", "0"):
        return False
    if val == "1":
        return True
    stages = {s.strip() for s in val.replace(",", ":").split(":") if s.strip()}
    if class_name is None:
        return bool(stages)
    return _TF32_STAGE_BY_CLASS.get(class_name) in stages
