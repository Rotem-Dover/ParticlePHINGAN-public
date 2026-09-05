"""
GPU parity + microbenchmark harness for the fused generator kernel.

Drives the two networks of the `phin_gan` preset — ContinuousGenerator and
SecondaryGenerator — built through benchmark.track_simulator_config, i.e. the
same objects the runtime uses, with the same pinned checkpoints and scalers.

    cd src && PHINGAN_FUSED_MLP=1 PYTHONPATH=. \
      python -m benchmark.runtime_profiling.test_fused_mlp --device cuda --n 1048576

Per net it reports:

  1. kernel vs emulator   -- `_launch` against `_emulate_packed_forward` on the
     identical packed spec and input. Isolates the Triton body from the
     packing. Reported PER OUTPUT COLUMN, plus the cross-paired diff
     (kernel col j vs emulator col 1-j) for a two-column net: a transposed
     store in the kernel's `rows * N_OUT + j` epilogue would swap columns,
     which a flattened norm cannot see but a per-column comparison makes
     obvious (correct pairing ~1e-7, cross pairing ~1e-1). The secondary net
     emits a single column ([e_sec], n_out=1), so this probe self-disables
     there (the `if n_out == 2:` guard) -- kept for whichever net is next
     trained at n_out=2.
  2. kernel vs eager module -- the fused forward against the untouched class
     forward, paired per lane on ONE shared latent tensor (torch.randn is
     patched for the duration so both sides consume the same draw; two
     independent randn calls would not match and the comparison would be
     meaningless).
  3. clamp probe (nets with a terminal op, `spec.clamp_kind != 0`) -- inputs
     are driven out of distribution until the UNCLAMPED output saturates
     above 1 and below 0 (`_clamp_search`); for clamp_kind 2 (reflect-top)
     the deep regime past 2 is NOT reachable that way on this architecture
     (LayerNorm before every Linear but the readout renormalizes away the
     input's scale) and is instead reached by amplifying the readout
     layer's weights 20x on a spec COPY, the same
     trick tests/physics/test_fused_mlp.py's
     test_reflect_epilogue_order_is_reflect_then_clamp uses. The kernel is
     then required to equal `_apply_terminal(raw, spec.clamp_kind)` on
     whichever regimes were reached. Without this the upper bound is never
     exercised (in-distribution the raw output stays below 1) and a
     clamp(0, 10)-class typo, or a reflect/clamp order swap, would pass.
  4. bench -- CUDA-event timing of the eager class forward, an
     inductor-compiled forward, and the fused kernel, and the realised
     per-kernel speedup k = t_compiled / t_fused. The COMPILED ratio is the
     Amdahl-relevant one: the production arm compiles the process phases and
     dynamo traces through predict(), so the network already runs as inductor
     code there.

`--sweep` additionally times `_launch` over BLOCK x num_warps and prints the
best config per net, for _LAUNCH_CONFIG in physics/interfaces/
fused_mlp.py. Launch configs that fail to compile (register pressure
on the wide tile pair) are reported, not swallowed.

The precision arm is a BUG DETECTOR, not a physics acceptance test:
eager-vs-eager on one seed and device is bitwise identical, so there is no
statistical null. IEEE fp32 tl.dot differs from cuBLAS only in summation
order, bounding drift at ~1e-6 relative. A |dz| at 1e-3 or above is a packing
bug (transposed weight, wrong ln_mult, mis-assigned tile, clamp on the wrong
side), not a precision matter.
"""
import argparse
import contextlib
import os
import sys

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench(fn, iters: int = 50, warmup: int = 10) -> float:
    """Mean ms per call, CUDA-event timed."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@contextlib.contextmanager
def _pinned_latent(latent: torch.Tensor):
    """Make every `torch.randn(B, n_noise, ...)` draw return `latent`.

    Both the eager class forward and the fused forward draw their latent
    internally with exactly that positional shape, so patching the draw is
    the only way to pair the two sides per lane without reimplementing (and
    thereby possibly mis-implementing) the forward body here.
    """
    real = torch.randn
    shape = tuple(latent.shape)

    def fake(*args, **kwargs):
        if tuple(args) == shape:
            return latent
        return real(*args, **kwargs)

    torch.randn = fake
    try:
        yield
    finally:
        torch.randn = real


def _as_2d(t: torch.Tensor, n_out: int) -> torch.Tensor:
    return t.view(-1, n_out)


def _col_stats(a: torch.Tensor, b: torch.Tensor, n_out: int) -> list[tuple]:
    """[(col, mean_abs_dz, max_abs_dz)] for two [B, n_out] tensors."""
    a, b = _as_2d(a, n_out), _as_2d(b, n_out)
    out = []
    for j in range(n_out):
        d = (a[:, j] - b[:, j]).abs()
        out.append((j, d.mean().item(), d.max().item()))
    return out


def _print_cols(title: str, stats: list[tuple], names: list[str]) -> None:
    for j, mean, mx in stats:
        print(f"  {title:<22s} col {j} ({names[j]:<5s}): "
              f"mean|dz| {mean:.3e}   max|dz| {mx:.3e}")


_COL_NAMES = {
    "ContinuousGenerator": ["e_cnt"],
    "SecondaryGenerator": ["e_sec"],
}


def _apply_terminal(raw: torch.Tensor, kind: int) -> torch.Tensor:
    """The eval-mode terminal op per _FusedSpec.clamp_kind (1 = Hardtanh(0,1),
    2 = reflect-top). Kind 0 nets never reach the probe."""
    if kind == 2:
        return torch.where(raw > 1.0, 2.0 - raw, raw).clamp(0.0, 1.0)
    return raw.clamp(0.0, 1.0)


# Sweep guard: the kernel keeps a [BLOCK, WB] fp32 activation tile live across
# the whole layer chain. This bound is EMPIRICAL, not derived from the card's
# register file: 8192 is the largest tile observed to compile and run in
# reasonable time, and 16384 is the smallest observed to fail (register
# pressure sends ptxas into a compile that never returns). The true spill
# point is somewhere between; the guard is set at the last known-good value
# rather than at the first known-bad one. If `physics/interfaces/
# fused_mlp.py`'s `_TILE_LADDER` grows past what this bound covers,
# re-derive it from the register file rather than raising it blind.
_MAX_TILE_FLOATS = 8192

# The sweep grid. These defaults are the ones that produced
# `measurements/fused_mlp_sweep/` and the current `_LAUNCH_CONFIG`;
# `--sweep-blocks` / `--sweep-warps` widen them without changing what a bare
# `--sweep` measures, so an extended sweep stays comparable to the recorded
# one.
_SWEEP_BLOCKS = (32, 64, 128, 256)
_SWEEP_WARPS = (2, 4, 8)


# ---------------------------------------------------------------------------
# Per-net arms
# ---------------------------------------------------------------------------

def _parity(stage, gen, spec, n, device, failures, tol):
    from physics.interfaces.fused_mlp import (
        _emulate_packed_forward, _launch)

    cls = type(gen).__name__
    names = _COL_NAMES.get(cls, [f"out{j}" for j in range(spec.n_out)])
    n_out = spec.n_out

    torch.manual_seed(4242)
    # The x-scalers are min-max normalisations, so scaled labels live in
    # [0, 1]; the first four lanes pin the corners so the extremes are always
    # in the sample.
    labels = torch.rand(n, spec.n_cond, device=device, dtype=torch.float32)
    labels[0] = 0.0
    labels[1] = 1.0
    labels[2] = 1e-6
    labels[3] = 1.0 - 1e-6
    latent = torch.randn(n, spec.n_u, device=device, dtype=torch.float32)

    with torch.no_grad(), _pinned_latent(latent):
        z_eager = type(gen).forward(gen, labels)
        z_fused = gen.forward(labels)
    x = torch.cat([labels, latent], dim=-1).contiguous().float()
    with torch.no_grad():
        z_emul = _emulate_packed_forward(spec, x)
        z_launch = _launch(spec, x)

    print(f"[{stage}] {cls}  spec: n_cond {spec.n_cond} n_u {spec.n_u} "
          f"n_layers {spec.n_layers} tiles ({spec.tile_a}, {spec.tile_b}) "
          f"n_out {n_out} clamp_kind {spec.clamp_kind} act {spec.act_kind} "
          f"launch ({spec.launch_block}, {spec.launch_warps}) "
          f"tf32 {spec.tf32}")
    print(f"  shapes: eager {tuple(z_eager.shape)} fused {tuple(z_fused.shape)} "
          f"emul {tuple(z_emul.shape)} launch {tuple(z_launch.shape)}")

    if tuple(z_fused.shape) != tuple(z_eager.shape):
        failures.append(f"{stage}: fused shape {tuple(z_fused.shape)} != "
                        f"eager {tuple(z_eager.shape)}")

    s_ke = _col_stats(z_launch, z_emul, n_out)
    _print_cols("kernel vs emulator", s_ke, names)
    s_km = _col_stats(z_fused, z_eager, n_out)
    _print_cols("kernel vs eager", s_km, names)

    # Column-order probe: with n_out == 2 the cross pairing must be LARGE. If
    # it is not, the per-column test has no power and a transposed store would
    # go unnoticed -- so a small cross diff is itself reported as a failure.
    if n_out == 2:
        a = _as_2d(z_launch, 2)
        b = _as_2d(z_emul, 2)
        cross = max((a[:, 0] - b[:, 1]).abs().mean().item(),
                    (a[:, 1] - b[:, 0]).abs().mean().item())
        correct = max(m for _, m, _ in s_ke)
        print(f"  column-order probe:    correct pairing mean|dz| {correct:.3e}   "
              f"cross pairing mean|dz| {cross:.3e}   "
              f"separation {cross / max(correct, 1e-30):.3e}x")
        if cross < 1e-3:
            failures.append(f"{stage}: cross-pairing mean|dz| {cross:.3e} is "
                            "too small for the per-column test to have power")

    worst = max(max(m for _, _, m in s_ke), max(m for _, _, m in s_km))
    if worst > tol:
        failures.append(f"{stage}: max|dz| {worst:.3e} > {tol:g}")
        print(f"  [FAIL] worst max|dz| {worst:.3e} > tol {tol:g}")
    else:
        print(f"  [ok]   worst max|dz| {worst:.3e} <= tol {tol:g}")

    if spec.clamp_kind:
        _clamp_probe(stage, gen, spec, n, device, failures)
    return labels, latent


def _clamp_search(spec, m: int, device, need_deep: bool):
    """Search over input scales -- and, only if that alone cannot reach the
    deep regime, readout-weight amplification on a COPY of `spec` -- for a
    batch whose raw (clamp_kind=0) output covers every regime the terminal
    epilogue branches on: `raw > 1` (upper bound), `raw < 0` (lower bound),
    and, for clamp_kind == 2 (reflect-top), `raw > 2` (deep enough that the
    reflect must fold negative before the trailing clamp floors it at 0).

    `raw > 2` is UNREACHABLE via input scaling alone on this architecture, by
    construction -- LayerNorm sits immediately before every Linear but the
    readout, so it renormalizes away nearly all of an out-of-distribution
    input's scale before the final Linear ever sees it. `raw > 1` and
    `raw < 0` ARE routinely reached that way, though.

    When input scaling alone cannot reach the deep regime, reach it
    the same way tests/physics/test_fused_mlp.py's
    test_reflect_epilogue_order_is_reflect_then_clamp does -- amplify the
    readout layer's weights/bias 20x, on a COPY (`spec` itself is never
    mutated: the caller's live spec is still needed by the bench/sweep arms
    that run after the clamp probe). This is a spec-internal consistency
    check on the kernel epilogue's branches, not a physics-accuracy
    measurement -- amplification only needs to reach every REGIME the
    epilogue distinguishes, not represent the trained checkpoint's real
    output distribution, exactly as in the unit test.

    Pure torch (`_emulate_packed_forward` only, never the CUDA-only
    kernel) -- CPU-safe and the testable core of `_clamp_probe`, which
    wraps this with the CUDA-only kernel-vs-`_apply_terminal` verification.

    Returns `(chosen, seen, used_spec, amplified)`:
    - `chosen`: `(scale, x, raw, hi, lo)` for the batch that satisfied every
      REQUIRED condition (deep is required only if `need_deep` and reachable
      at all -- see below), or the best available match with the deep
      requirement relaxed, or `None` if even `raw > 1`/`raw < 0` were never
      both reached.
    - `seen`: `{"hi": bool, "lo": bool, "deep": bool}` -- true iff that
      condition was reached at ANY tried scale/spec, even off the winning
      one, so a caller can report exactly which regime(s) were never
      reached rather than a blanket message.
    - `used_spec`: the spec (`spec` or its amplified copy) `chosen`'s `raw`
      was actually computed from -- the caller must verify the kernel
      against THIS spec, not `spec`, when `amplified` is True.
    - `amplified`: whether readout amplification was needed.
    """
    import dataclasses
    from physics.interfaces.fused_mlp import _emulate_packed_forward

    def _try(probe_spec, require_deep):
        torch.manual_seed(99)
        raw_spec = dataclasses.replace(probe_spec, clamp_kind=0)
        seen = {"hi": False, "lo": False, "deep": False}
        for scale in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
            labels = (torch.rand(m, spec.n_cond, device=device) - 0.5) * 2.0 * scale
            latent = torch.randn(m, spec.n_u, device=device) * scale
            x = torch.cat([labels, latent], dim=-1).contiguous().float()
            with torch.no_grad():
                raw = _as_2d(_emulate_packed_forward(raw_spec, x), spec.n_out)
            hi, lo = raw.max(dim=0).values, raw.min(dim=0).values
            cond = dict(hi=bool((hi > 1.0).all()), lo=bool((lo < 0.0).all()),
                       deep=bool((raw > 2.0).any()))
            for k in seen:
                seen[k] = seen[k] or cond[k]
            if cond["hi"] and cond["lo"] and (not require_deep or cond["deep"]):
                return (scale, x, raw, hi, lo), seen
        return None, seen

    used_spec, amplified = spec, False
    chosen, seen = _try(spec, require_deep=need_deep)

    if chosen is None and need_deep:
        amp_spec = dataclasses.replace(spec, weights=spec.weights.clone(),
                                       bias=spec.bias.clone())
        amp_spec.weights[spec.n_layers - 1] *= 20.0
        amp_spec.bias[spec.n_layers - 1] *= 20.0
        chosen, amp_seen = _try(amp_spec, require_deep=True)
        seen = {k: seen[k] or amp_seen[k] for k in seen}
        if chosen is not None:
            used_spec, amplified = amp_spec, True

    if chosen is None:
        # Deep regime unreachable even after amplification (not observed on
        # any real net, but handled defensively): fall back to verifying
        # only the reachable regimes, on the un-amplified spec.
        chosen, shallow_seen = _try(spec, require_deep=False)
        seen = {k: seen[k] or shallow_seen[k] for k in seen}
        used_spec, amplified = spec, False

    return chosen, seen, used_spec, amplified


def _clamp_missing(seen: dict, need_deep: bool) -> list:
    """Human-readable list of the regime(s) `_clamp_search`'s `seen` never
    reached (empty iff every regime the epilogue kind cares about fired)."""
    return [label for key, label in (
               ("hi", "raw > 1 (upper bound)"),
               ("lo", "raw < 0 (lower bound)"),
               ("deep", "raw > 2 (reflect's deep branch)"))
           if (key != "deep" or need_deep) and not seen[key]]


def _clamp_probe(stage, gen, spec, n, device, failures):
    """Drive the terminal op (Hardtanh(0, 1), or reflect-top for clamp_kind 2)
    through every regime it branches on (`_clamp_search`), then require the
    kernel to equal `_apply_terminal(raw, spec.clamp_kind)` exactly.

    The deep regime (clamp_kind == 2's `raw > 2`) being unreachable via
    input scaling alone is a STRUCTURAL, checkpoint-scale property (see
    `_clamp_search`'s docstring), not a bug, so it does not fail the probe
    for that net. It fails only if (a) the
    kernel actually disagrees with `_apply_terminal` on whatever regimes
    WERE reached, or (b) even the reachable regimes (`raw > 1`, `raw < 0`)
    were never reached at all.
    """
    from physics.interfaces.fused_mlp import _launch

    m = min(n, 1 << 16)
    need_deep = spec.clamp_kind == 2
    chosen, seen, used_spec, amplified = _clamp_search(spec, m, device, need_deep)
    missing = _clamp_missing(seen, need_deep)

    if chosen is None:
        print(f"  [FAIL] clamp probe: never reached {', '.join(missing)} "
              "(tried input scaling to 1000x" +
              (", then 20x readout amplification" if need_deep else "") + ")")
        failures.append(f"{stage}: clamp probe never reached the reachable "
                        f"regimes ({', '.join(missing)})")
        return
    if missing:
        print(f"  clamp probe: {', '.join(missing)} unreachable even after "
              "20x readout amplification -- verifying the reachable "
              "regimes only; NOT gated as a failure (structural, see "
              "_clamp_search's docstring).")

    scale, x, raw, hi, lo = chosen
    with torch.no_grad():
        got = _as_2d(_launch(used_spec, x), spec.n_out)
    want = _apply_terminal(raw, spec.clamp_kind)
    d = (got - want).abs()
    frac_hi = (got == 1.0).float().mean(dim=0)
    frac_lo = (got == 0.0).float().mean(dim=0)
    print(f"  clamp probe @scale {scale:g}"
          f"{' (readout amplified 20x)' if amplified else ''}: raw max per "
          f"col {[round(v, 4) for v in hi.tolist()]}  raw min per col "
          f"{[round(v, 4) for v in lo.tolist()]}")
    print(f"    at upper bound {[round(v, 4) for v in frac_hi.tolist()]}   "
          f"at lower bound {[round(v, 4) for v in frac_lo.tolist()]}   "
          f"max|kernel - clamp(raw)| {d.max().item():.3e}")
    if d.max().item() > 1e-4:
        failures.append(f"{stage}: clamped kernel differs from clamp(raw) by "
                        f"{d.max().item():.3e}")


def _bench_arm(stage, gen, labels, n, failures):
    """Eager / inductor-compiled / fused timings and the realised k."""
    cls = type(gen).__name__
    eager = lambda l: type(gen).forward(gen, l)  # noqa: E731
    with torch.no_grad():
        t_eager = _bench(lambda: eager(labels))
        compiled = torch.compile(eager, dynamic=True)
        compiled(labels)  # compile outside the timed region
        t_compiled = _bench(lambda: compiled(labels))
        t_fused = _bench(lambda: gen.forward(labels))
    k_eager = t_eager / t_fused
    k_compiled = t_compiled / t_fused
    print(f"  bench @{n:,} lanes: eager {t_eager:8.3f} ms | "
          f"compiled {t_compiled:8.3f} ms | fused {t_fused:8.3f} ms")
    print(f"  realised k: vs eager {k_eager:5.2f}x | "
          f"vs inductor-compiled {k_compiled:5.2f}x   <-- Amdahl-relevant k")
    return {"stage": stage, "class": cls, "eager_ms": t_eager,
            "compiled_ms": t_compiled, "fused_ms": t_fused,
            "k_vs_eager": k_eager, "k_vs_compiled": k_compiled}


def _sweep(stage, spec, labels, latent, n,
           blocks=_SWEEP_BLOCKS, warps=_SWEEP_WARPS,
           max_tile_floats=_MAX_TILE_FLOATS):
    from physics.interfaces.fused_mlp import _launch
    x = torch.cat([labels, latent], dim=-1).contiguous().float()

    # The incumbent `_LAUNCH_CONFIG` entry is forced into the grid. Only
    # within-job ratios are trustworthy on this cluster (the job-to-job offset
    # is ~3% in general and was once 68% on a single phase), so a candidate
    # measured without its own baseline in the same job cannot be compared to
    # the recorded table -- it would only look faster or slower than a number
    # from another card on another day.
    blocks = tuple(sorted(set(blocks) | {spec.launch_block}))
    warps = tuple(sorted(set(warps) | {spec.launch_warps}))

    print(f"  launch sweep @{n:,} lanes (tiles ({spec.tile_a}, {spec.tile_b}), "
          f"tf32 {spec.tf32}):")
    print(f"    grid: BLOCK {list(blocks)} x num_warps {list(warps)} "
          f"(incumbent ({spec.launch_block}, {spec.launch_warps}) forced in "
          f"as the within-job anchor)"
          + ("" if max_tile_floats == _MAX_TILE_FLOATS else
             f"   [GUARD OVERRIDDEN: {max_tile_floats} floats, default "
             f"{_MAX_TILE_FLOATS} -- a config past the default bound may hang "
             f"ptxas for tens of minutes; see _MAX_TILE_FLOATS]"))
    results = []
    with torch.no_grad():
        for blk in blocks:
            # Register-footprint guard, _MAX_TILE_FLOATS (see its definition
            # for why the bound is 8192 and what that excludes). Printed, not
            # silently dropped: a skipped config must never be mistaken for a
            # tested-and-rejected one.
            if blk * spec.tile_b > max_tile_floats:
                print(f"    BLOCK={blk:3d}: SKIPPED (BLOCK x WB = "
                      f"{blk * spec.tile_b} > {max_tile_floats} floats live "
                      "per program; spill-bound)")
                continue
            for wp in warps:
                try:
                    _launch(spec, x[:blk * 4], block=blk, num_warps=wp)
                    t = _bench(lambda: _launch(spec, x, block=blk, num_warps=wp))
                except Exception as err:  # register pressure / shape refusal
                    msg = str(err).strip().splitlines()
                    print(f"    BLOCK={blk:3d} num_warps={wp}: FAILED "
                          f"({type(err).__name__}: {msg[0][:110] if msg else ''})")
                    continue
                results.append((t, blk, wp))
                print(f"    BLOCK={blk:3d} num_warps={wp}: {t:8.3f} ms")
    if not results:
        print(f"  BEST {stage}: no config compiled")
        return None
    t_best, blk, wp = min(results)
    base = [t for t, b, w in results if (b, w) == (spec.launch_block,
                                                  spec.launch_warps)]
    ref = f"{base[0]:.3f} ms" if base else "n/a"
    print(f"  BEST {stage}: BLOCK={blk} num_warps={wp} ({t_best:.3f} ms); "
          f"spec was ({spec.launch_block}, {spec.launch_warps}) at {ref}")
    return {"stage": stage, "block": blk, "warps": wp, "ms": t_best,
            "spec_block": spec.launch_block, "spec_warps": spec.launch_warps,
            "spec_ms": base[0] if base else None}


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", "--batch", dest="n", type=int, default=1 << 20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tol", type=float, default=None,
                        help="max |z_fused - z_eager| allowed (z-space is "
                             "O(1)). Default 1e-3 in IEEE (~1000x the fp32 "
                             "summation-reorder scale); 1e-1 under TF32, "
                             "whose acceptance is the physics gate, not here")
    parser.add_argument("--stages", default="continuous,secondary")
    parser.add_argument("--sweep", action="store_true",
                        help="per net, time the kernel over BLOCK x num_warps "
                             "and print the best config for _LAUNCH_CONFIG")
    parser.add_argument("--sweep-blocks", default=None,
                        help="comma-separated BLOCK values for --sweep "
                             f"(default {','.join(map(str, _SWEEP_BLOCKS))})")
    parser.add_argument("--sweep-warps", default=None,
                        help="comma-separated num_warps values for --sweep "
                             f"(default {','.join(map(str, _SWEEP_WARPS))})")
    parser.add_argument("--sweep-max-tile-floats", type=int,
                        default=_MAX_TILE_FLOATS,
                        help="override the BLOCK x WB register-footprint guard "
                             f"(default {_MAX_TILE_FLOATS}). Raising it can "
                             "hand ptxas a config that compiles for tens of "
                             "minutes without returning -- run such an arm "
                             "last, and under a timeout")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--force-tile", type=int, default=None,
                        help="floor both tiles of every swept net's pair "
                             "(sweep arms only, e.g. 16 vs 32 for the "
                             "secondary net; never a shipped default "
                             "-- that is _MIN_TILE's job)")
    args = parser.parse_args()

    def _int_list(raw, default):
        if raw is None:
            return default
        vals = tuple(int(v) for v in raw.replace(":", ",").split(",") if v.strip())
        if not vals:
            sys.exit("empty sweep grid")
        return vals

    sweep_blocks = _int_list(args.sweep_blocks, _SWEEP_BLOCKS)
    sweep_warps = _int_list(args.sweep_warps, _SWEEP_WARPS)

    if args.device != "cuda":
        sys.exit("test_fused_mlp needs a CUDA device (the fused path is CUDA-only)")
    if not torch.cuda.is_available():
        sys.exit("test_fused_mlp needs a CUDA device; torch.cuda.is_available() is False")

    print(f"device_name: {torch.cuda.get_device_name(0)!r}   torch {torch.__version__}")
    try:
        import triton
        print(f"triton {triton.__version__}")
    except Exception as err:
        sys.exit(f"triton unavailable: {err}")
    print(f"PHINGAN_FUSED_MLP={os.environ.get('PHINGAN_FUSED_MLP', '0')!r}  "
          f"PHINGAN_FUSED_MLP_TF32={os.environ.get('PHINGAN_FUSED_MLP_TF32', '0')!r}")

    from benchmark.track_simulator_config import build_stepping_manager
    from benchmark.runtime_profiling.profile_runtime import _STAGE_ATTR, _unwrap
    from physics.interfaces.fused_mlp import (
        _SPEC_REGISTRY, build_spec, try_enable_fused_forward)

    sm = build_stepping_manager("phin_gan", device="cuda")
    (process,) = sm.processes

    failures: list[str] = []
    benches, sweeps = [], []
    for stage in args.stages.replace(":", ",").split(","):
        stage = stage.strip()
        if not stage:
            continue
        gen = _unwrap(getattr(process, _STAGE_ATTR[stage]))
        if not try_enable_fused_forward(gen):
            failures.append(f"{stage}: try_enable_fused_forward refused")
            print(f"[FAIL] {stage}: fusion refused (see the warning above)")
            continue
        # The registered spec is the one the runtime launches (it carries the
        # TF32 latch); build_spec is only a fallback for an unregistered gen.
        spec = _SPEC_REGISTRY.get(id(gen)) or build_spec(gen)
        if args.force_tile is not None:
            forced = build_spec(gen, min_tile=args.force_tile)
            forced.tf32 = spec.tf32
            forced.launch_block = spec.launch_block
            forced.launch_warps = spec.launch_warps
            # the fused forward launches the REGISTERED spec; re-register so
            # the kernel-vs-eager arm measures the forced tile too
            _SPEC_REGISTRY[id(gen)] = forced
            spec = forced
        tol = args.tol if args.tol is not None else (1e-1 if spec.tf32 else 1e-3)

        labels, latent = _parity(stage, gen, spec, args.n, "cuda",
                                 failures, tol)
        if not args.no_bench:
            benches.append(_bench_arm(stage, gen, labels, args.n, failures))
        if args.sweep:
            s = _sweep(stage, spec, labels, latent, args.n,
                       blocks=sweep_blocks, warps=sweep_warps,
                       max_tile_floats=args.sweep_max_tile_floats)
            if s:
                sweeps.append(s)
        print()

    if benches:
        print("=== realised per-kernel speedup k ===")
        for b in benches:
            print(f"  {b['stage']:<10s} {b['class']:<20s} "
                  f"eager {b['eager_ms']:8.3f} | compiled {b['compiled_ms']:8.3f} | "
                  f"fused {b['fused_ms']:8.3f} ms | k_vs_compiled "
                  f"{b['k_vs_compiled']:5.2f}x | k_vs_eager {b['k_vs_eager']:5.2f}x")
    if sweeps:
        print("=== _LAUNCH_CONFIG candidates ===")
        for s in sweeps:
            print(f"  {s['stage']:<10s} BLOCK={s['block']} num_warps={s['warps']} "
                  f"({s['ms']:.3f} ms)")

    if failures:
        sys.exit("FUSED-MLP PARITY FAILED:\n  " + "\n  ".join(failures))
    print("all nets: fused kernel parity OK")


if __name__ == "__main__":
    main()
