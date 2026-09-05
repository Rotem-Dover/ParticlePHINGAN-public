"""Runtime-profiling baseline harness for the full track simulation.

Runs a scaling sweep (n_events = 1e2 ... 1e6 by default) of TrackSimulator on
the only preset this harness supports, `phin_gan` (analytic-MC length +
WGAN-GP continuous + WGAN-GP secondary), and writes a single structured JSON
report per invocation so successive speedup iterations can be compared
against this baseline.

What it records:

- meta: particle/material/beam energy, preset, device (+ GPU name), torch
  version, thread count, git SHA (via the PROFILE_GIT_SHA env var forwarded
  by deploy_and_profile.sh), label, timestamps.
- generators: per stage (length/continuous/secondary), the class that
  ACTUALLY runs, its checkpoint path (None for the length stage's analytic
  MC sampler, which has no network), and whether it is running the fused-MLP
  kernel (`fused`) and in TF32 mode (`fused_tf32`).
- compile diagnostics: torch._dynamo counters after the timed runs — if zero
  graphs were compiled, torch.compile never actually ran (the known
  `.predict()` vs `__call__` pitfall in G4hIonisation.to()).
- sweep: per n_events — wall time, steps executed until all particles died,
  total particle-steps, particle-steps/s, peak CUDA memory, and the per-step
  alive-batch-size decay curve (downsampled).
- stage timing (optional): one run with CUDA-synchronizing wrappers around
  each process phase (compute_step_limit / along_step_do_it /
  post_step_do_it), splitting wall time into per-process-phase buckets plus
  the TrackSimulator geometry/bookkeeping remainder. The synchronization
  inflates the absolute total; only the relative shares are meaningful.
- torch profiler (optional): a short profiled run; exports the operator
  table (text) and a chrome trace next to the JSON report.

Usage (module form, from src/):

    python -m benchmark.runtime_profiling.profile_runtime \
        --preset phin_gan --label baseline \
        --event-counts 100,1000,10000,100000,1000000

The active particle/beam is fixed by common.run_context's unconditional
proton_in_aluminum_100MeV preset — there is no ACTIVE_PARTICLE env lookup, so nothing to select via CLI flag.
"""
import argparse
import datetime
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# NOTE: common.* / physics.* imports happen inside main() AFTER argparse, so
# `--help` works without loading physics tables, and so any ACTIVE_PARTICLE
# mishap is reported clearly.
# --------------------------------------------------------------------------

# The three stages. `length` is the analytic MC sampler
# (PhysicsLengthGenerator) -- it has no network and no checkpoint, which is
# why the report tolerates a None checkpoint for it.
PAPER_STAGES = ("length", "continuous", "secondary")

_STAGE_ATTR = {
    "length": "length_generator",
    "continuous": "continuous_generator",
    "secondary": "secondary_generator",
}


def _parse_counts(raw: str) -> list[int]:
    # Accept ',' or ':' separators — ':' survives PBS `qsub -v` (which splits
    # its value list on commas even inside quotes).
    return [int(float(tok)) for tok in raw.replace(":", ",").split(",") if tok.strip()]


def _now_tag() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")


def _downsample(values: list[int], max_points: int = 1000) -> dict:
    stride = max(1, len(values) // max_points)
    return {"stride": stride, "values": values[::stride], "n_total": len(values)}


def _window_times_ms(stamps: list[float], t0: float, window: int = 64) -> dict:
    """Average per-step wall time (ms) per `window`-step window. The
    simulator's periodic all-dead check syncs the device on that cadence, so
    window boundaries are true wall-clock points even when the CPU runs
    ahead of the GPU between them."""
    if not stamps:
        return {"window": window, "avg_step_ms": []}
    out, prev = [], t0
    for k in range(window - 1, len(stamps), window):
        out.append(round((stamps[k] - prev) * 1000.0 / window, 3))
        prev = stamps[k]
    return {"window": window, "avg_step_ms": out, "n_steps": len(stamps)}


# --------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------

def _unwrap(gen):
    """torch.compile wraps modules in OptimizedModule; the original lives in
    _orig_mod. For class reporting we always want the original."""
    return getattr(gen, "_orig_mod", gen)


def _generator_report(sm) -> dict:
    """Per-stage {class, checkpoint, fused, fused_tf32} for the report header.

    Keyed by PAPER_STAGES. Reads the ACTUAL kernel mode off the generator
    (`_fused_mlp` / `_fused_mlp_tf32`), not the env request, so a refused
    fusion is reported as refused instead of as requested.
    """
    (process,) = sm.processes
    report = {}
    for stage in PAPER_STAGES:
        gen = _unwrap(getattr(process, _STAGE_ATTR[stage]))
        ckpt = getattr(gen, "checkpoint", None)
        report[stage] = {
            "class": type(gen).__name__,
            "checkpoint": None if ckpt is None else str(ckpt),
            "fused": bool(getattr(gen, "_fused_mlp", False)),
            "fused_tf32": bool(getattr(gen, "_fused_mlp_tf32", False)),
        }
    return report


def _verify_backends(preset: str, gen_report: dict, allow_fallback: bool):
    """Fail loudly if a stage did not load the backend the preset promises.

    Only the two network stages carry checkpoints; `length` is the analytic MC
    sampler and legitimately has none.
    """
    missing = [s for s in ("continuous", "secondary")
               if gen_report[s]["checkpoint"] is None]
    if missing and not allow_fallback:
        raise RuntimeError(
            f"preset {preset!r}: stages {missing} loaded no checkpoint; "
            "re-run with --allow-fallback only if that is intended")


def _dynamo_counters() -> dict:
    """Snapshot torch._dynamo counters — the ground truth on whether
    torch.compile ever compiled/ran anything this process."""
    try:
        from torch._dynamo.utils import counters
        return {group: dict(vals) for group, vals in counters.items()}
    except Exception as err:
        return {"error": f"{type(err).__name__}: {err}"}


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------

class CountingManager:
    """Transparent proxy around SteppingManager that records the alive-batch
    size of every step. The masked tensor already exists when we read
    .shape[0], so this adds no extra GPU sync."""

    def __init__(self, sm):
        self._sm = sm
        self.batch_sizes: list[int] = []

    def __call__(self, kinetic_energy):
        self.batch_sizes.append(int(kinetic_energy.shape[0]))
        return self._sm(kinetic_energy)

    def to(self, device, dtype=None):
        self._sm.to(device, dtype=dtype)
        return self

    @property
    def processes(self):
        return self._sm.processes

    def reset(self):
        self.batch_sizes = []

    # TrackSimulator.run() calls reset_nan_guard() before its step loop and
    # check_nan_guard() after it (SteppingManager's NaN-accumulator contract,
    # common/config.py's ContinuousYScaler lookup) -- without these the proxy
    # is not actually transparent and every run() crashes at reset_nan_guard.
    def reset_nan_guard(self) -> None:
        self._sm.reset_nan_guard()

    def check_nan_guard(self) -> None:
        self._sm.check_nan_guard()


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _timed_run(simulator, counter: CountingManager, e0: float, n_events: int,
               n_steps: int, device: str) -> dict:
    counter.reset()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    # Per-step wall timestamps. Individual steps are noisy (the CPU queues
    # ahead of the GPU between syncs), but the periodic all-dead check syncs
    # the device, so windowed averages over >= that interval are accurate.
    step_stamps: list[float] = []
    t0 = time.perf_counter()
    simulator.run(E_0=e0, n_events=n_events, n_steps=n_steps, save_steps_states=False,
                  progress_cb=lambda i, total: step_stamps.append(time.perf_counter()))
    _sync(device)
    wall = time.perf_counter() - t0

    # Prefer the simulator's on-device alive-count recording: stepping is
    # fixed-shape, so the stepping manager sees the full batch every step
    # and counter.batch_sizes stays a constant n_events rather than
    # tracking the alive decay. Falls back to the counter when the
    # simulator does not expose alive_counts_per_step.
    alive_counts = getattr(simulator, "alive_counts_per_step", None)
    if isinstance(alive_counts, torch.Tensor) and alive_counts.numel() > 0:
        decay = [int(v) for v in alive_counts.detach().cpu().tolist()]
    else:
        decay = counter.batch_sizes
    particle_steps = int(sum(decay))
    lanes_stepped = int(sum(counter.batch_sizes))
    alive_end = int(simulator.alive_particles.sum().item())
    entry = {
        "n_events": n_events,
        "wall_s": round(wall, 4),
        "steps_executed": len(counter.batch_sizes),
        "particle_steps": particle_steps,
        "lanes_stepped": lanes_stepped,
        "particle_steps_per_s": round(particle_steps / wall) if wall > 0 else None,
        "events_per_s": round(n_events / wall, 2) if wall > 0 else None,
        "alive_at_end": alive_end,
        "batch_size_decay": _downsample(decay),
        "step_window_ms": _window_times_ms(step_stamps, t0),
    }
    if device.startswith("cuda"):
        entry["peak_cuda_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return entry


def _stage_timing_run(simulator, counter: CountingManager, e0: float,
                      n_events: int, n_steps: int, device: str) -> dict:
    """Wrap every process phase with synchronizing timers and run once.
    Instance-attribute assignment shadows the bound methods, so this only
    affects this run's SteppingManager instance."""
    acc: dict[str, float] = {}

    def _wrap(proc, method_name):
        orig = getattr(proc, method_name)
        key = f"{type(proc).__name__}.{method_name}"
        acc[key] = 0.0

        def timed(*args, **kwargs):
            _sync(device)
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            _sync(device)
            acc[key] += time.perf_counter() - t0
            return out

        setattr(proc, method_name, timed)
        return orig

    originals = []
    for proc in counter.processes:
        for method in ("compute_step_limit", "along_step_do_it", "post_step_do_it"):
            if hasattr(proc, method):
                originals.append((proc, method, _wrap(proc, method)))

    try:
        entry = _timed_run(simulator, counter, e0, n_events, n_steps, device)
    finally:
        for proc, method, orig in originals:
            setattr(proc, method, orig)

    stages_total = sum(acc.values())
    entry["per_phase_s"] = {k: round(v, 4) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
    entry["phases_total_s"] = round(stages_total, 4)
    # Geometry (rotation matrices), state bookkeeping, alive-mask reduction —
    # everything outside the physics processes.
    entry["simulator_other_s"] = round(entry["wall_s"] - stages_total, 4)
    entry["note"] = ("per-phase timers synchronize CUDA around every call; absolute wall_s is "
                     "inflated vs the sweep runs — read the relative shares only")
    return entry


class _FineTimer:
    """Nested CUDA-synchronizing wall timers installed by shadowing bound
    methods / module globals. `set_phase` wrappers additionally tag inner
    (phase-scoped) buckets so cross-phase helpers (e.g. the range-table
    interpolation kernel, called from both the length stage's predict
    and along_step_do_it's compute_average_loss) are attributed to
    their caller."""

    def __init__(self, device: str):
        self.device = device
        self.acc: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.phase = ""
        self._restores: list[tuple] = []

    def wrap(self, obj, attr: str, key: str, phase_scoped: bool = False,
             set_phase: str | None = None):
        orig = getattr(obj, attr)

        def timed(*args, **kwargs):
            prev = self.phase
            if set_phase is not None:
                self.phase = set_phase
            _sync(self.device)
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                _sync(self.device)
                full = f"{prev}::{key}" if (phase_scoped and prev) else key
                self.acc[full] = self.acc.get(full, 0.0) + (time.perf_counter() - t0)
                self.calls[full] = self.calls.get(full, 0) + 1
                self.phase = prev

        setattr(obj, attr, timed)
        self._restores.append((obj, attr, orig))

    def restore(self):
        for obj, attr, orig in reversed(self._restores):
            setattr(obj, attr, orig)
        self._restores.clear()


def _install_fine_timers(simulator, timer: _FineTimer) -> None:
    """Instrument the eager simulator. Buckets (nested — inner times are
    included in their enclosing phase bucket):

      simulator.generate_step > stepping_manager > G4hIonisation.<phase> >
        <stage>.predict > <stage>.{x_scale,forward,y_rescale}
      plus the average-loss / zero-prob / secondary-generation-prob table
      lookups, tagged by calling phase.

    There is exactly one process (G4hIonisation) and three stages
    (PAPER_STAGES); the hook count is asserted so a stage-list mismatch
    silently mis-attributes the profile instead of failing loudly. The count
    only credits a stage when `_wrap_generator` actually installed the
    `predict` timer -- a `None` generator (or one whose `_STAGE_ATTR`
    attribute is missing) does NOT count as hooked.
    """

    def _wrap_generator(gen, stage: str) -> bool:
        if gen is None:
            return False
        timer.wrap(gen, "predict", f"{stage}.predict")
        timer.wrap(gen, "forward", f"{stage}.forward")
        for scaler_attr, key in (("x_scaler", f"{stage}.x_scale"),
                                 ("y_scaler", f"{stage}.y_rescale")):
            scaler = getattr(gen, scaler_attr, None)
            if scaler is not None:
                method = "scale" if scaler_attr == "x_scaler" else "rescale"
                if hasattr(scaler, method):
                    timer.wrap(scaler, method, key)
        return True

    (process,) = simulator.stepping_manager.processes
    cls = type(process).__name__
    for method, tag in (("compute_step_limit", "limit"),
                        ("along_step_do_it", "along"),
                        ("post_step_do_it", "post")):
        if hasattr(process, method):
            timer.wrap(process, method, f"{cls}.{method}", set_phase=f"ion.{tag}")

    hooked = 0
    for stage in PAPER_STAGES:
        gen = _unwrap(getattr(process, _STAGE_ATTR[stage]))
        if _wrap_generator(gen, stage):
            hooked += 1
    assert hooked == len(PAPER_STAGES), (
        f"fine timers hooked {hooked} stages, expected {len(PAPER_STAGES)}; "
        "a stage-list mismatch silently mis-attributes the profile")

    timer.wrap(process, "compute_average_loss", "avg_loss_splines", phase_scoped=True)
    timer.wrap(process, "compute_zero_prob", "zero_prob_kernel", phase_scoped=True)
    timer.wrap(process, "sec_gen_prob", "sec_prob_splines", phase_scoped=True)

    # Simulator-level buckets.
    timer.wrap(simulator, "_TrackSimulator__generate_step", "simulator.generate_step")
    timer.wrap(simulator, "_get_delta_state_and_post_dir", "simulator.geometry")
    timer.wrap(simulator, "_update_particles_state", "simulator.state_update")

    # SteppingManager total: __call__ can't be shadowed on the instance
    # (dunder lookup goes through the type), so proxy it.
    class _TimedSM:
        def __init__(self, sm):
            self._sm = sm

        def __call__(self, kinetic_energy):
            _sync(timer.device)
            t0 = time.perf_counter()
            try:
                return self._sm(kinetic_energy)
            finally:
                _sync(timer.device)
                timer.acc["stepping_manager"] = (
                    timer.acc.get("stepping_manager", 0.0) + (time.perf_counter() - t0))
                timer.calls["stepping_manager"] = timer.calls.get("stepping_manager", 0) + 1

        def to(self, device, dtype=None):
            self._sm.to(device, dtype=dtype)
            return self

        @property
        def processes(self):
            return self._sm.processes

    simulator.stepping_manager = _TimedSM(simulator.stepping_manager)


def _fine_timing_run(preset: str, e0: float, n_events: int, n_steps: int,
                     device: str) -> dict:
    """Deep per-part breakdown on an EAGER (uncompiled) simulator. Absolute
    times are inflated vs production (no inductor fusion + a device sync
    around every timer); read the relative shares, anchored to the compiled
    stage-timing run's phase totals."""
    from benchmark.runtime_profiling.simulator_builders import build_eager_simulator
    simulator, _ = build_eager_simulator(preset, device)

    # Shape warm-up (TorchScript kernel re-specialization, allocator).
    simulator.run(E_0=e0, n_events=n_events, n_steps=96, save_steps_states=False)
    _sync(device)

    timer = _FineTimer(device)
    _install_fine_timers(simulator, timer)
    try:
        _sync(device)
        t0 = time.perf_counter()
        simulator.run(E_0=e0, n_events=n_events, n_steps=n_steps, save_steps_states=False)
        _sync(device)
        wall = time.perf_counter() - t0
    finally:
        timer.restore()

    acc = timer.acc
    phase_keys = [k for k in acc if "_do_it" in k or "compute_step_limit" in k]
    phases_total = sum(acc[k] for k in phase_keys)
    derived = {
        "stepping_manager_glue_s": round(acc.get("stepping_manager", 0.0) - phases_total, 4),
        "generate_step_masking_s": round(
            acc.get("simulator.generate_step", 0.0) - acc.get("stepping_manager", 0.0), 4),
        "run_loop_other_s": round(
            wall - acc.get("simulator.generate_step", 0.0)
            - acc.get("simulator.geometry", 0.0)
            - acc.get("simulator.state_update", 0.0), 4),
    }
    steps = int(simulator.alive_counts_per_step.shape[0])
    return {
        "n_events": n_events,
        "wall_s": round(wall, 4),
        "steps_executed": steps,
        "buckets_s": {k: round(v, 4) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])},
        "calls": timer.calls,
        "derived": derived,
        "note": ("EAGER simulator (phase/geometry compile latched off) with a device sync "
                 "around every timer: absolute wall_s is inflated vs production — read "
                 "relative shares, anchored to the compiled stage_timing phase totals. "
                 "Inner buckets are included in their enclosing phase bucket."),
    }


def _torch_profile_run(simulator, e0: float, n_events: int, n_steps: int,
                       device: str, out_base: Path) -> dict:
    from torch.profiler import profile, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        simulator.run(E_0=e0, n_events=n_events, n_steps=n_steps, save_steps_states=False)
        _sync(device)

    sort_key = "cuda_time_total" if device.startswith("cuda") else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
    table_path = out_base.with_suffix(".profiler.txt")
    table_path.write_text(table)
    trace_path = out_base.with_suffix(".trace.json.gz")
    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception as err:
        trace_path = None
        print(f">> chrome trace export failed: {err}")
    print(table)
    return {
        "n_events": n_events,
        "n_steps": n_steps,
        "sort_key": sort_key,
        "table_file": str(table_path),
        "trace_file": str(trace_path) if trace_path else None,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime-profiling baseline for TrackSimulator")
    parser.add_argument("--preset", default="phin_gan", choices=["phin_gan"],
                        help="Backend preset (this harness supports exactly "
                             "one: phin_gan -- analytic-MC length + WGAN "
                             "continuous + WGAN secondary)")
    parser.add_argument("--eager", action="store_true",
                        help="build with build_eager_simulator (no compile, no fusion)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--e0", type=float, default=None,
                        help="Initial kinetic energy in eV (default: beam energy of the active particle)")
    parser.add_argument("--n-steps", type=int, default=10_250,
                        help="Step cap per event (the run breaks early once every particle stops)")
    parser.add_argument("--event-counts", default="100,1000,10000,100000,1000000",
                        help="Comma-separated n_events sweep")
    parser.add_argument("--label", default="baseline", help="Tag stored in the report (e.g. baseline, iter1_compile_fix)")
    parser.add_argument("--stage-timing-events", type=int, default=100_000,
                        help="n_events for the per-phase breakdown run (0 disables)")
    parser.add_argument("--torch-profile-events", default="10000",
                        help="n_events for the torch.profiler run(s) (0 disables). "
                             "Accepts a ':'- or ','-separated list to profile several "
                             "batch sizes in ONE job (same GPU — the A6000 pool mixes "
                             "cards, so cross-job kernel comparisons are confounded)")
    parser.add_argument("--torch-profile-steps", type=int, default=100)
    parser.add_argument("--fine-timing-events", type=int, default=0,
                        help="n_events for the deep per-part breakdown on an EAGER "
                             "(uncompiled) simulator (0 disables). Runs last; read "
                             "relative shares only.")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="Do not hard-fail when a net stage silently fell back to the physics-MC sampler")
    parser.add_argument("--out", default=None, help="Output JSON path (default: STORAGE/runtime_profiling/...)")
    args = parser.parse_args()

    # Threading: honor OMP_NUM_THREADS if set, else leave torch's default.
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or None
    if n_threads:
        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(n_threads)
        except RuntimeError:
            pass  # already started parallel work

    from common.config import DEVICE, BEAM_ENERGY_eV
    from common.run_context import PARTICLE, TARGET_MATERIAL
    from common.paths import STORAGE
    from benchmark.runtime_profiling.simulator_builders import (
        build_eager_simulator, build_optimized_simulator)

    device = DEVICE if args.device == "auto" else args.device
    e0 = args.e0 if args.e0 is not None else float(BEAM_ENERGY_eV)
    event_counts = _parse_counts(args.event_counts)

    ts = _now_tag()
    out_dir = STORAGE / "runtime_profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = Path(args.out) if args.out else (
        out_dir / f"{ts}_{args.label}_{PARTICLE.name}_{args.preset}_{device}.json"
    )

    print(f"\n{'=' * 70}")
    print("Runtime profiling")
    print(f"{'=' * 70}")
    print(f"label:    {args.label}")
    print(f"particle: {PARTICLE.name} in {TARGET_MATERIAL.name}, E0 = {e0 / 1e6:.2f} MeV")
    print(f"preset:   {args.preset}   device: {device}")
    print(f"sweep:    {event_counts}  (n_steps cap {args.n_steps})")
    print(f"report:   {out_base}")
    print(f"{'=' * 70}\n")

    build = build_eager_simulator if args.eager else build_optimized_simulator
    simulator, sm = build(args.preset, device)
    counter = CountingManager(sm)
    simulator.stepping_manager = counter  # count per-step alive batches

    gen_report = _generator_report(sm)
    for stage, info in gen_report.items():
        print(f">> {stage}: {info['class']} "
              f"(checkpoint: {info['checkpoint']}, "
              f"fused: {info['fused']}, fused_tf32: {info['fused_tf32']})")
    _verify_backends(args.preset, gen_report, args.allow_fallback)

    # Incremental report: re-written after every completed section so a crash
    # in a later diagnostic (e.g. a fine-timing device bug) never loses the
    # sweep and stage-timing results already gathered.
    report = {
        "meta": {
            "label": args.label,
            "timestamp": ts,
            "hostname": socket.gethostname(),
            "particle": PARTICLE.name,
            "material": TARGET_MATERIAL.name,
            "beam_energy_eV": e0,
            "preset": args.preset,
            "eager": args.eager,
            "device": device,
            "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "n_threads": torch.get_num_threads(),
            "git_sha": os.environ.get("PROFILE_GIT_SHA", ""),
            "compile_mode": os.environ.get("PHINGAN_COMPILE_MODE", "default"),
            "n_steps_cap": args.n_steps,
            "argv": sys.argv[1:],
        },
        "generators": gen_report,
        "dynamo_counters": None,
        "sweep": [],
        "stage_timing": None,
        "torch_profile": None,
        "fine_timing": None,
    }

    def _dump_report():
        out_base.parent.mkdir(parents=True, exist_ok=True)
        out_base.write_text(json.dumps(report, indent=2))

    # Warm-up: JIT-script compilation, cuDNN autotune, CUDA context, and any
    # torch.compile activity all land here instead of in the timed sweep.
    print("\n>> warm-up run ...")
    simulator.run(E_0=e0, n_events=min(event_counts[0], 1000), n_steps=20, save_steps_states=False)
    _sync(device)

    sweep = []
    for n_events in event_counts:
        # Per-scale warmup at THIS batch shape: TorchScript's profiling
        # executor re-specializes its kernel plans per input shape, a one-time
        # ~tens-of-seconds cost that otherwise lands in the first timed
        # window of every scale (measured: 586 ms/step for the first 64-step
        # window at 1e4, 5.7 ms/step after). Production runs use one fixed
        # n_events, so steady-state is the honest metric; the one-time cost
        # is reported separately as warmup_s.
        _sync(device)
        tw = time.perf_counter()
        simulator.run(E_0=e0, n_events=n_events, n_steps=96, save_steps_states=False)
        _sync(device)
        warmup_s = time.perf_counter() - tw
        print(f"\n>> sweep: {n_events:,} events (shape warmup {warmup_s:.1f} s) ...")
        entry = _timed_run(simulator, counter, e0, n_events, args.n_steps, device)
        entry["warmup_s"] = round(warmup_s, 3)
        sweep.append(entry)
        report["sweep"] = sweep
        _dump_report()
        print(f"   {entry['wall_s']:.2f} s | {entry['steps_executed']} steps | "
              f"{entry['particle_steps']:,} particle-steps "
              f"({entry['particle_steps_per_s']:,}/s)"
              + (f" | peak {entry.get('peak_cuda_mem_mb')} MB" if device.startswith("cuda") else ""))

    stage_timing = None
    if args.stage_timing_events > 0:
        print(f"\n>> stage-timing run ({args.stage_timing_events:,} events, CUDA-sync timers) ...")
        stage_timing = _stage_timing_run(
            simulator, counter, e0, args.stage_timing_events, args.n_steps, device)
        report["stage_timing"] = stage_timing
        _dump_report()
        for k, v in stage_timing["per_phase_s"].items():
            print(f"   {k:55s} {v:10.3f} s")
        print(f"   {'<simulator geometry / bookkeeping / other>':55s} "
              f"{stage_timing['simulator_other_s']:10.3f} s")

    torch_profile_counts = [n for n in _parse_counts(args.torch_profile_events) if n > 0]
    if torch_profile_counts:
        # A list (even for one entry) — older reports hold a single dict here.
        report["torch_profile"] = []
        for n in torch_profile_counts:
            print(f"\n>> torch.profiler run ({n:,} events, "
                  f"{args.torch_profile_steps} steps) ...")
            # Per-count file names, so multi-size traces don't overwrite.
            base_n = out_base.with_name(f"{out_base.stem}_{n}ev.json") \
                if len(torch_profile_counts) > 1 else out_base
            report["torch_profile"].append(_torch_profile_run(
                simulator, e0, n, args.torch_profile_steps, device, base_n))
            _dump_report()

    fine_timing = None
    if args.fine_timing_events > 0:
        print(f"\n>> fine-timing run ({args.fine_timing_events:,} events, EAGER simulator, "
              f"deep sync timers) ...")
        fine_timing = _fine_timing_run(
            args.preset, e0, args.fine_timing_events, args.n_steps, device)
        report["fine_timing"] = fine_timing
        _dump_report()
        for k, v in fine_timing["buckets_s"].items():
            print(f"   {k:55s} {v:10.3f} s")
        for k, v in fine_timing["derived"].items():
            print(f"   {k:55s} {v:10.3f} s")

    dynamo = _dynamo_counters()
    frames_compiled = (dynamo.get("stats") or {}).get("unique_graphs", 0)
    print(f"\n>> torch._dynamo unique graphs compiled this process: {frames_compiled}")
    if not args.eager and device.startswith("cuda") and not frames_compiled:
        print(">> NOTE: this is a compiled build but dynamo compiled ZERO graphs — "
              "the compiled path never executed (predict() bypasses the "
              "OptimizedModule __call__).")

    report["dynamo_counters"] = dynamo
    _dump_report()
    print(f"\n>> report written: {out_base}")


if __name__ == "__main__":
    main()
