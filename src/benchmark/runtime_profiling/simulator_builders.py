"""Shared optimized / eager simulator construction for the profiling and
physics-gate harnesses.

PHINGAN_FUSED_MLP is an env var latched at `.to()` time inside
G4hIonisation.to(), so an optimized and a fully eager simulator can coexist in
one process as long as the env is toggled around each build. torch.compile of
the phase methods + geometry latches per-instance (`_compiled` /
`_geometry_compiled`), which the eager builder pre-sets before the CUDA move.

Both builders take a preset NAME rather than a config dict: checkpoints are
resolved by explicit path through `benchmark.track_simulator_config`, so
there is no config-dict plumbing to carry.

PHINGAN_FUSED_MLP_TF32 (tensor-core dots inside the fused kernel) is
deliberately NOT managed here: it passes through from the job env to the
optimized build, and it cannot contaminate the eager build because the eager
builder holds PHINGAN_FUSED_MLP itself off -- TF32 is a sub-mode of the fused
path, dead without it.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

_LATCH_KEYS = ("PHINGAN_FUSED_MLP",)


@contextmanager
def env_latches(fused: bool):
    """Scoped setting of the fused-MLP latch. `True` sets the var to "1";
    `False` removes it entirely (the constructors treat any non-"1" as off,
    but removing keeps the env clean). Always restores the previous state,
    including on exceptions."""
    saved = {k: os.environ.get(k) for k in _LATCH_KEYS}
    try:
        if fused:
            os.environ["PHINGAN_FUSED_MLP"] = "1"
        else:
            os.environ.pop("PHINGAN_FUSED_MLP", None)
        yield
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def build_optimized_simulator(preset: str, device: str):
    """Production-optimized simulator: fused MLPs latched on during
    construction; on CUDA the phase methods + geometry compile in the
    constructors' `.to()` calls (mode from TORCH_COMPILE_MODE)."""
    from particle_propagation.track_simulator import TrackSimulator
    from benchmark.track_simulator_config import build_stepping_manager

    with env_latches(fused=True):
        sm = build_stepping_manager(preset, device=device)
        simulator = TrackSimulator(stepping_manager=sm, device=device)
    return simulator, sm


def build_eager_simulator(preset: str, device: str):
    """Build a simulator whose process phases and geometry are NOT
    torch.compile'd, by constructing on CPU and latching the compile flags
    (`_compiled` / `_geometry_compiled`) before the move to CUDA; the fused
    MLP env latch is held OFF for the whole construction. The
    @torch.jit.script kernels still run scripted, exactly as in production."""
    from particle_propagation.track_simulator import TrackSimulator
    from benchmark.track_simulator_config import build_stepping_manager

    with env_latches(fused=False):
        sm = build_stepping_manager(preset, device="cpu")
        for proc in sm.processes:
            if hasattr(proc, "_compiled"):
                proc._compiled = True
        simulator = TrackSimulator(stepping_manager=sm, device="cpu")
        simulator._geometry_compiled = True
        simulator.to(device)
    return simulator, sm
