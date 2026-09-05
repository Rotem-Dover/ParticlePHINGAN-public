"""Pure statistical helpers for the GPU physics gate (verify_physics_gate).

Everything here is CPU-safe and free of storage / CUDA / config imports so
the gating math is unit-testable on any machine. Tensors are accepted on any
device; computation happens where the data lives except where noted.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch


def two_sample_ks(a: torch.Tensor, b: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D = sup |F_a - F_b|."""
    a = a.detach().flatten().double().sort().values
    b = b.detach().flatten().double().sort().values
    grid = torch.cat([a, b]).sort().values
    cdf_a = torch.searchsorted(a, grid, right=True).double() / a.numel()
    cdf_b = torch.searchsorted(b, grid, right=True).double() / b.numel()
    return float((cdf_a - cdf_b).abs().max())


def two_sample_ks_bound(n: int, m: int, alpha: float = 1e-3,
                        margin: float = 2.0) -> float:
    """Same-distribution KS acceptance bound at level `alpha`, with a
    multiplicative safety `margin`: D_crit = c(alpha) * sqrt((n+m)/(n*m)),
    c(alpha) = sqrt(-ln(alpha/2)/2)."""
    c = math.sqrt(-0.5 * math.log(alpha / 2.0))
    return margin * c * math.sqrt((n + m) / (n * m))


def cdf_from_quantiles(q: torch.Tensor, u: torch.Tensor
                       ) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a theoretical CDF callable from a dense quantile grid q(u).

    Handles atoms (many u mapping to one q value, e.g. the length stage's
    L_eloss plateau) by assigning each distinct q its MAXIMUM u — the
    right-continuous CDF jump. Returned callable maps a tensor of physical
    values to CDF probabilities on the same device/dtype (as required by
    benchmark.ks_stats.kolmogorov_smirnov_distance).
    """
    q_np = q.detach().cpu().double().flatten().numpy()
    u_np = u.detach().cpu().double().flatten().numpy()
    order = np.argsort(q_np, kind="stable")
    qs, us = q_np[order], u_np[order]
    us = np.maximum.accumulate(us)  # enforce a monotone CDF
    q_unique = np.unique(qs)
    last_idx = np.searchsorted(qs, q_unique, side="right") - 1
    u_at_q = us[last_idx]

    def cdf(x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().double().flatten().numpy()
        out = np.interp(x_np, q_unique, u_at_q, left=0.0, right=1.0)
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)

    return cdf


def relative_errors(opt: torch.Tensor, ref: torch.Tensor,
                    floor: float = 1e-30) -> torch.Tensor:
    """|opt - ref| / |ref|, defined as exactly 0 where both are 0 (dead lanes,
    zero-theta rows). `floor` only guards the division."""
    opt = opt.detach().double().flatten()
    ref = ref.detach().double().flatten()
    diff = (opt - ref).abs()
    rel = diff / ref.abs().clamp(min=floor)
    both_zero = (opt == 0) & (ref == 0)
    return torch.where(both_zero, torch.zeros_like(rel), rel)


def sorted_quantile(x: torch.Tensor, p: float) -> float:
    """Sort-based quantile — torch.quantile rejects very large tensors."""
    s = x.detach().flatten().double().sort().values
    if s.numel() == 0:
        return float("nan")
    idx = min(int(p * (s.numel() - 1)), s.numel() - 1)
    return float(s[idx])


def decade_stats(ref: torch.Tensor, rel_err: torch.Tensor,
                 min_count: int = 100) -> dict[int, dict]:
    """Bucket `rel_err` by floor(log10(ref)) over ref > 0; drop buckets with
    fewer than `min_count` samples. Returns {decade: {n, p50, p999, max}}."""
    ref = ref.detach().double().flatten()
    rel_err = rel_err.detach().double().flatten()
    pos = ref > 0
    ref_pos, err_pos = ref[pos], rel_err[pos]
    decades = torch.floor(torch.log10(ref_pos)).long()
    out: dict[int, dict] = {}
    for d in torch.unique(decades).tolist():
        mask = decades == d
        n = int(mask.sum())
        if n < min_count:
            continue
        e = err_pos[mask]
        out[int(d)] = {
            "n": n,
            "p50": sorted_quantile(e, 0.5),
            "p999": sorted_quantile(e, 0.999),
            "max": float(e.max()),
        }
    return out


def collapse_count(opt: torch.Tensor, ref: torch.Tensor,
                   ref_floor: float = 1e-20) -> int:
    """Samples where the optimized path returned exactly 0 while the fp64
    reference resolves a value above `ref_floor`."""
    opt = opt.detach().double().flatten()
    ref = ref.detach().double().flatten()
    return int(((opt <= 0) & (ref > ref_floor)).sum())


def distinct_value_ratio(opt: torch.Tensor, ref: torch.Tensor) -> float:
    """#distinct(opt) / #distinct(ref) — detects output quantization."""
    n_ref = int(torch.unique(ref.detach().flatten()).numel())
    n_opt = int(torch.unique(opt.detach().flatten()).numel())
    return n_opt / max(n_ref, 1)


def aggregate_passed(arms: dict[str, dict]) -> bool:
    """Overall verdict: fail iff any arm explicitly failed. `passed: None`
    means skipped / downgraded — reported, not failing."""
    return all(arm.get("passed") is not False for arm in arms.values())
