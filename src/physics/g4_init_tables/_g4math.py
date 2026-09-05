"""
Bit-faithful Python ports of G4's `G4Exp` and `G4Log` (Padé-based fast
math from `G4Exp.hh` / `G4Log.hh` in geant4-11.2.0).

`grid.py`'s log-spaced energy grid and the Bethe-Bloch / Bragg dE/dx models
under `g4_init_tables/models/` call these instead of `math.exp` / `math.log`
wherever the result must reproduce a G4-computed value to the last ULP —
most directly in the grid construction itself:

    invdBin = (n - 1) / G4Log(emax / emin)
    E[i]    = emin * G4Exp(i / invdBin)

A single-ULP delta here propagates into every downstream table.

Both `G4Exp` and `G4Log` agree with libm to within a few ULPs at most
arguments and exactly elsewhere. They are NOT correctly rounded — so
treat them as opaque definitions and reproduce their bit-pattern, do
not try to reason about them as math.exp / math.log + ε.

Implementation notes:

* The C++ source uses unions for `double <-> uint64` reinterpretation;
  Python doesn't have those, so we use `struct.pack/unpack` on scalars
  and `numpy.ndarray.view` on arrays. Both are bit-faithful.

* `math.fma` is NOT used by G4 (the .hh files write `a *= x; a += b;`
  expecting compiler-controlled rounding). We mirror that explicitly.

* For tensors, the same algorithm runs elementwise via torch ops; the
  bit-cast uses `t.contiguous().view(torch.int64)` (both dtypes are
  8 bytes so the view is well-defined).
"""
from __future__ import annotations

import math
import struct
from typing import overload

import torch


# ---- G4Exp constants (G4Exp.hh:75-83) -------------------------------- #

_PX1_EXP = 1.26177193074810590878e-4
_PX2_EXP = 3.02994407707441961300e-2
_PX3_EXP = 9.99999999999999999910e-1
_QX1_EXP = 3.00198505138664455042e-6
_QX2_EXP = 2.52448340349684104192e-3
_QX3_EXP = 2.27265548208155028766e-1
_QX4_EXP = 2.00000000000000000009e0
_LOG2E   = 1.4426950408889634073599   # 1 / ln(2)
_EXP_LIMIT = 708.0

# Two halves of ln(2), split for extra precision in argument reduction
# (G4Exp.hh:187-188). Their sum equals ln(2) to ~80 bits.
_LN2_HI = 6.93145751953125e-1
_LN2_LO = 1.42860682030941723212e-6


# ---- G4Log constants (G4Log.hh:78-141) ------------------------------- #

_SQRTH    = 0.70710678118654752440  # 1/sqrt(2)
_PX1_LOG  = 1.01875663804580931796e-4
_PX2_LOG  = 4.97494994976747001425e-1
_PX3_LOG  = 4.70579119878881725854e0
_PX4_LOG  = 1.44989225341610930846e1
_PX5_LOG  = 1.79368678507819816313e1
_PX6_LOG  = 7.70838733755885391666e0
_QX1_LOG  = 1.12873587189167450590e1
_QX2_LOG  = 4.52279145837532221105e1
_QX3_LOG  = 8.29875266912776603211e1
_QX4_LOG  = 7.11544750618563894466e1
_QX5_LOG  = 2.31251620126765340583e1

# The constant `res -= fe * 2.12...e-4; ...; res += fe * 0.693359375;`
# in G4Log.hh:251-255 splits ln(2) into a precise pair (high = 0.693359375,
# low = 2.121944400546905827679e-4) just like G4Exp's argument reduction.
_LN2_LOG_HI = 0.693359375
_LN2_LOG_LO = 2.121944400546905827679e-4


# Signed-int64 view of the 64-bit mantissa+sign mask used inside G4Log.
# In C++ the mask is the unsigned literal `0x800FFFFFFFFFFFFFULL`, which sets
# bit 63 (the sign bit) — i.e. doesn't fit in a signed int64. Pack/unpack
# round-trips the bit pattern to the signed view PyTorch will accept.
_MANTISSA_AND_SIGN_MASK = struct.unpack("<q", struct.pack("<Q", 0x800FFFFFFFFFFFFF))[0]
_HALF_EXPONENT_OR_MASK = 0x3FE0000000000000  # fits int64 unchanged


# --------------------------- scalar (float) --------------------------- #

def _u64_to_f64(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & 0xFFFFFFFFFFFFFFFF))[0]


def _f64_to_u64(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def g4_exp_scalar(initial_x: float) -> float:
    """Bit-faithful Python port of `G4Exp(double)` for scalar `float`."""
    if initial_x > _EXP_LIMIT:
        return math.inf
    if initial_x < -_EXP_LIMIT:
        return 0.0
    x = initial_x

    # Argument reduction: write x = n*ln(2) + r with |r| <= 0.5*ln(2),
    # then exp(x) = 2^n * exp(r). The split of ln(2) into _LN2_HI/_LO
    # buys ~80 bits of precision for the reduction.
    px = math.floor(_LOG2E * x + 0.5)
    n = int(px)
    x = x - px * _LN2_HI
    x = x - px * _LN2_LO

    xx = x * x

    pxv = _PX1_EXP
    pxv = pxv * xx + _PX2_EXP
    pxv = pxv * xx + _PX3_EXP
    pxv = pxv * x

    qxv = _QX1_EXP
    qxv = qxv * xx + _QX2_EXP
    qxv = qxv * xx + _QX3_EXP
    qxv = qxv * xx + _QX4_EXP

    r = pxv / (qxv - pxv)
    r = 1.0 + 2.0 * r

    # Multiply by 2^n. Constructed directly from the exponent bits.
    two_to_n = _u64_to_f64((n + 1023) << 52)
    return r * two_to_n


def g4_log_scalar(x: float) -> float:
    """Bit-faithful Python port of `G4Log(double)` for scalar `float`.

    Mirrors G4Log.hh:227-263. Argument must be > 0 (returns NaN otherwise,
    matching the C++ behaviour for `original_x < LOG_LOWER_LIMIT`).
    """
    original_x = x
    if original_x <= 0.0:
        return float("nan")

    # Split x = m * 2^e with m in [sqrt(0.5), sqrt(2)) (i.e. 0.5 < m <= 1
    # then doubled). The C++ uses bit manipulation; we mirror that.
    n = _f64_to_u64(x)
    le = n >> 52
    fe = float(le - 1023)
    n = n & 0x800FFFFFFFFFFFFF
    n = n | 0x3FE0000000000000  # forces exponent bits to encode 0.5
    x = _u64_to_f64(n)

    if x > _SQRTH:
        fe += 1.0
    else:
        x = x + x
    x = x - 1.0

    # Rational approximation of log(1 + x) for x in (-0.293, 0.414).
    px = _PX1_LOG
    px = px * x + _PX2_LOG
    px = px * x + _PX3_LOG
    px = px * x + _PX4_LOG
    px = px * x + _PX5_LOG
    px = px * x + _PX6_LOG

    x2 = x * x
    px = px * x
    px = px * x2

    qx = x + _QX1_LOG
    qx = qx * x + _QX2_LOG
    qx = qx * x + _QX3_LOG
    qx = qx * x + _QX4_LOG
    qx = qx * x + _QX5_LOG

    res = px / qx
    res -= fe * _LN2_LOG_LO
    res -= 0.5 * x2
    res = x + res
    res += fe * _LN2_LOG_HI

    return res


# --------------------------- tensor (torch) --------------------------- #

def _bitcast_f64_to_i64(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int64)


def _bitcast_i64_to_f64(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.float64)


def g4_exp(x: torch.Tensor) -> torch.Tensor:
    """Bit-faithful elementwise `G4Exp` over a float64 tensor.

    Saturating branches (|x| > 708) are handled with `torch.where` so the
    function is differentiable-safe and broadcasts naturally.
    """
    if x.dtype != torch.float64:
        raise TypeError(f"g4_exp requires float64 input, got {x.dtype}")

    initial_x = x
    x = initial_x

    px = torch.floor(_LOG2E * x + 0.5)
    n = px.to(torch.int64)
    x = x - px * _LN2_HI
    x = x - px * _LN2_LO

    xx = x * x

    pxv = torch.full_like(x, _PX1_EXP)
    pxv = pxv * xx + _PX2_EXP
    pxv = pxv * xx + _PX3_EXP
    pxv = pxv * x

    qxv = torch.full_like(x, _QX1_EXP)
    qxv = qxv * xx + _QX2_EXP
    qxv = qxv * xx + _QX3_EXP
    qxv = qxv * xx + _QX4_EXP

    r = pxv / (qxv - pxv)
    r = 1.0 + 2.0 * r

    two_to_n = _bitcast_i64_to_f64((n + 1023) << 52)
    out = r * two_to_n
    out = torch.where(initial_x > _EXP_LIMIT, torch.full_like(out, float("inf")), out)
    out = torch.where(initial_x < -_EXP_LIMIT, torch.zeros_like(out), out)
    return out


def g4_log(x: torch.Tensor) -> torch.Tensor:
    """Bit-faithful elementwise `G4Log` over a float64 tensor.

    Returns NaN where the input is non-positive (matching the C++).
    """
    if x.dtype != torch.float64:
        raise TypeError(f"g4_log requires float64 input, got {x.dtype}")

    pos_mask = x > 0.0
    safe_x = torch.where(pos_mask, x, torch.ones_like(x))

    n = _bitcast_f64_to_i64(safe_x).clone()
    le = n >> 52
    fe = (le - 1023).to(torch.float64)
    n = n & _MANTISSA_AND_SIGN_MASK
    n = n | _HALF_EXPONENT_OR_MASK
    m = _bitcast_i64_to_f64(n).clone()

    use_high = m > _SQRTH
    fe = torch.where(use_high, fe + 1.0, fe)
    m = torch.where(use_high, m, m + m)
    z = m - 1.0

    px = torch.full_like(z, _PX1_LOG)
    px = px * z + _PX2_LOG
    px = px * z + _PX3_LOG
    px = px * z + _PX4_LOG
    px = px * z + _PX5_LOG
    px = px * z + _PX6_LOG

    z2 = z * z
    px = px * z
    px = px * z2

    qx = z + _QX1_LOG
    qx = qx * z + _QX2_LOG
    qx = qx * z + _QX3_LOG
    qx = qx * z + _QX4_LOG
    qx = qx * z + _QX5_LOG

    res = px / qx
    res = res - fe * _LN2_LOG_LO
    res = res - 0.5 * z2
    res = z + res
    res = res + fe * _LN2_LOG_HI

    return torch.where(pos_mask, res, torch.full_like(res, float("nan")))


# --------------------- TorchScript-compatible variant --------------------- #

def _g4_log_script_impl(x: torch.Tensor) -> torch.Tensor:
    """TorchScript-compatible G4Log, bitwise identical to `g4_log`.

    torch 2.12 TorchScript supports neither closed-over module globals nor
    the `Tensor.view(dtype)` bitcast, so this variant (a) repeats the Padé
    constants as locals and (b) replaces the exponent/mantissa bit surgery
    with `torch.frexp`. Both encodings place the mantissa in [0.5, 1) for
    every normal positive double: G4Log forces the IEEE-754 biased-exponent
    bits to 0x3FE (true exponent -1), and frexp returns m in [0.5, 1) by
    definition, so the mantissas are identical. The exponents differ by one:
    frexp returns e with x = m * 2^e while G4Log computes le - 1023 = e - 1,
    hence fe = frexp_e - 1. Verified bitwise-equal to `g4_log` over a
    2e6-point sweep of [10 eV, 10 TeV].
    Keep the constants in sync with the module-level _PX*_LOG/_QX*_LOG.
    """
    _SQRTH = 0.70710678118654752440
    _PX1 = 1.01875663804580931796e-4
    _PX2 = 4.97494994976747001425e-1
    _PX3 = 4.70579119878881725854e0
    _PX4 = 1.44989225341610930846e1
    _PX5 = 1.79368678507819816313e1
    _PX6 = 7.70838733755885391666e0
    _QX1 = 1.12873587189167450590e1
    _QX2 = 4.52279145837532221105e1
    _QX3 = 8.29875266912776603211e1
    _QX4 = 7.11544750618563894466e1
    _QX5 = 2.31251620126765340583e1
    _LN2_HI = 0.693359375
    _LN2_LO = 2.121944400546905827679e-4

    # Explicit raise (not assert): eager asserts vanish under `python -O`,
    # and this guard protects a float64 bit-contract.
    if x.dtype != torch.float64:
        raise TypeError("g4_log_jit requires float64 input")
    pos_mask = x > 0.0
    safe_x = torch.where(pos_mask, x, torch.ones_like(x))

    m, e = torch.frexp(safe_x)
    m = m.to(torch.float64)
    fe = (e - 1).to(torch.float64)

    use_high = m > _SQRTH
    fe = torch.where(use_high, fe + 1.0, fe)
    m = torch.where(use_high, m, m + m)
    z = m - 1.0

    px = torch.full_like(z, _PX1)
    px = px * z + _PX2
    px = px * z + _PX3
    px = px * z + _PX4
    px = px * z + _PX5
    px = px * z + _PX6

    z2 = z * z
    px = px * z
    px = px * z2

    qx = z + _QX1
    qx = qx * z + _QX2
    qx = qx * z + _QX3
    qx = qx * z + _QX4
    qx = qx * z + _QX5

    res = px / qx
    res = res - fe * _LN2_LO
    res = res - 0.5 * z2
    res = z + res
    res = res + fe * _LN2_HI

    return torch.where(pos_mask, res, torch.full_like(res, float("nan")))


# Scripted on purpose: this pins the arithmetic order and rounding of the
# operations below, so the result stays bit-for-bit consistent with the
# other scripted G4-bit-exact kernels this module and interpolator.py define.
g4_log_jit = torch.jit.script(_g4_log_script_impl)
