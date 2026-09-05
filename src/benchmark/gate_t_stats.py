"""Gate T's statistics: the simulator vs GEANT4 truth.

`E_SEC` and `THETA` carry a large exact-zero spike (most steps produce no
secondary / no discrete scatter). KS on a distribution with a Dirac atom is
all-or-nothing: it mostly reads the atom's MASS difference, not the shape of
the continuum, so a well-matched continuum can still score ~0.5 and look
catastrophic. Those two features are therefore split here: KS on the
nonzero continuum, plus the atom's mass gated separately.

Order statistics use plain `scipy`/`numpy`, never `torch.Tensor.quantile`:
torch raises "quantile() input tensor is too large" above ~16M elements, and
GEANT4 truth files run to hundreds of millions of rows. This module never
imports torch at all, so that failure mode cannot be reintroduced by
accident.
"""
from typing import Optional

import numpy as np
from scipy.stats import ks_2samp

# Features that carry a large exact-zero Dirac atom and must be split into
# nonzero-continuum KS + separately reported atom mass.
_ATOM_FEATURES = ("E_SEC", "THETA")

# The feature the energy-distance statistic is computed on: E_CNT / log10
# continuous energy loss specifically, not an arbitrary feature.
_ENERGY_DISTANCE_FEATURE = "E_CNT"

_MIN_NONZERO = 100


def _energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """1-D energy distance on log10 of the two samples, over a subsample.

    A random (not positional) subsample of up to 4000 points per side, so
    this stays tractable on truth files with hundreds of millions of rows.
    """
    rng = np.random.default_rng(0)
    ix = rng.choice(len(x), size=min(4000, len(x)), replace=False)
    iy = rng.choice(len(y), size=min(4000, len(y)), replace=False)
    lx = np.log10(np.clip(np.asarray(x, dtype=np.float64)[ix], 1e-12, None))
    ly = np.log10(np.clip(np.asarray(y, dtype=np.float64)[iy], 1e-12, None))
    d_xy = np.abs(lx[:, None] - ly[None, :]).mean()
    d_xx = np.abs(lx[:, None] - lx[None, :]).mean()
    d_yy = np.abs(ly[:, None] - ly[None, :]).mean()
    return float(2 * d_xy - d_xx - d_yy)


def compare_distributions(sim: dict, ref: dict) -> dict:
    """Compare matching features of two empirical samples.

    `sim` / `ref` are `{feature_name: 1-D array}` and MUST carry exactly the
    same feature keys -- callers may pass a single feature (as gate T's own
    tests do) or the full set, but the two sides must agree on which. This
    is deliberately strict rather than silently comparing the intersection:
    a caller that builds `sim` and `ref` independently (as gate T's real
    acceptance run does) could otherwise drop a feature to a typo or a
    missing column and have the gate silently stop checking it -- which
    looks exactly like a pass.

    Returns `{"ks": {feat: float}, "atom_mass": {feat: float},
    "energy_distance": float | None, "n_rows": {feat: {"sim": int, "ref": int}}}`.
    `E_SEC` / `THETA` contribute to `atom_mass` in addition to `ks`; the other
    features contribute to `ks` only. `energy_distance` is `None` when
    neither side carries `E_CNT`.
    """
    if set(sim) != set(ref):
        raise ValueError(
            "compare_distributions: sim/ref feature sets differ "
            f"(sim only: {sorted(set(sim) - set(ref))}, "
            f"ref only: {sorted(set(ref) - set(sim))}) -- a gate that "
            "silently drops a mismatched feature is not checking it")
    feats = list(sim)
    ks: dict[str, float] = {}
    atom_mass: dict[str, float] = {}
    n_rows: dict[str, dict[str, int]] = {}

    for name in feats:
        xa = np.asarray(sim[name])
        xb = np.asarray(ref[name])
        n_rows[name] = {"sim": int(len(xa)), "ref": int(len(xb))}

        if name in _ATOM_FEATURES:
            ca, cb = xa[xa > 0], xb[xb > 0]
            if len(ca) < _MIN_NONZERO or len(cb) < _MIN_NONZERO:
                raise RuntimeError(
                    f"{name}: fewer than {_MIN_NONZERO} nonzero entries "
                    f"(sim={len(ca)}, ref={len(cb)}) -- statistic is "
                    "meaningless; supply more rows")
            ks[name] = float(ks_2samp(ca, cb).statistic)
            atom_mass[name] = float(abs((xa == 0).mean() - (xb == 0).mean()))
        else:
            ks[name] = float(ks_2samp(xa, xb).statistic)

    energy_distance: Optional[float] = None
    if _ENERGY_DISTANCE_FEATURE in feats:
        energy_distance = _energy_distance(sim[_ENERGY_DISTANCE_FEATURE],
                                           ref[_ENERGY_DISTANCE_FEATURE])

    return {"ks": ks, "atom_mass": atom_mass,
            "energy_distance": energy_distance, "n_rows": n_rows}
