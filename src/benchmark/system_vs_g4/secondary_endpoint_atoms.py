"""Boundary-atom census of the secondary generator: how much mass the loaded
`phin_gan` secondary net puts EXACTLY at z == 0 (e_sec == Tc) and z == 1
(e_sec == t_max(E)), against GEANT4 truth on the same conditioning energies.

A Hardtanh(0, 1) terminal has zero gradient outside [0, 1], so WGAN-GP
training cannot pull back mass the generator pushes past the clamp; the
`hardtanh_reflect_top` terminal folds the z > 1 spill back down (z -> 2 - z)
before the trailing clamp instead of saturating it there, which removes the
z == 1 atom the plain Hardtanh terminal would otherwise carry, but the z == 0
atom remains (Hardtanh's own dead gradient below 0). A z == 1 atom shows up
downstream as a low-theta shelf: at the kinematic endpoint theta -> 0, and
float rounding scatters those rows over a narrow band where G4 has
~nothing.

This is the acceptance instrument for the secondary net's boundary atoms: it
exits nonzero if either generator atom exceeds ATOM_TOLERANCE_X times G4's
(with an absolute floor so a G4 zero does not demand a generator zero).

    cd src && PYTHONPATH=. python -m benchmark.system_vs_g4.secondary_endpoint_atoms
    ... --seed 1 --tol 10
"""
import argparse
import pickle as pkl
import sys

import torch

from common.enums import SecondaryFeats
from common.paths import DATASETS_DIR, PATH_TO_GEANT4_DATA

ATOM_TOLERANCE_X = 10.0
ATOM_ABS_FLOOR = 1e-6   # a generator atom below this passes regardless of G4


def census(gen, X: torch.Tensor, Y: torch.Tensor) -> dict:
    """z-space boundary fractions for the generator and for the truth."""
    with torch.no_grad():
        z = gen.forward(gen.x_scaler.scale(X).view(-1, 1))[:, SecondaryFeats.E_SEC_IDX]
        zt = gen.y_scaler.scale(Y.view(-1, 1), X.view(-1, 1)).flatten()
    n = float(len(z))
    return {
        "n_rows": int(n),
        "gen_z_eq_1": float((z >= 1).sum()) / n,
        "gen_z_eq_0": float((z <= 0).sum()) / n,
        "gen_z_ge_0.99999": float((z >= 0.99999).sum()) / n,
        "g4_z_ge_1": float((zt >= 1).sum()) / n,
        "g4_z_le_0": float((zt <= 0).sum()) / n,
        "g4_z_ge_0.99999": float((zt >= 0.99999).sum()) / n,
        "g4_z_max": float(zt.max()),
        "output_activation": getattr(gen, "output_activation", "?"),
    }


def verdict(c: dict, tol: float = ATOM_TOLERANCE_X) -> tuple[bool, list[str]]:
    reasons = []
    for gen_key, g4_key in (("gen_z_eq_1", "g4_z_ge_1"), ("gen_z_eq_0", "g4_z_le_0")):
        g, t = c[gen_key], c[g4_key]
        if g > ATOM_ABS_FLOOR and g > tol * max(t, ATOM_ABS_FLOOR):
            reasons.append(f"{gen_key}={g:.3e} exceeds {tol:g}x G4 ({g4_key}={t:.3e})")
    return (not reasons), reasons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=ATOM_TOLERANCE_X)
    ap.add_argument("--preset", default="phin_gan", choices=["phin_gan"])
    args = ap.parse_args(argv)

    from benchmark.system_vs_g4.net_sampling import preset_ionisation
    torch.manual_seed(args.seed)
    gen = preset_ionisation(args.preset).secondary_generator
    stem = PATH_TO_GEANT4_DATA.stem
    with open(DATASETS_DIR / f"{stem}_secondary_X_filtered.pkl", "rb") as f:
        X = torch.from_numpy(pkl.load(f)).float()
    with open(DATASETS_DIR / f"{stem}_secondary_Y_filtered.pkl", "rb") as f:
        Y = torch.from_numpy(pkl.load(f)).float()

    c = census(gen, X, Y)
    for k, v in c.items():
        print(f"{k:>18}: {v:.4e}" if isinstance(v, float) else f"{k:>18}: {v}")
    ok, reasons = verdict(c, args.tol)
    print("PASS" if ok else "FAIL: " + "; ".join(reasons))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
