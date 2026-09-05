"""Is the primary's scattering angle determined by conservation?

`physics.g4h_ionisation.kinematics.compute_primary_theta` computes theta from
energy-momentum conservation rather than sampling it. That is only
legitimate if GEANT4's own recorded `angleDiscrete` is already a deterministic
function of the delta-ray energy. This instrument measures it directly, and it
is deliberately cheap: it needs no training, no checkpoints and no simulator --
only a truth file.

Reads (data_handling.reader column names):
    KineticEnergy   pre-step primary KE [eV]
    ContinuousLoss  along-step continuous loss [eV]
    SecondaryLoss   delta-ray KE [eV]  (0 where no secondary was produced)
    angleDiscrete   G4's recorded primary deflection [rad]

Expected: median relative residual ~1e-8.

A floor near ~8e-8 instead is the signature of a CONSTANTS mismatch, not a
physics failure: `physics.definitions` may not match the CLHEP version in
the G4 build that produced the file. See
tests/physics/test_theta_kinematics.py.

Usage:
    cd src && PYTHONPATH=. python -m benchmark.validate_theta_determinism \\
        --root /path/to/100MeV_1k.root --out /tmp/stage0.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.run_context import PARTICLE
from physics.definitions.physical_constants import electron_mass_eV
from physics.g4h_ionisation.kinematics import compute_primary_theta

log = logging.getLogger(__name__)


def validate(df: pd.DataFrame) -> dict:
    """Compare recorded `angleDiscrete` against the reconstruction.

    Rows with no secondary (`SecondaryLoss == 0`) are dropped: theta is
    identically zero there, so the relative residual is 0/0 -- counting them
    would either dilute the statistic or poison it with NaN.

    Returns a dict of `n_rows`, `median_rel`, `p99_rel`, `max_rel`, `max_abs`.
    """
    has_secondary = df["SecondaryLoss"].to_numpy() > 0.0
    sub = df.loc[has_secondary]
    if len(sub) == 0:
        raise ValueError("no rows with a secondary; nothing to validate")

    # float64 throughout: stage 0 chases ~1e-8 agreement, which float32's
    # ~1e-7 cannot resolve. The RUNTIME path is float32 (see kinematics.py);
    # this is the reference arm, not the shipped one.
    post_cnt = torch.tensor(
        (sub["KineticEnergy"] - sub["ContinuousLoss"]).to_numpy(), dtype=torch.float64)
    e_sec = torch.tensor(sub["SecondaryLoss"].to_numpy(), dtype=torch.float64)
    recorded = sub["angleDiscrete"].to_numpy().astype("float64")

    predicted = compute_primary_theta(
        post_cnt, e_sec, float(PARTICLE.mass_eV), float(electron_mass_eV)).numpy()

    abs_resid = np.abs(predicted - recorded)
    rel_resid = abs_resid / np.maximum(np.abs(recorded), 1e-30)

    # numpy, NOT torch, for the order statistics. torch.quantile raises
    # "quantile() input tensor is too large" above ~16M elements, and a real
    # truth file runs to ~224M rows -- so a torch implementation passes every
    # unit test and then fails on the one run that matters. np.quantile has no
    # such limit. Do not "simplify" this back to rel_resid.quantile(0.99).
    return {
        "n_rows": int(len(sub)),
        "median_rel": float(np.median(rel_resid)),
        "p99_rel": float(np.quantile(rel_resid, 0.99)),
        "max_rel": float(rel_resid.max()),
        "max_abs": float(abs_resid.max()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="GEANT4 truth ROOT file (tree 'Data')")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the report as JSON here")
    args = parser.parse_args()

    from data_handling.reader import read_geant4_simulation_output
    df = read_geant4_simulation_output(args.root)

    report = validate(df)
    report["source"] = str(args.root)
    report["primary_mass_eV"] = float(PARTICLE.mass_eV)
    report["electron_mass_eV"] = float(electron_mass_eV)

    text = json.dumps(report, indent=2)
    log.info("stage-0 theta determinism\n%s", text)
    if args.out is not None:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
