"""Write a tiny synthetic GEANT4-style ROOT file (tree 'Data') that exercises
every row-local rule in data_handling.reader: StepNum==0 rows, KineticEnergy
below MIN_ENERGY_CUTOFF, ProcessName=='Transportation', and
NumOfSecondaries==0 rows with nonzero SecondaryLoss."""
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from common.config import MIN_ENERGY_CUTOFF


def write_synthetic_root(path: Path, n_events: int = 50,
                         steps_per_event: int = 10, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    n = n_events * steps_per_event
    ev = np.repeat(np.arange(n_events), steps_per_event).astype(np.int32)
    step = np.tile(np.arange(steps_per_event), n_events).astype(np.int32)

    ke = rng.uniform(0.0, 1.0e8, n)
    ke[rng.random(n) < 0.15] = MIN_ENERGY_CUTOFF * 0.5   # below-cutoff rows
    proc = np.where(rng.random(n) < 0.2, "Transportation", "eIoni")
    proc[rng.random(n) < 0.05] = "eBrem"                 # sparse brems rows
    num_sec = rng.integers(0, 3, n).astype(np.int32)
    sec = rng.uniform(0.0, 1.0e4, n)                     # nonzero even when num_sec==0

    with uproot.recreate(path) as f:
        # NB: assigning a dict via f["Data"] = {...} writes an RNTuple on
        # this uproot version (5.7.2) rather than a classic TTree, and
        # data_handling.reader's `.arrays(library="pd")` doesn't group an
        # RNTuple with a string field into a real pandas DataFrame (it
        # silently returns an Awkward Array instead, breaking `.query()`).
        # Real GEANT4 output files are classic TTrees, so use the explicit
        # `mktree` call to match that format.
        f.mktree("Data", {
            "EventNum": ev,
            "StepNum": step,
            "X": rng.uniform(-0.5, 0.5, n),
            "Y": rng.normal(0.0, 0.01, n),
            "Z": rng.normal(0.0, 0.01, n),
            "StepLength": 10.0 ** rng.uniform(-9.0, -4.0, n),
            "angleMSC": 10.0 ** rng.uniform(-7.0, -1.0, n),
            "angleDiscrete": 10.0 ** rng.uniform(-7.0, -1.0, n),
            "KineticEnergy": ke,
            "SecondaryLoss": sec,
            "ContinuousLoss": rng.uniform(0.0, 1.0e4, n),
            "NumOfSecondaries": num_sec,
            "ProcessName": ak.Array([str(s) for s in proc]),
        })
