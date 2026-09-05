import struct

import uproot
import pickle
import numpy  as np
import pandas as pd
import logging

from pathlib    import Path

import common.units as u
from common.enums import G4Columns, TOTAL_ENERGY_LOSS
from common.config import MIN_ENERGY_CUTOFF
from common.paths import PATH_TO_GEANT4_DATA


log = logging.getLogger(__name__)


def read_geant4_simulation_output(path_to_samples: Path = PATH_TO_GEANT4_DATA, verbose: bool = True) -> pd.DataFrame:
    """
    Reads the GEANT4 simulation output csv file and filters some irrelevant rows.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    if path_to_samples.with_suffix('.pkl').exists():
        log.info("Reading GEANT4 simulation output pickle file...")
        with open(path_to_samples.with_suffix('.pkl'), 'rb') as f:
            df: pd.DataFrame = pickle.load(f)

        return df

    log.info("Reading GEANT4 simulation output root file...")
    # Open the binary ROOT file
    with uproot.open(path_to_samples, num_workers=8) as doc:
        # "Data" is the name of the NTuple tree assigned in Geant4
        df = doc["Data"].arrays(library="pd")

    df = (
        df.query(f'{G4Columns.StepNum} > 0')  # First step is in vacuum
        .assign(StepNum=lambda x: x[G4Columns.StepNum] - 1)
    )

    # Correcting Secondary Energy Loss (numerical errors)
    df.loc[df[G4Columns.NumOfSecondaries] == 0, G4Columns.SecondaryLoss] = 0

    df = _filter(df)

    df = _add_extension_columns(df)
    log.info(f"Finished reading GEANT4 output file. Total steps: {df.shape[0]}")

    df.to_pickle(path_to_samples.with_suffix('.pkl'))
    logging.info(f"Saved GEANT4 simulation output to {path_to_samples.with_suffix('.pkl')}")

    return df


def _add_extension_columns(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Adding extension columns to the dataframe...")

    log.info("\tAdding TotalEnergyLoss column...")
    df[TOTAL_ENERGY_LOSS] = df[G4Columns.ContinuousLoss] + df[G4Columns.SecondaryLoss]

    return df


def _filter(df: pd.DataFrame):
    N0 = df.shape[0]
    df = df.query(f"{G4Columns.KineticEnergy} > {MIN_ENERGY_CUTOFF}").reset_index(drop=True)
    N1 = df.shape[0]
    log.info(f"\tFiltered out {N1 - N0} steps with kinetic energy below {MIN_ENERGY_CUTOFF} eV...")
    df = df.query(f"{G4Columns.ProcessName} != 'Transportation'").reset_index(drop=True)
    N2 = df.shape[0]
    log.info(f"\tFiltered out {N2-N1} Transportation Steps...")

    return df.sort_values(by=[G4Columns.EventNum, G4Columns.StepNum], ignore_index=True)


def read_secondary_frac_data(path: Path | str) -> pd.DataFrame:
    """
    Read Geant4 lambda debug binary file into a pandas DataFrame.

    Record layout (written by G4VEnergyLossProcess):
      [preStepEnergy_MeV, preSecEnergy_MeV, preStepInvLambda_per_cm, lx_cm]
    where each field is a C++ double (8 bytes).
    Despite the GEANT4 member being named `preStepLambda`, it is the
    macroscopic cross section Σ = 1/MFP [1/cm], not the mean free path
    itself. The probability of generating a secondary on a step of length
    lx is therefore lx * preStepInvLambda (≈ 1 - exp(-lx·Σ)).

    Parameters
    ----------
    path : Path | str
        Path to G4VEnergyLossProcess_lambda_debug.bin

    Returns
    -------
    pd.DataFrame
        Columns: preStepEnergy, preSecEnergy, preStepLambda, lx
    """
    record_format = "=dddd"  # native endianness, 4 doubles, no padding

    with open(path, "rb") as f:
        data = f.read()

    rows = list(struct.iter_unpack(record_format, data))
    df = pd.DataFrame(rows, columns=["preStepEnergy", "preSecEnergy", "preStepLambda", "lx"])
    df[["preStepEnergy", "preSecEnergy"]] *= u.MeV2eV
    # df[["preStepLambda", "lx"]] *= u.cm2m
    return df
