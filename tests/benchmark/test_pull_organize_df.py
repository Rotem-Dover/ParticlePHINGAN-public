"""organize_df computes TOTAL_ENERGY_LOSS as ContinuousLoss + SecondaryLoss.

The simulated process is proton-only, ionisation-only: there is no
bremsstrahlung process and no brems energy column to fold in. `organize_df`
must not double-count a precomputed TOTAL_ENERGY_LOSS (the G4 reader
precomputes it for the truth files) and must derive it correctly when
absent.
"""
import numpy as np
import pandas as pd

from benchmark.system_vs_g4.pull_analysis.energy_deposition_pull_analysis import organize_df
from common import units as U
from common.enums import G4Columns, TOTAL_ENERGY_LOSS


def _base_df(**extra_cols) -> pd.DataFrame:
    data = {
        G4Columns.EventNum: [0, 0, 1],
        G4Columns.X: [0.0, 0.01, 0.02],
        G4Columns.Y: [0.0, 0.001, -0.001],
        G4Columns.ContinuousLoss: [1.0e6, 2.0e6, 3.0e6],
        G4Columns.SecondaryLoss: [0.5e6, 0.0, 1.5e6],
    }
    data.update(extra_cols)
    return pd.DataFrame(data)


def test_brems_column_is_not_folded_into_total_energy_loss():
    # The simulated process is proton-only, ionisation-only: there is no
    # brems process, so a brems-named column ("E_BREMS" or
    # G4Columns.BremsLoss, e.g. from a G4 truth file that carries one) must
    # NOT be folded into TOTAL_ENERGY_LOSS -- there is nothing to account
    # for.
    df = _base_df(**{
        "E_BREMS": [10.0e6, 0.0, 20.0e6],
        G4Columns.BremsLoss: [5.0e6, 0.0, 1.0e6],
    })
    out = organize_df(df.copy())
    expected_MeV = np.array([1.5e6, 2.0e6, 4.5e6]) * U.eV2MeV
    np.testing.assert_allclose(out[TOTAL_ENERGY_LOSS].values, expected_MeV)


def test_without_brems_column_total_is_cnt_plus_sec():
    out = organize_df(_base_df())
    expected_MeV = np.array([1.5e6, 2.0e6, 4.5e6]) * U.eV2MeV
    np.testing.assert_allclose(out[TOTAL_ENERGY_LOSS].values, expected_MeV)


def test_precomputed_total_energy_loss_is_untouched():
    # The G4 reader precomputes TOTAL_ENERGY_LOSS (photon k already inside
    # SecondaryLoss) — organize_df must not double-count anything on top.
    df = _base_df(**{
        TOTAL_ENERGY_LOSS: [7.0e6, 8.0e6, 9.0e6],
        "E_BREMS": [10.0e6, 0.0, 20.0e6],
    })
    out = organize_df(df)
    np.testing.assert_allclose(
        out[TOTAL_ENERGY_LOSS].values, np.array([7.0e6, 8.0e6, 9.0e6]) * U.eV2MeV
    )
