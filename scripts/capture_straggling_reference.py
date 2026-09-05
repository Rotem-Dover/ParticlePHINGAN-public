"""Capture the straggling-model reference values that
tests/physics/test_straggling_reference_values.py compares against.

Usage (from the repository root):
    PYTHONPATH=src python scripts/capture_straggling_reference.py \\
        --out tests/data/straggling_reference.npz

The beam is asserted, not assumed: the file is only meaningful for the
100 MeV proton-in-aluminium beam.
"""
import argparse
from pathlib import Path

import numpy as np

import common.units as U
from common.run_context import PARTICLE, TARGET_MATERIAL
from common.utils import PointInPhaseSpace
from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import (
    ContinuousStragglingModel,
)
from physics.g4h_ionisation.explicit_physics.discrete.straggling_function import (
    DiscreteStragglingModel,
)
from physics.g4h_ionisation.explicit_physics.step_length.straggling_function import (
    StepLengthModel,
)

N_PDF = 512
# Lower bound is 0.5 MeV, not the beam's 0.4 MeV kill sentinel: at exactly
# 0.4 MeV DiscreteStragglingModel.set_state's own assertion (t_max >= Tc)
# fails for every step length -- the kinematic max delta-ray energy dips
# below the production cut right at that floor. This is a genuine boundary
# of the physics model, so nudging the grid above it is the correct fix
# rather than weakening any assertion.
KE_GRID_eV = np.logspace(np.log10(0.5 * U.MeV2eV), np.log10(100.0 * U.MeV2eV), 11)
L_GRID_m = np.logspace(np.log10(1e-11), np.log10(1e-4), 11)


def _beam_tag() -> str:
    return f"{PARTICLE.name}_in_{TARGET_MATERIAL.name}"


def _scalar_state(model) -> dict[str, float]:
    """Every public numeric attribute the model set for this state.

    Name-driven rather than a hardcoded list: set_fluct_params/set_glandz_params
    assign ~25 attributes and a hardcoded list would silently stop covering a
    new one. bool is excluded via the int check because bool is an int subclass.
    """
    return {
        k: float(v)
        for k, v in model.__dict__.items()
        if not k.startswith("_")
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    }


def capture(out_path: Path) -> None:
    assert _beam_tag() == "proton_in_Aluminum", (
        f"wrong beam: {_beam_tag()}. Run with the default PHINGAN_BEAM "
        "(unset selects proton_in_aluminum_100MeV)."
    )

    cnt, sec, length = ContinuousStragglingModel(), DiscreteStragglingModel(), StepLengthModel()

    ke_out, l_out, rows, types = [], [], [], []
    pdf_x, pdf_cnt, cdf_cnt, pdf_sec, pdf_len = [], [], [], [], []

    for ke in KE_GRID_eV:
        for l_m in L_GRID_m:
            point = PointInPhaseSpace(step_length=l_m * U.m2cm, primary_energy=float(ke))
            cnt.set_state(point)
            sec.set_state(point)
            length.set_state(point)

            # Grid depends on the INPUT only, so both sides evaluate on the
            # identical abscissa even if a model constant moved. Leading 0.0
            # is required, not cosmetic: few_ionization_collisions_pdf's own
            # normalization asserts x[0] == 0 ("should be called only with a
            # full range of x"); a pure logspace(0.0, ...) starts at 10**0 =
            # 1 and trips that assertion in the few-collisions regime.
            xs = np.concatenate(([0.0], np.logspace(0.0, np.log10(ke), N_PDF - 1)))

            ke_out.append(float(ke))
            l_out.append(float(l_m))
            rows.append(_scalar_state(cnt))
            types.append("|".join(sorted(t.name for t in cnt.straggling_type)))
            pdf_x.append(xs)
            pdf_cnt.append(np.asarray(cnt.straggling_function(xs), dtype=np.float64))
            cdf_cnt.append(np.asarray(cnt.straggling_cdf(xs), dtype=np.float64))
            pdf_sec.append(np.asarray(sec.straggling_function(xs), dtype=np.float64))
            # The step-length model's abscissa is a length in cm, not an energy.
            pdf_len.append(np.asarray(
                length.straggling_function(np.logspace(-9.0, 1.0, N_PDF)), dtype=np.float64))

    names = sorted({k for r in rows for k in r})
    scalars = np.array([[r.get(n, np.nan) for n in names] for r in rows], dtype=np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ke_ev=np.array(ke_out), l_m=np.array(l_out),
        scalar_names=np.array(names), scalars=scalars,
        straggling_type=np.array(types), beam=np.array(_beam_tag()),
        pdf_x=np.array(pdf_x), pdf_cnt=np.array(pdf_cnt), cdf_cnt=np.array(cdf_cnt),
        pdf_sec=np.array(pdf_sec), pdf_len=np.array(pdf_len),
    )
    print(f"wrote {out_path}: {scalars.shape[0]} points x {scalars.shape[1]} scalars")
    print(f"scalars captured: {', '.join(names)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    capture(p.parse_args().out)
