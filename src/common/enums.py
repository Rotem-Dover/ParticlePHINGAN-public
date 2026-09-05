"""
Enum definitions for indexing step features (step length, scattering angle,
continuous and secondary energy losses) and for mapping GEANT4 simulation
output column names used throughout data loading and analysis.
"""
from enum import IntEnum, StrEnum


class ContinuousFeats(IntEnum):
    E_CNT_IDX = 0


class SecondaryFeats(IntEnum):
    # Column layout of the SecondaryGenerator's output. A checkpoint contract,
    # not a convention — do not reorder or extend. The generator emits only
    # the secondary energy loss; the primary's scattering angle is
    # reconstructed from energy-momentum conservation
    # (`physics.g4h_ionisation.kinematics`) rather than sampled, so the net
    # does not need an angle column. `N_SECONDARY_FEATURES` also drives the
    # trunk's width (`N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES`), so
    # this column count is a width knob as well as an output shape.
    E_SEC_IDX = 0


class StepFeaturesIdxes(IntEnum):
    # Layout of the per-step feature tensor `G4hIonisation` and
    # `SteppingManager` produce: step length, the primary's scattering angle,
    # the continuous energy loss, the secondary energy loss, and the
    # geometric step length. GEOM_STEP always equals L, since nothing here
    # shortens the true step into a shorter geometric one; it is kept as its
    # own column so `TrackSimulator`'s geometry transform can read it
    # unconditionally.
    L         = 0
    THETA     = 1
    E_CNT     = 2
    E_SEC     = 3
    GEOM_STEP = 4


class G4Columns(StrEnum):
    EventNum         = "EventNum"
    StepNum          = "StepNum"
    X                = "X"
    Y                = "Y"
    Z                = "Z"
    StepLength       = "StepLength"
    GeomStepLength   = "GeomStepLength"
    LateralDisplacement = "LateralDisplacement"
    angleMSC         = "angleMSC"
    angleDiscrete    = "angleDiscrete"
    KineticEnergy    = "KineticEnergy"
    SecondaryLoss    = "SecondaryLoss"
    ContinuousLoss   = "ContinuousLoss"
    BremsLoss        = "BremsLoss"
    NumOfSecondaries = "NumOfSecondaries"
    ProcessName      = "ProcessName"
    Time             = "Time"


TOTAL_ENERGY_LOSS = "TotalEnergyLoss"
