"""
Pure-Python port of GEANT4's initialisation-time physics tables for
ionisation energy loss.

Reproduces what G4VEnergyLossProcess::BuildDEDXTable / BuildRangeTable /
BuildLambdaTable emit at process initialisation, on the G4-standard
log-spaced energy grid, with G4-style cubic-spline interpolation.
Outputs land under `common.paths.INIT_TABLES_DIR`.
"""
