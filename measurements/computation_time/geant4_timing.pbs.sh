#!/bin/bash
#PBS -j oe
# PBS wrapper for the GEANT4 proton timing sweep. See
# geant4_timing_node.sh for the physics rationale.
#
#   qsub -q N -l select=1:ncpus=1:mem=8gb -l io=16 -l walltime=12:00:00 \
#        -N g4_proton_timing  geant4_timing.pbs.sh
#
# io=16 is mandatory; see project rule on PBS io throttling.
set -x
bash /srv01/agrp/rotemdo/StepsGenerator/timing_proton/geant4_timing_node.sh
echo "=== DONE rc=$? ==="
