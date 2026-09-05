#!/bin/bash
#PBS -j oe
# PBS wrapper for the GEANT4 per-phase timing job. See
# phase_timing_node.sh for rationale and install_on_cluster.sh
# for the one-time shim install + rebuild.
#
#   qsub -q N -l select=1:ncpus=1:mem=8gb -l io=16 -l walltime=06:00:00 \
#        -N g4_phase_timing  phase_timing.pbs.sh
set -x
bash /srv01/agrp/rotemdo/StepsGenerator/timing_phases/phase_timing_node.sh
echo "=== DONE rc=$? ==="
