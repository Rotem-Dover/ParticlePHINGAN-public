#!/bin/bash
# GEANT4 proton CPU timing sweep for Figure 14 (no saving).
#
# src/benchmark/system_vs_g4/plot_computation_time.py reads
# measurements/computation_time/geant4_cpu_times.csv, the CSV this script
# produces. Saving is left off: an earlier electron timing sweep measured
# saving-ON at ~4x saving-OFF, so this is not a cosmetic choice.
#
# PHYSICS: ionisation only, already. PhysListEmStandard.cc:142 registers only
# G4hIonisation for proton/pi+-; G4hMultipleScattering, G4hBremsstrahlung and
# G4hPairProduction are commented out. This macro's /particle/process/dump
# confirms it at runtime (Transportation, hIoni, UserMaxStep only), so every
# log carries its own proof.
#
# The ONLY edit versus the recorded ~/StepsGenerator/PhDMacro.mac is
# setAbsThick 10 cm -> 5 cm, which is cost-neutral: a 100 MeV proton's CSDA
# range in Al is ~3.6 cm, so it stops inside the absorber either way. Saving
# is therefore the single measured variable.
#
# The chrono timer in TestEm5.cc wraps the WHOLE macro, so /run/initialize and
# physics-table construction are inside the measured time. That is the same
# convention the electron sweep also uses; it is why the n=10 point is
# overhead-dominated, and why the figure's linear fit is
# median(t/n) -- a fit robust to the low-n points rather than anchored on them.
#
# Submit with (io=16 is NOT optional -- an io=1 job on this cluster spent its
# wall clock blocked on /storage and was killed at 35 s of CPU):
#
#   qsub -q N -l select=1:ncpus=1:mem=8gb -l io=16 -l walltime=12:00:00 \
#        -N g4_proton_timing  geant4_timing.pbs.sh
#
. /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

DIR=/srv01/agrp/rotemdo/StepsGenerator/timing_proton
BIN=/srv01/agrp/rotemdo/StepsGenerator/build/TestEm5
JOBID=${PBS_JOBID%%.*}
LOG=$HOME/g4_timing_proton_${JOBID}.log
CSV=$DIR/g4_cpu_times_proton.csv

cd "$DIR" || exit 1
{
echo "=== GEANT4 proton CPU timing sweep (no saving) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
lscpu | grep "Model name"
echo "Binary: $BIN (GEANT4 11.2.0, LCG_105)"
echo "Config: 100 MeV proton, G4_Al 5 cm / YZ 30 cm, cut 1 um,"
echo "        killSecondaries 2, eLoss/fluct true, seeds 42 42,"
echo "        /runAction/disableSaving true, ionisation only"
echo ""
: > "$CSV"
for N in 10 100 1000 10000 100000; do
  sed "s/__N_EVENTS__/$N/" geant4_macro_template.mac > "run_$N.mac"
  echo "--- beamOn $N start: $(date) ---"
  "$BIN" "run_$N.mac" > "out_$N.log" 2>&1
  rc=$?
  T=$(grep "Time taken by function" "out_$N.log" | awk "{print \$5}")
  if [ $rc -ne 0 ] || [ -z "$T" ]; then
    echo "FAILED at N=$N (rc=$rc, timer line missing) — see $DIR/out_$N.log"
    exit 1
  fi
  echo "$N,$T" >> "$CSV"
  echo "N=$N -> $T s"
done
echo ""
echo "--- process dump from out_10.log (ionisation-only proof) ---"
sed -n '/G4ProcessManager: particle\[proton\]/,/### Run 0 starts/p' out_10.log \
  | grep -E "^\[[0-9]\]===" || echo "(dump not found)"
echo ""
echo "SWEEP COMPLETE: $(date)"
echo "CSV: $CSV"
} 2>&1 | tee "$LOG"
