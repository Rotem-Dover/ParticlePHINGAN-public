#!/bin/bash
# GEANT4 proton per-phase CPU attribution: how the GEANT4 arm's ms/event
# (measurements/computation_time/geant4_cpu_times.csv) splits between
# hIoni's step-length (AlongStepGPIL+PostStepGPIL), continuous-loss
# (AlongStepDoIt) and secondary (PostStepDoIt) phases, plus "other" (stepping
# manager, Transportation, StepMax, tracking/stacking, user actions). The
# PHIN-GAN counterpart is profile_runtime's --stage-timing-events (see
# measurements/phase_timing/phin_gan_phases.pbs.sh).
#
# Same macro as the Figure 14 GEANT4 sweep (geant4_macro_template.mac) plus
# ONE line, `/testem/phys/timeProcesses __TIMING__`. With false the binary
# path is identical to the Figure 14 run (no wrapper object exists); with
# true hIoni is wrapped in TimedProcess (a G4WrapperProcess that forwards
# every call and reads the TSC around it). Verified locally: wrapped vs
# unwrapped ntuples are bit-identical over 500 events (4,902,177 steps), with
# an identical RNG end-state.
#
# Arms ALTERNATE off/on/off/on at N=1e4 so the in-situ wrapper overhead is
# (mean wall_on - mean wall_off)/calls within one job on one core -- the only
# CPU comparison this cluster supports, since shared nodes make cross-job CPU
# timing unreliable -- then one on-arm at 1e5 for the share statistics.
# Shares inside one run are immune to node contention (everything slows
# together); only the absolute ms/event is not.
#
# PREREQ (once): the shim must be installed in ~/StepsGenerator and the
# binary rebuilt -- see install_on_cluster.sh beside this file.
#
#   qsub -q N -l select=1:ncpus=1:mem=8gb -l io=16 -l walltime=06:00:00 \
#        -N g4_phase_timing  phase_timing.pbs.sh
#
. /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

DIR=/srv01/agrp/rotemdo/StepsGenerator/timing_phases
BIN=/srv01/agrp/rotemdo/StepsGenerator/build/TestEm5
JOBID=${PBS_JOBID%%.*}
LOG=$HOME/g4_phase_timing_${JOBID}.log
CSV=$DIR/g4_phase_timing.csv

mkdir -p "$DIR"; cd "$DIR" || exit 1
{
echo "=== GEANT4 proton per-phase timing (no saving) ==="
echo "Date: $(date)"; echo "Host: $(hostname)"; lscpu | grep "Model name"
echo "Binary: $BIN (GEANT4 11.2.0, LCG_105)"
echo "arm,N,beamOn_wall_s,ms_per_event,stepLen_raw_s,cont_raw_s,sec_raw_s,other_raw_s,calls_alongGPIL,calls_postGPIL,calls_alongDoIt,calls_postDoIt,tick_MHz,timer_ns_per_call" > "$CSV"
run_arm () {  # $1 = off|on, $2 = N, $3 = tag
  local T=$([ "$1" = on ] && echo true || echo false)
  sed -e "s/__N_EVENTS__/$2/" -e "s/__TIMING__/$T/" macro_template.mac > "run_$3.mac"
  echo "--- arm=$1 N=$2 start: $(date) ---"
  "$BIN" "run_$3.mac" > "out_$3.log" 2>&1; local rc=$?
  if [ $rc -ne 0 ]; then echo "FAILED $3 rc=$rc"; exit 1; fi
  if [ "$1" = on ]; then
    local W=$(grep "beamOn wall" "out_$3.log" | awk '{print $4}')
    local SL=$(grep "step length (GPIL" "out_$3.log" | awk '{print $5}')
    local CN=$(grep "continuous  (AlongDoIt)" "out_$3.log" | awk '{print $3}')
    local SE=$(grep "secondary   (PostDoIt)" "out_$3.log" | awk '{print $3}')
    local OT=$(grep "other (kernel" "out_$3.log" | awk '{print $3}')
    local CALLS=$(grep "calls           :" "out_$3.log" | sed 's/,//g' | awk '{print $4","$6","$8","$10}')
    local MHZ=$(grep "tick rate" "out_$3.log" | awk '{print $4}')
    local NS=$(grep "timer cost" "out_$3.log" | awk '{print $4}')
    echo "on,$2,$W,$(awk "BEGIN{print 1000*$W/$2}"),$SL,$CN,$SE,$OT,$CALLS,$MHZ,$NS" >> "$CSV"
  else
    local W=$(grep "ProcessTimer: off" "out_$3.log" | awk '{print $6}')
    echo "off,$2,$W,$(awk "BEGIN{print 1000*$W/$2}"),,,,,,,,,," >> "$CSV"
  fi
  grep -h "beamOn wall\|ProcessTimer: off\|of hIoni only" "out_$3.log"
}
run_arm off 10000 a1_off_1e4
run_arm on  10000 a2_on_1e4
run_arm off 10000 a3_off_1e4
run_arm on  10000 a4_on_1e4
run_arm on  100000 a5_on_1e5
echo ""; echo "--- process dump (on arm; must read Transportation, hIoni, UserMaxStep) ---"
grep -E "^\[[0-9]\]===" out_a2_on_1e4.log
echo ""; echo "--- full ProcessTimer block, 1e5 arm ---"
sed -n '/ProcessTimer (hIoni/,/=====================$/p' out_a5_on_1e5.log
echo ""; echo "SWEEP COMPLETE: $(date)"; echo "CSV: $CSV"
} 2>&1 | tee "$LOG"
