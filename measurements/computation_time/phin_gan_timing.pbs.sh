#!/bin/bash
#PBS -j oe
# PHIN-GAN GPU sweep for Figure 14 -- the full 10..1e6 event-count grid on
# the accelerated build, ALL SIX POINTS IN ONE JOB.
#
#   qsub -q gpu -l select=1:ncpus=4:ngpus=1:gputype=A6000 -l io=16 \
#        -l walltime=02:00:00 -N fig_runtime_gpu  phin_gan_timing.pbs.sh
#
# WHY ONE JOB: the job-to-job offset is ~3 % in general, and one job showed a
# 68 % inflation in a single phase while its network phases matched to ~1 %.
# A curve stitched from six jobs would encode that scatter as physics.
#
# WHY THE TIMERS ARE OFF (--stage-timing-events 0, --torch-profile-events 0):
# the recorded wall_s must be the UN-INSTRUMENTED sweep wall. Phase timers
# insert syncs that re-attribute rather than create time.
#
# NO LEADING POINT IS DISCARDED: profile_runtime does its own warmup run
# before the sweep.
#
# PHINGAN_FUSED_MLP=1 is set explicitly even though build_optimized_simulator
# latches it anyway; each JSON's generator report records the ACTUAL kernel
# mode, which is how a mislabelled arm gets caught.
#
# Four alternating arms this run, not two: fused r1 / eager r1 / fused r2 /
# eager r2, so the fused-vs-eager comparison sits inside one job the same way
# a pair of fused-only repeats would.
set -x
PY=/storage/agrp/rotemdo/venv/bin/python
SRC=/srv01/agrp/rotemdo/src
OUT=/storage/agrp/rotemdo/runtime_profiling/m1_fig14
mkdir -p "$OUT"
export OMP_NUM_THREADS=4

hostname
nvidia-smi --query-gpu=index,name,driver_version,clocks.max.sm,power.limit --format=csv
# gputype=A6000 on this cluster resolves to an Ada. Assert, never trust the label.
$PY -c "import torch;print('device_name',torch.cuda.get_device_name(0))"
lscpu | grep -E "Model name" | head -2
uptime

cd "$SRC" || exit 1

# The console-visible summary is filtered by grep, but the FULL unfiltered
# output is always tee'd to "$OUT/$1.raw.log" first -- a bare grep alone
# silently drops anything that doesn't match, Traceback lines included. Read
# the .raw.log on any RC_* != 0.
run_sweep () {   # $1 = label, $2 = extra args
  PHINGAN_FUSED_MLP=1 PYTHONPATH=. $PY -m benchmark.runtime_profiling.profile_runtime \
    --preset phin_gan --device cuda --label "$1" --n-steps 10250 $2 \
    --event-counts 10,100,1000,10000,100000,1000000 \
    --stage-timing-events 0 --torch-profile-events 0 \
    --out "$OUT/$1.json" 2>&1 | tee "$OUT/$1.raw.log" \
    | grep -E "^>>|wall|events/s|sweep:|device|Traceback|Error|error|RuntimeError|AttributeError|assert" | tail -30
  echo "RC_$1=${PIPESTATUS[0]}"
}

run_sweep m1_fused_r1 ""
run_sweep m1_eager_r1 "--eager"
run_sweep m1_fused_r2 ""
run_sweep m1_eager_r2 "--eager"

echo "=== DONE ==="
