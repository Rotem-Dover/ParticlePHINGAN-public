#!/bin/bash
#PBS -j oe
# PHIN-GAN GPU per-phase breakdown at 1e6 events -- fused and eager, one job.
#
#   qsub -q gpu -l select=1:ncpus=4:ngpus=1:gputype=A6000 -l io=16 \
#        -l walltime=03:00:00 -N m2_phases  phin_gan_phases.pbs.sh
#
# WHY ONE JOB: comparing a fused-arm phase timing to an eager-arm phase
# timing from two different jobs would encode the job-to-job offset (a few
# percent in general, and sometimes much worse -- one job showed 68 %
# inflation in a single phase) as physics.
#
# INSTRUMENTED WALLS ARE ATTRIBUTION-ONLY, NEVER QUOTED AS THE HEADLINE
# NUMBER. --stage-timing-events 1000000 turns phase timers ON here, which is
# the point of this job (a per-phase breakdown), but the timers insert syncs
# that re-attribute time rather than create or destroy it. Read this job's
# wall_s figures for phase SHARES only; the end-to-end number comes from the
# un-instrumented sweep in measurements/computation_time/.
#
# PHINGAN_FUSED_MLP=1 is set explicitly on both invocations even though
# build_optimized_simulator latches it anyway (and --eager bypasses it); each
# JSON's generator report records the ACTUAL kernel mode, which is how a
# mislabelled arm gets caught.
set -x
PY=/storage/agrp/rotemdo/venv/bin/python
SRC=/srv01/agrp/rotemdo/src
OUT=/storage/agrp/rotemdo/runtime_profiling/m2_phases
mkdir -p "$OUT"
export OMP_NUM_THREADS=4

hostname
nvidia-smi --query-gpu=index,name,driver_version,clocks.max.sm,power.limit --format=csv
# gputype=A6000 on this cluster resolves to an Ada. Assert, never trust the label.
$PY -c "import torch;print('device_name',torch.cuda.get_device_name(0))"
lscpu | grep -E "Model name" | head -2
uptime

cd "$SRC" || exit 1

PHINGAN_FUSED_MLP=1 PYTHONPATH=. $PY -m benchmark.runtime_profiling.profile_runtime \
  --preset phin_gan --device cuda --label m2_fused --n-steps 10250 \
  --event-counts 1000000 --stage-timing-events 1000000 --torch-profile-events 0 \
  --out "$OUT/m2_fused_phases.json"
echo "RC_m2_fused=$?"

PHINGAN_FUSED_MLP=1 PYTHONPATH=. $PY -m benchmark.runtime_profiling.profile_runtime \
  --preset phin_gan --device cuda --label m2_eager --eager --n-steps 10250 \
  --event-counts 1000000 --stage-timing-events 1000000 --torch-profile-events 0 \
  --out "$OUT/m2_eager_phases.json"
echo "RC_m2_eager=$?"

echo "=== DONE ==="
