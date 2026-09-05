#!/bin/bash
#PBS -j oe
# Gate B calibration, run in its OWN job, blind to the gating run: measures
# the eager-vs-eager seed-to-seed null at the pinned calibration size
# (GATE_B_CALIBRATION_RUN_SIZE = 200 events x 12000 steps,
# verify_physics_gate.py's own --n-events/--n-steps defaults, so nothing
# extra needs passing here). Never widen a threshold to turn a red gate
# green -- this run only PRODUCES the null; thresholds are set to 2x it.
#
#   qsub -q gpu -l select=1:ncpus=4:ngpus=1:gputype=A6000 -l io=16 \
#        -l walltime=04:00:00 -N p3_gateb_cal gate_b_calibrate.pbs.sh
set -x
PY=/storage/agrp/rotemdo/venv/bin/python
SRC=/srv01/agrp/rotemdo/src

export OMP_NUM_THREADS=4
OUT=/storage/agrp/rotemdo/runtime_profiling/p3_gateb
mkdir -p "$OUT"

echo "=== host / gpu ==="
hostname
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

cd "$SRC" || exit 1

# `gputype=A6000` on this cluster resolves to an RTX 6000 Ada -- but the pool
# also holds real A6000s, and gate B's own thresholds are pinned per-device.
# Refuse a non-Ada card rather than silently calibrating against the wrong
# population. ALLOW_ANY_GPU=1 overrides, and the log then says so.
EXPECT="NVIDIA RTX 6000 Ada Generation"
GOT=$($PY -c "import torch;print(torch.cuda.get_device_name(0))")
echo "device_name=$GOT"
if [ "$GOT" != "$EXPECT" ]; then
  if [ "${ALLOW_ANY_GPU:-0}" = "1" ]; then
    echo "WARNING: card is '$GOT', not '$EXPECT'. ALLOW_ANY_GPU=1 -- continuing."
    echo "         This null is NOT comparable to an Ada-measured threshold."
  else
    echo "REFUSING: card is '$GOT', not '$EXPECT'. Resubmit, or set"
    echo "          ALLOW_ANY_GPU=1 deliberately."
    exit 2
  fi
fi

$PY -c "import torch, triton, platform; print('torch', torch.__version__); print('triton', triton.__version__); print('python', platform.python_version())"

cd "$SRC" || exit 1
PYTHONPATH=. $PY -m benchmark.runtime_profiling.verify_physics_gate \
  --gate b --device cuda --calibrate 2>&1 | tee "$OUT/gate_b_null_p3.log"
echo "RC_calibrate=${PIPESTATUS[0]}"

echo "=== DONE ==="
ls -la "$OUT"
