#!/bin/bash
#PBS -j oe
# Gate B gating run: the optimization gate, comparing the optimized build
# against the eager build. Submit ONLY after the thresholds derived from
# gate_b_calibrate.pbs.sh's null are committed and deployed -- this run
# gates the ACTUAL optimized-vs-eager delta against those thresholds and
# exits nonzero on a breach.
#
#   qsub -q gpu -l select=1:ncpus=4:ngpus=1:gputype=A6000 -l io=16 \
#        -l walltime=04:00:00 -N p3_gateb_run gate_b.pbs.sh
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
# also holds real A6000s, and gate B's thresholds were calibrated on an Ada.
# Refuse a non-Ada card rather than gating against the wrong population.
# ALLOW_ANY_GPU=1 overrides, and the log then says so.
EXPECT="NVIDIA RTX 6000 Ada Generation"
GOT=$($PY -c "import torch;print(torch.cuda.get_device_name(0))")
echo "device_name=$GOT"
if [ "$GOT" != "$EXPECT" ]; then
  if [ "${ALLOW_ANY_GPU:-0}" = "1" ]; then
    echo "WARNING: card is '$GOT', not '$EXPECT'. ALLOW_ANY_GPU=1 -- continuing."
    echo "         This gating run is NOT comparable to the Ada-calibrated"
    echo "         thresholds."
  else
    echo "REFUSING: card is '$GOT', not '$EXPECT'. Resubmit, or set"
    echo "          ALLOW_ANY_GPU=1 deliberately."
    exit 2
  fi
fi

$PY -c "import torch, triton, platform; print('torch', torch.__version__); print('triton', triton.__version__); print('python', platform.python_version())"

cd "$SRC" || exit 1

# Parity prelude: re-run the fused-MLP parity harness once, on-card, in THIS
# job, so the on-card evidence for the fused path sits beside the gate B run
# it certifies, per the within-one-job-on-one-card rule. `--stages` defaults
# to "continuous,secondary" (both nets); `--no-bench` skips the timing
# sweep -- only the numeric kernel-vs-emulator/eager parity check matters
# here.
PHINGAN_FUSED_MLP=1 PYTHONPATH=. $PY -m \
  benchmark.runtime_profiling.test_fused_mlp \
  --device cuda --n 1048576 --no-bench 2>&1 | tee "$OUT/parity_prefix_gateb.log"
PARITY_RC="${PIPESTATUS[0]}"
echo "PARITY_RC=${PARITY_RC}"
# Structural, not just logged: a failed parity prelude must never be
# followed by a quotable gate B verdict in the same log -- a red parity
# check means the on-card fused path is not trustworthy, so the gate B run
# below would be certifying nothing. Fail loud and stop here instead.
[ "$PARITY_RC" = "0" ] || exit 3

PHINGAN_FUSED_MLP=1 PYTHONPATH=. $PY -m \
  benchmark.runtime_profiling.verify_physics_gate \
  --gate b --device cuda --out "$OUT" 2>&1 | tee "$OUT/gate_b_run_p3.log"
echo "GATE_B_RC=${PIPESTATUS[0]}"

echo "=== DONE ==="
ls -la "$OUT"
