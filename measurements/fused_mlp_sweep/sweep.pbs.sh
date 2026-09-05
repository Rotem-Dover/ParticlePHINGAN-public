#!/bin/bash
#PBS -j oe
# Tile/launch sweep for the fused secondary-generator MLP kernel, one job,
# one card. The secondary net packs at (16, 16) tiles by default
# (`--force-tile` on test_fused_mlp.py picks the tile pair explicitly). This
# sweep asks whether a WIDER tile pair -- (32, 32), the next rung
# `--force-tile 32` resolves to (`_choose_tiles` with min_tile=32 and the
# secondary's widest real layer at 16: need=32 -> tile_b=32, no narrower rung
# fits -> tile_a=tile_b=32 too) -- beats the minimal (16, 16) one. The two
# tile arms are COMPARED to each other, so the within-job rule binds them
# together on one card: absolute walls from two different jobs are not
# comparable to better than a few percent.
#
#   qsub -q gpu -l select=1:ncpus=4:ngpus=1:gputype=A6000 -l io=16 \
#        -l walltime=02:00:00 -N p3_sweep sweep.pbs.sh
#
# Read repeat AGREEMENT, never BEST: each tile arm sweeps twice (r1 = full
# pass with the eager/inductor/fused microbench, one inductor compile; r2 =
# --no-bench, a repeat that hits the triton cache from r1 and exists only to
# put the run-to-run spread on record). A gap between tile arms smaller than
# the r1-vs-r2 spread on the SAME arm is not a win.
#
# The incumbent (32, 16)-block/warp launch config (_LAUNCH_CONFIG's
# SecondaryGenerator entry) is auto-anchored in BOTH tile arms by _sweep's
# force-incumbent rule, so a candidate always has a within-job baseline even
# though the launch config itself is not what --force-tile varies.
set -x
PY=/storage/agrp/rotemdo/venv/bin/python
SRC=/srv01/agrp/rotemdo/src

export OMP_NUM_THREADS=4
OUT=/storage/agrp/rotemdo/runtime_profiling/p3_sweep
mkdir -p "$OUT"
BLOCKS=16,32,64,128,256
WARPS=2,4,8,16

echo "=== host / gpu ==="
hostname
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

cd "$SRC" || exit 1

# `gputype=A6000` on this cluster resolves to an RTX 6000 Ada -- but the pool
# also holds real A6000s. This sweep's whole output is compared against a
# fixed baseline (the (16,16) minimal-tile arm measured in the SAME job), but
# absolute walls are only meaningful against an Ada-measured record, so
# refuse a different card. ALLOW_ANY_GPU=1 overrides, and the log then says
# so.
EXPECT="NVIDIA RTX 6000 Ada Generation"
GOT=$($PY -c "import torch;print(torch.cuda.get_device_name(0))")
echo "device_name=$GOT"
if [ "$GOT" != "$EXPECT" ]; then
  if [ "${ALLOW_ANY_GPU:-0}" = "1" ]; then
    echo "WARNING: card is '$GOT', not '$EXPECT'. ALLOW_ANY_GPU=1 -- continuing."
    echo "         These timings are NOT comparable to the Ada-measured record."
  else
    echo "REFUSING: card is '$GOT', not '$EXPECT'. Resubmit, or set"
    echo "          ALLOW_ANY_GPU=1 deliberately."
    exit 2
  fi
fi

$PY -c "import torch, triton, platform; print('torch', torch.__version__); print('triton', triton.__version__); print('python', platform.python_version())"
md5sum physics/interfaces/_fused_mlp_kernel.py \
       physics/interfaces/fused_mlp.py \
       benchmark/runtime_profiling/test_fused_mlp.py

echo "=== incumbent _LAUNCH_CONFIG ==="
PYTHONPATH=. $PY -c "
from physics.interfaces.fused_mlp import _LAUNCH_CONFIG as L
for k in ('ContinuousGenerator', 'SecondaryGenerator'):
    print(f'{k}: {L.get(k)}')"

# Self-documenting: print the ACTUAL resolved (tile_a, tile_b) pair for each
# --force-tile value below, rather than relying on this comment's arithmetic.
# build_spec is CPU-safe (only weight-tensor placement reads
# gen.parameters().device, which is fine on CPU) -- no CUDA needed for this
# check alone.
echo "=== resolved tile pairs per --force-tile ==="
PYTHONPATH=. $PY - <<'EOF'
import torch
from physics.g4h_ionisation.generators.secondary_generator.neural_net import SecondaryGenerator
from physics.interfaces.fused_mlp import build_spec
gen = SecondaryGenerator().eval()
for mt in (16, 32):
    s = build_spec(gen, min_tile=mt)
    print(f"force-tile {mt} -> tiles ({s.tile_a}, {s.tile_b})")
EOF

# $1 = tile (--force-tile value), $2 = pass label (r1 | r2). Only r2 gets
# --no-bench -- r1 carries the parity check AND the eager/inductor/fused
# bench (one inductor compile); r2 is the cheap warm-cache repeat that exists
# only for the run-to-run spread. NOTE: a naive `${2:+--no-bench}` would fire
# on BOTH r1 and r2 (since $2 is non-empty either way) -- guarded here with an
# explicit string test instead.
run_arm () {
  local no_bench_flag=""
  if [ "$2" = "r2" ]; then
    no_bench_flag="--no-bench"
  fi
  PHINGAN_FUSED_MLP=1 PYTHONPATH=. timeout 3600 $PY -u -m \
    benchmark.runtime_profiling.test_fused_mlp \
    --device cuda --n 1048576 --stages secondary \
    --force-tile "$1" --sweep --sweep-blocks "$BLOCKS" --sweep-warps "$WARPS" \
    $no_bench_flag 2>&1 | tee "$OUT/sweep_tile$1_$2.log"
  echo "RC_tile$1_$2=${PIPESTATUS[0]}"
}
# parity runs inside each harness invocation BEFORE the sweep, per tile arm.
echo "=== tile 16 (minimal -- the live secondary's own packing) ==="
run_arm 16 r1
run_arm 16 r2
echo "=== tile 32 (candidate: does a wider tile pay off) ==="
run_arm 32 r1
run_arm 32 r2

echo "=== secondary: BEST per repeat ==="
grep -h "BEST secondary:" "$OUT"/sweep_tile16_r1.log "$OUT"/sweep_tile16_r2.log \
                         "$OUT"/sweep_tile32_r1.log "$OUT"/sweep_tile32_r2.log

# continuous no-change control: parity only, default tiles, no sweep. The
# continuous net's tiling is untouched by this sweep -- this just confirms
# the fused path still parities cleanly in the same environment/checkpoint
# pair.
PHINGAN_FUSED_MLP=1 PYTHONPATH=. $PY -u -m \
  benchmark.runtime_profiling.test_fused_mlp \
  --device cuda --n 1048576 --stages continuous --no-bench \
  2>&1 | tee "$OUT/parity_continuous.log"
echo "RC_parity_continuous=${PIPESTATUS[0]}"

# Fusion-refusal check: confirm try_enable_fused_forward's refusal->fallback
# composition on a REAL structural mismatch, on-card (zero CPU coverage of
# this path -- the device check short-circuits before build_spec ever runs
# on CPU). A freshly-constructed SecondaryGenerator(output_activation=
# "sigmoid") is the mismatch: _extract_layers explicitly raises on a
# terminal nn.Sigmoid (the fused epilogue implements only the Hardtanh(0,1)
# clamp and the HardtanhReflectTop reflect rule), so
# try_enable_fused_forward must catch that ValueError and return False, not
# raise and not silently miscompute. forward() takes a 1-D `labels` tensor
# of shape [B] (see SecondaryGenerator.forward: labels.unsqueeze(1) feeds
# embed's Linear(1, E)), which torch.full((8,), 0.5) already matches -- no
# shape fix needed.
PYTHONPATH=. $PY - <<'EOF' 2>&1 | tee "$OUT/fusion_refusal_check.log"
import torch
from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
    SecondaryGenerator)
from physics.interfaces.fused_mlp import try_enable_fused_forward
gen = SecondaryGenerator(output_activation="sigmoid").eval().to("cuda")
assert try_enable_fused_forward(gen) is False   # refused, logged, no raise
with torch.inference_mode():
    out = gen.forward(torch.full((8,), 0.5, device="cuda"))
print("fusion refusal check OK: refused ->", tuple(out.shape))
EOF
echo "RC_fusion_refusal_check=${PIPESTATUS[0]}"

# Graph-break check: dynamo graph count for along_step_do_it with fusion on.
# Signature verified against src/physics/g4h_ionisation/g4h_ionisation.py:
#   def along_step_do_it(self, kinetic_energy: torch.Tensor,
#                         step_length: torch.Tensor, **kwargs) -> dict
# -- two positional tensor args after self, exactly as called below.
PHINGAN_FUSED_MLP=1 PYTHONPATH=. $PY - <<'EOF' 2>&1 | tee "$OUT/graph_break_check.log"
import torch
from benchmark.track_simulator_config import build_stepping_manager
sm = build_stepping_manager("phin_gan", device="cuda")
(proc,) = sm.processes
n = 1024
ke = torch.full((n,), 5e7, device="cuda")
L = torch.full((n,), 1e-6, device="cuda")
ex = torch._dynamo.explain(proc.along_step_do_it)(ke, L)
print("graph_count", ex.graph_count, "break_count", ex.graph_break_count)
EOF
echo "RC_graph_break_check=${PIPESTATUS[0]}"

echo "=== DONE ==="
ls -la "$OUT"
