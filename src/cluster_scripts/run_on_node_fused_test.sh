#!/bin/bash
# Node-side runner for the fused MLP parity + microbenchmark harness
# (benchmark.runtime_profiling.test_fused_mlp). Env via qsub -v from
# submit_fused_test_job.py.
source ~/.bashrc
source /storage/agrp/rotemdo/venv/bin/activate

echo "===== System Info ====="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
nvidia-smi || echo "nvidia-smi not available"

if [ -n "$NCPUS" ] && [ "$NCPUS" -gt 0 ]; then
    export OMP_NUM_THREADS=$NCPUS
fi

cd $RUN_DIR
set -o pipefail
python -u -m benchmark.runtime_profiling.test_fused_mlp ${FUSED_TEST_ARGS:-} 2>&1 \
    | tee "$HOME/fused_test_${PBS_JOBID%%.*}.log"
exit $?
