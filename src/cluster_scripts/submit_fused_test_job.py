"""
Submits a short GPU job running the fused MLP parity +
microbenchmark harness (benchmark.runtime_profiling.test_fused_mlp).
Log: ~/fused_test_<jobid>.log on the cluster.

Runs on the login node WITHOUT the venv — must not import common.*.
"""
import argparse
import os

CLUSTER_RUN_DIR = "/srv01/agrp/rotemdo/src/"
SCRIPT_PATH = "/srv01/agrp/rotemdo/src/cluster_scripts/run_on_node_fused_test.sh"

parser = argparse.ArgumentParser(description="Submit fused-MLP parity/bench job")
parser.add_argument('--ncpus', '-nc',   type=str, default='4')
parser.add_argument('--mem', '-mem',    type=str, default='16gb')
parser.add_argument('--walltime', '-wt', type=str, default='02:00:00')
parser.add_argument('--gputype', '-gt', type=str, default='A6000')
# No --particle: the active beam is fixed by common.run_context via
# PHINGAN_BEAM, not by a submitter flag.
parser.add_argument('--tf32',           type=str, nargs='?', const='1', default='0',
                    help='PHINGAN_FUSED_MLP_TF32 for the job: bare --tf32 -> "1" (all '
                         'stages); or a PBS-safe \':\'-separated stage list, e.g. '
                         '--tf32 length:continuous:secondary. TF32 stages use a wide '
                         '1e-1 parity tol (dz measured + reported; gate judges)')
parser.add_argument('--extra-args',     type=str, default='',
                    help='Forwarded verbatim to test_fused_mlp (e.g. "--batch 1048576")')
parser.add_argument('--dry-run', '-n',  action='store_true')
args = parser.parse_args()

if ',' in args.tf32:  # PBS -v splits on commas
    args.tf32 = args.tf32.replace(',', ':')

resources = (f"walltime={args.walltime},mem={args.mem},ncpus={args.ncpus},"
             f"io=1,ngpus=1,gputype={args.gputype}")
env = (
    f'RUN_DIR="{CLUSTER_RUN_DIR}"'
    f',NCPUS={args.ncpus}'
    f',PHINGAN_FUSED_MLP_TF32="{args.tf32}"'
    f',FUSED_TEST_ARGS="{args.extra_args}"'
)
command = f'qsub -q gpu -N fused_mlp_test -l {resources} -v {env} {SCRIPT_PATH}'
print(command)
if args.dry_run:
    print("[DRY RUN] not submitted")
else:
    os.system(command)
