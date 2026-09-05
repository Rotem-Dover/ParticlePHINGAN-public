#!/usr/bin/env bash
# Deploy src/ to the cluster and submit a WGAN-GP training job.
#
# Usage:
#   ./deploy_and_train.sh continuous                     # fresh run
#   ./deploy_and_train.sh secondary
#   ./deploy_and_train.sh continuous --resume <run-id>   # continue a run
#                                                        # (old version_N
#                                                        # names still work)
#   ./deploy_and_train.sh continuous --dry-run           # no runs/ branch;
#                                                        # code still rsyncs,
#                                                        # submitter only prints
#                                                        # the qsub line
#   ./deploy_and_train.sh continuous --beam=proton_in_iron_100MeV
#                                                        # non-default material;
#                                                        # default is
#                                                        # proton_in_aluminum_100MeV.
#                                                        # See physics/definitions/
#                                                        # beam.py PRESETS for the
#                                                        # full set of valid values.
#   ./deploy_and_train.sh secondary --sec-output-act=sigmoid  # Sigmoid terminal
#                                                        # (PHINGAN_SEC_OUTPUT_ACT;
#                                                        # default = config's
#                                                        # SECONDARY_OUTPUT_ACTIVATION)
#   ./deploy_and_train.sh secondary --sec-output-act=stretched_sigmoid --sec-stretch-eps=0.01
#                                                        # stretched-sigmoid terminal
#                                                        # (PHINGAN_SEC_STRETCH_EPS; token
#                                                        # _stretched_sigmoid_eps0.01)
#   ./deploy_and_train.sh secondary --sec-width=16       # trunk width multiplier
#                                                        # (PHINGAN_SEC_WIDTH;
#                                                        # default = config's
#                                                        # N_NEURON_MULTIPLIER_SEC)
#   ./deploy_and_train.sh secondary --seed=7             # per-run RNG seed
#                                                        # (PHINGAN_SEED in the
#                                                        # job env; default 42);
#                                                        # RUN_ID's stage token
#                                                        # becomes secondary_s7
#   ./deploy_and_train.sh continuous --ckpt-every=10     # checkpoint cadence
#                                                        # in epochs
#                                                        # (PHINGAN_CKPT_EVERY;
#                                                        # default 100 -- mind
#                                                        # the /storage inode
#                                                        # quota, save_top_k=-1
#                                                        # keeps every file)
#   ./deploy_and_train.sh secondary --gputype=A5000      # any submitter arg
#                                                        # is forwarded verbatim
#   ./deploy_and_train.sh continuous --no-phys           # no-physics-
#                                                        # regularization
#                                                        # ablation arm; RUN_ID's
#                                                        # stage token becomes
#                                                        # continuous_nophys
#
# The stage is the first argument; everything after it is forwarded verbatim
# to the matching submit_<stage>_job.py.
#
# Every deploy is tracked two ways:
#   * RUN_ID = <ts>_<stage>_<mat>_<short-sha>_<uniq> names the TensorBoard
#     run directory. It travels in the qsub job env (PHINGAN_RUN_ID, via the
#     submitter's --run-id) — NOT in a shared src/.run_id file, which the
#     next deploy would overwrite while this job is still queued.
#   * The exact deployed tree is snapshotted to a git branch
#     runs/<stage>/<timestamp>, committing dirty state if needed, so the
#     code behind any run is recoverable with `git checkout` of that branch.
#
# THIS SCRIPT SYNCS CODE ONLY. storage/ is never rsynced -- see
# ./scripts/stage_artifacts.sh, which must have been run once before
# the first job. The submitters preflight for those artifacts and refuse
# rather than queueing a job that would die on a GPU node.
#
# Follow progress live:
#   ssh gpu2 tail -f 'train_<stage>.o<jobid>'
set -euo pipefail

CLUSTER_HOST="gpu2"
CLUSTER_REPO="/srv01/agrp/rotemdo"

# Set the moment before `git stash push` and cleared right after a successful
# `git stash pop`; the EXIT trap below reads it to tell an ordinary abort
# apart from one that leaves uncommitted work sitting in the stash.
STASH_SAVED=0
on_exit() {
  local status=$?
  if [[ "$STASH_SAVED" == "1" && "$status" != "0" ]]; then
    echo "!! aborted mid-snapshot: your uncommitted changes are in the git stash -- inspect 'git stash list', restore with 'git stash pop'" >&2
  fi
}
trap on_exit EXIT

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {continuous|secondary} [submitter args...]" >&2
  exit 2
fi
STAGE="$1"; shift
case "$STAGE" in
  continuous|secondary) ;;
  *) echo "unknown stage '$STAGE' (expected continuous or secondary)" >&2; exit 2 ;;
esac
SUBMIT_SCRIPT="src/cluster_scripts/submit_${STAGE}_job.py"

# Non-destructive scan of the forwarded args for --resume (both the
# space-separated and --resume=<id> forms) and --dry-run: resuming reuses the
# prior RUN_ID (so the curves continue in one directory) instead of minting a
# fresh one, and a dry run makes no run to track at all. The args still
# forward to the submitter untouched either way.
RESUME_ID=""
DRY_RUN=0
NO_PHYS=0
SEED=""
SEC_OUTPUT_ACT=""
SEC_STRETCH_EPS=""
SEC_WIDTH=""
BEAM="proton_in_aluminum_100MeV"
prev=""
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    if [[ "$prev" == "--resume" ]]; then
      RESUME_ID="$arg"
    fi
    if [[ "$arg" == --resume=* ]]; then
      RESUME_ID="${arg#--resume=}"
    fi
    if [[ "$arg" == "--dry-run" ]]; then
      DRY_RUN=1
    fi
    if [[ "$arg" == "--no-phys" ]]; then
      NO_PHYS=1
    fi
    if [[ "$prev" == "--seed" ]]; then
      SEED="$arg"
    fi
    if [[ "$arg" == --seed=* ]]; then
      SEED="${arg#--seed=}"
    fi
    if [[ "$prev" == "--sec-output-act" ]]; then
      SEC_OUTPUT_ACT="$arg"
    fi
    if [[ "$arg" == --sec-output-act=* ]]; then
      SEC_OUTPUT_ACT="${arg#--sec-output-act=}"
    fi
    if [[ "$prev" == "--sec-stretch-eps" ]]; then
      SEC_STRETCH_EPS="$arg"
    fi
    if [[ "$arg" == --sec-stretch-eps=* ]]; then
      SEC_STRETCH_EPS="${arg#--sec-stretch-eps=}"
    fi
    if [[ "$prev" == "--sec-width" ]]; then
      SEC_WIDTH="$arg"
    fi
    if [[ "$arg" == --sec-width=* ]]; then
      SEC_WIDTH="${arg#--sec-width=}"
    fi
    if [[ "$prev" == "--beam" ]]; then
      BEAM="$arg"
    fi
    if [[ "$arg" == --beam=* ]]; then
      BEAM="${arg#--beam=}"
    fi
    prev="$arg"
  done
fi
if [[ "$prev" == "--resume" && -z "$RESUME_ID" ]]; then
  echo "--resume requires a run id (the TensorBoard run-directory name)" >&2
  exit 2
fi
if [[ "$prev" == --resume=* && -z "$RESUME_ID" ]]; then
  echo "--resume requires a run id (the TensorBoard run-directory name)" >&2
  exit 2
fi
if [[ "$RESUME_ID" == --* ]]; then
  echo "--resume requires a run id, got flag-like '$RESUME_ID'" >&2
  exit 2
fi

case "$BEAM" in
  proton_in_aluminum_100MeV|proton_in_iron_100MeV|proton_in_beryllium_100MeV) ;;
  *) echo "unknown --beam '$BEAM' (see physics/definitions/beam.py PRESETS)" >&2; exit 2;;
esac
MAT="${BEAM#proton_in_}"; MAT="${MAT%_100MeV}"

cd "$(git rev-parse --show-toplevel)"

TS="$(date +%Y-%m-%d-%H%M%S)"
BRANCH="runs/${STAGE}/${TS}"
ORIG_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ "$DRY_RUN" == "1" ]]; then
  # No run is actually happening, so there is nothing to snapshot: skip both
  # the branch-collision check and the stash round-trip and just read the
  # code that would have been deployed.
  SHA="$(git rev-parse HEAD)"
  echo ">> dry run: no runs/ branch created"
else
  # A same-second re-run would otherwise collide on `git checkout -b`
  # mid-snapshot with the dirty tree already stashed -- check up front, while
  # nothing has been touched yet, so the failure is safe rather than leaving
  # a stash to recover from.
  if git rev-parse --verify --quiet "refs/heads/$BRANCH" >/dev/null; then
    echo "!! branch $BRANCH already exists -- refusing to overwrite a run snapshot" >&2
    exit 1
  fi

  # Snapshot what is actually being rsynced to a run-tracking branch. Created
  # on resume too: a resumed leg's code can differ from an earlier leg's,
  # and the branch is what records it. Dirty state is committed onto the
  # branch, then restored uncommitted to the working tree so iteration
  # continues where it left off.
  if [[ -n "$(git status --porcelain)" ]]; then
    echo ">> dirty tree, snapshotting to $BRANCH"
    git stash push -u -m "deploy_and_train $STAGE @ $TS"
    STASH_SAVED=1
    STASH_REF="$(git rev-parse 'stash@{0}')"
    git checkout -b "$BRANCH"
    git stash apply "$STASH_REF"
    git add -A
    git commit -m "run snapshot: $STAGE @ $TS" --no-verify
    SHA="$(git rev-parse HEAD)"
    git checkout "$ORIG_BRANCH"
    git stash pop
    STASH_SAVED=0
  else
    SHA="$(git rev-parse HEAD)"
    git branch "$BRANCH" "$SHA"
    echo ">> clean tree, tagged $BRANCH -> $SHA"
  fi
fi
SHORT_SHA="${SHA:0:10}"

STAGE_TOKEN="$STAGE"
if [[ "$NO_PHYS" == "1" ]]; then
  STAGE_TOKEN="${STAGE}_nophys"
fi
if [[ -n "$SEED" ]]; then
  if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "--seed must be a non-negative integer, got '$SEED'" >&2
    exit 2
  fi
  STAGE_TOKEN="${STAGE_TOKEN}_s${SEED}"
fi
if [[ -n "$SEC_OUTPUT_ACT" ]]; then
  if [[ "$SEC_OUTPUT_ACT" != "hardtanh" && "$SEC_OUTPUT_ACT" != "sigmoid" \
        && "$SEC_OUTPUT_ACT" != "stretched_sigmoid" && "$SEC_OUTPUT_ACT" != "hardtanh_open_top" \
        && "$SEC_OUTPUT_ACT" != "hardtanh_reflect_top" ]]; then
    echo "--sec-output-act must be one of hardtanh, sigmoid, stretched_sigmoid, hardtanh_open_top, hardtanh_reflect_top; got '$SEC_OUTPUT_ACT'" >&2
    exit 2
  fi
  # the flag itself is forwarded to the submitter via "$*" below, which
  # exports it to the job as PHINGAN_SEC_OUTPUT_ACT; here we only name the run
  STAGE_TOKEN="${STAGE_TOKEN}_${SEC_OUTPUT_ACT}"
fi
if [[ -n "$SEC_STRETCH_EPS" ]]; then
  if ! [[ "$SEC_STRETCH_EPS" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "--sec-stretch-eps must be a positive decimal, got '$SEC_STRETCH_EPS'" >&2
    exit 2
  fi
  if [[ "$SEC_OUTPUT_ACT" != "stretched_sigmoid" ]]; then
    echo "--sec-stretch-eps only applies with --sec-output-act=stretched_sigmoid" >&2
    exit 2
  fi
  STAGE_TOKEN="${STAGE_TOKEN}_eps${SEC_STRETCH_EPS}"
fi
if [[ -n "$SEC_WIDTH" ]]; then
  if ! [[ "$SEC_WIDTH" =~ ^[0-9]+$ ]]; then
    echo "--sec-width must be a positive integer, got '$SEC_WIDTH'" >&2
    exit 2
  fi
  STAGE_TOKEN="${STAGE_TOKEN}_w${SEC_WIDTH}"
fi

if [[ -n "$RESUME_ID" ]]; then
  RUN_ID="$RESUME_ID"
  echo ">> RESUMING RUN_ID = $RUN_ID"
else
  UNIQ="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])' 2>/dev/null || echo "$$$RANDOM")"
  RUN_ID="${TS}_${STAGE_TOKEN}_${MAT}_${SHORT_SHA}_${UNIQ}"
  echo ">> RUN_ID = $RUN_ID"
fi

# Same rsync contract as deploy_and_profile.sh: exclude .run_id so a queued
# job's run id is never clobbered. A job's own RUN_ID travels in its qsub
# env instead, but the cluster src/ tree is shared across deploys, and
# another queued job may still be reading that file.
echo ">> rsync src/ -> ${CLUSTER_HOST}:${CLUSTER_REPO}/src/"
rsync -az --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  --exclude '.run_id' \
  src/ "${CLUSTER_HOST}:${CLUSTER_REPO}/src/"

# The submitter imports `common.paths` for its preflight, so it needs the
# project's dependencies, and the head node's default `python` is not the
# project's venv. Activate the same venv the node scripts source.
# PYTHONPATH=src because the submitter is run from the repo root, not from
# src/.
echo ">> submitting $SUBMIT_SCRIPT on $CLUSTER_HOST (git sha $SHA)"
ssh "$CLUSTER_HOST" "bash -lc 'source /storage/agrp/rotemdo/venv/bin/activate && cd ${CLUSTER_REPO} && PYTHONPATH=src PHINGAN_BEAM=${BEAM} python ${SUBMIT_SCRIPT} --git-sha ${SHA} --run-id ${RUN_ID} $*'"

echo ">> done. follow progress with:  ssh ${CLUSTER_HOST} tail -f 'train_${STAGE}.o<jobid>'"
if [[ "$DRY_RUN" != "1" ]]; then
  echo ">> inspect this run's exact code with:  git checkout ${BRANCH}"
fi
