#!/usr/bin/env bash
# Stage the storage artifacts a training job reads onto the cluster.
#
#   ./scripts/stage_artifacts.sh              # copy what is missing
#                                                     # (default: proton/aluminum)
#   ./scripts/stage_artifacts.sh --dry-run    # show what would move
#   ./scripts/stage_artifacts.sh --beam=proton/iron
#                                                     # stage a non-default
#                                                     # material; valid values
#                                                     # are proton/aluminum,
#                                                     # proton/iron,
#                                                     # proton/beryllium
#
# Run this ONCE before the first training submission. `deploy_and_train.sh`
# rsyncs `src/` only, and nothing else syncs `storage/` -- the cluster's
# storage tree is populated by hand.
#
# WHAT MOVES, under `resources/<beam>/`, and why each one is required:
#
#   datasets/       the training pairs for both stages.
#   scalers/        ContinuousYScaler.state.pt -- the analytic mean/std
#                   lookup in the portable form the scaler reads.
#                   ContinuousYScaler.load() RAISES without it; nothing
#                   builds it at load time.
#   tables/         the analytic mean/std pickle the scaler state was built
#                   FROM. Not read at training time; staged so the scaler
#                   state can be regenerated on the cluster directly.
#   cdfs/           continuous_cdfs.pkl and secondaries_cdfs.pkl, the
#                   theoretical CDFs the physics penalty scores against.
#                   Both GAN constructors REFUSE when absent rather than
#                   training unregularized.
#   init_tables/    dedx / range / scaled_kin_energy / lambda_ion .npz. NOT
#                   obvious: the CONTINUOUS datamodule needs them too. Its
#                   Bethe-Bloch filter builds a ContinuousStragglingModel,
#                   whose Geant4Tables base calls get_init_table_set() at
#                   __init__, so a missing table fails setup() with
#                   FileNotFoundError before training starts. Cheap to
#                   stage, and `python -m physics.g4_init_tables.build_all`
#                   rebuilds them if needed.
#
# The cdfs/ `_xs` / `_ys` variants are deliberately NOT staged: they are
# inputs to the offline `generate_cdfs.py` builder, and nothing on the
# training path opens them.
#
# Every leg is one-directional (local -> cluster) and none uses --delete.
set -euo pipefail

CLUSTER_HOST="gpu2"
CLUSTER_STORAGE="/storage/agrp/rotemdo"
BEAM="proton/aluminum"

DRY=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY="--dry-run"; echo ">> DRY RUN -- nothing will be written";;
    --beam=*)  BEAM="${arg#--beam=}";;
    *) echo "usage: $0 [--dry-run] [--beam=proton/aluminum|proton/iron|proton/beryllium]" >&2; exit 2;;
  esac
done
case "$BEAM" in
  proton/aluminum|proton/iron|proton/beryllium) ;;
  *) echo "unknown --beam '$BEAM'" >&2; exit 2;;
esac

# storage/ lives in the git common directory's checkout, which `paths.py`
# resolves the same way, so this script finds the same tree the code reads.
LOCAL_STORAGE="$(git rev-parse --path-format=absolute --git-common-dir)"
LOCAL_STORAGE="$(dirname "${LOCAL_STORAGE}")/storage"
if [[ ! -d "${LOCAL_STORAGE}" ]]; then
  echo "!! local storage tree not found at ${LOCAL_STORAGE}" >&2
  exit 1
fi
echo ">> local storage: ${LOCAL_STORAGE}"

copy_leg() {  # <label> <relative dir> [file...]
  local label="$1"; shift
  local rel="$1"; shift
  local src="${LOCAL_STORAGE}/${rel}"
  local dst="${CLUSTER_HOST}:${CLUSTER_STORAGE}/${rel}"

  if [[ ! -d "${src}" ]]; then
    echo "!! missing locally: ${src}" >&2
    exit 1
  fi

  echo ">> ${label}"
  ssh "${CLUSTER_HOST}" "mkdir -p ${CLUSTER_STORAGE}/${rel}"
  if [[ $# -eq 0 ]]; then
    rsync -avh --progress ${DRY} "${src}/" "${dst}/"
  else
    local files=()
    for f in "$@"; do files+=("${src}/${f}"); done
    rsync -avh --progress ${DRY} "${files[@]}" "${dst}/"
  fi
}

copy_leg "training datasets" \
  "resources/${BEAM}/datasets"

copy_leg "continuous Y-scaler state" \
  "resources/${BEAM}/scalers"

copy_leg "analytic mean/std table" \
  "resources/${BEAM}/tables"

copy_leg "theoretical CDFs for the physics penalty" \
  "resources/${BEAM}/cdfs" \
  continuous_cdfs.pkl secondaries_cdfs.pkl

copy_leg "GEANT4 init tables" \
  "resources/${BEAM}/init_tables"

echo
echo ">> staged. verify with the submitters' own preflight:"
echo "     ./deploy_and_train.sh continuous --dry-run"
echo "     ./deploy_and_train.sh secondary  --dry-run"
