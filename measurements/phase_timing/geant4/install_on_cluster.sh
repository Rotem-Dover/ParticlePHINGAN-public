#!/bin/bash
# One-time: install the hIoni timing shim into the CLUSTER StepsGenerator copy
# and rebuild TestEm5. Run ON the cluster after copying this directory there:
#
#   (laptop)  scp -r measurements/phase_timing/geant4 \
#                 gpu2:/srv01/agrp/rotemdo/StepsGenerator/timing_phases
#   (cluster) bash /srv01/agrp/rotemdo/StepsGenerator/timing_phases/install_on_cluster.sh
#
# The cluster copy of TestEm5 has /runAction/disableSaving and its sources
# differ slightly from the laptop copy, which is why the wiring is an
# anchored, idempotent python edit rather than a git patch. It asserts on
# every anchor -- a drifted file fails loud instead of half-patching.
set -eo pipefail   # no -u: the LCG setup.sh reads unset vars
. /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh
SRC=/srv01/agrp/rotemdo/StepsGenerator
HERE=$(cd "$(dirname "$0")" && pwd)
python3 "$HERE/patch/apply_timing_patch.py" "$SRC"
cd "$SRC/build"
cmake . > cmake_reconf_timing.log 2>&1      # GLOB picks up the two new .cc
make TestEm5 -j4 2>&1 | tail -5
echo "rebuilt: $SRC/build/TestEm5"
mkdir -p "$SRC/timing_phases"
if [ "$HERE" != "$SRC/timing_phases" ]; then
  cp "$HERE/macro_template.mac" "$HERE/phase_timing_node.sh" "$SRC/timing_phases/"
fi
echo "smoke (10 events, timing on):"
sed -e "s/__N_EVENTS__/10/" -e "s/__TIMING__/true/" "$HERE/macro_template.mac" > "$SRC/timing_phases/smoke.mac"
(cd "$SRC/timing_phases" && "$SRC/build/TestEm5" smoke.mac | grep -E "^\[[0-9]\]===|beamOn wall|of hIoni only")
