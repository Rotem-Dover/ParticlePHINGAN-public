"""The PHINGAN_BEAM contract: registry-only, fail-loud, default aluminum."""
import os
import re
import subprocess
import sys
from pathlib import Path

from physics.definitions.beam import PRESETS

REPO = Path(__file__).resolve().parents[2]


def _run(code: str, beam: str | None):
    env = dict(os.environ)
    env.pop("PHINGAN_BEAM", None)
    if beam is not None:
        env["PHINGAN_BEAM"] = beam
    env["PYTHONPATH"] = "src"
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, cwd=REPO)


def test_default_is_aluminum():
    out = _run("from common.run_context import ACTIVE_NAME, TARGET_MATERIAL;"
               "print(ACTIVE_NAME, TARGET_MATERIAL.name)", beam=None)
    assert out.returncode == 0, out.stderr
    assert "proton_in_aluminum_100MeV Aluminum" in out.stdout


def test_iron_selects_iron_and_repoints_paths():
    out = _run("from common import paths; from common.run_context import TARGET_MATERIAL;"
               "print(TARGET_MATERIAL.name); print(paths.TRACKS_DIR)",
               beam="proton_in_iron_100MeV")
    assert out.returncode == 0, out.stderr
    assert "Iron" in out.stdout
    assert "/proton/iron/" in out.stdout.replace("\\\\", "/")


def test_beryllium_selects_beryllium():
    out = _run("from common.run_context import TARGET_MATERIAL, GRID_STEP_LEN_MAX_m;"
               "print(TARGET_MATERIAL.name, GRID_STEP_LEN_MAX_m)",
               beam="proton_in_beryllium_100MeV")
    assert out.returncode == 0, out.stderr
    assert "Beryllium 0.0002" in out.stdout


def test_unknown_beam_fails_loud():
    out = _run("import common.run_context", beam="proton_in_unobtainium_100MeV")
    assert out.returncode != 0
    assert "not a registered beam" in out.stderr


def _case_arm_tokens(script_text: str, case_var: str) -> set[str]:
    """Extract the pipe-separated tokens of the first case arm on `$<case_var>`
    in a shell script (i.e. the accepted-value list before its `);;` /
    `) ;;`), robust to whitespace but exact about the tokens themselves. This
    is a drift pin, not a shell parser: it deliberately fails loud (via the
    assert) rather than guessing if a script's case-block shape changes, so a
    future edit to either script is expected to update this helper or its
    caller, not silently pass.
    """
    m = re.search(
        r'case\s+"\$' + re.escape(case_var) + r'"\s+in\s*\n\s*(.+?)\)\s*;;',
        script_text, re.DOTALL,
    )
    assert m, f'no `case "${case_var}" in` block found'
    return {tok.strip() for tok in m.group(1).split("|")}


def test_deploy_and_train_beam_case_matches_presets():
    """deploy_and_train.sh's `--beam` case block must accept exactly the
    preset names in physics.definitions.beam.PRESETS -- nothing hardcoded
    twice, nothing silently out of sync."""
    text = (REPO / "deploy_and_train.sh").read_text()
    tokens = _case_arm_tokens(text, "BEAM")
    assert tokens == set(PRESETS)


def test_stage_artifacts_beam_case_matches_presets():
    """scripts/stage_artifacts.sh's `--beam=` case block must accept
    exactly `proton/<material>` for each material PRESETS actually has a
    beam for -- same drift risk as the deploy script, different token
    format (`proton/<material>`, not the preset name)."""
    text = (REPO / "scripts" / "stage_artifacts.sh").read_text()
    tokens = _case_arm_tokens(text, "BEAM")
    expected = {f"proton/{cfg.material.name.lower()}" for cfg in PRESETS.values()}
    assert tokens == expected
