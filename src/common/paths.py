"""Filesystem layout of every artifact the package reads or writes.

Two roots under ``STORAGE``:

* ``resources/<particle>/<material>/`` — GEANT4 truth, built tables, CDFs,
  scaler states, training datasets, trainer figures (``MATERIAL_DIR``).
* ``runs/<particle>/<material>/`` — training runs: checkpoints, TensorBoard
  event files and ``meta.json`` (``RUNS_DIR``).

``STORAGE`` is ``/storage/agrp/rotemdo`` on the cluster (detected by the
presence of ``/srv01/agrp/rotemdo``) and ``<repo>/storage`` otherwise; in a
linked git worktree it resolves to the main checkout's ``storage/``.

The material subtree follows the beam selected by ``common.run_context``
(``PHINGAN_BEAM``), so every constant here retargets with the beam.
"""
from pathlib import Path

from common.run_context import PARTICLE, TARGET_MATERIAL, BEAM_ENERGY_TAG


def _main_worktree_storage(project_dir: Path) -> Path | None:
    """Resolve the main working tree's ``storage/`` when running in a worktree.

    ``storage/`` is gitignored and only ever populated in the main checkout
    (~16 GB, never duplicated per worktree). In a *linked* git worktree the
    ``.git`` entry is a file (``gitdir: <main>/.git/worktrees/<name>``) rather
    than a directory; we parse it to find the shared ``.git`` and from there the
    main working tree root, returning ``<main_root>/storage``. Returns ``None``
    for a normal checkout (``.git`` is a directory) or a non-repository, so the
    caller keeps the default local ``storage/``.
    """
    git_path = project_dir / ".git"
    if not git_path.is_file():
        return None  # normal checkout (.git is a dir) or not a git repo
    try:
        content = git_path.read_text().strip()
    except OSError:
        return None
    if not content.startswith("gitdir:"):
        return None
    gitdir = Path(content[len("gitdir:"):].strip())
    # gitdir == <main>/.git/worktrees/<name>; `commondir` back-links to the
    # shared .git (usually "../.."), whose parent is the main working tree.
    commondir_file = gitdir / "commondir"
    if commondir_file.is_file():
        try:
            common_git = (gitdir / commondir_file.read_text().strip()).resolve()
        except OSError:
            common_git = gitdir.parent.parent
    else:
        common_git = gitdir.parent.parent
    return common_git.parent / "storage"


# ------------------------------- Roots ------------------------------- #
ON_CLUSTER = True
PROJECT_DIR = Path("/srv01/agrp/rotemdo/")
STORAGE     = Path("/storage/agrp/rotemdo")
if not PROJECT_DIR.exists():
    PROJECT_DIR = Path(__file__).parent.parent.parent
    STORAGE = PROJECT_DIR / "storage"
    ON_CLUSTER = False
    # storage/ is gitignored and populated only in the main checkout; a
    # linked git worktree reads that copy rather than an empty local tree.
    _shared_storage = _main_worktree_storage(PROJECT_DIR)
    if _shared_storage is not None and _shared_storage.exists():
        STORAGE = _shared_storage

assert PROJECT_DIR.exists(), f"Project directory was not found. Got {PROJECT_DIR}"

_PARTICLE_DIR = PARTICLE.name.lower()          # "proton"
_MATERIAL_DIR = TARGET_MATERIAL.name.lower()   # "aluminum", "iron", "beryllium"

RESOURCES_ROOT = STORAGE / "resources"
MATERIAL_DIR   = RESOURCES_ROOT / _PARTICLE_DIR / _MATERIAL_DIR
RUNS_DIR       = STORAGE / "runs" / _PARTICLE_DIR / _MATERIAL_DIR

# Figures and caches. The aluminium beam (the paper's) writes at the root;
# any other beam writes under its own material subdirectory so the referee
# materials' figures never overwrite the published set.
BENCHMARKS_ROOT = PROJECT_DIR / "artifacts" / "benchmark_outputs"   # beam-independent
BENCHMARKS_OUTPUTS = BENCHMARKS_ROOT
if _MATERIAL_DIR != "aluminum":
    BENCHMARKS_OUTPUTS = BENCHMARKS_ROOT / _MATERIAL_DIR
VERSION_NUMBER = 1   # tag on some benchmark figure filenames

# ------------------------------- GEANT4 truth ------------------------------- #
TRACKS_DIR = MATERIAL_DIR / "tracks"
PATH_TO_GEANT4_DATA        = TRACKS_DIR / f"{BEAM_ENERGY_TAG}_1k.root"
PATH_TO_GEANT4_DATA_SEED42 = TRACKS_DIR / f"{BEAM_ENERGY_TAG}_10k.root"

# Reference dumps read only by tests and diagnostics, never by the runtime.
VALIDATION_DIR   = MATERIAL_DIR / "validation_references"
G4_REFERENCE_DIR = VALIDATION_DIR / "g4_reference"
PATH_G4_DUMP_SEC_GEN_FRAC = VALIDATION_DIR / "ke_to_sec_gen_frac_100.bin"

# ------------------------------- Built tables ------------------------------- #
# Four GEANT4 initialisation tables built by `physics.g4_init_tables.build_all`
# from documented models; the runtime's table source (InitTableSet).
INIT_TABLES_DIR      = MATERIAL_DIR / "init_tables"
PATH_INIT_DEDX       = INIT_TABLES_DIR / "dedx.npz"
PATH_INIT_RANGE      = INIT_TABLES_DIR / "range.npz"
PATH_INIT_SCALED_KIN = INIT_TABLES_DIR / "scaled_kin_energy.npz"
PATH_INIT_LAMBDA_ION = INIT_TABLES_DIR / "lambda_ion.npz"

# The continuous-loss mean/std lookup, built analytically from
# `explicit_physics` by `build_analytic_mean_std_table`; ContinuousYScaler's
# state is derived from it by scripts/build_continuous_scaler_state.py.
TABLES_DIR = MATERIAL_DIR / "tables"
PATH_ANALYTIC_MEAN_STD_TABLE = TABLES_DIR / "continuous_mean_std_lookup_table_steps_5000_analytic.pkl"

# Theoretical CDFs for the physics penalty (generate_cdfs.py of each stage).
CDFS_DIR = MATERIAL_DIR / "cdfs"
PATH_CONTINUOUS_CDFS_XS  = CDFS_DIR / "continuous_cdfs_xs.pkl"
PATH_CONTINUOUS_CDFS_YS  = CDFS_DIR / "continuous_cdfs_ys.pkl"
PATH_SECONDARIES_CDFS_XS = CDFS_DIR / "secondaries_cdfs_xs.pkl"
PATH_SECONDARIES_CDFS_YS = CDFS_DIR / "secondaries_cdfs_ys.pkl"
PATH_TO_CONTINUOUS_CDFS  = CDFS_DIR / "continuous_cdfs.pkl"
PATH_TO_SECONDARIES_CDFS = CDFS_DIR / "secondaries_cdfs.pkl"

# ------------------------------- Scalers, datasets, figures ------------------------------- #
# <ScalerClassName>.state.pt, one per scaler, fitted by the training
# datamodules. `nophys_continuous/` holds the two states the no-physics
# ablation's continuous checkpoint was trained against (its ContinuousXScaler
# bounds differ from the current fit).
SCALERS_DIR            = MATERIAL_DIR / "scalers"
NOPHYS_CNT_SCALERS_DIR = SCALERS_DIR / "nophys_continuous"
DATASETS_DIR           = MATERIAL_DIR / "datasets"
FIGURES_DIR            = MATERIAL_DIR / "figures"   # trainer diagnostics (PHINGAN_SAVE_FIGS=1)

# ------------------------------- Pinned checkpoints ------------------------------- #
# Pinned by explicit path, not resolved by recency; each is the checkpoint the
# `phin_gan` / `gan` presets load (benchmark/track_simulator_config.py). One
# (run, epoch) pair per stage per material; every material's secondary run is
# the same `hardtanh_open_top` W=16 configuration, loaded as
# `hardtanh_reflect_top` (common/config.py). The picks were made with
# `benchmark.system_vs_g4.checkpoint_scan` (continuous: KS in the
# generator's scaled space over every saved epoch at two noise seeds, then
# fig 12's energy distance on the shortlist; secondary: a conditional
# chi-square of z given E, which catches the spectrum ripples KS averages
# away, then the tail fractions P(z > 0.99), P(z > 0.999) and the boundary
# atoms). The no-physics ablation was
# trained for aluminium only, so the `gan` preset exists for that beam alone.
_PHIN_GAN_PINS = {
    "aluminum": (
        ("2026-08-12-164426_continuous_aluminum_60281d7c67_e29e061b", 23499),
        ("2026-08-19-192831_secondary_s2_hardtanh_open_top_w16_aluminum_1eabc3aea3_ef3bec7d", 17999),
    ),
    "iron": (
        ("2026-08-10-205505_continuous_iron_79d03274be_f80bf0ea", 18899),
        ("2026-08-22-155334_secondary_s3_hardtanh_open_top_w16_iron_6885824c7d_6f298002", 29699),
    ),
    "beryllium": (
        ("2026-08-11-173628_continuous_beryllium_39936807d7_66579441", 37699),
        ("2026-08-22-155401_secondary_s3_hardtanh_open_top_w16_beryllium_4a01a3fce4_40873f3f", 56499),
    ),
}
(_CNT_RUN, _CNT_EPOCH), (_SEC_RUN, _SEC_EPOCH) = _PHIN_GAN_PINS[_MATERIAL_DIR]
CNT_CKPT = RUNS_DIR / "logger_cnt" / _CNT_RUN / "checkpoints" / f"epoch={_CNT_EPOCH}.ckpt"
SEC_CKPT = RUNS_DIR / "logger_sec" / _SEC_RUN / "checkpoints" / f"epoch={_SEC_EPOCH}.ckpt"
NOPHYS_CNT_CKPT = RUNS_DIR / "logger_cnt_nophys" / "version_2" / "checkpoints" / "epoch=1599.ckpt"
NOPHYS_SEC_CKPT = RUNS_DIR / "logger_sec_nophys" / "2026-08-14-111344_secondary_nophys_aluminum_ecc6089cfa_f2411d16" / "checkpoints" / "epoch=3199.ckpt"

for _d in (TRACKS_DIR, VALIDATION_DIR, G4_REFERENCE_DIR, INIT_TABLES_DIR,
           TABLES_DIR, CDFS_DIR, SCALERS_DIR, NOPHYS_CNT_SCALERS_DIR,
           DATASETS_DIR, FIGURES_DIR, RUNS_DIR, BENCHMARKS_OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)
