"""Every artifact path lives under one of two roots, and both are created on
import."""
import common.paths as paths

ROOTED = {
    "TRACKS_DIR", "INIT_TABLES_DIR", "CDFS_DIR", "VALIDATION_DIR",
    "G4_REFERENCE_DIR", "TABLES_DIR", "SCALERS_DIR", "NOPHYS_CNT_SCALERS_DIR",
    "DATASETS_DIR", "FIGURES_DIR", "PATH_TO_GEANT4_DATA",
    "PATH_TO_GEANT4_DATA_SEED42", "PATH_INIT_DEDX", "PATH_INIT_RANGE",
    "PATH_INIT_SCALED_KIN", "PATH_INIT_LAMBDA_ION", "PATH_TO_CONTINUOUS_CDFS",
    "PATH_TO_SECONDARIES_CDFS", "PATH_G4_DUMP_SEC_GEN_FRAC",
    "PATH_ANALYTIC_MEAN_STD_TABLE",
}
RUN_ROOTED = {"CNT_CKPT", "SEC_CKPT", "NOPHYS_CNT_CKPT", "NOPHYS_SEC_CKPT"}


def test_resource_constants_sit_under_the_material_dir():
    for name in ROOTED:
        p = getattr(paths, name)
        assert paths.MATERIAL_DIR in p.parents or p == paths.MATERIAL_DIR, name


def test_material_dir_is_resources_root_particle_material():
    assert paths.MATERIAL_DIR == paths.RESOURCES_ROOT / paths._PARTICLE_DIR / paths._MATERIAL_DIR
    assert paths.RESOURCES_ROOT == paths.STORAGE / "resources"


def test_checkpoints_sit_under_runs_dir():
    assert paths.RUNS_DIR == paths.STORAGE / "runs" / paths._PARTICLE_DIR / paths._MATERIAL_DIR
    for name in RUN_ROOTED:
        assert paths.RUNS_DIR in getattr(paths, name).parents, name


def test_directories_exist_after_import():
    # The module creates them on import; never delete anything under
    # storage/ from a test.
    for name in ("TRACKS_DIR", "INIT_TABLES_DIR", "CDFS_DIR", "TABLES_DIR",
                 "SCALERS_DIR", "NOPHYS_CNT_SCALERS_DIR", "DATASETS_DIR",
                 "FIGURES_DIR", "RUNS_DIR", "BENCHMARKS_OUTPUTS"):
        assert getattr(paths, name).is_dir(), name
