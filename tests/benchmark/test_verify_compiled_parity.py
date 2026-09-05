import benchmark.runtime_profiling.verify_compiled_parity as vcp


def test_module_does_not_import_g4_init_tables():
    """verify_compiled_parity drives the simulator through the built
    presets and never reaches into the table builder directly."""
    import inspect
    assert "g4_init_tables" not in inspect.getsource(vcp)


def test_compare_runs_on_cpu_and_reports_the_device():
    r = vcp.compare_compiled_vs_eager("phin_gan", device="cpu",
                                      n_events=2, n_steps=100, seed=1)
    assert r["device"] == "cpu"
    assert r["preset"] == "phin_gan"
    assert "max_abs_dz" in r and "mean_abs_dz" in r
