"""Per-epoch training-diagnostic PNGs (FIGURES_DIR/<Stage>GAN/version_<run>/
<kind>/<epoch>.png) are OFF by default: the same figures already reach
TensorBoard via add_figure(), and a dense checkpoint cadence writing PNGs
on top risks the cluster's storage inode quota. PHINGAN_SAVE_FIGS=1
re-enables them."""
import importlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize("stage,cls", [("continuous", "ContinuousGAN"),
                                       ("secondary", "SecondaryGAN")])
def test_savefig_is_gated_by_env(monkeypatch, tmp_path, stage, cls):
    mod = importlib.import_module(
        f"physics.g4h_ionisation.generators.train_networks.{stage}_generator.wgan")
    # savefig only reads self.current_epoch; call it unbound on a stub so no
    # Lightning module (scalers, CDFs, GPU) has to be constructed.
    gan = type("Stub", (), {"current_epoch": 7})()
    fig = plt.figure()
    monkeypatch.delenv("PHINGAN_SAVE_FIGS", raising=False)
    getattr(mod, cls).savefig(gan, fig, tmp_path / "off")
    assert not (tmp_path / "off").exists()
    monkeypatch.setenv("PHINGAN_SAVE_FIGS", "1")
    getattr(mod, cls).savefig(gan, fig, tmp_path / "on")
    assert (tmp_path / "on" / "7.png").is_file()
    plt.close(fig)
