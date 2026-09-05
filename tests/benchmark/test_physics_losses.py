"""Fig E.1 (physics regularization along training) -- the checkpoint walk,
the incremental cache and the arm wiring against the pinned REVISED_* runs."""
import numpy as np
import pytest

from benchmark.system_vs_g4 import physics_losses as pl
from benchmark.track_simulator_config import PRESETS
from common.paths import (NOPHYS_CNT_CKPT, CNT_CKPT, NOPHYS_SEC_CKPT, SEC_CKPT, SCALERS_DIR)


def _touch_ckpts(d, epochs, last=True):
    ck = d / "checkpoints"
    ck.mkdir(parents=True)
    for e in epochs:
        (ck / f"epoch={e}.ckpt").touch()
    if last:
        (ck / "last.ckpt").touch()
    (ck / "notes.txt").touch()
    return d


def test_list_checkpoints_sorted_capped_no_last(tmp_path):
    run = _touch_ckpts(tmp_path / "run", [299, 99, 199, 40099])
    got = pl.list_checkpoints(run, max_epoch=40000)
    assert [e for e, _ in got] == [99, 199, 299]
    assert all(p.name == f"epoch={e}.ckpt" for e, p in got)
    assert [e for e, _ in pl.list_checkpoints(run, max_epoch=None)] == [99, 199, 299, 40099]


def test_cache_roundtrip_and_resume_skips_scored(tmp_path):
    cache = tmp_path / "c.npz"
    assert pl.load_cache(cache) == {}
    pl.save_cache(cache, {99: 0.5, 199: 0.4})
    assert pl.load_cache(cache) == {99: 0.5, 199: 0.4}
    run = _touch_ckpts(tmp_path / "run", [99, 199, 299])
    todo = pl.pending_checkpoints(run, pl.load_cache(cache), max_epoch=None)
    assert [e for e, _ in todo] == [299]


def test_arms_wire_the_pinned_runs_and_preset_classes():
    arms = {a.key: a for a in pl.ARMS}
    assert set(arms) == {"phingan_cnt", "phingan_sec", "gan_cnt", "gan_sec"}
    pins = {"phingan_cnt": CNT_CKPT, "phingan_sec": SEC_CKPT,
            "gan_cnt": NOPHYS_CNT_CKPT, "gan_sec": NOPHYS_SEC_CKPT}
    for key, ckpt in pins.items():
        a = arms[key]
        assert a.run_dir == ckpt.parent.parent
        assert a.pinned_epoch == int(ckpt.stem.split("=")[1])
    # same classes the simulator presets load -- one source of truth
    for preset, prefix in (("phin_gan", "phingan"), ("gan", "gan")):
        for stage in ("cnt", "sec"):
            gen_cls, _, y_cls, state_dir = PRESETS[preset][stage]
            a = arms[f"{prefix}_{stage}"]
            assert a.gen_cls is gen_cls and a.y_scaler_cls is y_cls
            assert a.state_dir == state_dir


def test_plot_draws_four_curves_and_optional_pick_markers():
    import matplotlib
    matplotlib.use("Agg")
    curves = {a.key: (np.array([100, a.pinned_epoch, 99999]), np.array([0.3, 0.2, 0.1])) for a in pl.ARMS}
    fig = pl.plot_physics_losses(curves, mark_picks=False)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 4
    assert ax.get_yscale() == "log"
    fig2 = pl.plot_physics_losses(curves, mark_picks=True)
    assert len(fig2.axes[0].get_lines()) == 8   # 4 curves + 4 single-point markers


def test_import_csv_seeds_caches_without_clobbering(tmp_path):
    csv = tmp_path / "curves.csv"
    csv.write_text("arm,epoch,physics_regularization\n"
                   "phingan_cnt,99,3.0e-02\nphingan_cnt,199,2.0e-02\ngan_sec,99,1.0e-01\n")
    arm = {a.key: a for a in pl.ARMS}["phingan_cnt"]
    pl.save_cache(pl.cache_path(tmp_path, arm), {99: 0.5})       # pre-existing entry wins
    pl.import_csv(csv, tmp_path)
    assert pl.load_cache(pl.cache_path(tmp_path, arm)) == {99: 0.5, 199: 0.02}
    gan_sec = {a.key: a for a in pl.ARMS}["gan_sec"]
    assert pl.load_cache(pl.cache_path(tmp_path, gan_sec)) == {99: 0.1}
    (tmp_path / "bad.csv").write_text("x,y\n")
    with pytest.raises(ValueError):
        pl.import_csv(tmp_path / "bad.csv", tmp_path)


def test_arm_init_seeds_match_the_runs():
    seeds = {a.key: a.init_seed for a in pl.ARMS}
    assert seeds == {"gan_cnt": 42, "gan_sec": 42, "phingan_cnt": 42, "phingan_sec": 2}


def test_initial_generator_is_reproducible_and_seed_dependent():
    """seed_everything + bare constructor is what the trainers do; the same
    seed must rebuild the same weights, a different seed different ones."""
    import torch
    from physics.interfaces.scaler_interface import ScalerInterface
    arm = {a.key: a for a in pl.ARMS}["phingan_cnt"]

    class _NoScaler(ScalerInterface):
        def to(self, device): return self
        def scale(self, x, **kw): return x
        def rescale(self, y, **kw): return y
    dummy = _NoScaler()
    a = pl.initial_generator(arm, "cpu", dummy, dummy)
    b = pl.initial_generator(arm, "cpu", dummy, dummy)
    c = pl.initial_generator(arm._replace(init_seed=arm.init_seed + 1), "cpu", dummy, dummy)
    differs = False
    for (ka, va), (kb, vb), (kc, vc) in zip(a.state_dict().items(), b.state_dict().items(), c.state_dict().items()):
        assert ka == kb == kc
        assert torch.equal(va, vb)
        differs |= not torch.equal(va, vc)
    assert differs      # norm gains start at ones for every seed; the linear weights must not
    assert not any(p.requires_grad for p in a.parameters())
