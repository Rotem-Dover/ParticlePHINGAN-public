import benchmark.runtime_profiling.profile_runtime as pr
from benchmark.runtime_profiling.simulator_builders import build_eager_simulator


def test_stage_list_is_the_three_stages():
    assert pr.PAPER_STAGES == ("length", "continuous", "secondary")


def test_generator_report_covers_exactly_the_three_stages():
    _, sm = build_eager_simulator("phin_gan", device="cpu")
    report = pr._generator_report(sm)
    assert set(report) == set(pr.PAPER_STAGES)


def test_generator_report_names_the_pinned_checkpoints():
    from common.paths import CNT_CKPT, SEC_CKPT
    _, sm = build_eager_simulator("phin_gan", device="cpu")
    report = pr._generator_report(sm)
    assert str(CNT_CKPT) == str(report["continuous"]["checkpoint"])
    assert str(SEC_CKPT) == str(report["secondary"]["checkpoint"])


def test_length_stage_reports_no_checkpoint():
    """The length stage is the analytic MC sampler -- no network, no ckpt."""
    _, sm = build_eager_simulator("phin_gan", device="cpu")
    report = pr._generator_report(sm)
    assert report["length"]["checkpoint"] is None
    assert report["length"]["class"] == "PhysicsLengthGenerator"
