"""Convenience sampling over the phin_gan preset, for the figure scripts.

`generate_continuous` / `generate_secondary` are free functions taking the
`G4hIonisation` process explicitly -- `SteppingManager` carries no per-stage
convenience surface. The secondary output is `(N, 1)`: `SecondaryFeats` has
one column, `E_SEC_IDX`. Where a figure needs theta it is reconstructed
downstream from e_sec via energy-momentum conservation, the same way the
runtime does.
"""
from typing import Optional

import torch

from benchmark.track_simulator_config import build_stepping_manager
from common.enums import SecondaryFeats
from physics.g4h_ionisation.g4h_ionisation import G4CONSTS, G4hIonisation


def preset_ionisation(name: str, device: str = "cpu") -> G4hIonisation:
    """The named preset's single ionisation process."""
    (proc,) = build_stepping_manager(name, device=device).processes
    return proc


def phin_gan_ionisation(device: str = "cpu") -> G4hIonisation:
    """The phin_gan preset's single ionisation process."""
    return preset_ionisation("phin_gan", device)


def generate_continuous(
        proc: G4hIonisation,
        continuous_input: torch.Tensor,
        *,
        correct_zero: bool = True,
        correct_deterministic: bool = True,
        repeat_interleave: Optional[int] = None,
) -> torch.Tensor:
    if correct_zero and not correct_deterministic:
        raise ValueError(
            "Zero correction without deterministic correction can lead to "
            "unphysical results. Please enable deterministic correction "
            "when enabling zero correction.")
    e_cnt = proc.continuous_generator.predict(
        continuous_input, repeat_interleave=repeat_interleave)
    if repeat_interleave is not None:
        continuous_input = continuous_input.repeat_interleave(
            repeat_interleave, dim=0)

    kinetic_energy = continuous_input[:, 0]
    step_length = continuous_input[:, 1]

    if correct_deterministic:
        # Correcting complete deterministic losses
        alive_particles = torch.ones(
            kinetic_energy.shape[0], device=kinetic_energy.device,
            dtype=torch.bool)

        global_alive_indices = torch.where(alive_particles)[0]
        average_loss = proc.compute_average_loss(
            kinetic_energy[alive_particles], step_length[alive_particles])
        alive_kill = average_loss > kinetic_energy[alive_particles]
        e_cnt[global_alive_indices[alive_kill]] = \
            kinetic_energy[global_alive_indices[alive_kill]]
        alive_particles[global_alive_indices[alive_kill]] = False

        average_loss_surviving = average_loss[~alive_kill]

        is_deterministic = average_loss_surviving < G4CONSTS.min_energy_loss

        alive_idx = torch.where(alive_particles)[0]

        # Deterministic case
        deterministic_idx = alive_idx[is_deterministic]
        e_cnt[deterministic_idx] = average_loss_surviving[is_deterministic]

    if correct_zero:
        # Zero continuous loss case
        not_deterministic = ~is_deterministic
        not_deterministic_idx = alive_idx[not_deterministic]

        zero_prob = proc.compute_zero_prob(
            average_loss_surviving[not_deterministic],
            kinetic_energy[not_deterministic_idx])
        is_zero_cnt = torch.rand(
            zero_prob.shape[0],
            device=average_loss_surviving.device) < zero_prob

        zero_cnt_idx = not_deterministic_idx[is_zero_cnt]
        e_cnt[zero_cnt_idx] = 0.0

    # Kill slow particles
    post_cnt_energy = kinetic_energy - e_cnt
    kill_particles = post_cnt_energy < 0.
    e_cnt[kill_particles] = kinetic_energy[kill_particles]

    # The deterministic-region overwrite above (correct_deterministic) is
    # what replaces every NaN the analytic mean/std lookup gathers on its
    # ~29% NaN interior (see CLAUDE.md, "The continuous mean/std lookup").
    # Unlike the runtime's along_step_do_it, this offline sampler carries no
    # device-side NaN accumulator -- so a NaN reaching here would propagate
    # silently into a figure instead of raising. Fail loudly instead.
    assert torch.isfinite(e_cnt).all(), (
        "generate_continuous produced non-finite E_CNT -- the "
        "deterministic-region overwrite should have replaced every NaN "
        "gathered from the continuous mean/std lookup; investigate before "
        "trusting any figure built from this output")

    return e_cnt


def generate_secondary(
        proc: G4hIonisation,
        kinetic_energy: torch.Tensor,
        e_cnt: torch.Tensor,
        *,
        correct_zero: bool = True,
        repeat_interleave: Optional[int] = None,
) -> torch.Tensor:
    post_cnt_energy = kinetic_energy - e_cnt
    secondary = proc.secondary_generator.predict(
        post_cnt_energy, repeat_interleave=repeat_interleave)

    post_cnt_energy_expanded = post_cnt_energy.repeat_interleave(
        repeat_interleave, dim=0) \
        if repeat_interleave is not None else post_cnt_energy

    if correct_zero:
        ke_expanded = kinetic_energy.repeat_interleave(
            repeat_interleave, dim=0) \
            if repeat_interleave is not None else kinetic_energy

        secondary_fractions = proc.sec_gen_prob(
            ke_expanded, post_cnt_energy_expanded)
        n_secondaries = (torch.rand(
            ke_expanded.shape[0], device=ke_expanded.device,
            dtype=secondary.dtype) < secondary_fractions).to(secondary.dtype)
        secondary[:, SecondaryFeats.E_SEC_IDX] *= n_secondaries

    post_step_energy = post_cnt_energy_expanded \
        - secondary[:, SecondaryFeats.E_SEC_IDX]
    dont_make_secondary = post_step_energy < 0.
    secondary[dont_make_secondary, SecondaryFeats.E_SEC_IDX] = 0

    return secondary
