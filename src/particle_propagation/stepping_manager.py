"""
Orchestrates a single simulation step by chaining a physics process's three
GEANT4-mirroring phases: (1) each process proposes a step-length limit and
the minimum wins, (2) along-step accumulates continuous energy loss, and
(3) post-step accumulates secondary energy loss and the scattering angle.
"""
import logging
import torch
from typing import Self, Optional, Type
from pathlib import Path

from common.config import DEVICE
from physics.interfaces.generator_interface import GeneratorInterface
from physics.process import G4VProcess

log = logging.getLogger(__name__)


def _summarise_load_failure(err: BaseException) -> str:
    """
    Compress a neural-network load failure into a single short sentence so the
    fallback warning is readable. PyTorch's `load_state_dict` mismatch dumps
    can be dozens of lines long; we surface the cause and (for state-dict
    mismatches) a compact "N tensors mismatched, e.g. <key>: ckpt <shape>
    vs model <shape>" hint.
    """
    if isinstance(err, FileNotFoundError):
        return f"checkpoint or scaler file missing ({err.filename or err})"
    if isinstance(err, KeyError):
        return f"config refers to unknown class or scaler name {err}"
    text = str(err)
    if "size mismatch" in text:
        import re
        mismatches = re.findall(
            r"size mismatch for (\S+?):.*?shape torch\.Size\(\[([^\]]+)\]\).*?shape in current model is torch\.Size\(\[([^\]]+)\]\)",
            text,
        )
        if mismatches:
            key, ckpt_shape, model_shape = mismatches[0]
            return (
                f"state-dict shape mismatch ({len(mismatches)} tensor(s); "
                f"e.g. {key}: checkpoint [{ckpt_shape}] vs current model [{model_shape}]) "
                "- network architecture changed since this checkpoint was saved"
            )
        return "state-dict mismatch (architecture changed since checkpoint was saved)"
    if "Missing key(s)" in text or "Unexpected key(s)" in text:
        return "state-dict key mismatch (architecture changed since checkpoint was saved)"
    first_line = text.strip().splitlines()[0] if text else ""
    return f"{type(err).__name__}: {first_line}" if first_line else type(err).__name__


class SteppingManager:
    def __init__(self, processes: list[G4VProcess], device: str = DEVICE):
        self.device = device
        self.processes = processes
        self.to(device)

    def __call__(
            self,
            kinetic_energy: torch.Tensor,
    ) -> torch.Tensor:
        output_dtype = kinetic_energy.dtype

        # Step 1: propose step lengths across all processes
        process_limits = []
        for process in self.processes:
            limit = process.compute_step_limit(kinetic_energy)
            process_limits.append(limit)

        # Take the minimum over all proposed processes
        if len(process_limits) == 1:
            step_length = process_limits[0]
        else:
            step_length = torch.min(torch.stack(process_limits, dim=0), dim=0).values
        step_length = step_length.to(dtype=output_dtype)

        # Winner masks among post-step competitors (None when <2 competitors:
        # with a single discrete process, it is always eligible).
        comp_idx = [i for i, p in enumerate(self.processes)
                    if getattr(p, "competes_post_step", False)]
        won_masks: dict[int, Optional[torch.Tensor]] = {i: None for i in comp_idx}
        if len(comp_idx) > 1:
            comp_limits = torch.stack([process_limits[i] for i in comp_idx], dim=0)
            winner = torch.argmin(comp_limits, dim=0)          # per-event
            # Compare against step_length in the SAME dtype it was cast to above:
            # a raw-dtype vs output_dtype-cast mismatch could round step_length
            # below the exact per-competitor min and silently suppress a vertex.
            realized = comp_limits.to(dtype=output_dtype).min(dim=0).values <= step_length
            for slot, i in enumerate(comp_idx):
                won_masks[i] = (winner == slot) & realized

        # Step 2: along-step do-it
        e_cnt_total = torch.zeros_like(kinetic_energy)
        # Nothing shortens the true step into a shorter geometric one, so
        # geom_step_m is a value copy of step_length and stays bit-identical
        # to it by construction (not by numerical coincidence).
        geom_step_m = step_length.clone()

        for process in self.processes:
            along_out = process.along_step_do_it(kinetic_energy, step_length, e_cnt=e_cnt_total)
            if 'e_cnt' in along_out:
                e_cnt_total += along_out['e_cnt'].to(dtype=output_dtype)

        # Step 3: post-step do-it
        e_sec_total = torch.zeros_like(kinetic_energy)
        theta_total = torch.zeros_like(kinetic_energy)

        for i, process in enumerate(self.processes):
            kwargs = {"e_cnt": e_cnt_total}
            if i in won_masks and won_masks[i] is not None:
                kwargs["won"] = won_masks[i]
            post_out = process.post_step_do_it(kinetic_energy, step_length, **kwargs)
            if 'e_sec' in post_out:
                e_sec_total += post_out['e_sec'].to(dtype=output_dtype)
            if 'theta' in post_out:
                theta_total += post_out['theta'].to(dtype=output_dtype)

        return torch.stack([
            step_length, theta_total, e_cnt_total, e_sec_total, geom_step_m,
        ], dim=1)

    def to(self, device: str, dtype: torch.dtype = None):
        for process in self.processes:
            if hasattr(process, 'to'):
                process.to(device, dtype=dtype)
        return self

    # --- per-run diagnostic latches ----------------------------------------
    # Pure fan-out, no state of its own. A process that keeps a device-side
    # diagnostic latch (only G4hIonisation's along-step NaN guard does)
    # exposes reset/check; the run loop owner drives them once per run.
    # `hasattr` rather than a required interface: G4VProcess subclasses
    # without a latch simply have nothing to reset.

    def reset_nan_guard(self) -> None:
        for process in self.processes:
            if hasattr(process, 'reset_nan_guard'):
                process.reset_nan_guard()

    def check_nan_guard(self) -> None:
        for process in self.processes:
            if hasattr(process, 'check_nan_guard'):
                process.check_nan_guard()

    @classmethod
    def from_config(
            cls,
            device: str,
            processes: list[dict],
    ) -> Self:
        """
        Build a SteppingManager from a list of process configurations.

        Each entry in `processes` is a dict with a `'name'` key naming the
        physics process (only `'g4h_ionisation'` is registered) plus
        per-stage generator/scaler subconfigs. Presence of a process in the
        list controls whether it runs — there is no separate enable flag.

        Each generator subconfig has shape:
            {'name': <ClassName>, 'checkpoint': <path, optional>}
        and each scaler entry is the bare class name as a string.
        Generators with a `'checkpoint'` key are loaded from the Lightning
        checkpoint; otherwise they are default-constructed (used by the
        physics-MC backends).
        """
        built = []
        for proc_cfg in processes:
            name = proc_cfg['name']
            if name == 'g4h_ionisation':
                built.append(cls._build_g4h_ionisation(proc_cfg, device))
            else:
                raise ValueError(f"Unknown process name: {name!r}")
        return cls(processes=built, device=device)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_generator(
            gen_cfg: dict,
            classes: dict,
            x_scaler=None,
            y_scaler=None,
            fallback_class: Optional[Type[GeneratorInterface]] = None,
    ):
        """
        Build a generator from config. If `fallback_class` is provided and the
        neural-net path fails for any reason (missing/corrupt checkpoint,
        state-dict mismatch after a refactor, missing/incompatible scaler
        pickle), fall back to a default-constructed physics-MC sampler with
        no scalers (the MC samplers work in physical units and bypass
        @scaled_forward).
        """
        gen_class = classes[gen_cfg['name']]
        checkpoint = gen_cfg.get('checkpoint')
        try:
            if checkpoint is not None:
                # infer_arch: this dict-config path is consumed only by the
                # playground's sim_runner, whose checkpoint browser must load
                # checkpoints trained at architecture widths other than the
                # ones in config.py. The runtime preset builder
                # (benchmark/track_simulator_config.py) calls from_checkpoint
                # directly, without this flag, and stays strict.
                gen = gen_class.from_checkpoint(Path(checkpoint),
                                                infer_arch=True)
            else:
                gen = gen_class()
            gen.set_scalers(x_scaler, y_scaler)
            return gen
        except Exception as err:
            if fallback_class is None:
                raise
            reason = _summarise_load_failure(err)
            log.warning(
                "Could not load neural network %s%s -> %s. "
                "Falling back to analytical physics-MC sampler %s "
                "(predictions will use the Geant4-style explicit sampler instead of the trained network).",
                gen_class.__name__,
                f" from checkpoint '{checkpoint}'" if checkpoint else "",
                reason,
                fallback_class.__name__,
            )
            gen = fallback_class()
            gen.set_scalers(None, None)
            return gen

    @classmethod
    def _build_g4h_ionisation(cls, cfg: dict, device: str):
        """Dict-config path for the ionisation process.

        Wires the step-length sampler plus a continuous and a secondary
        generator, each either the trained WGAN-GP network or the analytic
        physics-MC sampler the playground's "MC baseline" selects. There is
        no trained length network, and cnt/sec have no `fallback_class`: a
        bad checkpoint or scaler must raise, not silently swap in a
        different physics model.
        """
        from physics.g4h_ionisation.g4h_ionisation import G4hIonisation
        from physics.g4h_ionisation.generators.length_generator.physics_mc      import PhysicsLengthGenerator
        from physics.g4h_ionisation.generators.continuous_generator.neural_net  import ContinuousGenerator
        from physics.g4h_ionisation.generators.secondary_generator.neural_net   import SecondaryGenerator
        from physics.g4h_ionisation.generators.continuous_generator.physics_mc  import PhysicsContinuousGenerator
        from physics.g4h_ionisation.generators.secondary_generator.physics_mc   import PhysicsSecondaryGenerator
        from physics.g4h_ionisation.generators.continuous_generator.scaler      import ContinuousXScaler, ContinuousYScaler
        from physics.g4h_ionisation.generators.secondary_generator.scaler       import SecondaryXScaler, SecondaryYScaler

        scalers = {
            'ContinuousXScaler': ContinuousXScaler, 'ContinuousYScaler': ContinuousYScaler,
            'SecondaryXScaler': SecondaryXScaler, 'SecondaryYScaler': SecondaryYScaler,
        }
        length_classes = {c.__name__: c for c in (PhysicsLengthGenerator,)}
        cnt_classes    = {c.__name__: c for c in (ContinuousGenerator, PhysicsContinuousGenerator)}
        sec_classes    = {c.__name__: c for c in (SecondaryGenerator, PhysicsSecondaryGenerator)}

        # .load(), not the bare constructor: a scaler carries no state after
        # __init__ and would die on .to(device) with all-None fields. The
        # classmethod reads the portable state off disk.
        def _scaler(name): return scalers[name].load() if name else None

        length_gen = cls._build_generator(
            cfg['length_generator'], length_classes,
            _scaler(cfg.get('length_x_scaler')), _scaler(cfg.get('length_y_scaler')),
            fallback_class=PhysicsLengthGenerator,
        )
        continuous_gen = cls._build_generator(
            cfg['continuous_generator'], cnt_classes,
            _scaler(cfg.get('continuous_x_scaler')), _scaler(cfg.get('continuous_y_scaler')),
        )
        secondary_gen = cls._build_generator(
            cfg['secondary_generator'], sec_classes,
            _scaler(cfg.get('secondary_x_scaler')), _scaler(cfg.get('secondary_y_scaler')),
        )
        return G4hIonisation(
            length_generator=length_gen,
            continuous_generator=continuous_gen,
            secondary_generator=secondary_gen,
            device=device,
        )


if __name__ == "__main__":
    from benchmark.track_simulator_config import build_stepping_manager
    from common.config import BEAM_ENERGY_eV

    step_generator = build_stepping_manager("phin_gan", device='cpu')

    kinetic_energy = torch.tensor([BEAM_ENERGY_eV]*1_000, device="cpu")
    output = step_generator(kinetic_energy)
    print(output)
