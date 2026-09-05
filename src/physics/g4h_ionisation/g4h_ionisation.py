"""The ionisation process for a charged particle stepped through matter.

`G4hIonisation` implements a three-phase `G4VProcess`: `compute_step_limit`
proposes the next step length, `along_step_do_it` accumulates the
continuous energy loss over that step, and `post_step_do_it` samples a
discrete secondary energy loss and the primary's resulting scattering
angle. Every number it produces comes from a locally built `InitTableSet`
(dE/dx, range, inverse-range and ionisation cross-section, built from
documented GEANT4 models). The length stage is an analytic Monte Carlo
sampler; the continuous and secondary stages are WGAN-GP generators.

Two design choices worth noting:

* `theta` is computed from energy-momentum conservation with `atan2`, not
  sampled by a network; see `physics/g4h_ionisation/kinematics.py` for the
  numerical reason `atan2` is used rather than `acos`.
* There is no restricted-dE/dx hook for a second energy-loss process: this
  is the only process stepping the particle, so there is nothing else to
  fold into the along-step mean.
"""
import torch
from typing import Optional

import common.units as U
from common.enums import SecondaryFeats
from common.run_context import PARTICLE
from physics.definitions.physical_constants import electron_mass_eV
from physics.g4h_ionisation.kinematics import compute_primary_theta

from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import GeantConstants
from physics.g4_init_tables.interpolator import g4pv_inverse_range_eval
from physics.g4_init_tables.runtime import InitTableSet
# compute_zero_prob is a closed-form G4UniversalFluctuation expression over
# material constants (no table lookup), so it does not go through
# InitTableSet -- it calls interp_kernels' JIT kernel and constants
# directly. Module-qualified on purpose: this module's own
# _PRIMARY_MASS_EV/_ELECTRON_MASS_EV are deliberately a different mass pair
# (see the comment below), so the zero-prob constants must come from
# interp_kernels itself rather than being re-derived here.
from physics.g4h_ionisation import interp_kernels

from physics.interfaces.generator_interface import GeneratorInterface
from physics.process import G4VProcess

G4CONSTS = GeantConstants()

# Module-level floats so the compiled post_step_do_it graph embeds them as
# bare constants rather than reading attributes per step. Both come from
# physics.definitions, via common.run_context for the active particle --
# the sole constant set this process uses.
_PRIMARY_MASS_EV: float = PARTICLE.mass_eV
_ELECTRON_MASS_EV: float = electron_mass_eV


# --- fluctuation-model helpers ----------------------------------------------
def _average_loss_fp64(tables: InitTableSet, kinetic_energy: torch.Tensor,
                       step_length: torch.Tensor) -> torch.Tensor:
    """G4 mean restricted energy loss over a step [eV].

    dE/dx * L for short steps; above the linear-loss limit, the
    range-difference form E - E_inv(range(E) - L) (the fluctuation-mean
    path of G4VEnergyLossProcess::AlongStepDoIt).
    """
    min_kin = G4CONSTS.min_kin_energy
    ke = kinetic_energy.to(torch.float64)
    step_cm = step_length.to(torch.float64) * U.m2cm

    average_loss = tables.dedx(ke) * step_cm  # eV

    long_step_mask = average_loss > ke * G4CONSTS.linear_loss_limit

    f_range = torch.clamp(tables.range(ke), min=0.0)  # cm
    scale_factor = torch.where(
        ke < min_kin,
        torch.sqrt(ke / min_kin),
        torch.ones_like(ke),
    )
    f_range = f_range * scale_factor

    x_remaining = f_range - step_cm  # cm
    inv = tables.inverse_range
    scaled_energy = g4pv_inverse_range_eval(
        inv.bin, inv.data, inv.sec_deriv, min_kin, x_remaining,
    )

    de = ke - scaled_energy
    use_range = long_step_mask & (de > 0.0)
    return torch.where(use_range, de, average_loss)


def _sec_gen_prob_fp64(tables: InitTableSet, pre_step_ke: torch.Tensor,
                       post_step_ke: torch.Tensor) -> torch.Tensor:
    """Acceptance probability of G4's integral approach.

    Fresh sigma(E_post) over the cached preStepLambda
    (tables.sigma_ion_biased -- the fEmOnePeak branch of
    G4VEnergyLossProcess::ComputeLambdaForScaledEnergy, v11.2.0).
    mfpKinEnergy resets after every discrete interaction (PostStepDoIt),
    so this stateless form matches a G4 reference dump's logged preStepLambda
    row-for-row.
    """
    pre_sigma = tables.sigma_ion_biased(pre_step_ke)
    post_sigma = tables.sigma_ion(post_step_ke.to(torch.float64))
    ratio = torch.where(pre_sigma > 0.0, post_sigma / pre_sigma,
                        torch.zeros_like(post_sigma))
    return torch.clamp(ratio, min=0.0, max=1.0)


class G4hIonisation(G4VProcess):
    competes_post_step = True

    def __init__(self,
                 length_generator: GeneratorInterface,
                 continuous_generator: GeneratorInterface,
                 secondary_generator: GeneratorInterface,
                 device: str = "cpu"):
        super().__init__("G4hIonisation")
        self.length_generator = length_generator
        self.continuous_generator = continuous_generator
        self.secondary_generator = secondary_generator

        self._compiled = False

        # Device-resident latch for the along-step NaN guard. Created BEFORE
        # `to()` so it exists for the whole lifetime of the object: a direct
        # `along_step_do_it(...)` call (unit tests) needs no `run()` around it,
        # it just accumulates into an already-zeroed flag which
        # `check_nan_guard()` reads whenever the caller chooses.
        self._nan_seen = torch.zeros((), dtype=torch.bool, device=device)

        self.tables = InitTableSet.load(device)
        self.to(device)

    def to(self, device: str, dtype: torch.dtype = None):
        # Rebind rather than mutate: this runs before any graph is traced, and
        # a cross-device `.to()` cannot be in-place anyway. Every later
        # reset uses `zero_()` so the object identity the compiled graph
        # captured stays stable.
        self._nan_seen = self._nan_seen.to(device)

        self.length_generator.to(device)
        self.continuous_generator.to(device)
        self.secondary_generator.to(device)

        for gen in (self.length_generator, self.continuous_generator, self.secondary_generator):
            if hasattr(gen, "x_scaler") and gen.x_scaler is not None:
                if dtype is not None and hasattr(gen.x_scaler, "to_dtype"):
                    gen.x_scaler.to_dtype(dtype)
                if hasattr(gen.x_scaler, "to"):
                    gen.x_scaler.to(device)
            if hasattr(gen, "y_scaler") and gen.y_scaler is not None:
                if dtype is not None and hasattr(gen.y_scaler, "to_dtype"):
                    gen.y_scaler.to_dtype(dtype)
                if hasattr(gen.y_scaler, "to"):
                    gen.y_scaler.to(device)

        if dtype is not None:
            self.length_generator.to(dtype)
            self.continuous_generator.to(dtype)
            self.secondary_generator.to(dtype)

        if device == 'cuda':
            # Opt-in whole-net fused CUDA kernel (PHINGAN_FUSED_MLP=1).
            # try_enable_fused_forward introspects each generator's
            # structure and falls back to its eager forward on any
            # structural mismatch, logging the refusal rather than
            # miscomputing.
            from physics.interfaces.fused_mlp import (
                fused_mlp_requested, try_enable_fused_forward)
            if fused_mlp_requested():
                for gen in (self.length_generator, self.continuous_generator,
                            self.secondary_generator):
                    try_enable_fused_forward(gen)

        if device == 'cuda' and not self._compiled:
            # Compile the bound process-phase methods (instance-attribute
            # shadows of the class methods). torch.compile(module) would not
            # help here, since call sites use predict(), which forwards
            # through the module's own forward rather than a compiled one -- so
            # the compile boundary sits at these phase methods instead: the
            # where-chains, zero-prob glue and rand draws fuse into the same
            # graphs as each generator's scale -> forward -> rescale
            # pipeline (dynamo traces through @torch.jit.script callees, so
            # the scripted G4 kernels -- e.g. compute_primary_theta -- inline
            # into the same graph rather than causing a graph break).
            # Each generator's own predict() stays uncompiled: nesting a
            # compiled callable inside a traced region forces graph breaks.
            # dynamic=True keeps the traced code shape-generic across the
            # power-of-two compaction buckets.
            #
            # The along-step entry compiled here is `_along_step_core`, NOT
            # the public `along_step_do_it`. The public method is a thin,
            # uncompiled wrapper that runs the NaN-guard accumulate after the
            # core returns -- see `along_step_do_it` below for why: compiling
            # the public entry would let the compiler fuse the guard's
            # `.any()` reduction into the phase body, pinning the whole
            # kernel's launch geometry to whatever shape first triggered
            # compilation.
            from common.config import TORCH_COMPILE_MODE
            for method in ("compute_step_limit", "_along_step_core", "post_step_do_it"):
                setattr(self, method, torch.compile(
                    getattr(self, method), dynamic=True, mode=TORCH_COMPILE_MODE))
            self._compiled = True

        self.tables.to(device)

        return self

    def compute_step_limit(self, kinetic_energy: torch.Tensor) -> torch.Tensor:
        return self.length_generator.predict(kinetic_energy)

    def _along_step_core(self, kinetic_energy: torch.Tensor, step_length: torch.Tensor, **kwargs) -> dict:
        # The compiled half of the along-step phase: all of the physics, and
        # nothing else. The NaN-guard accumulate deliberately does not live
        # here -- it runs in the uncompiled `along_step_do_it` wrapper below.
        #
        # Per-lane elementwise decision chain (torch.where over disjoint
        # masks), rather than data-dependent boolean indexing: a
        # `tensor[mask]` gather/scatter runs aten::nonzero, forcing a
        # GPU->CPU sync in the hot loop. The three masks (full_deposit /
        # is_deterministic / is_zero_cnt) are pairwise disjoint, so the
        # where-chain is equivalent to applying them as sequential indexed
        # assignments in this same order.
        continuous_input = torch.cat([kinetic_energy.unsqueeze(1), step_length.unsqueeze(1)], dim=1)
        e_cnt = self.continuous_generator.predict(continuous_input)

        average_loss = self.compute_average_loss(kinetic_energy, step_length)
        # Lanes whose mean loss exceeds their energy deposit everything
        # (killed within the step).
        full_deposit = average_loss > kinetic_energy

        # A lane counts as deterministic -- no fluctuation, no network
        # sample -- when EITHER this process's own mean-loss threshold says
        # so, OR the continuous y-scaler's mean/std lookup table says so.
        # The two predicates are evaluated over different tables (this one
        # over the built init tables; the lookup over its own analytic
        # mean/std table), and the lookup is a nearest-cell gather with no
        # interpolation, so a lane can sit exactly on the boundary where the
        # lookup's cell is still NaN while this process's own average_loss
        # has already crossed the threshold. ORing the lookup's NaN mask
        # into is_deterministic means the two predicates can never disagree
        # on such a lane: whatever the lookup declares deterministic is
        # absorbed here too, so a NaN cell in the lookup never reaches the
        # network or the guard below.
        #
        # average_loss is the physically correct value on these lanes,
        # since it is exactly what the deterministic branch of the
        # fluctuation model returns, so absorbing more lanes into this mask
        # changes no other lane's outcome. This is deliberately not
        # `isnan(e_cnt)`: a NaN the network itself emits must still reach
        # the run-end guard rather than being silently replaced (see
        # `along_step_do_it` below).
        lookup_is_nan = getattr(
            getattr(self.continuous_generator, "y_scaler", None),
            "lookup_is_nan", None)
        if lookup_is_nan is None:
            # This generator's y-scaler carries no mean/std lookup (e.g. an
            # ablation with no continuous z-space), so there is no NaN
            # region to absorb. Python-level branch: resolved once at trace
            # time, never a device-side condition.
            table_deterministic = torch.zeros_like(full_deposit)
        else:
            table_deterministic = lookup_is_nan(continuous_input)

        is_deterministic = ~full_deposit & (
            (average_loss < G4CONSTS.min_energy_loss) | table_deterministic)

        # The generator outputs a continuous density (trained on the
        # nonzero data only), so the discrete P(loss=0) atom of the
        # fluctuation model is injected here. A generator that already
        # samples the zero-loss atom itself (samples_zero_loss=True) must
        # skip this, or the atom is double-counted and the small-loss
        # region is depleted by the extra factor of (1 - z).
        if not getattr(self.continuous_generator, "samples_zero_loss", False):
            zero_prob = self.compute_zero_prob(average_loss, kinetic_energy)
            u = torch.rand(kinetic_energy.shape[0], device=kinetic_energy.device)
            is_zero_cnt = ~full_deposit & ~is_deterministic & (u < zero_prob)
            e_cnt = torch.where(is_zero_cnt, torch.zeros_like(e_cnt), e_cnt)

        # .to(e_cnt.dtype) keeps the return dtype fixed regardless of the
        # operand dtypes: torch.where promotes to the common dtype of its
        # operands, so casting the replacement value first pins the result
        # to e_cnt's own dtype.
        e_cnt = torch.where(is_deterministic, average_loss.to(e_cnt.dtype), e_cnt)
        e_cnt = torch.where(full_deposit, kinetic_energy.to(e_cnt.dtype), e_cnt)

        kill_particles = (kinetic_energy.to(e_cnt.dtype) - e_cnt) < 0.
        e_cnt = torch.where(kill_particles, kinetic_energy.to(e_cnt.dtype), e_cnt)

        return {"e_cnt": e_cnt}

    def along_step_do_it(self, kinetic_energy: torch.Tensor, step_length: torch.Tensor, **kwargs) -> dict:
        """Public along-step entry: the compiled core, then the NaN guard.

        This wrapper is never compiled, so `SteppingManager.step()` (plain
        Python) never forces it to trace, and its NaN-guard accumulate
        always runs outside the compiled `_along_step_core` graph.
        """
        out = self._along_step_core(kinetic_energy, step_length, **kwargs)
        e_cnt = out["e_cnt"]

        # The continuous y-scaler's mean/std lookup stores NaN over the
        # region its own builder calls deterministic -- a contiguous
        # small-step prefix of every kinetic-energy row -- rather than
        # fabricating a placeholder standard deviation there. Those lanes
        # are absorbed by the `is_deterministic` / `full_deposit`
        # where-chain in the core, which ORs in the lookup's own NaN mask,
        # so the absorbed set contains every cell the table declares
        # deterministic by construction and the two can never disagree.
        # What remains reachable here is a NaN from somewhere else entirely
        # -- the network itself, or `average_loss` -- which is exactly what
        # this guard should stay loud about. `kill_particles` above cannot
        # catch a NaN, since any comparison against NaN is False.
        #
        # WHY THE ACCUMULATE IS OUT HERE AND NOT IN THE CORE. Folding this
        # `.any()` reduction into the compiled core lets the compiler fuse
        # it together with the entire phase body into one kernel whose
        # launch geometry is baked from the shape that first triggered
        # compilation and never revisited afterward. Hoisting the
        # accumulate out of the compiled scope leaves the phase math to
        # fuse into ordinary pointwise kernels and costs one extra, small
        # reduction launch per step -- not a synchronisation.
        #
        # ACCUMULATE UNCONDITIONALLY, READ ONCE. The detection itself runs
        # on every device and in both the eager and compiled arms -- there
        # is no device or tracing condition on it. What is deferred is the
        # HOST read: `|=` into a device-resident 0-d bool tensor is a pure
        # device-side op, while `bool(...)`/`.item()` on it would force a
        # host synchronisation. `check_nan_guard` below does that read once
        # per run, from the caller that owns the step loop, outside any
        # compiled region.
        #
        # WHY THE READ CANNOT BE PER-STEP. A per-step host read of a
        # data-dependent CUDA reduction forces a synchronisation on every
        # step, inflating the cost of whichever arm performs it for reasons
        # that have nothing to do with the physics being compared.
        #
        # WHAT THIS IS. A diagnostic, not physics: it changes no value the
        # pipeline produces, only whether a NaN escape is loud or silent.
        # The OR makes it lossless: a single bad lane at any step of a run
        # latches the flag until `reset_nan_guard()` clears it.
        self._nan_seen |= torch.isnan(e_cnt).any()

        return out

    # --- along-step NaN guard: reset / read ---------------------------------
    # Split out of `along_step_do_it` so the only host read happens once per
    # run. `TrackSimulator.run()` calls `reset_nan_guard()` before its step
    # loop and `check_nan_guard()` after it, via `SteppingManager`. A caller
    # that drives `along_step_do_it` directly (unit tests) calls these itself.

    def reset_nan_guard(self) -> None:
        """Clear the latch, in place.

        `.zero_()` rather than rebinding to a fresh tensor: rebinding was
        tested under `aot_eager` and mutations still landed (dynamo re-guards
        the attribute on read), so a captured-graph-identity hazard is not
        the operative reason and is not claimed here. `.zero_()` is still
        preferred because it avoids invalidating the compiled graph's guards
        (a rebind is a new object identity, which would force a recompile)
        and avoids a per-run allocation.
        """
        self._nan_seen.zero_()

    def check_nan_guard(self) -> None:
        """Read the latch once, on the host, and refuse if it is set.

        This is the ONLY sync the guard costs, and it must stay outside any
        compiled region -- call it from the run loop's owner, never from a
        phase method.
        """
        # Not a bare `assert`: asserts are stripped under `python -O`, and this
        # read is now the ONLY place the accumulated flag is ever consulted --
        # an `-O` build would make the whole mechanism silently inert, which is
        # precisely the failure mode it exists to eliminate.
        if bool(self._nan_seen):
            raise RuntimeError(
                "NaN e_cnt escaped along_step_do_it. is_deterministic ORs "
                "in the mean/std lookup's own NaN mask, so a lane that "
                "merely gathered the table's deterministic NaN cannot be "
                "the cause -- that set is absorbed by construction. Suspect "
                "a NaN from elsewhere: the continuous network's own "
                "output, or compute_average_loss (an out-of-range spline, "
                "or a non-finite kinetic energy or step length reaching "
                "the phase). The flag is latched across the whole run, so "
                "the offending step is not necessarily the last one."
            )

    def post_step_do_it(self, kinetic_energy: torch.Tensor, step_length: torch.Tensor,
                        e_cnt: torch.Tensor, won: Optional[torch.Tensor] = None) -> dict:
        post_cnt_energy = kinetic_energy - e_cnt
        secondary = self.secondary_generator.predict(post_cnt_energy)
        # int(): FX codegen cannot serialize an IntEnum tensor index inside
        # a compiled phase graph.
        #
        # The generator emits only e_sec. theta is not read from its
        # output: theta is a deterministic function of e_sec (see below), so
        # sampling it would spend network capacity reproducing a closed-form
        # relationship instead of learning the energy-loss distribution.
        e_sec = secondary[:, int(SecondaryFeats.E_SEC_IDX)]

        secondary_fractions = self.sec_gen_prob(kinetic_energy, post_cnt_energy)
        n_secondaries = (torch.rand(kinetic_energy.shape[0],
                                    device=kinetic_energy.device,
                                    dtype=e_sec.dtype) < secondary_fractions).to(e_sec.dtype)
        e_sec = e_sec * n_secondaries

        post_step_energy = post_cnt_energy - e_sec
        e_sec = torch.where(post_step_energy < 0., torch.zeros_like(e_sec), e_sec)

        if won is not None:
            e_sec = torch.where(won, e_sec, torch.zeros_like(e_sec))

        # theta LAST, from the FINAL e_sec: every zeroing above therefore
        # propagates into theta on its own (e_sec = 0 => pe = 0 =>
        # p1_perp = 0 => atan2(0, p0) = 0), which is why there are no
        # separate theta masks. Moving this call above any of the maskings
        # would silently reintroduce the need for them.
        theta = compute_primary_theta(
            post_cnt_energy, e_sec, _PRIMARY_MASS_EV, _ELECTRON_MASS_EV)

        return {"theta": theta, "e_sec": e_sec}

    def compute_average_loss(self, kinetic_energy: torch.Tensor,
                             step_length: torch.Tensor) -> torch.Tensor:
        return _average_loss_fp64(
            self.tables, kinetic_energy, step_length).to(kinetic_energy.dtype)

    @staticmethod
    def compute_zero_prob(mean_loss: torch.Tensor,
                          kinetic_energy: torch.Tensor) -> torch.Tensor:
        return interp_kernels._compute_zero_prob_jit(
            mean_loss, kinetic_energy,
            interp_kernels._PROTON_MASS_EV, interp_kernels._ELECTRON_MASS_EV,
            interp_kernels._MATERIAL_TC, interp_kernels._MATERIAL_I,
            interp_kernels._IONIZATION_RATE, interp_kernels._FW, interp_kernels._A0,
            interp_kernels._NMAX_CONT, interp_kernels._E0, interp_kernels._N_MIN_BOHR,
            interp_kernels._LOG_W1, interp_kernels._SCALING,
        )

    def sec_gen_prob(self, pre_step_ke: torch.Tensor,
                     post_step_ke: torch.Tensor) -> torch.Tensor:
        return _sec_gen_prob_fp64(
            self.tables, pre_step_ke, post_step_ke).to(pre_step_ke.dtype)
