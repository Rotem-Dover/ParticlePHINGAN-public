"""
Neural network generators for the secondary energy-loss and scattering-angle stage.
SecondaryGenerator is the physics-informed WGAN-GP generator conditioned on kinetic
energy, outputting E_SEC. NoPhysSecondaryGenerator is a BatchNorm-based
variant serving as a non-physics baseline. (Theta is reconstructed deterministically
downstream via energy-momentum conservation, not sampled.)
"""
import torch

from torch import nn

from common.config import (N_EMBEDDING_SEC, N_NOISE_SEC, N_SECONDARY_FEATURES,
                           N_NEURON_MULTIPLIER_SEC, N_NEURON_MULTIPLIER_SEC_NOPHYS,
                           SECONDARY_OUTPUT_ACTIVATION, SECONDARY_STRETCH_EPS)
from common.enums import SecondaryFeats

from physics.interfaces.generator_interface import GeneratorInterface

# Terminal-activation kinds SecondaryGenerator can be built with, and the
# integer each is recorded as in the state dict (`output_activation_id`
# buffer):
#
# * "hardtanh" -- Hardtanh(0, 1). Its zero gradient outside [0, 1] lets the
#   generator accumulate density at the z == 0 / z == 1 boundaries.
# * "sigmoid" -- has no dead-gradient region, but the WGAN gradient through
#   sigma'(x) = z(1 - z) vanishes in the tails, so the trained density
#   cannot approach the endpoints closely.
# * "stretched_sigmoid" -- StretchedSigmoid maps a reachable interior
#   ceiling onto z = 1, recovering an endpoint plain Sigmoid cannot reach.
# * "hardtanh_open_top" -- keeps Hardtanh's lower clip at 0 in both training
#   and eval, but applies the upper clip only in eval, letting the trained
#   density run to (and past) z = 1 during training.
# * "hardtanh_reflect_top" -- open_top's training behaviour with a
#   reflecting eval rule: spill past z = 1 is folded back (z -> 2 - z)
#   instead of piling onto the z == 1 atom.
#
# Ids are append-only: a checkpoint written before a given id existed has no
# key and is treated as hardtanh (0).
OUTPUT_ACTIVATION_IDS = {"hardtanh": 0, "sigmoid": 1, "stretched_sigmoid": 2,
                         "hardtanh_open_top": 3, "hardtanh_reflect_top": 4}
_OUTPUT_ACTIVATION_BY_ID = {v: k for k, v in OUTPUT_ACTIVATION_IDS.items()}


class StretchedSigmoid(nn.Module):
    """z = (1 + 2 eps) * sigmoid(x) - eps.

    Training mode: unclamped, so the critic sees mass the generator puts past
    the endpoints and pushes it back with a live gradient (no dead region as
    Hardtanh has, no unreachable endpoint as plain Sigmoid has). Eval mode:
    clamped to [0, 1] -- z = 1 is reached at a finite logit
    ln((1 + eps) / eps), unlike plain Sigmoid where it is only approached in
    the limit. Whatever mass still overflows at inference lands on the
    endpoint atoms, which `benchmark.system_vs_g4.secondary_endpoint_atoms`
    measures.
    """
    def __init__(self, eps: float):
        super().__init__()
        eps = float(eps)
        if not eps > 0:
            raise ValueError(f"StretchedSigmoid eps must be > 0, got {eps!r}")
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (1.0 + 2.0 * self.eps) * torch.sigmoid(x) - self.eps
        return y if self.training else y.clamp(0.0, 1.0)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class HardtanhOpenTop(nn.Module):
    """Hardtanh(0, 1) with the upper clip applied in eval only.

    Training mode: `clamp(min=0)` -- Hardtanh's own bottom half, with the
    same dead gradient below 0; no upper bound, so the critic sees
    mass past z = 1 and the generator can put density right up to the
    endpoint without a sigmoid-style ceiling. Eval mode: `clamp(0, 1)`; the
    spill past 1 lands on the z == 1 atom, which
    `benchmark.system_vs_g4.secondary_endpoint_atoms` measures.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0.0) if self.training else x.clamp(0.0, 1.0)


class HardtanhReflectTop(nn.Module):
    """hardtanh_open_top's training behaviour with a reflecting eval rule.

    Training mode: `clamp(min=0)` -- byte-identical to HardtanhOpenTop, so an
    open_top checkpoint IS a reflect_top checkpoint (the load guard admits
    that one direction). Eval mode: spill past 1 is reflected, z -> 2 - z,
    then clamped to [0, 1] -- branch-free, no extra network cost.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x.clamp(min=0.0)
        return torch.where(x > 1.0, 2.0 - x, x).clamp(0.0, 1.0)


def _output_activation_module(name: str, stretch_eps: float = SECONDARY_STRETCH_EPS) -> nn.Module:
    if name == "hardtanh":
        return nn.Hardtanh(min_val=0, max_val=1, inplace=True)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "stretched_sigmoid":
        return StretchedSigmoid(stretch_eps)
    if name == "hardtanh_open_top":
        return HardtanhOpenTop()
    if name == "hardtanh_reflect_top":
        return HardtanhReflectTop()
    raise ValueError(f"output_activation must be one of "
                     f"{sorted(OUTPUT_ACTIVATION_IDS)}, got {name!r}")


class NoPhysSecondaryGenerator(GeneratorInterface):
    """
    Non-physics secondary generator for secondary energy loss.
    Uses BatchNorm1d instead of LayerNorm, InstanceNorm1d in embedding.
    Trunk width is N_NEURON_MULTIPLIER_SEC_NOPHYS (32) -- the ablation's own
    checkpoint contract, decoupled from the physics net's N_NEURON_MULTIPLIER_SEC.
    Input: kinetic_energy (1 feature)
    Output: E_SEC (1 feature). Scattering angle theta is reconstructed deterministically
    downstream via energy-momentum conservation, not sampled.
    """
    def __init__(
            self,
            n_embedding:   int  = N_EMBEDDING_SEC,
            n_noise:       int  = N_NOISE_SEC,
    ):
        super(NoPhysSecondaryGenerator, self).__init__()
        self.n_noise     = n_noise
        self.n_embedding = n_embedding
        self.embed       = self.embedding()
        self.model = nn.Sequential(
            self._block(
                n_noise + self.n_embedding,
                N_NEURON_MULTIPLIER_SEC_NOPHYS * N_SECONDARY_FEATURES,
                bias=False
            ),
            self._block(
                N_NEURON_MULTIPLIER_SEC_NOPHYS * N_SECONDARY_FEATURES,
                N_NEURON_MULTIPLIER_SEC_NOPHYS * N_SECONDARY_FEATURES,
                bias=False
            ),
            self._last_block()
        )

    @staticmethod
    def _block(
            in_channels:  int,
            out_channels: int,
            bias:         bool = True
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.BatchNorm1d(out_channels, affine=True),
            nn.ReLU()
        )

    @staticmethod
    def _last_block() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(N_NEURON_MULTIPLIER_SEC_NOPHYS * N_SECONDARY_FEATURES, N_NEURON_MULTIPLIER_SEC_NOPHYS // 2 * N_SECONDARY_FEATURES, bias=False),
            nn.BatchNorm1d(N_NEURON_MULTIPLIER_SEC_NOPHYS // 2 * N_SECONDARY_FEATURES, affine=True),
            nn.ReLU(),
            nn.Linear(
                N_NEURON_MULTIPLIER_SEC_NOPHYS // 2 * N_SECONDARY_FEATURES,
                N_SECONDARY_FEATURES,
                bias=True
            ),
        )

    def embedding(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(1, self.n_embedding, bias=True),
            nn.InstanceNorm1d(1, affine=True),
            nn.ReLU(),
            nn.Linear(self.n_embedding, self.n_embedding, bias=True),
            nn.Flatten()
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        latent_vector = torch.randn(labels.shape[0], self.n_noise, device=self.device, dtype=labels.dtype)
        # InstanceNorm1d needs 3D input [N, C, L], so reshape to [N, 1, 1]
        labels = self.embed(labels.view(-1, 1, 1))
        gen_input = torch.cat([labels, latent_vector], dim=-1)

        output = self.model(gen_input)

        # ReLU on e_sec (sharp cut on minimum energy). The indexing is kept
        # column-explicit rather than whole-tensor so a future added column
        # does not silently inherit the rectifier.
        output[:, [SecondaryFeats.E_SEC_IDX]] = \
            nn.ReLU()(output[:, [SecondaryFeats.E_SEC_IDX]])

        return output


class SecondaryGenerator(GeneratorInterface):
    """
    Generator for the secondary (delta-ray) energy loss.
    Input: kinetic_energy (1 feature)
    Output: [E_SEC] (N_SECONDARY_FEATURES features)

    The width arguments all default to the `common.config` constants — a bare
    `SecondaryGenerator()` is the checkpoint contract, unchanged. Explicit
    widths exist for loading checkpoints trained at a different width than
    the current config constants; see `ctor_kwargs_from_state_dict`.

    `output_activation` ("hardtanh" | "sigmoid" | "stretched_sigmoid" |
    "hardtanh_open_top" | "hardtanh_reflect_top", default the config
    constant) picks the terminal layer -- "hardtanh_reflect_top" trains
    identically to "hardtanh_open_top" but at eval reflects any z>1 spill
    back as 2-z before clamping to [0, 1] -- and `output_stretch_eps`
    (default the config constant; meaningful only for "stretched_sigmoid",
    stored as 0.0 otherwise) its stretch. All variants have identical
    parameter shapes, so the choice is recorded in the state dict as the
    `output_activation_id` and `output_stretch_eps` buffers and checked at
    load: a checkpoint of one kind (or stretch) refuses to load into a net
    of another (`_load_from_state_dict`).
    """
    def __init__(
            self,
            n_embedding:   int  = N_EMBEDDING_SEC,
            n_noise:       int  = N_NOISE_SEC,
            n_trunk:       int  = N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES,
            n_half:        int  = (N_NEURON_MULTIPLIER_SEC // 2) * N_SECONDARY_FEATURES,
            n_features:    int  = N_SECONDARY_FEATURES,
            output_activation: str = SECONDARY_OUTPUT_ACTIVATION,
            output_stretch_eps: float = SECONDARY_STRETCH_EPS,
    ):
        super(SecondaryGenerator, self).__init__()
        if output_activation not in OUTPUT_ACTIVATION_IDS:
            raise ValueError(f"output_activation must be one of "
                             f"{sorted(OUTPUT_ACTIVATION_IDS)}, got {output_activation!r}")
        self.output_activation = output_activation
        # eps is part of the contract only for the stretched kind; the other
        # kinds record 0.0 so their state dicts stay comparable across kinds.
        self.stretch_eps = (float(output_stretch_eps)
                            if output_activation == "stretched_sigmoid" else 0.0)
        self.n_noise     = n_noise
        self.n_embedding = n_embedding
        self.embed       = self.embedding()
        self.model = nn.Sequential(
            self._block(n_noise + self.n_embedding, n_trunk, bias=False),
            self._block(n_trunk, n_trunk, bias=False),
            self._last_block(n_trunk, n_half, n_features, output_activation,
                             self.stretch_eps),
        )
        # Persistent so it lands in every checkpoint written from now on;
        # int64 0-d so it survives torch.save/load and dtype casts untouched.
        self.register_buffer(
            "output_activation_id",
            torch.tensor(OUTPUT_ACTIVATION_IDS[output_activation], dtype=torch.int64))
        self.register_buffer(
            "output_stretch_eps",
            torch.tensor(self.stretch_eps, dtype=torch.float64))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "output_activation_id"
        if key not in state_dict:
            # Checkpoint saved before this buffer existed: no key, and
            # such a checkpoint was always trained with Hardtanh.
            state_dict[key] = torch.tensor(OUTPUT_ACTIVATION_IDS["hardtanh"],
                                           dtype=torch.int64)
        stored = int(state_dict[key])
        want = OUTPUT_ACTIVATION_IDS[self.output_activation]
        if (stored == OUTPUT_ACTIVATION_IDS["hardtanh_open_top"]
                and want == OUTPUT_ACTIVATION_IDS["hardtanh_reflect_top"]):
            # One-directional compat: open_top and reflect_top are the SAME
            # net at training time (identical clamp(min=0) forward), so an
            # open_top checkpoint loads into a reflect net -- only the eval
            # spill rule differs, and that is exactly what the reflect kind
            # was pinned to change. The reverse direction stays refused: an
            # open_top net loading a reflect checkpoint would silently
            # restore the atom-producing clamp the pick rejected.
            state_dict[key] = torch.tensor(want, dtype=torch.int64)
            stored = want
        if stored != want:
            error_msgs.append(
                f"checkpoint output_activation is "
                f"{_OUTPUT_ACTIVATION_BY_ID.get(stored, stored)!r} but this "
                f"SecondaryGenerator was built with {self.output_activation!r}; "
                f"the parameter shapes are identical so nothing else would "
                f"catch this -- set common.config.SECONDARY_OUTPUT_ACTIVATION "
                f"(or pass output_activation=) to match the checkpoint")
        ekey = prefix + "output_stretch_eps"
        if ekey not in state_dict:
            # Checkpoint saved before this buffer existed: no key. Hardtanh
            # and plain sigmoid checkpoints record 0.0 here anyway.
            state_dict[ekey] = torch.tensor(0.0, dtype=torch.float64)
        stored_eps = float(state_dict[ekey])
        if stored == want and abs(stored_eps - self.stretch_eps) > 1e-12:
            error_msgs.append(
                f"checkpoint output_stretch_eps is {stored_eps!r} but this "
                f"SecondaryGenerator was built with {self.stretch_eps!r}; "
                f"set common.config.SECONDARY_STRETCH_EPS (or pass "
                f"output_stretch_eps=) to match the checkpoint")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def ctor_kwargs_from_state_dict(cls, state_dict: dict) -> dict:
        """Recover the constructor widths from a generator state dict.

        Every width is readable off the layer shapes:
        `embed.0.weight` is `(n_embedding, 1)`, `model.0.0.weight` is
        `(n_trunk, n_noise + n_embedding)`, `model.2.0.weight` is
        `(n_half, n_trunk)` and `model.2.3.weight` is `(n_features, n_half)`;
        the terminal activation is the `output_activation_id` buffer (absent
        => hardtanh) and its stretch the `output_stretch_eps` buffer
        (absent => 0.0).
        Consumed by `from_checkpoint(..., infer_arch=True)` — a tooling path;
        the runtime keeps the strict config-shaped default.
        """
        n_embedding = int(state_dict["embed.0.weight"].shape[0])
        n_trunk, first_in = (int(d) for d in state_dict["model.0.0.weight"].shape)
        act_id = int(state_dict.get("output_activation_id",
                                    OUTPUT_ACTIVATION_IDS["hardtanh"]))
        return {
            "n_embedding": n_embedding,
            "n_noise":     first_in - n_embedding,
            "n_trunk":     n_trunk,
            "n_half":      int(state_dict["model.2.0.weight"].shape[0]),
            "n_features":  int(state_dict["model.2.3.weight"].shape[0]),
            "output_activation": _OUTPUT_ACTIVATION_BY_ID[act_id],
            "output_stretch_eps": float(state_dict.get("output_stretch_eps", 0.0)),
        }

    @staticmethod
    def _block(
            in_channels:  int,
            out_channels: int,
            bias:         bool = True
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

    @staticmethod
    def _last_block(n_trunk: int, n_half: int, n_features: int,
                    output_activation: str = SECONDARY_OUTPUT_ACTIVATION,
                    output_stretch_eps: float = SECONDARY_STRETCH_EPS) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(n_trunk, n_half, bias=False),
            nn.LayerNorm(n_half),
            nn.ReLU(),
            nn.Linear(n_half, n_features, bias=True),
            _output_activation_module(output_activation, output_stretch_eps),
        )

    def embedding(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(1, self.n_embedding, bias=True),
            nn.LayerNorm(self.n_embedding),
            nn.ReLU(),
            nn.Linear(self.n_embedding, self.n_embedding, bias=True),
            nn.Flatten()
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        latent_vector = torch.randn(labels.shape[0], self.n_noise, device=self.device, dtype=labels.dtype)
        labels = self.embed(labels.unsqueeze(1))
        gen_input = torch.cat([labels, latent_vector], dim=-1)

        return self.model(gen_input)
