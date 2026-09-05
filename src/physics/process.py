import torch


class G4VProcess:
    # True for processes that compete for the discrete post-step vertex
    # (per-event winner masking in SteppingManager); MSC/continuous-only
    # processes leave it False.
    competes_post_step: bool = False

    def __init__(self, name: str):
        self.name = name

    def compute_step_limit(self, kinetic_energy: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def along_step_do_it(self, kinetic_energy: torch.Tensor, step_length: torch.Tensor, **kwargs) -> dict:
        return {}

    def post_step_do_it(self, kinetic_energy: torch.Tensor, step_length: torch.Tensor, **kwargs) -> dict:
        return {}

