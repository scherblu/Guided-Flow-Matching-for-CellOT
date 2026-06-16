from .base import BaseDistribution
from ..utils.misc import get_device
import torch

class SpiralDistribution(BaseDistribution):
    def __init__(self, noise: float = 0.1, energy_fn: str = None, energy_kwargs: dict = None):
        super().__init__(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
        self.noise = noise

    def get_default_energy_fn_name(self) -> str:
        return 'origin_quadratic_distance'

    def origin_quadratic_distance(self, x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
        return x.square().sum(dim=-1) * scale

    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = get_device()
        theta = torch.linspace(0, 4 * torch.pi, batch_size, device=device)
        r = torch.linspace(0, 4, batch_size, device=device)
        x1 = r * torch.cos(theta) + self.noise * torch.rand(batch_size, device=device)
        x2 = r * torch.sin(theta) + self.noise * torch.rand(batch_size, device=device)
        return torch.stack([x1, x2], dim=-1) / 3

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Log probability for spiral is not implemented yet")
