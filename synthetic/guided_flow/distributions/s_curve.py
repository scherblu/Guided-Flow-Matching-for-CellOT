from .base import BaseDistribution, EPS
from ..utils.misc import get_device
import torch
import numpy as np
import sklearn

class SCurveDistribution(BaseDistribution):
    def __init__(self, noise: float = 0.05, energy_fn: str = None, energy_kwargs: dict = None):
        super().__init__(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
        self.noise = noise

    def get_default_energy_fn_name(self) -> str:
        return 'diagonal_difference_distance'

    def diagonal_difference_distance(self, x: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
        """
        Allows grad.
        """
        return (x[:, 0] - x[:, 1]).abs() * scale

    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = get_device()
        X, _ = sklearn.datasets.make_s_curve(n_samples=batch_size, noise=self.noise) # (B, 2)
        x = torch.tensor(X[:, [0]], device=device, dtype=torch.float32)
        y = torch.tensor(X[:, [2]], device=device, dtype=torch.float32)
        return torch.cat([x, y / 2], dim=-1)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for s-curve is complicated, not implemented yet")
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for s-curve is complicated, not implemented yet")

    def __name__(self):
        return f'S-Curve'
